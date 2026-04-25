import asyncio
import logging
import time
from datetime import datetime
import numpy as np
import pandas as pd
from collections import deque
from core.historical_fetcher import HistoricalFetcher
from core.feature_engineer import AdvancedFeatureEngineer
from core.scheduler import MarketState

class LiveTradingEngine:
    """
    개별 종목에 대해 실시간 틱(Tick) 수신, 1분봉 병합(OHLCV), 웜업, 
    Feature Engineering 및 RL Agent 시그널 추론을 전담하는 엔진입니다.
    """
    def __init__(self, symbol: str, config_manager, order_manager, shared_agent):
        self.symbol = symbol
        self.config_manager = config_manager
        self.order_manager = order_manager
        self.agent = shared_agent
        self.logger = logging.getLogger(f"LiveEngine[{self.symbol}]")
        
        # 1분봉 버퍼 (MA20 등을 계산하기 위해 최소 40개 이상 유지, 넉넉히 100개)
        self.minute_buffer = deque(maxlen=100)
        
        # 현재 병합 중인 캔들 상태
        self.current_minute_str = None
        self.current_candle = None
        
        # 동시성(Race Condition) 방지 락
        self.lock = asyncio.Lock()
        
        # 쿨다운 관리
        self.last_action_time = 0.0
        self.cooldown_seconds = 3.0
        
        # 하드 스탑로스/테이크프로핏 임계값
        self.tick_stop_loss = -0.015
        self.tick_take_profit = 0.03

        # 웜업 상태 플래그
        self.is_warmed_up = False

    async def warmup(self, access_token: str):
        """부팅 시 당일 과거 1분봉을 로드하여 버퍼를 미리 채웁니다 (Seed Data)."""
        fetcher = HistoricalFetcher(self.config_manager)
        today_str = datetime.now().strftime("%Y%m%d")
        
        self.logger.info("과거 데이터를 로드하여 버퍼 웜업을 시작합니다...")
        data = await fetcher.fetch_historical_data(self.symbol, today_str, access_token)
        
        if data:
            # 키움 API는 최신 순으로 올 수 있으므로 시간 역순 정렬
            sorted_data = sorted(data, key=lambda x: x["timestamp"])
            added_count = 0
            async with self.lock:
                for row in sorted_data[-100:]:
                    self.minute_buffer.append(row)
                    added_count += 1
            self.logger.info(f"웜업 완료: 초기 1분봉 데이터 {added_count}개 버퍼 적재됨.")
            self.is_warmed_up = True
        else:
            self.logger.warning("웜업 실패 또는 데이터 없음 (장 시작 직후일 수 있습니다). 빈 버퍼로 시작합니다.")
            self.is_warmed_up = True

    async def update_tick(self, price: float, volume: int, timestamp):
        """실시간 틱 수신 및 1분봉 조합"""
        if not self.is_warmed_up:
            return

        if isinstance(timestamp, str):
            ts_str = timestamp
        else:
            ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # [신규] 1. 틱 단위 실시간 감시 (Stop Loss / Take Profit)
        holdings = getattr(self.order_manager, 'holdings', {}).get(self.symbol, 0)
        if holdings > 0:
            avg_price = getattr(self.order_manager, 'avg_entry_prices', {}).get(self.symbol, 0.0)
            if avg_price > 0:
                pnl_pct = (price - avg_price) / avg_price
                if pnl_pct <= self.tick_stop_loss or pnl_pct >= self.tick_take_profit:
                    # 조건 충족 시 즉각 매도 (정각 대기 안 함)
                    self.logger.error(f"[긴급] 틱 단위 스탑로스/익절 발동! (현재 수익률: {pnl_pct*100:.2f}%)")
                    
                    # 즉각 주문 실행 (비동기)
                    asyncio.create_task(self.order_manager.send_order("SELL", self.symbol, int(price), holdings))
                    
                    # AI 상태 즉시 강제 동기화 (헛발질 방지)
                    self.order_manager.holdings[self.symbol] = 0
                    self.order_manager.avg_entry_prices[self.symbol] = 0.0
                    return # 캔들 병합 및 시스템 진행 스킵 (이미 팔았음)
            
        # 'YYYY-MM-DD HH:MM' 분 단위까지만 절사
        minute_str = ts_str[:16]
        
        async with self.lock:
            # 새로운 분이 시작되었을 때
            if self.current_minute_str != minute_str:
                if self.current_candle is not None:
                    await self._finalize_candle()
                
                # 새로운 캔들 생성
                self.current_minute_str = minute_str
                self.current_candle = {
                    "timestamp": minute_str + ":00",
                    "symbol": self.symbol,
                    "open": price,
                    "high": price,
                    "low": price,
                    "price": price,   # close
                    "volume": volume
                }
            else:
                # 기존 캔들 업데이트
                self.current_candle["high"] = max(self.current_candle["high"], price)
                self.current_candle["low"] = min(self.current_candle["low"], price)
                self.current_candle["price"] = price # close
                self.current_candle["volume"] += volume

    async def _finalize_candle(self):
        """1분이 지나 캔들이 확정되었을 때 호출되며, AI 추론 파이프라인을 가동합니다."""
        if not self.current_candle:
            return
            
        self.minute_buffer.append(self.current_candle)
        self.logger.info(f"[1분봉 확정] {self.current_candle['timestamp'][11:]} | 종가={self.current_candle['price']:,} | 거래량={self.current_candle['volume']}")
        
        # 추론 가동 (락이 걸린 상태에서 안전하게 수행)
        await self._run_inference()

    async def check_empty_minute(self, current_time: datetime):
        """
        [Edge Case 방어] 거래량이 없어 해당 분(Minute)에 틱이 전혀 안 들어온 경우,
        직전 종가를 토대로 Volume 0짜리 임시 캔들을 밀어 넣습니다.
        (스케줄러나 메인 루프에서 1분마다 호출 필요)
        """
        minute_str = current_time.strftime("%Y-%m-%d %H:%M")
        async with self.lock:
            if self.current_minute_str != minute_str and self.current_minute_str is not None:
                # 시간이 지났는데 새 틱이 안 들어왔다면 이전 캔들 확정
                if self.current_candle is not None:
                    await self._finalize_candle()
                
                # 빈 캔들 생성 (종가 복사, 거래량 0)
                last_close = self.minute_buffer[-1]["price"] if len(self.minute_buffer) > 0 else 0
                if last_close > 0:
                    self.current_minute_str = minute_str
                    self.current_candle = {
                        "timestamp": minute_str + ":00",
                        "symbol": self.symbol,
                        "open": last_close,
                        "high": last_close,
                        "low": last_close,
                        "price": last_close,
                        "volume": 0
                    }
                    self.logger.debug(f"[0거래량 방어] {minute_str}에 빈 캔들(종가 {last_close})을 생성했습니다.")

    async def _run_inference(self):
        """Feature Engineer 통과 후 Agent 추론 및 주문 실행"""
        # 최소 30개 분봉(Feature 계산용) + 10개(Observation 용도) = 안전하게 40개 이상일 때 발동
        if len(self.minute_buffer) < 30:
            self.logger.debug(f"버퍼 데이터 부족 ({len(self.minute_buffer)}/30). AI 추론을 대기합니다.")
            return

        # 1. 지표(Feature) 계산 규격 100% 동기화 (Advanced 모드)
        # deque 버퍼를 List[dict]로 변환하여 그대로 밀어넣음
        buffer_list = list(self.minute_buffer)
        
        try:
            # Pandas 행렬 생성으로 VWAP 등 당일 리셋 로직 자동 반영됨
            features_1d_list = AdvancedFeatureEngineer.process_historical_data(buffer_list)
        except Exception as e:
            self.logger.error(f"[피처계산 오류] {e}")
            return

        seq_len = getattr(self.agent, 'seq_len', 10)
        if len(features_1d_list) < seq_len:
            return

        # 2. 가장 최근 10스텝(sequence) 추출 및 Flatten (10 x 11차원 = 110차원)
        obs = features_1d_list[-seq_len:]
        obs_1d = obs.flatten()

        # 3. Action Masking (잔고/보유량 물리적 제약)
        action_masks = [True, True, True] # Hold, Buy, Sell
        
        # (A) Buy 제약
        current_price = self.current_candle['price']
        max_invest = self.config_manager.get_dict().get("max_invest_per_symbol", 1000000) if hasattr(self.config_manager, "get_dict") else 1000000
        balance = self.order_manager.get_balance() if hasattr(self.order_manager, 'get_balance') else 10000000
        if balance < current_price:
            action_masks[1] = False
            
        # (B) Sell 제약
        holdings = self.order_manager.holdings.get(self.symbol, 0)
        if holdings <= 0:
            action_masks[2] = False
            
        action_masks_np = np.array(action_masks, dtype=bool)

        # 4. 모델 예측 (Inference)
        obs_batch = np.expand_dims(obs_1d, axis=0)
        result = self.agent.predict(obs_batch, action_masks=action_masks_np, return_probs=True)
        action, probs = result
        if isinstance(action, np.ndarray): action = int(action[0])
        
        # 5. 신뢰도(Confidence) 임계값 적용
        ai_threshold = self.config_manager.get("ai_confidence_threshold", 0.5) if hasattr(self.config_manager, "get") else 0.5
        max_prob = float(max(probs))
        raw_action = action
        if max_prob < ai_threshold:
            action = 0 # Hold 하향 변환

        action_names = {0: "Hold", 1: "Buy", 2: "Sell"}
        if action in [1, 2]:
            self.logger.error(
                f"[AI추론] 신호: {action_names.get(raw_action,'?')} -> 최종: {action_names.get(action,'?')} "
                f"| C={current_price:,} | H={probs[0]:.2f} B={probs[1]:.2f} S={probs[2]:.2f} "
                f"| 확신도: {max_prob:.2f}"
            )

        # 6. UI 업데이트 연결 및 텔레그램 알림
        self._update_ui_signals(action, action_names.get(action, "Hold"), probs)
        
        # [신규] 텔레그램 시그널 알림 전송 (Buy, Sell인 경우에만)
        if action in [1, 2]:
            notifier = getattr(self.order_manager, 'notifier', None)
            if notifier:
                action_str = "매수(BUY)" if action == 1 else "매도(SELL)"
                conf_pct = max_prob * 100
                msg = f"⏱️ <b>AI 시그널 감지</b>\n━━━━━━━━━━━━━━━\n• 종목: {self.symbol}\n• 신호: <b>{action_str}</b>\n• 1분종가: ₩{current_price:,.0f}\n• 확신도: {conf_pct:.1f}%\n(H:{probs[0]*100:.0f}% B:{probs[1]*100:.0f}% S:{probs[2]*100:.0f}%)"
                import asyncio
                asyncio.create_task(notifier.send_message(msg))

        # 7. 실제 주문 실행 (쿨다운 및 장 상태 체크)
        current_time = time.time()
        if current_time - self.last_action_time < self.cooldown_seconds:
            return

        scheduler = getattr(self.config_manager, "_injected_scheduler", None)
        if scheduler and scheduler.current_state != MarketState.TRADING:
            if action in [1, 2]:
                self.logger.warning("가상 신호 발생 (장외 시간)")
            return

        if action == 1: # Buy
            qty = int(max_invest // current_price)
            if qty > 0:
                asyncio.create_task(self.order_manager.send_order("BUY", self.symbol, int(current_price), qty))
                self.last_action_time = current_time
        elif action == 2: # Sell
            qty = holdings
            if qty > 0:
                asyncio.create_task(self.order_manager.send_order("SELL", self.symbol, int(current_price), qty))
                self.last_action_time = current_time

    def _update_ui_signals(self, action, signal_text, probs):
        vm = getattr(self.config_manager, "_injected_live_vm", None)
        if vm:
            if self.symbol not in vm.symbols_summary:
                vm.symbols_summary[self.symbol] = {"price": 0, "ai_signal": "-", "holdings": 0}
            vm.symbols_summary[self.symbol]["ai_signal"] = signal_text
            vm.symbols_summary[self.symbol]["price"] = self.current_candle.get('price', 0)
            vm.sig_symbols_summary_updated.emit(vm.symbols_summary)
            
            # 선택된 종목일 경우 상세 확률 전달
            if vm.selected_symbol == self.symbol or not vm.selected_symbol:
                conf = {"Hold": int(probs[0]*100), "Buy": int(probs[1]*100), "Sell": int(probs[2]*100)}
                vm.sig_ai_confidence_updated.emit(conf)
