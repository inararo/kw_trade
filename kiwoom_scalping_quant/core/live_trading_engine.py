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

        # [신규] 주문 진행 상태 Lock 플래그 (중복 주문 방지)
        self._is_order_pending = False

    async def warmup(self, access_token: str):
        """부팅 시 최근 약 1시간 정도의 데이터를 로드하여 지표 계산 기반을 마련합니다."""
        fetcher = HistoricalFetcher(self.config_manager)
        today_str = datetime.now().strftime("%Y%m%d")
        
        self.logger.info(f"최소 웜업 시작... (Token 보유 여부: {bool(access_token)})")
        data_result = await fetcher.fetch_historical_data(self.symbol, today_str, access_token, max_pages=1)
        
        if data_result is None or (hasattr(data_result, 'is_failure') and data_result.is_failure()):
            self.logger.warning(f"웜업 실패: {getattr(data_result, 'failure', lambda: 'Data is None')()}")
            self.is_warmed_up = True
            return

        data = data_result.unwrap() if hasattr(data_result, 'unwrap') else data_result
        if hasattr(data, '_inner_value'): data = data._inner_value

        if data and isinstance(data, list):
            sorted_data = sorted(data, key=lambda x: x["timestamp"])
            async with self.lock:
                existing_timestamps = {c["timestamp"] for c in self.minute_buffer}
                added_count = 0
                for row in sorted_data:
                    if row["timestamp"] in existing_timestamps: continue
                    self.minute_buffer.append(row)
                    added_count += 1
            self.logger.info(f"웜업 완료: 과거 데이터 {added_count}개 적재됨.")
            self.is_warmed_up = True
            
            if self.minute_buffer:
                last_candle = self.minute_buffer[-1]
                if self.current_candle is None: self.current_candle = last_candle.copy()
                vm = getattr(self.config_manager, "_injected_live_vm", None)
                if vm and self.symbol in vm.symbols_summary:
                    vm.symbols_summary[self.symbol]["price"] = last_candle["price"]
                    vm._ui_dirty = True
                asyncio.create_task(self._run_inference())
        else:
            self.is_warmed_up = True

    # 앞의 불필요한 인자(symbol, state)는 스펀지처럼 흡수하고, 핵심 데이터만 빼서 씁니다.
    async def update_tick(self, *args, **kwargs):
        """실시간 틱 수신 및 1분봉 조합 (직접 콜백 대응 및 심볼 필터링 추가)"""
        try:
            # 0. 심볼 필터링 (DataCollector에서 직접 호출 시 타 종목 데이터 유입 방지)
            target_symbol = args[0] if args else kwargs.get('symbol')
            if target_symbol and target_symbol.split('_')[0].strip() != self.symbol.split('_')[0].strip():
                return

            # 1. 인자 유연 추출
            price = kwargs.get('price')
            volume = kwargs.get('volume')
            timestamp = kwargs.get('timestamp')

            # 위치 인자(args)로 들어왔을 경우를 대비한 방어 로직 (DataCollector 호출 포맷 대응)
            if price is None and len(args) >= 3:
                # DataCollector: callback(symbol, state, price=p, volume=v, timestamp=t)
                # 만약 위치 인자로만 왔다면 뒤에서부터 추출
                price = args[-3] if isinstance(args[-3], (int, float)) else price
                volume = args[-2] if isinstance(args[-2], (int, float)) else volume
                timestamp = args[-1]

            # 형변환 보장
            if price is None or volume is None: return
            price = float(price)
            volume = int(volume)

            # 2. 틱 유입 생존 신고 (너무 많으면 안되니 500틱마다 한 번씩 터미널에 보고)
            if not hasattr(self, '_tick_alive_cnt'): self._tick_alive_cnt = 0
            self._tick_alive_cnt += 1
            if self._tick_alive_cnt % 500 == 0:
                self.logger.info(f"[{self.symbol}] 엔진 내부 틱 수신 중... (현재가: {price}, 웜업: {self.is_warmed_up})")

            # 3. 웜업이 안 끝났으면 1분봉 조립을 대기
            if not self.is_warmed_up: return

            ts_str = timestamp if isinstance(timestamp, str) else timestamp.strftime("%Y-%m-%d %H:%M:%S")

            # 4. 기존 스탑로스 / 익절 로직 유지
            if not self._is_order_pending:
                holdings = getattr(self.order_manager, 'holdings', {}).get(self.symbol, 0)
                if holdings > 0:
                    avg_price = getattr(self.order_manager, 'avg_entry_prices', {}).get(self.symbol, 0.0)
                    if avg_price > 0:
                        pnl_pct = (price - avg_price) / avg_price
                        if pnl_pct <= self.tick_stop_loss or pnl_pct >= self.tick_take_profit:
                            self.logger.error(f"🚨 [긴급] 틱 단위 스탑로스/익절 발동! (수익률: {pnl_pct * 100:.2f}%)")
                            self._is_order_pending = True
                            asyncio.create_task(self._execute_order_background("SELL", int(price), holdings))
                            return

                            # 5. 1분봉 병합 핵심 로직
            minute_str = ts_str[:16]
            async with self.lock:
                if self.current_minute_str != minute_str:
                    # 분이 바뀌면 이전 캔들을 확정하고 AI 추론으로 넘김!
                    if self.current_candle is not None:
                        await self._finalize_candle()

                    self.current_minute_str = minute_str
                    self.current_candle = {
                        "timestamp": minute_str + ":00", "symbol": self.symbol,
                        "open": price, "high": price, "low": price, "price": price, "volume": volume
                    }
                else:
                    self.current_candle["high"] = max(self.current_candle["high"], price)
                    self.current_candle["low"] = min(self.current_candle["low"], price)
                    self.current_candle["price"] = price  # close
                    self.current_candle["volume"] += volume

            # UI 업데이트 요청 (안전장치)
            vm = getattr(self.config_manager, "_injected_live_vm", None)
            if vm: vm._on_data_received({"symbol": self.symbol, "price": price, "volume": volume})

        except Exception as e:
            # 🚨 암살당하던 에러를 멱살 잡고 끌어올려 터미널에 전시합니다!
            import traceback
            self.logger.error(f"[{self.symbol}] 🚨 엔진 update_tick 치명적 에러 발생!\n{traceback.format_exc()}")

    async def _finalize_candle(self):
        if not self.current_candle: return
        self.minute_buffer.append(self.current_candle)
        self.logger.info(f"[1분봉 확정] {self.current_candle['timestamp'][11:]} | 종가={self.current_candle['price']:,}")
        await self._run_inference()

    async def check_empty_minute(self, current_time: datetime):
        minute_str = current_time.strftime("%Y-%m-%d %H:%M")
        async with self.lock:
            if self.current_minute_str != minute_str and self.current_minute_str is not None:
                if self.current_candle is not None: await self._finalize_candle()
                last_close = self.minute_buffer[-1]["price"] if len(self.minute_buffer) > 0 else 0
                if last_close > 0:
                    self.current_minute_str = minute_str
                    self.current_candle = {
                        "timestamp": minute_str + ":00", "symbol": self.symbol,
                        "open": last_close, "high": last_close, "low": last_close, "price": last_close, "volume": 0
                    }

    async def _run_inference(self):
        """AI 추론 및 주문 결정"""
        # [디버그 로그 추가] AI가 깨어있는지 확인
        self.logger.info(f"[{self.symbol}] AI 추론 시도... (현재 버퍼 크기: {len(self.minute_buffer)})")

        if self._is_order_pending:
            self.logger.debug(f"[{self.symbol}] 주문 파이프라인 가동 중 - 추론 생략")
            return

        if len(self.minute_buffer) < 30: return

        # 1. Feature Engineering
        buffer_list = list(self.minute_buffer)
        try:
            features = AdvancedFeatureEngineer.process_historical_data(buffer_list)
        except Exception as e:
            self.logger.error(f"피처 계산 에러: {e}")
            return

        seq_len = getattr(self.agent, 'seq_len', 10)
        if len(features) < seq_len: return
        obs_1d = features[-seq_len:].flatten()

        # 2. Observation Padding (Target Dim 동기화)
        obs_shape = getattr(self.agent.env, 'observation_space', None)
        if obs_shape:
            target_dim = obs_shape.shape[0]
            if len(obs_1d) < target_dim:
                padded = np.pad(obs_1d, (0, target_dim - len(obs_1d)), 'constant')
                obs_1d = padded

        # 3. Action Masking 및 예측
        action_masks = [True, True, True]
        current_price = self.current_candle['price'] if self.current_candle else 0
        balance = self.order_manager.get_balance() if hasattr(self.order_manager, 'get_balance') else 0
        holdings = self.order_manager.holdings.get(self.symbol, 0)
        
        if balance < current_price: action_masks[1] = False
        if holdings <= 0: action_masks[2] = False
        
        action, probs = self.agent.predict(np.expand_dims(obs_1d, axis=0), action_masks=np.array(action_masks), return_probs=True)
        if isinstance(action, np.ndarray): action = int(action[0])
        
        # 신뢰도 필터
        ai_threshold = self.config_manager.get("ai_confidence_threshold", 0.5)
        if max(probs) < ai_threshold: action = 0
        
        self._update_ui_signals(action, {0:"Hold", 1:"Buy", 2:"Sell"}.get(action, "Hold"), probs)

        # 4. 주문 실행 (쿨다운 & 장상태 체크)
        current_time = time.time()
        if current_time - self.last_action_time < self.cooldown_seconds: return
        
        scheduler = getattr(self.config_manager, "_injected_scheduler", None)
        if scheduler and scheduler.current_state != MarketState.TRADING: return

        max_invest = self.config_manager.get("max_invest_per_symbol", 1000000)
        if action == 1: # BUY
            qty = int(max_invest // current_price)
            if qty > 0:
                self._is_order_pending = True  # 태스크 생성 전에 즉시 문부터 잠금!
                asyncio.create_task(self._execute_order_background("BUY", int(current_price), qty))
                self.last_action_time = current_time
        elif action == 2: # SELL
            if holdings > 0:
                self._is_order_pending = True  # 👈 [여기에 추가!] 매수와 동일하게 즉시 락 설정
                asyncio.create_task(self._execute_order_background("SELL", int(current_price), holdings))
                self.last_action_time = current_time

    async def _execute_order_background(self, side, price, qty):
        """
        [핵심 파이프라인] 주문 전송 -> 5초 대기 -> 미체결 시 자동 취소
        """
        internal_id = None
        
        try:
            self.logger.info(f"📤 주문 실행 파이프라인 가동: {side} {self.symbol} {qty}주 @ {price:,}원")
            
            # 1. 주문 전송
            result = await self.order_manager.send_order(side, self.symbol, price, qty)
            if hasattr(result, 'is_failure') and result.is_failure():
                self.logger.error(f"❌ 전송 실패: {result.failure()}")
                return

            internal_id = result.unwrap() if hasattr(result, 'unwrap') else result
            
            # [알림] 텔레그램 전송
            notifier = getattr(self.order_manager, 'notifier', None)
            if notifier:
                msg = f"🚀 <b>주문 집행 완료</b>\n• 종목: {self.symbol}\n• 구분: {side}\n• 수량: {qty}주\n• 가격: ₩{price:,}\n(5초 후 미체결분 자동 취소 예정)"
                asyncio.create_task(notifier.send_message(msg))

            # 2. 5초간 체결 대기 (타임아웃 감시)
            await asyncio.sleep(5.0)

            # 3. 미체결 잔량 확인 및 자동 취소 프로세스
            order_info = self.order_manager.active_orders.get(internal_id)
            if order_info:
                unexecuted = order_info.get('unexecuted_qty', 0)
                status = order_info.get('status')
                
                if unexecuted > 0 and status not in ["FILLED", "CANCELLED", "FAILED"]:
                    self.logger.error(f"⏰ 5초 타임아웃! 미체결 잔량 {unexecuted}주 취소 절차를 시작합니다.")
                    await self.order_manager.cancel_order(internal_id)
                    await asyncio.sleep(2.0) # 서버 처리 시간 대기
                else:
                    self.logger.info(f"🎯 주문 처리 완료 (결과 상태: {status})")

        except Exception as e:
            self.logger.error(f"🔥 주문 파이프라인 치명적 오류: {e}")
        finally:
            self._is_order_pending = False
            self.logger.info(f"🔒 {self.symbol} 주문 락 해제 완료.")

    def _update_ui_signals(self, action, signal_text, probs):
        vm = getattr(self.config_manager, "_injected_live_vm", None)
        if vm:
            if self.symbol not in vm.symbols_summary:
                vm.symbols_summary[self.symbol] = {"name": self.symbol, "price": 0, "ai_signal": "-", "holdings": 0}
            vm.symbols_summary[self.symbol]["ai_signal"] = signal_text
            if vm.selected_symbol == self.symbol:
                conf = {"Hold": int(probs[0]*100), "Buy": int(probs[1]*100), "Sell": int(probs[2]*100)}
                vm.sig_ai_confidence_updated.emit(conf)
            vm._ui_dirty = True
