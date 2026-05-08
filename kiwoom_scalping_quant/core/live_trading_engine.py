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
from utils.math_jit import get_valid_tick_price
from utils.trade_logger import trade_logger
from models.lstm_extractor import OnlineRollingNormalizer

class LiveTradingEngine:
    """
    개별 종목에 대해 실시간 틱(Tick) 수신, 1분봉 병합(OHLCV), 웜업, 
    Feature Engineering 및 RL Agent 시그널 추론을 전담하는 엔진입니다.
    """
    def __init__(self, symbol: str, config_manager, order_manager, shared_agent, strategy_manager=None, broker_api=None):
        self.symbol = symbol
        self.config_manager = config_manager
        self.order_manager = order_manager
        self.agent = shared_agent
        self.strategy_manager = strategy_manager # [신규] 글로벌 쿨타임 확인용
        self.broker_api = broker_api # [신규] 토큰 재발급용 API 핸들
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
        self.last_change_rate = 0.0  # [신규] 당일 등락률 추적용
        
        # [수정] 하드 스탑로스/익절 임계값 (설정 파일과 연동)
        # AI의 자율성을 보장하기 위해 기본 하드 손절은 넉넉하게 -3.5%로 설정
        sl_config = float(self.config_manager.get("stop_loss_pct", -3.5))
        tp_config = float(self.config_manager.get("take_profit_pct", 4.0))
        
        # 변수명에 _pct가 명시되어 있으므로 사용자는 무조건 퍼센트로 입력했다고 신뢰함
        # 무조건 100으로 나누어 소수점으로 변환 (예: -3.5 -> -0.035)
        self.tick_stop_loss = sl_config / 100.0
        self.tick_take_profit = tp_config / 100.0
        
        self.logger.info(f"엔진 초기화: 하드 손절라인 {sl_config:.2f}%, 하드 익절라인 {tp_config:.2f}% 설정됨")

        # 웜업 상태 플래그
        self.is_warmed_up = False

        # [신규] 주문 진행 상태 Lock 플래그 (중복 주문 방지)
        self._is_order_pending = False

        # [동적 유니버스] 이탈 대상 상태 플래그
        self.is_condition_deleted = False

        # [신규] 실시간 피처 정규화기 (Z-score Scaling)
        # AI 모델의 신뢰도 포화(Saturation) 현상을 방지하기 위해 학습 시와 동일한 통계량으로 정규화 수행
        self.normalizer = OnlineRollingNormalizer(window_size=200) 

        # [좀비 방어] 완전히 파괴되었음을 알리는 플래그
        self._is_destroyed = False

    async def warmup(self, access_token: str):
        """부팅 시 최근 약 1시간 정도의 데이터를 로드하여 지표 계산 기반을 마련합니다."""
        fetcher = HistoricalFetcher(self.config_manager)
        fetcher.broker_api = self.broker_api # [중요] 토큰 재발급을 위해 API 주입
        today_str = datetime.now().strftime("%Y%m%d")
        
        self.logger.info(f"최소 웜업 시작... (Token 보유 여부: {bool(access_token)})")
        # [수정] max_pages=5로 변경하여 약 하루치(500분)의 데이터를 수집합니다 (1페이지=100분)
        data_result = await fetcher.fetch_historical_data(self.symbol, today_str, access_token, max_pages=5)
        
        # [좀비 방어] 데이터 수집 대기 중에 엔진 파괴 명령이 내려졌다면 즉시 중단
        if self._is_destroyed:
            self.logger.warning(f"[{self.symbol}] 웜업 중 파괴 명령 감지! 웜업을 중단합니다.")
            return

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
            
            # [신규] 정규화기(Normalizer) 사전 워밍업
            # 과거 데이터를 피처로 변환하여 정규화기에 미리 주입해야 부팅 즉시 정상적인 AI 추론이 가능합니다.
            if len(self.minute_buffer) >= 30:
                try:
                    all_features = AdvancedFeatureEngineer.process_historical_data(list(self.minute_buffer))
                    seq_len = getattr(self.agent, 'seq_len', 10)
                    
                    # [핵심 수정] _run_inference와 동일한 차원(seq_len * num_features)으로 주입
                    # 기존에 단일 시점(11차원)으로 주입하여 발생하던 shape 불일치 에러를 해결합니다.
                    self.normalizer.history.clear() # 웜업 전 초기화
                    for i in range(seq_len, len(all_features) + 1):
                        window_obs = all_features[i-seq_len : i].flatten()
                        if len(window_obs) > 0:
                            self.normalizer.normalize(window_obs)
                    
                    self.logger.info(f"정규화기 웜업 완료: {len(all_features)}개 분봉 기반 {len(self.normalizer.history)}개 윈도우 학습됨.")
                except Exception as e:
                    self.logger.error(f"정규화기 웜업 중 오류: {e}")

            if added_count == 0:
                self.logger.warning(f"⚠️ [{self.symbol}] 웜업 완료되었으나 적재된 데이터가 0개입니다. (서버 응답 없음 또는 형식 불일치)")
            else:
                self.logger.info(f"✅ [{self.symbol}] 웜업 완료: 과거 데이터 {added_count}개 적재됨.")
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
            change_rate = kwargs.get('change_rate', 0.0)
            
            # [신규] 당일 등락률 실시간 업데이트
            self.last_change_rate = change_rate

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

            # 4. 기존 스탑로스 / 익절 로직 유지 (단, 장 운영 시간 중에만 작동)
            now = datetime.now()
            market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
            market_end = now.replace(hour=15, minute=20, second=0, microsecond=0)
            
            if (market_start <= now <= market_end) and not self._is_order_pending:
                holdings = getattr(self.order_manager, 'holdings', {}).get(self.symbol, 0)
                if holdings > 0:
                    avg_price = getattr(self.order_manager, 'avg_entry_prices', {}).get(self.symbol, 0.0)
                    if avg_price > 0:
                        pnl_pct = (price - avg_price) / avg_price
                        if pnl_pct <= self.tick_stop_loss or pnl_pct >= self.tick_take_profit:
                            reason = "스탑로스" if pnl_pct <= self.tick_stop_loss else "익절"
                            msg = f"🚨 [긴급] {self.symbol} 틱 단위 {reason} 발동! (수익률: {pnl_pct * 100:.2f}%)"
                            self.logger.error(msg)
                            self._ui_log(msg)
                            self._is_order_pending = True
                            
                            # [개선] 하드 손절 시 체결 확률을 높이기 위해 현재가보다 1호가 아래로 주문 (Slippage 대응)
                            # 매도의 경우 price * 0.999 정도면 충분히 최우선 매수호가에 체결됨
                            raw_sell_price = price * 0.999 if reason == "스탑로스" else price
                            sell_price = get_valid_tick_price(raw_sell_price, "SELL")
                            asyncio.create_task(self._execute_order_background("SELL", sell_price, holdings))
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
        # [신규] 장 운영 시간(Market Hours) 하드락: 09:00:00 ~ 15:20:00
        now = datetime.now()
        market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=20, second=0, microsecond=0)
        
        if not (market_start <= now <= market_end):
            self.logger.debug(f"[{self.symbol}] 장외 시간(현재 {now.strftime('%H:%M:%S')}) - 추론 및 주문 파이프라인 바이패스")
            return

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
        
        # 실전 매매 잔고 강제 동기화 (예측 전에 실제 포트폴리오 상태 파악)
        if hasattr(self.order_manager, 'fetch_real_balance'):
            await self.order_manager.fetch_real_balance()
            
        # 1.5. 피처 및 Observation 통합 구축 (219차원)
        final_obs = self._build_live_observation(obs_1d, buffer_list)

        # 2. Observation Padding 안전 방어 (혹시라도 차원이 모자란 경우)
        obs_shape = getattr(self.agent.env, 'observation_space', None)
        if obs_shape:
            target_dim = obs_shape.shape[0]
            if len(final_obs) < target_dim:
                final_obs = np.pad(final_obs, (0, target_dim - len(final_obs)), 'constant')
            elif len(final_obs) > target_dim:
                final_obs = final_obs[:target_dim]
        
        obs_1d = final_obs

        # 3. Action Masking 및 예측
        # 0:Hold  1:Buy40%  2:Buy60%  3:Sell60%  4:Sell40%
        action_masks = [True, False, False, False, False]
        current_price = self.current_candle['price'] if self.current_candle else 0
        balance = self.order_manager.get_balance() if hasattr(self.order_manager, 'get_balance') else 0
        holdings = self.order_manager.holdings.get(self.symbol, 0)

        # 매수 가능 조건: 잔고 충분 & 주문 미진행
        can_buy = (balance >= current_price and not self._is_order_pending and not self.is_condition_deleted)
        filter_reasons = []
        
        # [하드 필터] 자체 2중 안전장치 검사 (매수 시그널 허용 전)
        if can_buy:
            # Indicator에서 계산된 SMA20, VWAP, ATR을 활용
            # final_obs 내 indicator 값은 스케일링 되어 있으므로 원본 Pandas에서 재추출
            try:
                df = pd.DataFrame(buffer_list)
                close_series = pd.to_numeric(df.get('price', 0), errors='coerce').fillna(0)
                high_series = pd.to_numeric(df.get('high', close_series), errors='coerce').fillna(0)
                low_series = pd.to_numeric(df.get('low', close_series), errors='coerce').fillna(0)
                vol_series = pd.to_numeric(df.get('volume', 0), errors='coerce').fillna(0)
                
                sma20 = float(close_series.rolling(window=20, min_periods=1).mean().iloc[-1])
                
                if 'timestamp' in df.columns:
                    date_str = df['timestamp'].astype(str).str[:8]
                    vp = close_series * vol_series
                    cum_vp = vp.groupby(date_str).cumsum()
                    cum_v = vol_series.groupby(date_str).cumsum()
                    vwap = float((cum_vp / (cum_v + 1e-9)).iloc[-1])
                else:
                    vwap = current_price
                    
                tr1 = high_series - low_series
                tr2 = (high_series - close_series.shift(1)).abs()
                tr3 = (low_series - close_series.shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr14 = float(tr.rolling(window=14, min_periods=1).mean().iloc[-1])
                
                # 거래대금 필터 (최근 1시간 기준 거래대금 등, 여기서는 당일 거래대금이 파싱 안되므로 생략 또는 보수적 접근)
                # 필터 1: SMA20 상향 돌파 (정배열 혹은 반등 확인)
                if current_price < sma20:
                    can_buy = False
                    filter_reasons.append("주가 20일선 이탈")
                    # self.logger.debug(f"[{self.symbol}] 🚫 하드 필터 차단: 현재가({current_price})가 SMA_20({sma20:.2f}) 아래에 있습니다.")
                
                # 필터 2: ATR 14가 임계값 (예: 최소 호가단위 2배 이상)
                min_atr = current_price * 0.005 # 0.5% 변동성
                if atr14 < min_atr:
                    can_buy = False
                    filter_reasons.append(f"변동성 부족(ATR {atr14:.1f} < {min_atr:.1f})")
                    # self.logger.debug(f"[{self.symbol}] 🚫 하드 필터 차단: ATR({atr14:.2f})이 최소 기준치({min_atr:.2f}) 미만입니다.")

                # 필터 3: 당일 급등 종목 매수 제한 (추격 매수 방지)
                max_rise = float(self.config_manager.get("max_daily_rise_pct", 30.0))
                if self.last_change_rate >= max_rise:
                    can_buy = False
                    filter_reasons.append(f"당일 급등({self.last_change_rate:.1f}%)")

                # 필터 4: 글로벌 매수 쿨타임 (전략적 분산)
                if self.strategy_manager and not self.strategy_manager.can_execute_buy():
                    can_buy = False
                    filter_reasons.append("글로벌 쿨타임")
                    
            except Exception as e:
                self.logger.warning(f"[{self.symbol}] ⚠️ 하드 필터 계산 에러: {e}")
                can_buy = False
        # 매도 가능 조건: 보유 수량 있음 & 주문 미진행
        can_sell = (holdings > 0 and not self._is_order_pending)

        if can_buy:
            action_masks[1] = True  # Buy 40%
            action_masks[2] = True  # Buy 60%
        if can_sell:
            action_masks[3] = True  # Sell 60%
            action_masks[4] = True  # Sell 40%

        # [검증] AI 모델 주입 직전 데이터 상태 정밀 로깅
        try:
            state_min = np.min(obs_1d)
            state_max = np.max(obs_1d)
            state_mean = np.mean(obs_1d)
            has_nan = np.isnan(obs_1d).any()
            has_inf = np.isinf(obs_1d).any()
            self.logger.info(f"🔍 [State 검증] {self.symbol} | Shape: {obs_1d.shape} | Min: {state_min:.4f} | Max: {state_max:.4f} | Mean: {state_mean:.4f} | NaN: {has_nan} | Inf: {has_inf}")
            self.logger.info(f"🔍 [State 샘플] {self.symbol} 데이터 앞부분: {obs_1d.flatten()[:5]}")
        except Exception as e:
            self.logger.info(f"🔍 [State 검증 실패] {e}")

        action, probs = self.agent.predict(np.expand_dims(obs_1d, axis=0), action_masks=np.array(action_masks), return_probs=True)
        if isinstance(action, np.ndarray): action = int(action[0])

        # [이중 안전장치] 마스킹된 액션이 선택되었을 경우 강제 홀딩
        if not action_masks[action]:
            action = 0

        # 신뢰도 필터 (매수/매도 임계값 분리 적용)
        ACTION_LABELS = {0: "Hold", 1: "Buy40%", 2: "Buy60%", 3: "Sell60%", 4: "Sell40%"}
        buy_threshold  = float(self.config_manager.get("ai_buy_threshold", 0.6))
        sell_threshold = float(self.config_manager.get("ai_sell_threshold", 0.6))

        # --- 심장박동 로그 (1분봉 확정마다 출력) ---
        buy_conf  = int(max(probs[1], probs[2]) * 100) if len(probs) >= 2 else 0
        sell_conf = int(max(probs[3], probs[4]) * 100) if len(probs) >= 4 else 0
        hold_conf = int(probs[0] * 100) if len(probs) >= 1 else 0

        # 현재 감시 현황 태그 (감시 n/m)
        status_tag = ""
        if self.strategy_manager:
            curr = len(self.strategy_manager.symbols)
            total = self.strategy_manager.MAX_CONCURRENT_STOCKS
            status_tag = f" (감시 {curr}/{total})"
        else:
            status_tag = ""

        if action in (1, 2):  # 매수 계열
            confidence = probs[action]
            if float(confidence) >= buy_threshold:
                self.logger.warning(
                    f"[🔥 매수 포착{status_tag}] 종목: {self.symbol} | 결과: {ACTION_LABELS[action]} {int(confidence*100)}%"
                    f" | (매수확신: {buy_conf}%) | ➡️ API 주문 전송!"
                )
            else:
                self.logger.warning(
                    f"[🧠 AI 판단{status_tag}] 종목: {self.symbol} | 결과: {ACTION_LABELS[action]} (임계값 미달)"
                    f" | (매수확신: {buy_conf}%, 매도확신: {sell_conf}%) | 🎯 타점 대기 중..."
                )
                action = 0
        elif action in (3, 4):  # 매도 계열
            confidence = probs[action]
            if float(confidence) >= sell_threshold:
                self.logger.warning(
                    f"[📉 매도 포착{status_tag}] 종목: {self.symbol} | 결과: {ACTION_LABELS[action]} {int(confidence*100)}%"
                    f" | (매도확신: {sell_conf}%) | ➡️ API 주문 전송!"
                )
            else:
                self.logger.warning(
                    f"[🧠 AI 판단{status_tag}] 종목: {self.symbol} | 결과: {ACTION_LABELS[action]} (임계값 미달)"
                    f" | (매수확신: {buy_conf}%, 매도확신: {sell_conf}%) | 🎯 타점 대기 중..."
                )
                action = 0
        else:  # Hold
            # [필터 차단 로깅] AI는 매수하고 싶어했으나(확신도 충족), 하드 필터가 막은 경우
            if buy_conf >= int(buy_threshold * 100) and filter_reasons:
                reason_str = ", ".join(filter_reasons)
                msg = f"[🚫 필터 차단{status_tag}] {self.symbol} | 매수확신 {buy_conf}% ➡️ Hold 변환 (사유: {reason_str})"
                self.logger.warning(msg)
                self._ui_log(msg)
            else:
                self.logger.info(
                    f"[🧠 AI 판단{status_tag}] 종목: {self.symbol} | 결과: Hold"
                    f" | (매수확신: {buy_conf}%, 매도확신: {sell_conf}%) | 🎯 타점 대기 중..."
                )

        self._update_ui_signals(action, ACTION_LABELS.get(action, "Hold"), probs)

        # 4. 주문 실행 (쿨다운 & 장상태 체크)
        current_time = time.time()
        if current_time - self.last_action_time < self.cooldown_seconds: return
        
        scheduler = getattr(self.config_manager, "_injected_scheduler", None)
        if scheduler and scheduler.current_state != MarketState.TRADING: return

        # [고도화] 최대 진입 자금 비율(max_position_pct)을 고려한 동적 투자 한도 적용
        risk_mgr = getattr(self.order_manager, 'risk_manager', None)
        if risk_mgr:
            max_invest = risk_mgr.get_dynamic_max_invest()
            # self.logger.debug(f"[{self.symbol}] 동적 투자 한도 적용: {max_invest:,.0f}원")
        else:
            max_invest = self.config_manager.get("max_invest_per_symbol", 1000000)

        if action in (1, 2):  # 매수 계열 (Buy40% / Buy60%)
            # 매수 비율: action=1 → 40%, action=2 → 60%
            buy_ratio = 0.40 if action == 1 else 0.60
            
            orderable_cash = getattr(self.order_manager, 'orderable_cash', 0.0)
            invest_amount = min(max_invest * buy_ratio, orderable_cash * 0.99) # 수수료 등 감안 99%

            if invest_amount < max_invest * buy_ratio and invest_amount > 0:
                self.logger.info(f"[{self.symbol}] 가용 현금 부족으로 투자 금액 하향 조정: {max_invest * buy_ratio:,.0f} -> {invest_amount:,.0f}")

            # 슬리피지 방지: 지정가(최우선 매도호가)를 추정하여 안전하게 주문 산출
            # 여기서는 편의상 current_price(시장가)를 기준으로 매수 주문. 
            # 스마트 오더 엔진(_execute_order_background) 내에서 get_valid_tick_price를 통해 최적 호가로 보정됩니다.
            qty = int(invest_amount // current_price)
            if qty > 0:
                if self.strategy_manager:
                    self.strategy_manager._pending_buy_symbols.add(self.symbol)
                    self.strategy_manager.record_buy()
                self._is_order_pending = True
                valid_price = get_valid_tick_price(current_price * 1.001, "BUY")
                self.logger.warning(
                    f"[📤 매수 주문 전송] {self.symbol} | {qty}주 @ {valid_price:,}원"
                    f" | 투자금: {invest_amount:,.0f}원 | 비율: {buy_ratio*100:.0f}%"
                )
                asyncio.create_task(self._execute_order_background("BUY", valid_price, qty))
                self.last_action_time = current_time
            else:
                self.logger.warning(f"[{self.symbol}] 주문 수량 0: 가용현금({orderable_cash:,.0f}), 투자비율({buy_ratio*100:.0f}%)")

        elif action in (3, 4):  # 매도 계열 (Sell60% / Sell40%)
            real_holdings = self.order_manager.holdings.get(self.symbol, 0)
            if real_holdings > 0:
                sell_ratio = 0.60 if action == 3 else 0.40
                sell_qty = max(1, int(real_holdings * sell_ratio))
                self._is_order_pending = True
                valid_price = get_valid_tick_price(current_price * 0.999, "SELL")
                self.logger.warning(
                    f"[📤 매도 주문 전송] {self.symbol} | {sell_qty}주 @ {valid_price:,}원"
                    f" | 보유: {real_holdings}주 | 비율: {sell_ratio*100:.0f}%"
                )
                asyncio.create_task(self._execute_order_background("SELL", valid_price, sell_qty))
                self.last_action_time = current_time
            else:
                self.logger.warning(f"[{self.symbol}] 중복 매도 신호 차단: 이미 보유 수량이 0입니다.")

    async def _execute_order_background(self, side, price, qty):
        """
        [핵심 파이프라인] 주문 전송 -> 5초 대기 -> 미체결 시 자동 취소
        """
        internal_id = None
        
        try:
            self.logger.error(f"📤 주문 실행 파이프라인 가동: {side} {self.symbol} {qty}주 @ {price:,}원")
            
            # 1. 주문 전송
            result = await self.order_manager.send_order(side, self.symbol, price, qty)
            
            # [방어 로직] returns 라이브러리의 Result 객체 안전한 언래핑
            from returns.pipeline import is_successful
            
            if hasattr(result, 'unwrap'):
                if is_successful(result):
                    # 성공 시 내부 ID(internal_id) 추출
                    internal_id = result.unwrap()
                else:
                    # 실패 시(Failure) 크래시 방지를 위해 unwrap()을 호출하지 않고 로그 출력 후 종료
                    # RiskManager 차단 등 정상적인 거부 사유를 로깅합니다.
                    failure_reason = result.failure()
                    self.logger.warning(f"🚫 [주문 스킵] RiskManager 차단 또는 에러: {failure_reason}")
                    return
            else:
                # Result 객체가 아닌 일반 값인 경우 그대로 사용
                internal_id = result
            
            # [신규] 주문 전송 성공 즉시 UI 알림
            success_msg = f"📤 [{self.symbol}] {side} 주문 {qty}주 @ {price:,}원 전송 성공"
            self.logger.error(success_msg)
            self._ui_log(success_msg)

            # [알림] 텔레그램 전송
            notifier = getattr(self.order_manager, 'notifier', None)
            if notifier:
                msg = f"🚀 <b>주문 집행 완료</b>\n• 종목: {self.symbol}\n• 구분: {side}\n• 수량: {qty}주\n• 가격: ₩{price:,}\n(5초 후 미체결분 자동 취소 예정)"
                asyncio.create_task(notifier.send_message(msg))

            # 2. 5초간 체결 대기 (타임아웃 감시)
            await asyncio.sleep(5.0)

            # 3. 체결 상태 확인 및 기록 (이미 체결되어 active_orders에 없을 수도 있음)
            order_info = self.order_manager.active_orders.get(internal_id)
            unexecuted = 0
            status = "FILLED (EXPECTED)"
            
            if order_info:
                unexecuted = order_info.get('unexecuted_qty', 0)
                status = order_info.get('status', 'PENDING')
                
                if unexecuted > 0 and status not in ["FILLED", "CANCELLED", "FAILED"]:
                    msg = f"⏰ [{self.symbol}] 5초 타임아웃! 미체결 잔량 {unexecuted}주 취소 절차 시작."
                    self.logger.error(msg)
                    self._ui_log(msg)
                    await self.order_manager.cancel_order(internal_id)
                    status = "TIMEOUT_CANCELLED"
                    await asyncio.sleep(1.5)
            else:
                # 주문 정보가 없다면 이미 체결되어 사라진 것으로 간주
                status = "FILLED"

            # 4. [파일/UI 기록] 최종 매매 결과 기록
            executed_qty = qty - unexecuted
            if executed_qty > 0:
                pnl, pnl_pct = 0, 0.0
                if side == "SELL":
                    avg_price = getattr(self.order_manager, 'avg_entry_prices', {}).get(self.symbol, 0.0)
                    if avg_price > 0:
                        pnl_pct = (price - avg_price) / avg_price
                        pnl = (price - avg_price) * executed_qty
                
                trade_logger.log_trade(self.symbol, side, executed_qty, price, pnl=pnl, pnl_pct=pnl_pct, note=f"Status: {status}")
                
                res_msg = f"🎯 [{self.symbol}] {side} 체결 완료 ({executed_qty}주, 수익률: {pnl_pct*100:+.2f}%)"
                self.logger.warning(res_msg) # visibility 강화
                self._ui_log(res_msg)

                # [실현 손익 업데이트] 글로벌 잔고 매니저 및 리스크 매니저에 반영
                if pnl != 0:
                    order_info = self.order_manager.active_orders.get(internal_id)
                    if order_info and not order_info.get('pnl_processed'):
                        self.order_manager.daily_realized_pnl += pnl
                        order_info['pnl_processed'] = True
                        if self.order_manager.risk_manager:
                            self.order_manager.risk_manager.update_pnl(pnl)
                        self.logger.warning(f"💰 [PnL 업데이트] {self.symbol} 매도로 인한 실현손익 반영: {pnl:,.0f}원 (당일 누적: {self.order_manager.daily_realized_pnl:,.0f}원)")

                # [Firebase] 최종 매매 결과 기록 (OrderManager의 체잔 데이터 누락 대비 백업)
                firebase_manager = getattr(self.order_manager, 'firebase_manager', None)
                if firebase_manager:
                    # 종목명 가져오기
                    symbol_name = self.symbol
                    if hasattr(self.order_manager, 'config') and hasattr(self.order_manager.config, 'get_symbols'):
                        for s in self.order_manager.config.get_symbols():
                            if s.get('code') == self.symbol:
                                symbol_name = s.get('name', self.symbol)
                                break

                    trade_data = {
                        "log_type":      side,
                        "symbol":        self.symbol,
                        "symbol_name":   symbol_name,
                        "price":         float(price),
                        "qty":           int(executed_qty),
                        "profit_loss":   float(pnl),
                        "profit_rate":   float(pnl_pct * 100),
                        "timestamp_str": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                        "note":          f"Engine Finalized (Status: {status})"
                    }
                    self.logger.warning(f"📤 [Firebase 전송 시도] {self.symbol} {side} 결과")
                    asyncio.create_task(firebase_manager.add_trade_log(trade_data))
                else:
                    self.logger.error("⚠️ [Firebase] FirebaseManager를 찾을 수 없어 로그를 전송하지 못했습니다.")

        except Exception as e:
            self.logger.error(f"🔥 주문 파이프라인 치명적 오류: {e}", exc_info=True)
        finally:
            self._is_order_pending = False
            # [안전장치] 주문 파이프라인 종료 시(성공/실패 무관) pending 최소화
            if self.strategy_manager:
                self.strategy_manager._pending_buy_symbols.discard(self.symbol)
            self.logger.info(f"🔒 {self.symbol} 주문 락 해제 완료.")

    def _build_live_observation(self, features_1d, buffer_list) -> np.ndarray:
        """
        훈련 환경(trading_env.py)의 _get_observation()과 100% 동일하게 
        Feature(110) + StockID(Max) + Portfolio(2) + Indicator(7) 를 결합합니다.
        """
        import pandas as pd
        
        # 1. Feature (110)
        obs_1d = features_1d
        if self.normalizer and len(obs_1d) > 0:
            obs_1d = self.normalizer.normalize(obs_1d)
            
        # 2. Stock ID (One-hot)
        try:
            # agent.env.get_attr()는 서브프로세스 래퍼용, 실패 시 기본값
            if hasattr(self.agent.env, 'get_attr'):
                all_symbols = self.agent.env.get_attr('all_symbols')[0]
                unwrapped = self.agent.env.get_attr('unwrapped')[0]
                max_num_symbols = getattr(unwrapped, 'max_num_symbols', len(all_symbols))
            else:
                all_symbols = getattr(self.agent.env, 'all_symbols', [self.symbol])
                max_num_symbols = getattr(self.agent.env, 'max_num_symbols', 140)
        except Exception:
            all_symbols = [self.symbol]
            max_num_symbols = 140 # Typical max limit
            
        stock_onehot = np.zeros(max_num_symbols, dtype=np.float32)
        if self.symbol in all_symbols:
            idx = all_symbols.index(self.symbol)
            if idx < max_num_symbols:
                stock_onehot[idx] = 1.0

        # 3. Portfolio State (2)
        current_price = float(buffer_list[-1]['price'])
        balance = getattr(self.order_manager, 'orderable_cash', 0.0)
        holdings = self.order_manager.holdings.get(self.symbol, 0)
        avg_entry = self.order_manager.avg_entry_prices.get(self.symbol, 0.0)
        
        net_worth = balance + holdings * current_price
        stock_value = holdings * current_price
        position_ratio = float(np.clip(stock_value / (net_worth + 1e-9), 0.0, 1.0))
        
        if holdings > 0 and avg_entry > 0:
            unrealized_pnl = (current_price - avg_entry) / (avg_entry + 1e-9)
        else:
            unrealized_pnl = 0.0
        unrealized_pnl = float(np.clip(unrealized_pnl, -1.0, 1.0))
        
        portfolio_state = np.array([position_ratio, unrealized_pnl], dtype=np.float32)
        
        # 4. Indicators (7) - 순수 Pandas 연산 동기화
        df = pd.DataFrame(buffer_list)
        close_series = pd.to_numeric(df.get('price', 0), errors='coerce').fillna(0)
        high_series = pd.to_numeric(df.get('high', close_series), errors='coerce').fillna(0)
        low_series = pd.to_numeric(df.get('low', close_series), errors='coerce').fillna(0)
        vol_series = pd.to_numeric(df.get('volume', 0), errors='coerce').fillna(0)
        
        sma20 = float(close_series.rolling(window=20, min_periods=1).mean().iloc[-1])
        sma60 = float(close_series.rolling(window=60, min_periods=1).mean().iloc[-1])
        
        delta = close_series.diff()
        gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-9)
        rsi14 = float((100.0 - (100.0 / (1.0 + rs))).iloc[-1])
        
        if 'timestamp' in df.columns:
            date_str = df['timestamp'].astype(str).str[:8]
            vp = close_series * vol_series
            cum_vp = vp.groupby(date_str).cumsum()
            cum_v = vol_series.groupby(date_str).cumsum()
            vwap = float((cum_vp / (cum_v + 1e-9)).iloc[-1])
        else:
            vwap = current_price
            
        std20 = float(close_series.rolling(window=20, min_periods=1).std().iloc[-1])
        bb_upper = sma20 + (std20 * 2)
        bb_lower = sma20 - (std20 * 2)
        
        tr1 = high_series - low_series
        tr2 = (high_series - close_series.shift(1)).abs()
        tr3 = (low_series - close_series.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = float(tr.rolling(window=14, min_periods=1).mean().iloc[-1])
        
        p = current_price + 1e-9
        sma20_scaled = float(np.clip((current_price - sma20) / p, -1.0, 1.0))
        sma60_scaled = float(np.clip((current_price - sma60) / p, -1.0, 1.0))
        rsi14_scaled = float(np.clip(rsi14 / 100.0, 0.0, 1.0))
        vwap_scaled = float(np.clip((current_price - vwap) / p, -1.0, 1.0))
        bb_upper_scaled = float(np.clip((bb_upper - current_price) / p, -1.0, 1.0))
        bb_lower_scaled = float(np.clip((current_price - bb_lower) / p, -1.0, 1.0))
        atr_scaled = float(np.clip(atr14 / p, 0.0, 1.0))
        
        indicator_obs = np.array([
            sma20_scaled, sma60_scaled, rsi14_scaled,
            vwap_scaled, bb_upper_scaled, bb_lower_scaled, atr_scaled
        ], dtype=np.float32)
        
        # 5. Combine All
        final_obs = np.concatenate([
            obs_1d,
            stock_onehot,
            portfolio_state,
            indicator_obs
        ]).astype(np.float32)
        
        return final_obs

    def _update_ui_signals(self, action, signal_text, probs):
        vm = getattr(self.config_manager, "_injected_live_vm", None)
        if vm:
            if self.symbol not in vm.symbols_summary:
                vm.symbols_summary[self.symbol] = {"name": self.symbol, "price": 0, "ai_signal": "-", "holdings": 0}
            vm.symbols_summary[self.symbol]["ai_signal"] = signal_text
            if vm.selected_symbol == self.symbol:
                # 5-액션: Buy = max(probs[1], probs[2]), Sell = max(probs[3], probs[4])
                buy_conf  = int(max(probs[1], probs[2]) * 100) if len(probs) >= 5 else int(probs[1] * 100)
                sell_conf = int(max(probs[3], probs[4]) * 100) if len(probs) >= 5 else int(probs[2] * 100)
                conf = {"Hold": int(probs[0]*100), "Buy": buy_conf, "Sell": sell_conf}
                vm.sig_ai_confidence_updated.emit(conf)
            vm._ui_dirty = True

    def _ui_log(self, message: str):
        """중요 메시지를 대시보드 UI 로그 창으로 전송"""
        vm = getattr(self.config_manager, "_injected_live_vm", None)
        if vm:
            # ViewModel의 append_log는 내부적으로 sig_log_appended 시그널을 emit함
            vm.append_log(message)

    async def destroy(self):
        """
        [동적 유니버스] 조건 이탈 및 잔고 0 확인 후 엔진을 안전하게 종료합니다.
        더 이상 이 종목에 대해 메모리를 낭비하지 않도록 버퍼를 비웁니다.
        """
        self.logger.info(f"[{self.symbol}] 🛑 엔진 안전 종료 (Graceful Shutdown) 절차 시작")
        
        # [좀비 방어] 즉시 파괴 상태 진입 (이후 틱/웜업 원천 차단)
        self._is_destroyed = True
        
        if getattr(self, 'warmup_task', None) and not self.warmup_task.done():
            self.warmup_task.cancel()
            self.logger.info(f"[{self.symbol}] 🛑 진행 중인 웜업 태스크 강제 취소 완료.")

        # 1. 펜딩 중인 주문 파이프라인이 끝날 때까지 대기 (최대 5초)
        wait_cnt = 0
        while self._is_order_pending and wait_cnt < 10:
            await asyncio.sleep(0.5)
            wait_cnt += 1
            
        # 2. 내부 버퍼 정리
        self.minute_buffer.clear()
        if hasattr(self.normalizer, 'history'):
            self.normalizer.history.clear()
            
        self.is_warmed_up = False
        self.logger.info(f"[{self.symbol}] 🗑️ 엔진 메모리 해제 및 파괴 완료.")
