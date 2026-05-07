import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import logging
from collections import deque
import pandas as pd

class ScalpingTradingEnv(gym.Env):
    """
    [Expert Baseline] 다종목 학습 최적화 환경 엔진.
    1. 보상 % 스케일링 (x10)
    2. 중도 리턴 버그 해결
    3. 스케일 불변 피처 적용
    4. 동적 종목 임베딩
    """
    def __init__(self, data_collector, order_manager, config):
        super().__init__()
        self.data_collector = data_collector
        self.order_manager = order_manager
        self.config = config
        self.logger = logging.getLogger("ScalpingTradingEnv")

        self.feature_mode = config.get('feature_mode', 'basic')
        self.window_size = config.get('window_size', 10)
        # [스마트 샘플링] Volume Spike 구간 우선 에피소드 시작 옵션
        self.use_smart_sampling = config.get('use_smart_sampling', False)
        # [추가] 에피소드 길이 개선을 위한 신규 옵션들
        self.always_start_day_begin = config.get('always_start_day_begin', False)
        self.allow_overnight_episodes = config.get('allow_overnight_episodes', False)
        
        # [모드 분기] 
        self.single_feature_dim = 11 if self.feature_mode == 'advanced' else 5
        self.feature_dim = self.single_feature_dim * self.window_size
        # [분할매수/매도] position_ratio, unrealized_pnl 2개 추가
        self.portfolio_state_dim = 2
        # [눌림목 지표] SMA_20, SMA_60, RSI_14 → 스케일링 후 3차원
        self.indicator_dim = 3
        # 현재 스텝의 지표값 캐시 (step()에서 보너스 판단용)
        self._current_indicators: dict = {}

        # [4] 다종목 학습을 위한 유니버스 정보 및 동적 임배딩 설정
        self.historical_data_dict = config.get("historical_data_dict", {})
        self.historical_data = config.get("historical_data", None) # [FIX] 백테스트 데이터 주입 복구
        
        # [FIX] 백테스트 및 실거래 시 차원 일치 보장
        global_symbols = config.get("all_symbols", [])
        if len(global_symbols) > 0:
            self.all_symbols = sorted(list(set(global_symbols)))
        else:
            self.all_symbols = sorted(list(self.historical_data_dict.keys())) if self.historical_data_dict else []
            
        self.symbol_to_idx = {sym: i for i, sym in enumerate(self.all_symbols)}
        self.current_symbol_idx = 0

        # [2] 종목 임베딩(One-hot) 차원 적응형 매칭 (Smart-Padding)
        # target_dim이 주어지면 모델에 맞춰 역산하고, 없으면 기본 100 사용
        target_dim = config.get("target_dim")
        if target_dim:
            # portfolio_state_dim(2) + indicator_dim(3)을 함께 차감하여 총 obs가 target_dim에 수렴하도록 역산
            self.stock_id_dim = max(1, target_dim - self.feature_dim - self.portfolio_state_dim - self.indicator_dim)
            self.max_num_symbols = self.stock_id_dim
            self.logger.info(
                f"Target Dimension Detected: Adapting stock_id_dim to {self.stock_id_dim} "
                f"(target={target_dim}, feature={self.feature_dim}, "
                f"portfolio={self.portfolio_state_dim}, indicator={self.indicator_dim})"
            )
        else:
            self.max_num_symbols = 100 
            self.stock_id_dim = self.max_num_symbols
        
        # [3] Observation 공간 차원: 특징 + 종목ID + 포트폴리오 상태(2) + 보조지표(3)
        obs_total_dim = self.feature_dim + self.stock_id_dim + self.portfolio_state_dim + self.indicator_dim
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_total_dim,), dtype=np.float32
        )
        self.lookback_buffer = deque(maxlen=self.window_size)
        self.logger.info(f"PPO Env Init: Obs Shape = {self.observation_space.shape} (Total:{obs_total_dim})")

        # 0:Hold, 1:Buy50%, 2:Buy100%, 3:Sell50%, 4:Sell100%
        self.action_space = spaces.Discrete(5)
        self.balance = config.get('initial_balance', 10000000)
        self.holdings = 0
        self.current_step = 0
        self.max_steps = config.get('max_steps', 2000)
        self.end_step = 0
        self.avg_entry_price = 0.0
        self.initial_price = 0
        
        # 쿨다운
        self.cooldown_steps = 10
        self.steps_since_buy = 0
        self.steps_since_sell = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        
        # 다중 종목 무작위 샘플링
        if self.historical_data_dict:
            available_symbols = list(self.historical_data_dict.keys())
            if available_symbols:
                selected_sym = random.choice(available_symbols)
                self.historical_data = self.historical_data_dict[selected_sym]
                self.config['symbol'] = selected_sym
                self.current_symbol_idx = self.symbol_to_idx.get(selected_sym, 0)
        
        self.balance = options.get('current_balance', self.config.get('initial_balance', 10000000))
        self.holdings = 0
        self.steps_since_buy = 0
        self.steps_since_sell = 100
        self.avg_entry_price = 0.0
        self._current_indicators = {}

        # 데이터 시작/종료점 설정
        if self.historical_data is not None:
            # [보조지표] SMA_20 / SMA_60 / RSI_14 자동 계산 (종목/캐시 단위)
            sym_key = self.config.get('symbol', '__default__')
            if not hasattr(self, '_indicator_cache'): self._indicator_cache = {}
            if sym_key not in self._indicator_cache:
                self._indicator_cache[sym_key] = self._compute_indicators(self.historical_data)
            self._indicator_df = self._indicator_cache[sym_key]

            data_len = len(self.historical_data)

            # [순서 중요] Advanced 상태 캐시를 먼저 초기화해야
            # _get_smart_start_step()이 precomputed_features를 참조할 수 있음
            if self.feature_mode == 'advanced':
                from core.feature_engineer import AdvancedFeatureEngineer
                if not hasattr(self, '_feature_cache'): self._feature_cache = {}
                sym = self.config.get('symbol')
                if sym not in self._feature_cache:
                    self._feature_cache[sym] = AdvancedFeatureEngineer.process_historical_data(self.historical_data)
                self.precomputed_features = self._feature_cache[sym]

            if self.config.get("mode") == "backtest":
                # [수정된 부분] 무작정 0부터 시작하지 않고, 최소 60번 캔들(SMA_60 계산을 위해) 또는 window_size부터 시작하도록 강제
                min_safe_step = max(self.window_size, 60)
                req_start_step = options.get('start_step', min_safe_step)

                # 방어 코드: 혹시 밖에서 0을 던졌더라도 안전한 위치로 보정
                self.current_step = max(req_start_step, min_safe_step)
                self.end_step = data_len - 1
                self.logger.info(f"Backtest Reset: Starting at {self.current_step} / End at {self.end_step}")
            else:
                # 학습 모드: 스마트 샘플링 or 순수 랜덤 분기
                min_safe_step = max(self.window_size, 60)  # 학습 때도 0이 아닌 안전 지대부터 시작
                max_start = max(min_safe_step, data_len - self.max_steps - 1)

                if 'start_step' in options:
                    self.current_step = max(options['start_step'], min_safe_step)
                elif self.use_smart_sampling:
                    # [개선] 스마트 샘플링 시 필요에 따라 장 시작 시점으로 보정
                    raw_start = self._get_smart_start_step(max_start)
                    if self.always_start_day_begin:
                        self.current_step = self._get_day_start_index(raw_start)
                    else:
                        self.current_step = raw_start
                else:
                    # [기존 유지] 순수 랜덤 샘플링
                    raw_start = random.randint(0, max_start)
                    if self.always_start_day_begin:
                        self.current_step = self._get_day_start_index(raw_start)
                    else:
                        self.current_step = raw_start

                self.end_step = min(data_len - 1, self.current_step + self.max_steps)
            
            self.initial_price = self._get_current_price()

        # [Zero-padding 방지] 버퍼 초기화
        # current_step 직전 window_size개의 실제 캔들 데이터로 채운다.
        # 기존 방식(동일 스텝 반복)은 ret=0, rel_p=0으로 모두 0이 되는 문제가 있었음.
        self.lookback_buffer.clear()
        buf_start = max(0, self.current_step - self.window_size + 1)
        for fill_idx in range(buf_start, self.current_step + 1):
            self.lookback_buffer.append(self._extract_single_feature(fill_idx))
        # 데이터가 window_size보다 부족하면 첫 피처로 앞부분 패딩
        if len(self.lookback_buffer) < self.window_size:
            first_feat = self._extract_single_feature(buf_start)
            while len(self.lookback_buffer) < self.window_size:
                self.lookback_buffer.appendleft(first_feat)

        return self._get_observation(), self._get_info()


    def _get_smart_start_step(self, max_start: int) -> int:
        """
        [스마트 에피소드 샘플링]
        - 80%: Volume Spike 구간(vol_activity > 1.5)에서 우선 샘플링
        - 20%: 과적합 방지를 위한 순수 랜덤 샘플링
        Volume Spike 후보군이 없으면 자동으로 순수 랜덤으로 폴백.
        """
        if max_start <= 0:
            return 0

        # Advanced 모드에서는 precomputed_features의 vol_activity 컬럼 활용
        # Basic 모드에서는 raw historical_data의 volume 직접 계산
        vol_activity = None
        try:
            if self.feature_mode == 'advanced' and hasattr(self, 'precomputed_features') and self.precomputed_features is not None:
                # vol_activity는 index 5 (AdvancedFeatureEngineer 컬럼 순서 기준)
                # ['disparity_ma5', 'disparity_ma20', 'disparity_vwap', 'bb_pos', 'rsi', 'v_activity', ...]
                V_ACTIVITY_IDX = 5
                if self.precomputed_features.shape[1] > V_ACTIVITY_IDX:
                    vol_activity = self.precomputed_features[:, V_ACTIVITY_IDX]
            else:
                # Basic 모드: raw volume에서 20봉 이동평균 대비 비율 직접 계산
                volumes = np.array([d.get('volume', 0) for d in self.historical_data], dtype=np.float32)
                rolling_mean = pd.Series(volumes).rolling(window=20).mean().fillna(volumes.mean()).values
                vol_activity = volumes / (rolling_mean + 1e-9)
        except Exception as e:
            self.logger.warning(f"[스마트 샘플링] vol_activity 계산 실패, 랜덤 폴백: {e}")

        # 80% 확률로 Volume Spike 구간에서 샘플링
        if vol_activity is not None and random.random() < 0.8:
            SPIKE_THRESHOLD = 1.5  # 20봉 평균 대비 1.5배 이상을 Volume Spike로 판단
            # max_start 이내의 인덱스만 후보로 (에피소드가 끝까지 실행될 수 있도록)
            candidate_indices = np.where(
                (vol_activity[:max_start + 1] > SPIKE_THRESHOLD)
            )[0]

            if len(candidate_indices) > 0:
                chosen = int(random.choice(candidate_indices))
                self.logger.debug(
                    f"[스마트 샘플링] Volume Spike 구간 선택: step={chosen}, "
                    f"vol_activity={vol_activity[chosen]:.2f}x (후보 {len(candidate_indices)}개)"
                )
                return chosen
            else:
                self.logger.debug("[스마트 샘플링] Volume Spike 후보 없음 → 랜덤 폴백")

        # 20% 확률 or 후보 없을 때: 순수 랜덤
        return random.randint(0, max_start)

    def _get_day_start_index(self, idx: int) -> int:
        """주어진 인덱스가 포함된 날짜의 첫 번째 데이터 인덱스를 찾습니다."""
        if self.historical_data is None: return idx
        try:
            target_date = str(self.historical_data[idx].get("timestamp", ""))[:10]
            # 이전 방향으로 탐색하여 날짜가 바뀌는 지점을 찾음
            curr = idx
            while curr > 0:
                prev_date = str(self.historical_data[curr-1].get("timestamp", ""))[:10]
                if prev_date != target_date:
                    break
                curr -= 1
            return curr
        except Exception:
            return idx

    def _compute_indicators(self, data: list) -> pd.DataFrame:
        """
        [보조지표 자동 계산]
        입력 데이터(list of dict)에서 SMA_20 / SMA_60 / RSI_14를 계산합니다.
        NaN이 발생하는 앞부분 행은 ffill 후 0으로 채웁니다.
        반환: 원본 데이터와 인덱스가 1:1 대응되는 DataFrame
        """
        prices = pd.Series([float(d.get('price', 0)) for d in data], dtype=np.float64)

        sma20 = prices.rolling(window=20, min_periods=1).mean()
        sma60 = prices.rolling(window=60, min_periods=1).mean()

        # RSI-14 수동 계산 (pandas_ta 없이도 동작)
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-9)
        rsi14 = 100.0 - (100.0 / (1.0 + rs))

        df = pd.DataFrame({'SMA_20': sma20, 'SMA_60': sma60, 'RSI_14': rsi14})
        df.ffill(inplace=True)
        df.fillna(0.0, inplace=True)
        return df

    def _get_indicator_obs(self, idx: int) -> np.ndarray:
        """
        현재 스텝의 보조지표를 0~1 범위로 스케일링하여 반환.
          - SMA_20 / SMA_60: 현재가 대비 비율 → -1~1로 클리핑 (0=동일, +1=가격이 SMA의 2배)
          - RSI_14: 0~100 → 0~1 정규화
        부수적으로 _current_indicators 캐시도 업데이트합니다.
        """
        if not hasattr(self, '_indicator_df') or self._indicator_df is None:
            self._current_indicators = {'SMA_20': 0.0, 'SMA_60': 0.0, 'RSI_14': 50.0}
            return np.zeros(self.indicator_dim, dtype=np.float32)

        idx = min(idx, len(self._indicator_df) - 1)
        row = self._indicator_df.iloc[idx]
        current_price = self._get_current_price()

        sma20 = float(row['SMA_20'])
        sma60 = float(row['SMA_60'])
        rsi14 = float(row['RSI_14'])

        # 캐시 갱신 (step()의 보너스 판단에서 사용)
        self._current_indicators = {'SMA_20': sma20, 'SMA_60': sma60, 'RSI_14': rsi14}

        # 스케일링
        p = current_price + 1e-9
        sma20_scaled = float(np.clip((current_price - sma20) / p, -1.0, 1.0))
        sma60_scaled = float(np.clip((current_price - sma60) / p, -1.0, 1.0))
        rsi14_scaled  = rsi14 / 100.0  # 0 ~ 1

        return np.array([sma20_scaled, sma60_scaled, rsi14_scaled], dtype=np.float32)

    def _extract_single_feature(self, idx):
        """[3] 스케일 불변 피처 추출 로직 (Basic/Advanced 통합)"""
        if self.historical_data is None: return np.zeros(self.single_feature_dim, dtype=np.float32)
        
        idx = min(idx, len(self.historical_data) - 1)
        
        if self.feature_mode == 'advanced' and hasattr(self, 'precomputed_features'):
            return self.precomputed_features[idx]

        # Basic 모드: 가격 수익률 및 변화율 기반
        row = self.historical_data[idx]
        prev_row = self.historical_data[max(0, idx-1)]
        curr_p, prev_p = float(row.get("price", 1000)), float(prev_row.get("price", 1000))
        
        ret = (curr_p - prev_p) / (prev_p + 1e-9) * 100.0
        rel_p = (curr_p - self.initial_price) / (self.initial_price + 1e-9) * 10.0
        vol_ret = (float(row.get("volume", 0)) - float(prev_row.get("volume", 0))) / (float(prev_row.get("volume", 0)) + 1e-9)
        
        return np.array([
            np.clip(ret, -5, 5), 
            np.clip(rel_p, -10, 10), 
            np.clip(vol_ret, -10, 10),
            row.get("OIR", 0.0), row.get("Volatility", 0.0)
        ], dtype=np.float32)

    def _get_portfolio_state(self) -> np.ndarray:
        """현재 포지션 비율과 평가 수익률을 반환 (0.0~1.0 범위)"""
        current_price = self._get_current_price()
        initial_balance = self.config.get('initial_balance', 10000000)
        net_worth = self.balance + self.holdings * current_price
        stock_value = self.holdings * current_price
        position_ratio = stock_value / (net_worth + 1e-9)
        position_ratio = float(np.clip(position_ratio, 0.0, 1.0))

        if self.holdings > 0 and self.avg_entry_price > 0:
            unrealized_pnl = (current_price - self.avg_entry_price) / (self.avg_entry_price + 1e-9)
        else:
            unrealized_pnl = 0.0
        unrealized_pnl = float(np.clip(unrealized_pnl, -1.0, 1.0))

        return np.array([position_ratio, unrealized_pnl], dtype=np.float32)

    def _get_observation(self):
        """[2] 특징 배열 + 종목 One-hot + 포트폴리오 상태(2) + 보조지표(3) 결합"""
        features = np.concatenate(list(self.lookback_buffer)).astype(np.float32)

        # 종목 One-hot (최대 max_num_symbols 고정 차원)
        stock_onehot = np.zeros(self.max_num_symbols, dtype=np.float32)
        if hasattr(self, 'current_symbol_idx') and self.current_symbol_idx < self.max_num_symbols:
            stock_onehot[self.current_symbol_idx] = 1.0

        portfolio_state = self._get_portfolio_state()

        # 보조지표: SMA_20_scaled, SMA_60_scaled, RSI_14_scaled
        # _get_indicator_obs는 _current_indicators 캐시도 함께 갱신
        indicator_obs = self._get_indicator_obs(self.current_step)

        return np.concatenate([features, stock_onehot, portfolio_state, indicator_obs])

    def _get_info(self):
        current_price = self._get_current_price()
        net_worth = self.balance + (self.holdings * current_price)
        return {
            "balance": self.balance, 
            "holdings": self.holdings, 
            "net_worth": net_worth, 
            "current_step": self.current_step
        }

    def action_masks(self):
        """5액션 기반 마스킹 (0:Hold 1:Buy50% 2:Buy100% 3:Sell50% 4:Sell100%)"""
        # 기본: Hold만 허용
        masks = [True, False, False, False, False]
        curr_p = self._get_current_price()
        if curr_p <= 0: return masks

        portfolio_state = self._get_portfolio_state()
        position_ratio = portfolio_state[0]

        # [당일 청산 규칙] 15:20 이후 매수 차단
        is_closing_time = False
        if self.historical_data is not None:
            import datetime
            idx = min(self.current_step, len(self.historical_data)-1)
            ts = self.historical_data[idx].get("timestamp")
            if ts and hasattr(ts, "hour"):
                kst_ts = ts + datetime.timedelta(hours=9)
                if kst_ts.hour == 15 and kst_ts.minute >= 20:
                    is_closing_time = True

        can_buy = (self.balance >= curr_p * 1.001
                   and position_ratio < 1.0
                   and self.steps_since_sell >= self.cooldown_steps
                   and not is_closing_time)
        can_sell = self.holdings > 0 and self.steps_since_buy >= self.cooldown_steps

        if can_buy:
            masks[1] = True  # Buy 50%
            masks[2] = True  # Buy 100%
        if can_sell:
            masks[3] = True  # Sell 50%
            masks[4] = True  # Sell 100%
        return masks

    def step(self, action):
        current_price = self._get_current_price()
        slippage = self.config.get('slippage', 0.0005)
        hard_stop_pct = self.config.get('hard_stop_pct', -0.03)  # 기본 -3%
        step_reward = 0.0
        action_executed = action

        # ──────────────────────────────────────────────
        # [오버라이드 1] 하드 스탑 (무한 물타기 방지)
        # unrealized_pnl 이 hard_stop_pct 이하이면 강제 전량 매도
        # ──────────────────────────────────────────────
        if self.holdings > 0 and self.avg_entry_price > 0:
            unrealized_pnl = (current_price - self.avg_entry_price) / (self.avg_entry_price + 1e-9)
            if unrealized_pnl <= hard_stop_pct:
                action = 4  # 전량 매도로 오버라이드
                action_executed = 4
                step_reward -= 1.5  # [완화] 강력한 페널티 축소 (-5.0 -> -1.5)
                self.logger.debug(
                    f"🚨 [HardStop] 평가손 {unrealized_pnl*100:.2f}% → 강제 전량 청산"
                )

        # ──────────────────────────────────────────────
        # [오버라이드 2] 15:20 당일 청산 규칙
        # ──────────────────────────────────────────────
        if self.historical_data is not None:
            import datetime
            idx = min(self.current_step, len(self.historical_data)-1)
            ts = self.historical_data[idx].get("timestamp")
            if ts and hasattr(ts, "hour"):
                kst_ts = ts + datetime.timedelta(hours=9)
                if kst_ts.hour == 15 and kst_ts.minute >= 20:
                    if self.holdings > 0:
                        action = 4  # 전량 매도로 통일
                        action_executed = 4
                        self.logger.debug(f"[DayTrading] 15:20 데드라인 - 강제 청산 ({kst_ts})")

        # ──────────────────────────────────────────────
        # 포트폴리오 상태 계산 (액션 실행 전 기준)
        # ──────────────────────────────────────────────
        net_worth = self.balance + self.holdings * current_price
        stock_value = self.holdings * current_price
        position_ratio = stock_value / (net_worth + 1e-9)

        # ──────────────────────────────────────────────
        # 1. Action Execution
        # ──────────────────────────────────────────────
        if action == 1:  # Buy 40% (총자산의 40% 분할 매수 / 물타기)
            if position_ratio < 1.0 and self.balance > current_price:
                invest_amount = net_worth * 0.40  # 총자산 40% 투자
                invest_amount = min(invest_amount, self.balance * 0.99)  # 잔고 초과 방지
                buy_price = current_price * (1 + slippage)
                shares = int(invest_amount / buy_price)
                if shares > 0:
                    total_cost = shares * buy_price
                    # 평균단가 재계산
                    prev_cost = self.holdings * self.avg_entry_price
                    self.holdings += shares
                    self.balance -= total_cost
                    self.avg_entry_price = (prev_cost + total_cost) / (self.holdings + 1e-9)
                    self.steps_since_buy = 0
                else:
                    step_reward -= 0.001
                    action_executed = 0
            else:
                # 이미 풀매수 상태이거나 잔고 부족
                step_reward -= 0.001
                action_executed = 0

        elif action == 2:  # Buy 60% (총자산의 60% 분할 매수)
            if position_ratio < 1.0 and self.balance > current_price:
                invest_amount = net_worth * 0.60  # 총자산 60% 투자
                invest_amount = min(invest_amount, self.balance * 0.99)  # 잔고 초과 방지
                buy_price = current_price * (1 + slippage)
                shares = int(invest_amount / buy_price)
                if shares > 0:
                    total_cost = shares * buy_price
                    prev_cost = self.holdings * self.avg_entry_price
                    self.holdings += shares
                    self.balance -= total_cost
                    self.avg_entry_price = (prev_cost + total_cost) / (self.holdings + 1e-9)
                    self.steps_since_buy = 0
                else:
                    step_reward -= 0.001
                    action_executed = 0
            else:
                step_reward -= 0.001
                action_executed = 0

        elif action == 3:  # Sell 60% (보유 물량의 60% 분할 매도)
            if self.holdings > 0:
                sell_shares = max(1, int(self.holdings * 0.6))
                sell_price = current_price * (1 - slippage)
                revenue = sell_shares * sell_price
                realized_pnl_pct = (sell_price - self.avg_entry_price) / (self.avg_entry_price + 1e-9) * 100.0
                reward_multiplier = 30.0 if realized_pnl_pct < 0 else 20.0
                step_reward += realized_pnl_pct * reward_multiplier
                if self.steps_since_buy < 5 and realized_pnl_pct > 0:
                    step_reward += 0.1
                self.balance += revenue
                self.holdings -= sell_shares
                # 평단가 유지 (매도 시 평단가는 변하지 않음)
                if self.holdings == 0:
                    self.avg_entry_price = 0.0
                self.steps_since_sell = 0
            else:
                step_reward -= 0.001
                action_executed = 0

        elif action == 4:  # Sell 40% (보유 물량의 40% 분할 매도)
            if self.holdings > 0:
                sell_shares = max(1, int(self.holdings * 0.4))
                sell_price = current_price * (1 - slippage)
                revenue = sell_shares * sell_price
                realized_pnl_pct = (sell_price - self.avg_entry_price) / (self.avg_entry_price + 1e-9) * 100.0
                reward_multiplier = 30.0 if realized_pnl_pct < 0 else 20.0
                step_reward += realized_pnl_pct * reward_multiplier
                if self.steps_since_buy < 5 and realized_pnl_pct > 0:
                    step_reward += 0.1
                    self.logger.debug(f"⚡ 속전속결 익절 보너스! (Hold: {self.steps_since_buy}스텝)")
                self.balance += revenue
                self.holdings -= sell_shares
                # 평단가 유지 (매도 시 평단가는 변하지 않음)
                if self.holdings == 0:
                    self.avg_entry_price = 0.0
                self.steps_since_sell = 0
            else:
                step_reward -= 0.001
                action_executed = 0

        # ──────────────────────────────────────────────
        # 1-1. Action 0: Hold (관망) 페널티
        # ──────────────────────────────────────────────
        if action == 0 and self.holdings == 0:
            # "아무것도 안 하는 죄" - 억지로라도 타점을 찾게 만듦
            step_reward -= 0.001

        # ──────────────────────────────────────────────
        # 1-2. 첫 매수 진입 보너스 (용기 장려)
        # ──────────────────────────────────────────────
        if action in (1, 2) and action_executed in (1, 2):
            # 이전 상태가 무포지션이었는데 매수했다면 (첫 진입)
            if self.holdings > 0 and (self.holdings - shares) == 0:
                step_reward += 0.05
                self.logger.debug(f"🚀 [EntryBonus] 첫 매수 진입 용기 보너스 +0.05")

        # ──────────────────────────────────────────────
        # [눌림목 타점 보너스]
        # 조건: SMA_20 > SMA_60 (상승 추세) AND RSI_14 < 40 (단기 과매도)
        # 위 조건에서 매수를 선택했다면 즉시 보너스 지급
        # ──────────────────────────────────────────────
        if action_executed in (1, 2):  # 실제로 매수가 체결된 경우만
            sma20 = self._current_indicators.get('SMA_20', 0.0)
            sma60 = self._current_indicators.get('SMA_60', 0.0)
            rsi14 = self._current_indicators.get('RSI_14', 50.0)
            is_pullback = (sma20 > sma60) and (rsi14 < 40.0)
            if is_pullback:
                if action_executed == 1:   # Buy 40% — 눌림목 진입 보너스
                    step_reward += 0.05
                    self.logger.debug(f"🎯 [눌림목] Buy40% 타점 보너스 +0.05 (RSI={rsi14:.1f}, SMA20={sma20:.0f}>SMA60={sma60:.0f})")
                elif action_executed == 2:  # Buy 60% — 확신 비중 보너스
                    step_reward += 0.08
                    self.logger.debug(f"🎯 [눌림목] Buy60% 확신 보너스 +0.08 (RSI={rsi14:.1f}, SMA20={sma20:.0f}>SMA60={sma60:.0f})")

        # ──────────────────────────────────────────────
        # 2. 시간 흐름 업데이트
        # ──────────────────────────────────────────────
        self.current_step += 1
        self.steps_since_buy += 1
        self.steps_since_sell += 1

        # ──────────────────────────────────────────────
        # 3. 보유 페널티 (장기 손실 방치 억제)
        # ──────────────────────────────────────────────
        if self.holdings > 0 and self.steps_since_buy > 20:
            current_profit = (current_price - self.avg_entry_price) / (self.avg_entry_price + 1e-9) * 100.0
            if current_profit < 0:
                step_reward -= 0.001
                if self.steps_since_buy % 10 == 0:
                    self.logger.debug(f"⏳ 손실 방치 페널티 (Hold: {self.steps_since_buy}스텝)")
        if self.holdings > 0:
            step_reward -= 0.005

        # ──────────────────────────────────────────────
        # 4. 상태 업데이트 및 종료 판정
        # ──────────────────────────────────────────────
        self.lookback_buffer.append(self._extract_single_feature(self.current_step))

        terminated = self.balance < 0
        truncated = False

        day_changed = False
        if self.historical_data is not None and self.current_step < len(self.historical_data):
            curr_date = str(self.historical_data[self.current_step-1].get("timestamp", ""))[:10]
            next_date = str(self.historical_data[self.current_step].get("timestamp", ""))[:10]
            if curr_date and next_date and curr_date != next_date:
                day_changed = True

        is_backtest = self.config.get("mode") == "backtest"
        should_truncate = False
        if self.current_step >= self.end_step:
            should_truncate = True
        elif day_changed and not is_backtest and not self.allow_overnight_episodes:
            should_truncate = True

        if should_truncate:
            truncated = True
            if self.holdings > 0:
                sell_price = current_price * (1 - slippage)
                revenue = self.holdings * sell_price
                profit_pct = (sell_price - self.avg_entry_price) / (self.avg_entry_price + 1e-9) * 100.0
                step_reward += profit_pct * 20.0
                self.balance += revenue
                self.holdings = 0
                self.avg_entry_price = 0.0
                action_executed = 4

        info = self._get_info()
        info["action_executed"] = action_executed

        return self._get_observation(), float(np.clip(step_reward, -10, 10)), terminated, truncated, info

    def _get_current_price(self):
        if self.historical_data is None: return 0.0
        idx = min(self.current_step, len(self.historical_data)-1)
        return float(self.historical_data[idx].get("price", 1000.0))
