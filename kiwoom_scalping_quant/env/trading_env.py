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
        # [스마트 샘플링] Volume Spike 구간 우선 에피소드 시작 옵션 (기본값: False)
        self.use_smart_sampling = config.get('use_smart_sampling', False)
        
        # [모드 분기] 
        self.single_feature_dim = 11 if self.feature_mode == 'advanced' else 5
        self.feature_dim = self.single_feature_dim * self.window_size

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
            self.stock_id_dim = max(1, target_dim - self.feature_dim)
            self.max_num_symbols = self.stock_id_dim
            self.logger.info(f"Target Dimension Detected: Adapting stock_id_dim to {self.stock_id_dim}")
        else:
            self.max_num_symbols = 100 
            self.stock_id_dim = self.max_num_symbols
        
        # [3] Observation 공간 차원: 특징 + 가변/고정 종목ID 차원 (최종 차원은 target_dim에 수렴)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.feature_dim + self.stock_id_dim,), dtype=np.float32
        )
        self.lookback_buffer = deque(maxlen=self.window_size)
        total_dim = self.feature_dim + self.stock_id_dim
        self.logger.info(f"PPO Env Init: Obs Shape = {self.observation_space.shape} (Total:{total_dim})")

        self.action_space = spaces.Discrete(3) # 0:Hold, 1:Buy, 2:Sell
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
        
        # 데이터 시작/종료점 설정
        if self.historical_data is not None:
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
                self.current_step = options.get('start_step', 0)
                self.end_step = data_len - 1
                self.logger.info(f"Backtest Reset: Starting at {self.current_step} / End at {self.end_step}")
            else:
                # 학습 모드: 스마트 샘플링 or 순수 랜덤 분기
                max_start = max(0, data_len - self.max_steps - 1)
                if 'start_step' in options:
                    self.current_step = options['start_step']
                elif self.use_smart_sampling:
                    self.current_step = self._get_smart_start_step(max_start)
                else:
                    # [기존 유지] 순수 랜덤 샘플링
                    self.current_step = random.randint(0, max_start)
                self.end_step = min(data_len - 1, self.current_step + self.max_steps)
            
            self.initial_price = self._get_current_price()

        # 버퍼 초기화
        self.lookback_buffer.clear()
        for _ in range(self.window_size):
            self.lookback_buffer.append(self._extract_single_feature(self.current_step))

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

    def _get_observation(self):
        """[2] 특징 배열 + 100차원 고정 Hard-Padding One-hot 결합"""
        features = np.concatenate(list(self.lookback_buffer)).astype(np.float32)
        
        # 무조건 100칸짜리 고정 배열 생성
        stock_onehot = np.zeros(self.max_num_symbols, dtype=np.float32)
        
        # 현재 종목의 인덱스가 100 이내인 경우에만 인코딩 (0 ~ 99)
        if hasattr(self, 'current_symbol_idx') and self.current_symbol_idx < self.max_num_symbols:
            stock_onehot[self.current_symbol_idx] = 1.0
            
        return np.concatenate([features, stock_onehot])

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
        """인위적인 마스킹 없이 잔고/보유량 기반 기본 마스킹만 수행"""
        masks = [True, False, False]
        curr_p = self._get_current_price()
        if curr_p <= 0: return masks
        
        if self.balance >= curr_p * 1.001 and self.holdings == 0 and self.steps_since_sell >= self.cooldown_steps:
            masks[1] = True
        if self.holdings > 0 and self.steps_since_buy >= self.cooldown_steps:
            masks[2] = True
        return masks

    def step(self, action):
        current_price = self._get_current_price()
        slippage = self.config.get('slippage', 0.0005)
        step_reward = 0.0
        action_executed = action
        
        # 1. Action Execution
        if action == 1: # Buy
            if self.balance > 0 and self.holdings == 0:
                # [FIX] 풀베팅 로직: 잔고의 99%를 사용하여 최대 수량 매수
                invest_amount = self.balance * 0.99
                buy_price = current_price * (1 + slippage)
                shares = int(invest_amount / buy_price)
                
                if shares > 0:
                    total_cost = shares * buy_price
                    self.balance -= total_cost
                    self.holdings = shares
                    self.avg_entry_price = buy_price
                    self.steps_since_buy = 0
                else:
                    action_executed = 0
            else:
                action_executed = 0
                
        elif action == 2: # Sell
            if self.holdings > 0:
                # [FIX] 일괄 매도 로직: 보유한 모든 수량 매도
                sell_price = current_price * (1 - slippage)
                revenue = self.holdings * sell_price
                self.balance += revenue
                
                # % 수익률 기반 보상 (x10 도파민 가중치 유지)
                profit_pct = (sell_price - self.avg_entry_price) / self.avg_entry_price * 100.0
                step_reward = profit_pct * 20.0
                
                self.holdings = 0
                self.avg_entry_price = 0.0
                self.steps_since_sell = 0
            else:
                action_executed = 0

        # 4. [FIX] 모든 시간 패널티 제거 (에이전트의 인내심 확보)
        # 포지션 보유 중 매 스텝 부과되던 감점(-0.0050 등)을 완전히 삭제하여 수익 구간까지 무한 홀딩을 가능하게 함
        self.current_step += 1
        self.steps_since_buy += 1
        self.steps_since_sell += 1
        
        # 상태 업데이트
        self.lookback_buffer.append(self._extract_single_feature(self.current_step))
        
        # 종료 판정
        terminated = self.balance < 0
        truncated = False
        
        day_changed = False
        if self.historical_data is not None and self.current_step < len(self.historical_data):
            curr_date = str(self.historical_data[self.current_step-1].get("timestamp", ""))[:10]
            next_date = str(self.historical_data[self.current_step].get("timestamp", ""))[:10]
            if curr_date and next_date and curr_date != next_date: day_changed = True
            
        # [FIX] 백테스트 모드: 실제 데이터의 끝에 도달했을 때만 종료 (날짜 변경 무시)
        is_backtest = self.config.get("mode") == "backtest"
        if self.current_step >= self.end_step or (day_changed and not is_backtest):
            truncated = True
            # 장 마감 시(학습 중) 또는 데이터 종료 시(백테스트) 강제 청산 보상 처리
            if self.holdings > 0:
                sell_price = current_price * (1 - slippage)
                revenue = self.holdings * sell_price
                
                # [FIX] 보상 폭발 버그 수정 (총액 revenue 대신 단가 sell_price 사용)
                profit_pct = (sell_price - self.avg_entry_price) / self.avg_entry_price * 100.0
                step_reward += profit_pct * 20.0
                
                self.balance += revenue
                self.holdings = 0

        return self._get_observation(), float(np.clip(step_reward, -10, 10)), terminated, truncated, self._get_info()

    def _get_current_price(self):
        if self.historical_data is None: return 0.0
        idx = min(self.current_step, len(self.historical_data)-1)
        return float(self.historical_data[idx].get("price", 1000.0))
