import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import logging

class ScalpingTradingEnv(gym.Env):
    """
    Maskable PPO와 호환되는 단일 종목 스캘핑 커스텀 환경
    """
    def __init__(self, data_collector, order_manager, config):
        super().__init__()
        self.data_collector = data_collector
        self.order_manager = order_manager
        self.config = config
        self.logger = logging.getLogger("ScalpingTradingEnv")

        # [Price, Volume, OIR, Volatility, Aggressiveness]
        self.single_feature_dim = 5
        self.seq_len = config.get('seq_len', 10)
        self.feature_dim = self.single_feature_dim * self.seq_len

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.feature_dim,), dtype=np.float32
        )

        # Action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)

        self.balance = config.get('initial_balance', 10000000)
        self.holdings = 0
        self.current_step = 0
        self.reward_history = []

        # Historical / Backtest 모드에서 사용할 데이터
        self.historical_data_dict = config.get("historical_data_dict", None)
        self.historical_data = config.get("historical_data", None)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 다중 종목 샘플링 모드 체크
        if self.historical_data_dict:
            available_symbols = list(self.historical_data_dict.keys())
            if available_symbols:
                selected_sym = random.choice(available_symbols)
                self.historical_data = self.historical_data_dict[selected_sym]
                self.config['symbol'] = selected_sym
                self.logger.info(f"에피소드 초기화: 랜덤 종목 선택 => {selected_sym} (데이터 {len(self.historical_data)}건)")

        self.balance = self.config.get('initial_balance', 10000000)
        self.holdings = 0
        self.current_step = 0
        self.reward_history = []

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _get_observation(self):
        # 만약 학습/백테스트 모드라서 historical_data가 주어졌다면,
        # data_collector 대신 historical_data 배열에서 상태를 구성
        if self.historical_data is not None:
            max_idx = len(self.historical_data) - 1
            idx = min(self.current_step, max_idx)

            # (단순화: historical_data에서 seq_len 만큼 추출하여 패딩)
            seq = []
            for i in range(self.seq_len):
                target_idx = max(0, idx - self.seq_len + 1 + i)
                # Assuming data is a dict with raw prices/volumes, we'd normally pass it to FeatureEngineer.
                # For this snippet's scope, we construct a dummy or simple normalized state.
                row = self.historical_data[target_idx]
                state_slice = np.array([
                    row.get("price", 1000),
                    row.get("volume", 0),
                    0.0, 0.0, 0.0 # OIR, Volatility, Agg (Mocked for historical if not pre-calculated)
                ], dtype=np.float32)
                seq.append(state_slice)
            return np.concatenate(seq)

        symbol = self.config.get('symbol')
        if hasattr(self.data_collector, "get_latest_state"):
            # DataCollector is now expected to return a sequence of states flattened
            state = self.data_collector.get_latest_state(symbol, seq_len=self.seq_len)
            if state is not None and len(state) == self.feature_dim:
                return state
        return np.zeros(self.feature_dim, dtype=np.float32)

    def _get_info(self):
        return {
            "balance": self.balance,
            "holdings": self.holdings,
            "unexecuted_orders": self.order_manager.has_unexecuted_orders()
        }

    def action_masks(self):
        masks = [True, False, False]

        symbol = self.config.get('symbol')

        # 격리(Isolation): 특정 종목의 미체결 주문만 확인
        if self.order_manager.has_unexecuted_orders(symbol=symbol):
            return masks

        # 실거래 동기화
        current_price = self._get_current_price()

        # 실제 계좌 잔고를 조회할 수 없으므로 가상 잔고 또는 글로벌/종목 리스크 한도를 참조 가능
        # If market state is LIQUIDATING, prevent BUY mask
        can_buy = True
        if hasattr(self.config, 'get'):
            # The env doesn't have a direct reference to MarketScheduler, but we can assume an external check or a flag
            # For now, we rely on MarketScheduler handling liquidation overrides itself
            pass

        # 백테스트나 시뮬레이션용 로직 (실전에서는 예수금 확인 로직 연동 필요)
        if self.balance >= current_price and can_buy:
            masks[1] = True

        # 보유 수량은 실제 order_manager의 상태와 동기화
        actual_holdings = self.order_manager.holdings.get(symbol, self.holdings)
        if actual_holdings > 0:
            masks[2] = True

        return masks

    def step(self, action):
        current_price = self._get_current_price()
        slippage = self.config.get('slippage', 0.0005)

        # 1. 행동 이전의 총자산 가치 (현금 잔고 + 보유 주식 가치)
        prev_net_worth = self.balance + (self.holdings * current_price)

        # 2. 액션 수행 (수수료/슬리피지 적용)
        if action == 1: # Buy
            cost = current_price * (1 + slippage)
            self.balance -= cost
            self.holdings += 1
        elif action == 2: # Sell
            revenue = current_price * (1 - slippage)
            self.balance += revenue
            self.holdings -= 1

        # 3. 행동 이후의 총자산 가치
        new_net_worth = self.balance + (self.holdings * current_price)

        # 4. 자산 증감분을 초기 자본금 대비 수익률(%)로 변환 및 스케일링
        initial_balance = float(self.config.get('initial_balance', 10000000))
        pct_change = (new_net_worth - prev_net_worth) / initial_balance
        step_reward = pct_change * 100.0

        # 극단적인 값이 나오지 않도록 클리핑 (예: -10 ~ 10 사이)
        step_reward = float(np.clip(step_reward, -10.0, 10.0))

        # 5. 시간 패널티 (Hold 방지) 스케일링된 보상에 맞게 미세 조정
        if action == 0 and self.holdings == 0:
            step_reward -= 0.001 # 무포지션 관망 시 미세한 패널티

        self.reward_history.append(step_reward)
        if len(self.reward_history) > 10:
            returns = np.array(self.reward_history)
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-9)
            # Sharpe ratio based bonus/penalty
            step_reward += sharpe_ratio * 0.1

        # 다시 한 번 클리핑 (샤프 지수 보너스 적용 후에도 안정성 유지)
        step_reward = float(np.clip(step_reward, -10.0, 10.0))

        self.current_step += 1
        obs = self._get_observation()
        info = self._get_info()

        terminated = self.balance < 0
        truncated = False

        if self.historical_data is not None and self.current_step >= len(self.historical_data) - 1:
            truncated = True

        return obs, step_reward, terminated, truncated, info

    def _get_current_price(self):
        if self.historical_data is not None:
            max_idx = len(self.historical_data) - 1
            idx = min(self.current_step, max_idx)
            return float(self.historical_data[idx].get("price", 1000.0))

        symbol = self.config.get('symbol')
        if hasattr(self.data_collector, "get_latest_price"):
            price = self.data_collector.get_latest_price(symbol)
            if price > 0:
                return price
        return 1000.0
