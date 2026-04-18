import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ScalpingTradingEnv(gym.Env):
    """
    Maskable PPO와 호환되는 단일 종목 스캘핑 커스텀 환경
    """
    def __init__(self, data_collector, order_manager, config):
        super().__init__()
        self.data_collector = data_collector
        self.order_manager = order_manager
        self.config = config

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.config.get('initial_balance', 10000000)
        self.holdings = 0
        self.current_step = 0
        self.reward_history = []

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _get_observation(self):
        if hasattr(self.data_collector, "get_latest_state"):
            # DataCollector is now expected to return a sequence of states flattened
            state = self.data_collector.get_latest_state(seq_len=self.seq_len)
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
        # 백테스트나 시뮬레이션용 로직 (실전에서는 예수금 확인 로직 연동 필요)
        if self.balance >= current_price:
            masks[1] = True

        # 보유 수량은 실제 order_manager의 상태와 동기화
        actual_holdings = self.order_manager.holdings.get(symbol, self.holdings)
        if actual_holdings > 0:
            masks[2] = True

        return masks

    def step(self, action):
        current_price = self._get_current_price()
        step_reward = 0.0
        slippage = self.config.get('slippage', 0.0005)

        if action == 1: # Buy
            cost = current_price * (1 + slippage)
            self.balance -= cost
            self.holdings += 1

        elif action == 2: # Sell
            revenue = current_price * (1 - slippage)
            self.balance += revenue
            self.holdings -= 1

        self.reward_history.append(step_reward)
        if len(self.reward_history) > 10:
            returns = np.array(self.reward_history)
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-9)
            step_reward += sharpe_ratio * 0.1

        self.current_step += 1
        obs = self._get_observation()
        info = self._get_info()

        terminated = self.balance < 0
        truncated = False

        return obs, step_reward, terminated, truncated, info

    def _get_current_price(self):
        if hasattr(self.data_collector, "get_latest_state"):
            # Get just the latest single tick to avoid unpacking the whole sequence
            state = self.data_collector.get_latest_state(seq_len=1)
            if state is not None and len(state) > 0:
                # Assuming price is at index 0, but it might be normalized.
                # In a real environment, we'd pull the unnormalized current price from the collector.
                # For this implementation's scope, we simulate it or rely on external mock wrapper.
                pass
        return 1000.0
