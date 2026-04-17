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

        # 임의의 특성 크기 50으로 가정
        self.feature_dim = 50
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
        return np.zeros(self.feature_dim, dtype=np.float32)

    def _get_info(self):
        return {
            "balance": self.balance,
            "holdings": self.holdings,
            "unexecuted_orders": self.order_manager.has_unexecuted_orders()
        }

    def action_masks(self):
        masks = [True, False, False]

        if self.order_manager.has_unexecuted_orders():
            return masks

        current_price = self._get_current_price()
        if self.balance >= current_price:
            masks[1] = True

        if self.holdings > 0:
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
        return 1000.0
