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
        self.max_steps = config.get('max_steps', 2000) # 한 에피소드당 최대 스텝 수
        self.end_step = 0
        self.reward_history = []

        # Historical / Backtest 모드에서 사용할 데이터
        self.historical_data_dict = config.get("historical_data_dict", None)
        self.historical_data = config.get("historical_data", None)

        # 뇌동매매 방지용 변수
        self.cooldown_steps = 5
        self.steps_since_buy = 0
        self.initial_price = 0

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
        
        # [기능 개선] 랜덤 시작점 로직 도입 (백테스트 모드일 경우 0부터 시작)
        if self.historical_data is not None:
            data_len = len(self.historical_data)
            if self.config.get("mode") == "backtest":
                self.current_step = 0
                self.end_step = data_len - 1
            else:
                # 학습 시에는 다양성을 위해 랜덤 시작점 사용
                max_start_idx = max(0, data_len - self.max_steps - 1)
                self.current_step = random.randint(0, max_start_idx)
                self.end_step = min(data_len - 1, self.current_step + self.max_steps)
            
            self.logger.info(f"에피소드 시작: 모드={self.config.get('mode', 'train')} / 시작점 {self.current_step} / 종료점 {self.end_step}")
        else:
            self.current_step = 0
            self.end_step = 0

        self.reward_history = []
        self.steps_since_buy = 0
        
        # 시작가 저장 (정규화 기준점)
        self.initial_price = self._get_current_price()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _get_observation(self):
        # 만약 학습/백테스트 모드라서 historical_data가 주어졌다면,
        # data_collector 대신 historical_data 배열에서 상태를 구성
        if self.historical_data is not None:
            max_idx = len(self.historical_data) - 1
            idx = min(self.current_step, max_idx)

            # (혁신: 원시 가격 -> 수익률 및 정규화 데이터로 변환)
            seq = []
            prices = [row.get("price", 1000) for row in self.historical_data]
            volumes = [row.get("volume", 0) for row in self.historical_data]
            
            # 기준값 계산
            local_prices = prices[max(0, idx-50):idx+1]
            local_volumes = volumes[max(0, idx-50):idx+1]
            mean_p, std_p = np.mean(local_prices), np.std(local_prices) + 1e-9
            mean_v, std_v = np.mean(local_volumes), np.std(local_volumes) + 1e-9

            for i in range(self.seq_len):
                target_idx = max(0, idx - self.seq_len + 1 + i)
                row = self.historical_data[target_idx]
                
                # Z-Score 정규화 및 상대 수익률 계산
                curr_p = row.get("price", 1000)
                norm_price = (curr_p - mean_p) / std_p
                rel_change = (curr_p - self.initial_price) / (self.initial_price + 1e-9)
                norm_vol = (row.get("volume", 0) - mean_v) / std_v
                
                state_slice = np.array([
                    norm_price, rel_change, norm_vol, 
                    row.get("OIR", 0.0), row.get("Volatility", 0.0)
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
        masks = [True, False, False]  # Hold만 기본 허용

        symbol = self.config.get('symbol')

        # 미체결 주문이 있으면 Hold만 허용
        if self.order_manager.has_unexecuted_orders(symbol=symbol):
            return masks

        # [버그 수정] 실제 가격 조회 - 0이면 매매 불가
        current_price = self._get_current_price()
        if current_price <= 0:
            return masks  # 가격 정보 없음 → Hold 강제

        # BUY: 잔고가 현재가 이상일 때만 허용
        if self.balance >= current_price:
            masks[1] = True

        # SELL: 실제 보유 수량 기준 + 쿨다운 체크
        actual_holdings = self.order_manager.holdings.get(symbol, self.holdings)
        if actual_holdings > 0:
            # 매수 후 최소 5스텝이 지나야 매도 가능
            if self.steps_since_buy >= self.cooldown_steps:
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

        # 4. 자산 증감분 계산
        delta_net_worth = new_net_worth - prev_net_worth
        initial_balance = float(self.config.get('initial_balance', 10000000))
        pct_change = delta_net_worth / initial_balance
        step_reward = pct_change * 100.0
        
        # [수수료 체감] 매매 시 발생하는 고정 비용(Transaction Cost) 부여
        transaction_cost = 0.05 # 수수료 + 슬리피지 추정치
        if action == 1 or action == 2:
            step_reward -= transaction_cost

        # [수익 강화] 매도(Action 2) 시 수익이 발생했다면 보상 증폭
        if action == 2 and delta_net_worth > 0:
            step_reward *= 2.0
            step_reward += 0.5  # 추가 수익 보너스

        # 쿨다운용 카운트 업데이트
        if action == 1:
            self.steps_since_buy = 0
        elif self.holdings > 0:
            self.steps_since_buy += 1

        # 극단적인 값이 나오지 않도록 클리핑 (예: -10 ~ 10 사이)
        step_reward = float(np.clip(step_reward, -10.0, 10.0))

        # 5. [패널티 조정] 시간 패널티 (Hold 방지)
        # 수수료(0.05)보다 작게 설정하여 억지 매매 방지
        if action == 0 and self.holdings == 0:
            step_reward -= 0.005

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

        # [기능 개선] 고정된 에피소드 길이(max_steps) 도달 시 종료
        if self.historical_data is not None and self.current_step >= self.end_step:
            truncated = True

        return obs, step_reward, terminated, truncated, info

    def _get_current_price(self):
        if self.historical_data is not None:
            max_idx = len(self.historical_data) - 1
            idx = min(self.current_step, max_idx)
            return float(self.historical_data[idx].get("price", 1000.0))

        symbol = self.config.get('symbol')
        if hasattr(self.data_collector, "get_latest_price"):
            # [버그 수정] 순수 코드와 _AL 접미사 양쪽 모두 시도
            price = self.data_collector.get_latest_price(symbol)
            if price > 0:
                return price
            price = self.data_collector.get_latest_price(symbol + "_AL")
            if price > 0:
                return price
        return 0.0  # 가격 미확인 시 0 반환 (caller가 BUY 차단)
