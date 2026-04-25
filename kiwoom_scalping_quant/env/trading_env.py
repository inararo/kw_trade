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

        self.feature_mode = config.get('feature_mode', 'basic')
        
        # [모드 분기] Basic: 5차원, Advanced: 10차원 (3개 추가 지표 반영)
        if self.feature_mode == 'advanced':
            self.single_feature_dim = 10
        else:
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
        self.cooldown_steps = 10  # [상향] 매수/매도 사이 최소 간격
        self.grace_period = 10     # [신규] 매수 후 패널티 면제 기간
        self.steps_since_buy = 0
        self.steps_since_sell = 0 # [신규] 매도 후 경과 스텝
        self.initial_price = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # [혁신] 외부 옵션을 통한 상태 주입 (백테스트 연속성 유지용)
        options = options or {}
        self.balance = options.get('current_balance', self.config.get('initial_balance', 10000000))
        forced_start_step = options.get('start_step', None)

        # 다중 종목 샘플링 모드 체크 (복구)
        if self.historical_data_dict:
            available_symbols = list(self.historical_data_dict.keys())
            if available_symbols:
                selected_sym = random.choice(available_symbols)
                self.historical_data = self.historical_data_dict[selected_sym]
                self.config['symbol'] = selected_sym
                self.logger.info(f"에피소드 초기화: 랜덤 종목 선택 => {selected_sym} (데이터 {len(self.historical_data)}건)")
        
        self.holdings = 0
        
        # [피처 전처리 캐싱] Advanced 모드일 경우 전체 배열을 한 번에 Pandas로 전처리
        if self.historical_data is not None and self.feature_mode == 'advanced':
            from core.feature_engineer import AdvancedFeatureEngineer
            symbol = self.config.get('symbol', 'unknown')
            
            if hasattr(self, '_feature_cache') is False:
                self._feature_cache = {}
                
            if symbol not in self._feature_cache:
                self.logger.info(f"[{symbol}] Advanced Feature DataFrame 계산 및 캐싱 중...")
                self._feature_cache[symbol] = AdvancedFeatureEngineer.process_historical_data(self.historical_data)
            self.precomputed_features = self._feature_cache[symbol]
        
        # [기능 개선] 시작점 결정 로직 (forced_start_step 우선)
        if self.historical_data is not None:
            data_len = len(self.historical_data)
            if forced_start_step is not None:
                self.current_step = forced_start_step
                self.end_step = data_len - 1
            elif self.config.get("mode") == "backtest":
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
        self.steps_since_sell = 100 # 초기에는 바로 매매 가능하도록 큰 값 설정
        self.avg_entry_price = 0.0  # [추가] 매수 단가 추적용
        
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

            # [O(1) 캐싱 대응] Advanced 모드면 캐시에서 바로 꺼내옴
            if self.feature_mode == 'advanced' and hasattr(self, 'precomputed_features'):
                seq = []
                for i in range(self.seq_len):
                    target_idx = max(0, idx - self.seq_len + 1 + i)
                    # 만약 데이터 길이가 짧아 target_idx가 범위를 벗어나면 패딩
                    if len(self.precomputed_features) > target_idx:
                        seq.append(self.precomputed_features[target_idx])
                    else:
                        seq.append(np.zeros(self.single_feature_dim, dtype=np.float32))
                return np.concatenate(seq)

            # (Basic 모드 로직: 원시 가격 -> 수익률 및 정규화 데이터로 변환)
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

        # BUY: 잔고가 현재가 이상일 때만 허용 + [추가] 매도 후 쿨다운 체크
        if self.balance >= current_price:
            if self.steps_since_sell >= self.cooldown_steps:
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
        action_executed = action
        invalid_action_penalty = 0.0
        
        if action == 1: # Buy
            cost = current_price * (1 + slippage)
            if self.holdings > 0:
                # [버그 수정] 이미 보유 중인데 또 매수하려 함 → Hold로 강제 전환 및 벌점
                action_executed = 0
                invalid_action_penalty = -0.01
                self.logger.debug(f"Action 1(BUY) 무효: 이미 {self.holdings}주 보유 중. 강제 Hold 전환.")
            elif self.balance < cost:
                # 잔고 부족
                action_executed = 0
                invalid_action_penalty = -0.01
                self.logger.debug(f"Action 1(BUY) 무효: 잔고 부족({self.balance} < {cost}). 강제 Hold 전환.")
            else:
                self.balance -= cost
                self.holdings += 1
                self.avg_entry_price = cost  # [추가] 매수 단가 저장
                
        elif action == 2: # Sell
            if self.holdings <= 0:
                # [버그 수정] 팔 주식이 없는데 매도하려 함 → Hold로 강제 전환 및 벌점
                action_executed = 0
                invalid_action_penalty = -0.01
                self.logger.debug("Action 2(SELL) 무효: 보유 주식 없음. 강제 Hold 전환.")
            else:
                revenue = current_price * (1 - slippage)
                self.balance += revenue
                self.holdings -= 1
                self.steps_since_sell = 0 # [추가] 매도 카운트 리셋

        # 3. 행동 이후의 총자산 가치 (실제 실행된 action_executed 기준)
        new_net_worth = self.balance + (self.holdings * current_price)

        step_reward = 0.0
        
        # [수수료 체감] 매매 시 발생하는 고정 비용(Transaction Cost) 부여
        transaction_cost = 0.05 # 수수료 + 슬리피지 추정치
        if action_executed == 1:
            step_reward -= transaction_cost
        elif action_executed == 2:
            step_reward -= transaction_cost
            # [혁신] 미실현 수익 보상 제거 & 실현 수익(Realized PnL) 중심 보상 체계 적용
            revenue = current_price * (1 - slippage)
            realized_profit = revenue - self.avg_entry_price
            profit_pct = (realized_profit / self.avg_entry_price) * 100.0 if self.avg_entry_price > 0 else 0.0

            if profit_pct > 0:
                # 수익 실현 시 강력한 도파민 보강 (10배 증폭 + 성공 보너스)
                step_reward += (profit_pct * 10.0) + 1.0
                self.logger.info(f"   >>> [DOPAMINE] 실현 수익 발생! 보상 증폭 적용. Reward: {step_reward:.4f}")
            else:
                # 손실 시에는 손실 분만큼 직접 차감
                step_reward += profit_pct
            
            self.avg_entry_price = 0.0 # 매도 후 평단가 리셋
        
        # [벌점] 불가능한 액션 시도에 대한 패널티 추가
        step_reward += invalid_action_penalty

        # 쿨다운용 카운트 업데이트
        if action_executed == 1:
            self.steps_since_buy = 0
        elif self.holdings > 0:
            self.steps_since_buy += 1

        # 4. [패널티 조정] 상태별 차등 시간 패널티 부여 (Hold Bias 및 존버 방지)
        if action_executed == 0:
            if self.holdings > 0:
                # [혁신] 유예 기간(Grace Period) 동안은 패널티 면제하여 패닉셀 방지
                if self.steps_since_buy > self.grace_period:
                    step_reward -= 0.0050
                else:
                    # 유예 기간 중에는 패널티 0 (인내심 유도)
                    pass
            else:
                # 무포지션 관망 패널티 (기존 0.002)
                step_reward -= 0.0020

        # 불필요한 연타 방지를 위해 모든 액션 시 카운트 증가
        self.steps_since_sell += 1

        # 극단적인 값이 나오지 않도록 클리핑 (예: -10 ~ 10 사이)
        step_reward = float(np.clip(step_reward, -10.0, 10.0))

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

        # [기능 개선] 날짜 변경 감지 (Day-Break Reset)
        # 같은 날짜 안에서만 에피소드가 이어지도록 강제하여 오버나잇 왜곡 방지
        day_changed = False
        if self.historical_data is not None and self.current_step < self.end_step:
            curr_ts = self.historical_data[self.current_step - 1].get("timestamp", "")
            next_ts = self.historical_data[self.current_step].get("timestamp", "")
            if curr_ts and next_ts and curr_ts[:10] != next_ts[:10]:
                day_changed = True
                self.logger.info(f"날짜 변경 감지 ({curr_ts[:10]} -> {next_ts[:10]}). 에피소드를 종료합니다.")

        # 고정된 에피소드 길이(max_steps) 도달 시 또는 날짜 변경 시 종료
        if self.historical_data is not None:
            if self.current_step >= self.end_step or day_changed:
                truncated = True

        # [추가] 에피소드 종료 시 강제 청산 (Force Close) 및 오버나잇 패널티
        if (terminated or truncated) and self.holdings > 0:
            revenue = current_price * (1 - slippage)
            self.balance += revenue
            self.holdings -= 1
            realized_profit = revenue - self.avg_entry_price
            profit_pct = (realized_profit / self.avg_entry_price) * 100.0 if self.avg_entry_price > 0 else 0.0
            
            # 오버나잇 강제 청산 패널티 부여 (-5.0)
            step_reward += profit_pct - 5.0
            self.logger.warning(f"에피소드 종료 강제 청산! (Overnight Penalty 부과) PnL: {profit_pct:.2f}%")

        # [추가] 실제 실행된 액션 정보를 info에 담아 시각화 도구가 필터링할 수 있게 도움
        info["action_executed"] = action_executed

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
