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
        
        # [모드 분기] 
        self.single_feature_dim = 11 if self.feature_mode == 'advanced' else 5
        self.feature_dim = self.single_feature_dim * self.window_size

        # [4] 다종목 학습을 위한 유니버스 정보 및 동적 임배딩 설정
        self.historical_data_dict = config.get("historical_data_dict", {})
        
        # 백테스트/실거래 시 차원 일치를 위한 유니버스 복구
        global_symbols = config.get("all_symbols", [])
        if len(global_symbols) > 0:
            self.all_symbols = sorted(list(set(global_symbols)))
        else:
            self.all_symbols = sorted(list(self.historical_data_dict.keys())) if self.historical_data_dict else []
            
        self.symbol_to_idx = {sym: i for i, sym in enumerate(self.all_symbols)}
        self.current_symbol_idx = 0
        self.stock_id_dim = len(self.all_symbols) if len(self.all_symbols) > 0 else 1
        
        # [3] Observation 공간 차원: 특징(Scale-invariant) + 종목ID(One-hot)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.feature_dim + self.stock_id_dim,), dtype=np.float32
        )
        self.lookback_buffer = deque(maxlen=self.window_size)

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
            max_start = max(0, data_len - self.max_steps - 1)
            self.current_step = random.randint(0, max_start) if self.config.get("mode") != "backtest" else 0
            self.end_step = min(data_len - 1, self.current_step + self.max_steps)
            self.initial_price = self._get_current_price()
            
            # Advanced 캐시 초기화
            if self.feature_mode == 'advanced':
                from core.feature_engineer import AdvancedFeatureEngineer
                if not hasattr(self, '_feature_cache'): self._feature_cache = {}
                sym = self.config.get('symbol')
                if sym not in self._feature_cache:
                    self._feature_cache[sym] = AdvancedFeatureEngineer.process_historical_data(self.historical_data)
                self.precomputed_features = self._feature_cache[sym]

        # 버퍼 초기화
        self.lookback_buffer.clear()
        for _ in range(self.window_size):
            self.lookback_buffer.append(self._extract_single_feature(self.current_step))

        return self._get_observation(), self._get_info()

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
        """[4] 특징 배열 + 동적 종목 One-hot 결합"""
        features = np.concatenate(list(self.lookback_buffer)).astype(np.float32)
        stock_onehot = np.zeros(self.stock_id_dim, dtype=np.float32)
        if self.current_symbol_idx < self.stock_id_dim:
            stock_onehot[self.current_symbol_idx] = 1.0
        return np.concatenate([features, stock_onehot])

    def _get_info(self):
        return {"balance": self.balance, "holdings": self.holdings, "current_step": self.current_step}

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
            cost = current_price * (1 + slippage)
            if self.balance >= cost and self.holdings == 0:
                self.balance -= cost
                self.holdings += 1
                self.avg_entry_price = cost
                self.steps_since_buy = 0
            else:
                action_executed = 0
                
        elif action == 2: # Sell
            if self.holdings > 0:
                revenue = current_price * (1 - slippage)
                self.balance += revenue
                self.holdings -= 1
                self.steps_since_sell = 0
                
                # [1] % 수익률 기반 보상 (x10 도파민 가중치)
                profit_pct = (revenue - self.avg_entry_price) / self.avg_entry_price * 100.0
                step_reward = profit_pct * 10.0
                
                self.avg_entry_price = 0.0
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
            
        if self.current_step >= self.end_step or day_changed:
            truncated = True
            # 장 마감 시 강제 청산 보상 처리
            if self.holdings > 0:
                revenue = current_price * (1 - slippage)
                profit_pct = (revenue - self.avg_entry_price) / self.avg_entry_price * 100.0
                step_reward += profit_pct * 10.0
                self.balance += revenue
                self.holdings = 0

        return self._get_observation(), float(np.clip(step_reward, -10, 10)), terminated, truncated, self._get_info()

    def _get_current_price(self):
        if self.historical_data is None: return 0.0
        idx = min(self.current_step, len(self.historical_data)-1)
        return float(self.historical_data[idx].get("price", 1000.0))
