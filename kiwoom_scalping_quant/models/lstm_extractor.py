import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import numpy as np
from collections import deque

class OnlineRollingNormalizer:
    """미래 데이터 참조(Data Leakage)가 없는 실시간 Z-score 정규화"""
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def normalize(self, new_feature_vector: np.ndarray) -> np.ndarray:
        self.history.append(new_feature_vector)

        if len(self.history) < 2:
            return new_feature_vector

        history_array = np.array(self.history)
        mean = np.mean(history_array, axis=0)
        std = np.std(history_array, axis=0) + 1e-8

        normalized_vector = (new_feature_vector - mean) / std
        return normalized_vector

class LSTMExtractor(BaseFeaturesExtractor):
    """
    State 벡터를 시퀀스 형태로 취급하여 LSTM을 통과시키는 커스텀 Feature Extractor
    """
    def __init__(self, observation_space: gym.spaces.Box, seq_len: int, features_dim: int = 128, hidden_size: int = 64, num_layers: int = 1):
        super().__init__(observation_space, features_dim)

        self.seq_len = seq_len
        # [FIX] 관측값 중 피처 영역과 종목 ID 영역을 명확히 분리
        # 어드밴스드 기준 11, 베이식 기준 5 차원 (seq_len으로 나눈 몫이 타당함)
        self.single_feature_dim = observation_space.shape[0] // seq_len
        self.feature_dim = self.seq_len * self.single_feature_dim
        self.stock_id_dim = observation_space.shape[0] - self.feature_dim

        self.lstm = nn.LSTM(
            input_size=self.single_feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # 최종 출력 레이어: LSTM 출력 + 종목 ID 차원 결합
        self.linear = nn.Linear(hidden_size + self.stock_id_dim, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # 1. 데이터 분리: (순차 피처 110개)와 (정적 종목 ID n개)
        sequence_data = observations[:, :self.feature_dim]
        static_data = observations[:, self.feature_dim:] # 종목 ID (One-hot)
        
        # 2. LSTM 통과를 위해 리셰이핑 (Batch, Seq, Dim)
        obs_reshaped = sequence_data.view(batch_size, self.seq_len, self.single_feature_dim)
        out, _ = self.lstm(obs_reshaped)
        
        # 3. 마지막 타임스텝의 출력 추출 및 종목 ID 결합
        last_hidden = out[:, -1, :]
        combined = torch.cat([last_hidden, static_data], dim=1)
        
        # 4. 최종 특징 추출
        features = self.relu(self.linear(combined))
        return features
