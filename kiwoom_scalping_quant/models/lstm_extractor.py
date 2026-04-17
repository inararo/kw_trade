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
        self.feature_size = observation_space.shape[0] // seq_len
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.linear = nn.Linear(self.hidden_size, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        obs_reshaped = observations.view(batch_size, self.seq_len, self.feature_size)

        out, (h_n, c_n) = self.lstm(obs_reshaped)

        last_hidden = out[:, -1, :]
        features = self.relu(self.linear(last_hidden))
        return features
