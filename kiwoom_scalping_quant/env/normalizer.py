import numpy as np
from collections import deque
from typing import List, Optional

class OnlineRollingNormalizer:
    """
    온라인 롤링 기반의 Z-score 정규화기.
    특정 피처(예: OIR)는 바이패스하여 원래 값(-1 ~ 1)을 유지할 수 있도록 지원합니다.
    """
    def __init__(self, window_size: int = 1000, bypass_indices: Optional[List[int]] = None):
        self.window_size = window_size
        self.bypass_indices = set(bypass_indices) if bypass_indices else set()
        self.buffer = deque(maxlen=window_size)

    def update_and_normalize(self, x: np.ndarray) -> np.ndarray:
        """
        새로운 관측치 x를 버퍼에 추가하고, 현재 버퍼의 통계량을 사용해 정규화합니다.
        """
        self.buffer.append(x)

        if len(self.buffer) < 2:
            return np.zeros_like(x)

        data = np.array(self.buffer)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        # 0으로 나누는 것 방지
        std[std < 1e-8] = 1.0

        normalized = (x - mean) / std

        # Bypass 처리
        for idx in self.bypass_indices:
            if idx < len(x):
                normalized[idx] = x[idx]

        return normalized
