import numpy as np
from numba import njit

@njit(nopython=True, fastmath=True)
def jit_rolling_z_score(history_array: np.ndarray, new_value: float) -> float:
    """
    Data Leakage 없이 새로운 값에 대한 Z-score를 반환합니다.
    Numba의 JIT 컴파일을 통해 파이썬 네이티브 연산 대비 압도적인 속도 향상을 제공합니다.

    :param history_array: shape (N,) 인 과거 가격/거래량 1차원 배열
    :param new_value: 현재 들어온 틱 데이터의 피처 값
    :return: Z-score로 정규화된 값
    """
    n = history_array.shape[0]
    if n < 2:
        return new_value

    mean_val = np.mean(history_array)
    std_val = np.std(history_array)

    if std_val < 1e-8:
        std_val = 1e-8

    return (new_value - mean_val) / std_val

@njit(nopython=True, fastmath=True)
def jit_calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Numba를 활용하여 C수준의 속도로 지수이동평균(EMA)을 계산합니다.
    """
    n = prices.shape[0]
    ema = np.empty(n, dtype=np.float64)
    if n == 0:
        return ema

    multiplier = 2.0 / (period + 1.0)
    ema[0] = prices[0]

    for i in range(1, n):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]

    return ema
