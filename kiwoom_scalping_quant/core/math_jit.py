import numpy as np
from numba import njit

@njit(fastmath=True, cache=True)
def calc_volatility_jit(prices: np.ndarray, period: int) -> float:
    """
    최근 period 틱 동안의 가격 표준편차(Volatility) 계산
    """
    if len(prices) < 2:
        return 0.0

    n = min(len(prices), period)
    # 최근 n개의 데이터 슬라이싱
    window = prices[-n:]

    mean = np.sum(window) / n
    var_sum = 0.0
    for i in range(n):
        diff = window[i] - mean
        var_sum += diff * diff

    return np.sqrt(var_sum / (n - 1)) if n > 1 else 0.0

@njit(fastmath=True, cache=True)
def calc_aggressiveness_jit(prices: np.ndarray, volumes: np.ndarray) -> float:
    """
    Tick Rule을 이용해 적극적 순매수 누적량(Aggressive Buy/Sell)을 계산합니다.
    """
    n = len(prices)
    if n < 2:
        return 0.0

    agg_vol = 0.0
    # 직전 틱과의 가격 비교
    for i in range(1, n):
        dp = prices[i] - prices[i-1]
        if dp > 0:
            agg_vol += volumes[i] # Aggressive Buy
        elif dp < 0:
            agg_vol -= volumes[i] # Aggressive Sell
        # dp == 0 인 경우는 무시 (또는 이전 방향성 추종 가능)

    return agg_vol
