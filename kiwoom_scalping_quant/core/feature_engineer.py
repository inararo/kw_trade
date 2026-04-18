import numpy as np
from typing import Dict
from core.math_jit import calc_volatility_jit, calc_aggressiveness_jit

class FeatureEngineer:
    """
    실시간 웹소켓 틱 및 호가창 데이터를 받아 미시구조 지표를 계산합니다.
    빠른 연산을 위해 고정 크기(Fixed-size) NumPy 배열의 링 버퍼(Ring Buffer) 패턴을 사용합니다.
    """
    def __init__(self, max_ticks: int = 100):
        self.max_ticks = max_ticks

        # 고정 크기 NumPy 배열 (Ring Buffer)
        self.price_buffer = np.zeros(max_ticks, dtype=np.float32)
        self.volume_buffer = np.zeros(max_ticks, dtype=np.float32)

        self.count = 0  # 현재까지 들어온 누적 데이터 수
        self.head = 0   # 링 버퍼 내 최신 데이터 인덱스

        self.current_oir = 0.0

    def update_orderbook(self, orderbook: Dict) -> float:
        """
        호가창 데이터 수신 시 OIR 갱신 (최우선 5호가)
        orderbook = {"asks": [{"price": p, "qty": q}, ...], "bids": [...]}
        """
        asks = orderbook.get("asks", [])[:5]
        bids = orderbook.get("bids", [])[:5]

        total_ask_qty = sum([item.get("qty", 0) for item in asks])
        total_bid_qty = sum([item.get("qty", 0) for item in bids])

        # OIR: (매수 총잔량 - 매도 총잔량) / (총잔량) => [-1.0 ~ 1.0]
        denom = total_bid_qty + total_ask_qty
        if denom > 0:
            self.current_oir = (total_bid_qty - total_ask_qty) / denom
        else:
            self.current_oir = 0.0

        return self.current_oir

    def _get_ordered_array(self, buffer: np.ndarray) -> np.ndarray:
        """
        링 버퍼를 시간순으로 정렬된 1차원 연속 배열로 반환합니다.
        (Numba JIT 함수들이 시간순 연속 배열을 가정하고 구현되어 있으므로 O(N) 복사가 발생하나,
        max_ticks가 100 정도로 매우 작으므로 C 레벨 복사는 무시할 수준입니다.)
        """
        if self.count < self.max_ticks:
            return buffer[:self.count]

        # 링 버퍼 합치기 (오래된 데이터 -> 최신 데이터)
        return np.concatenate((buffer[self.head:], buffer[:self.head]))

    def update_tick(self, price: float, volume: float) -> Dict[str, float]:
        """
        틱 데이터 수신 시 내역을 업데이트하고 변동성/체결강도를 계산합니다.
        """
        # 링 버퍼에 데이터 삽입
        self.price_buffer[self.head] = price
        self.volume_buffer[self.head] = volume

        self.head = (self.head + 1) % self.max_ticks
        self.count += 1

        # 시간순으로 정렬된 슬라이스 획득 (Numba JIT에 넘기기 위함)
        prices = self._get_ordered_array(self.price_buffer)
        volumes = self._get_ordered_array(self.volume_buffer)

        # JIT 함수 호출
        volatility = calc_volatility_jit(prices, period=60)
        aggressiveness = calc_aggressiveness_jit(prices, volumes)

        return {
            "OIR": self.current_oir,
            "Volatility": volatility,
            "Aggressiveness": aggressiveness
        }
