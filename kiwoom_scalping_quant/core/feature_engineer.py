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

class AdvancedFeatureEngineer:
    """
    백테스트/학습 시 Pandas를 활용하여 시계열 데이터를 일괄 처리하고,
    보조지표(RSI, 이격도, 볼린저 밴드, 거래량 스파이크 등)를 정규화하여 추출하는 모듈.
    """
    @staticmethod
    def process_historical_data(data_list: list) -> np.ndarray:
        import pandas as pd
        if not data_list:
            return np.array([])
            
        df = pd.DataFrame(data_list)
        if "price" not in df.columns or "volume" not in df.columns:
            return np.array([])
            
        # 1. Price Return (수익률)
        df['price_return'] = df['price'].pct_change().fillna(0)
        
        # 2. RSI (14기간)
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))
        # RSI 정규화: 0~100 -> -1.0 ~ 1.0 (50 기준)
        df['rsi_norm'] = (df['rsi'] - 50.0) / 50.0
        
        # 3. MA Disparity (이동평균 이격도 - 20틱 기준)
        df['ma20'] = df['price'].rolling(window=20, min_periods=1).mean()
        df['ma_disparity'] = (df['price'] - df['ma20']) / (df['ma20'] + 1e-9)
        # 이격도 정규화 (대략 -0.05 ~ 0.05 범위를 -1.0 ~ 1.0으로 스케일링, 극단값 클리핑)
        df['ma_disp_norm'] = (df['ma_disparity'] * 20.0).clip(-1.0, 1.0)
        
        # 4. Bollinger Bands Position (상/하단선 기준 위치)
        df['std20'] = df['price'].rolling(window=20, min_periods=1).std().fillna(0)
        df['upper'] = df['ma20'] + (2 * df['std20'])
        df['lower'] = df['ma20'] - (2 * df['std20'])
        band_range = df['upper'] - df['lower']
        # 하단=0, 중간=0.5, 상단=1.0. 이걸 -1.0 ~ 1.0으로
        df['bb_pos'] = np.where(band_range > 0, (df['price'] - df['lower']) / band_range, 0.5)
        df['bb_pos_norm'] = (df['bb_pos'] - 0.5) * 2.0
        df['bb_pos_norm'] = df['bb_pos_norm'].clip(-1.0, 1.0)
        
        # 5. Volume Spike (거래량 급증)
        df['vol_ma20'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['vol_spike'] = df['volume'] / (df['vol_ma20'] + 1e-9)
        # 평소=1.0. 0~5 범위를 대략 -1.0 ~ 1.0으로 스케일링
        df['vol_spike_norm'] = (df['vol_spike'] / 2.5 - 1.0).clip(-1.0, 1.0)
        
        # 6. 수익률 정규화 (최근 50 스텝 기준 Z-Score 모방, 통상 -1~1 사이)
        df['return_norm'] = (df['price_return'] * 100.0).clip(-1.0, 1.0)
        
        # OIR, Volatility 등 기타 데이터가 있다면 추가 패스스루
        df['oir'] = df['OIR'] if 'OIR' in df.columns else 0.0
        df['volatility'] = df['Volatility'] if 'Volatility' in df.columns else 0.0
        
        # 최종 Feature Matrix 구성
        feature_cols = [
            'return_norm',      # 수익률 정규화 [-1.0, 1.0]
            'vol_spike_norm',   # 거래량 스파이크 [-1.0, 1.0]
            'rsi_norm',         # RSI 정규화 [-1.0, 1.0]
            'ma_disp_norm',     # MA 이격도 [-1.0, 1.0]
            'bb_pos_norm',      # BB 위치 [-1.0, 1.0]
            'oir',              # OIR (기존 -1~1)
            'volatility'        # 변동성
        ]
        
        # 결측값 방어
        out_df = df[feature_cols].fillna(0.0)
        return out_df.to_numpy(dtype=np.float32)
