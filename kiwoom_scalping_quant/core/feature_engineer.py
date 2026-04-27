import numpy as np
import pandas as pd

class FeatureEngineer:
    """
    [Basic] 실시간 틱/호가 데이터를 받아 단기적 지표(OIR, Volatility 등)를 추출하는 유지형 클래스
    """
    def __init__(self, max_ticks=100):
        self.max_ticks = max_ticks
        self.price_buffer = np.zeros(max_ticks, dtype=float)
        self.volume_buffer = np.zeros(max_ticks, dtype=float)
        self.head = 0
        self.count = 0
        self.current_oir = 0.5
        
    def update_orderbook(self, orderbook):
        try:
            asks = orderbook.get("asks", [])
            bids = orderbook.get("bids", [])
            total_ask_qty = sum([item.get("qty", 0) for item in asks[:5]])
            total_bid_qty = sum([item.get("qty", 0) for item in bids[:5]])
            denom = total_bid_qty + total_ask_qty
            if denom > 0:
                self.current_oir = (total_bid_qty - total_ask_qty) / denom
        except Exception:
            pass

    def update_tick(self, price, volume):
        self.price_buffer[self.head] = price
        self.volume_buffer[self.head] = volume
        self.head = (self.head + 1) % self.max_ticks
        self.count += 1
        
        # 간단한 변동성 계산 (최근 10개 틱 기준)
        valid_count = min(self.count, self.max_ticks)
        if valid_count > 1:
            recent_prices = self.price_buffer[:valid_count] if self.count < self.max_ticks else np.concatenate((self.price_buffer[self.head:], self.price_buffer[:self.head]))
            returns = np.diff(recent_prices) / (recent_prices[:-1] + 1e-9)
            volatility = float(np.std(returns)) if valid_count > 2 else 0.0
        else:
            volatility = 0.0

        # 체결강도 근사
        aggressiveness = 0.5  # placeholder or simple calc
        
        return {
            "OIR": float(self.current_oir),
            "Volatility": volatility,
            "Aggressiveness": aggressiveness
        }

    @staticmethod
    def extract_features(data_list):
        if not data_list: return np.array([])
        df = pd.DataFrame(data_list)
        df['ret'] = df['price'].pct_change().fillna(0) * 100.0
        df['vol_ret'] = df['volume'].pct_change().fillna(0)
        return df[['ret', 'vol_ret']].values

class AdvancedFeatureEngineer:
    """
    다종목 학습 최적화: 모든 피처를 이격도(%), 비율(%)로 통일.
    """
    @staticmethod
    def process_historical_data(data_list):
        if not data_list: return np.array([])
        df = pd.DataFrame(data_list)
        df['price'] = df['price'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Timestamp를 미리 파싱 (시간 관련 피처 및 VWAP 그룹화 용도)
        if 'timestamp' in df.columns:
            try:
                temp_ts = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Asia/Seoul')
                df['_date'] = temp_ts.dt.date
            except Exception:
                df['_date'] = '1970-01-01'  # Fallback
        else:
            df['_date'] = '1970-01-01'

        # 1. 이동평균선 및 VWAP 이격도 (%)
        ma5 = df['price'].rolling(window=5).mean()
        ma20 = df['price'].rolling(window=20).mean()
        
        # 당일 시점(09:00:00)부터의 일일 단위 누적 거래대금/거래량으로 VWAP 리셋 계산
        df['cum_cash'] = (df['price'] * df['volume']).groupby(df['_date']).cumsum()
        df['cum_volume'] = df['volume'].groupby(df['_date']).cumsum()
        vwap = df['cum_cash'] / (df['cum_volume'] + 1e-9)
        
        # 2. 볼린저 밴드 위치 (%B)
        std20 = df['price'].rolling(window=20).std()
        df['bb_pos'] = (df['price'] - (ma20 - 2*std20)) / (4*std20 + 1e-9)
        
        # 3. 보상 및 피처용 수익률 계산
        df['ret_1'] = df['price'].pct_change().fillna(0) * 100.0
        df['ret_5'] = df['price'].pct_change(5).fillna(0) * 100.0
        
        # 4. 거래량 MA 대비 비율
        df['vol_activity'] = df['volume'] / (df['volume'].rolling(window=20).mean() + 1e-9)

        # 11차원 스케일 불변 지표 구성 (컬럼 인덱스 직접 모니터링하는 코드는 이 순서에 의존)
        features = pd.DataFrame()
        features['disparity_ma5']  = (df['price'] / ma5 - 1) * 100.0   # idx 0
        features['disparity_ma20'] = (df['price'] / ma20 - 1) * 100.0  # idx 1
        features['disparity_vwap'] = (df['price'] / vwap - 1) * 100.0  # idx 2
        features['bb_pos']         = df['bb_pos']                        # idx 3
        features['rsi']            = AdvancedFeatureEngineer._calc_rsi(df['price']) / 100.0  # idx 4
        # idx 5: 상대 거래량(Relative Volume) - 스마트 샘플링의 Volume Spike 후보군 탐지 기준
        features['rel_vol_20']     = df['volume'] / (df['volume'].rolling(window=20).mean() + 1e-9)  # idx 5
        features['ret_1']          = df['ret_1']                         # idx 6
        features['ret_5']          = df['ret_5']                         # idx 7
        features['oir']            = df.get('OIR', 0.5)                  # idx 8
        features['volatility']     = df.get('Volatility', 0.1)           # idx 9
        features['vol_change']     = df['volume'].pct_change().fillna(0) # idx 10

        final_array = features.fillna(0).values
        return np.clip(final_array, -10.0, 10.0).astype(np.float32)

    @staticmethod
    def _calc_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))
