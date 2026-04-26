import numpy as np
import pandas as pd

class FeatureEngineer:
    """
    [Basic] 스케일 불변 특징 추출기.
    """
    @staticmethod
    def extract_features(data_list):
        if not data_list: return np.array([])
        df = pd.DataFrame(data_list)
        # 절대 가격 대신 변화율 사용
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
        
        # 1. 이동평균선 및 VWAP 이격도 (%)
        ma5 = df['price'].rolling(window=5).mean()
        ma20 = df['price'].rolling(window=20).mean()
        vwap = (df['price'] * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-9)
        
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
