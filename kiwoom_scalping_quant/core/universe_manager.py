import os
import re
import asyncio
import logging
import aiohttp
from typing import List, Dict, Any, Optional
from returns.result import Result, Success, Failure
from returns.future import future_safe

class UniverseManager:
    """
    KOSPI, KOSDAQ 전체 종목 중 스캘핑(초단타)에 부적합한 종목(ETF, ETN, 스팩, 우선주 등)을 필터링하고
    거래대금 상위 Top N 종목을 추출하여 매매 유니버스를 구성합니다.
    """
    def __init__(self):
        self.base_url = os.getenv("KIWOOM_BASE_URL", "https://openapi.kiwoom.com")
        self.app_key = os.getenv("KIWOOM_APP_KEY")
        self.app_secret = os.getenv("KIWOOM_APP_SECRET")
        self.logger = logging.getLogger("UniverseManager")

    def _is_valid_scalping_symbol(self, name: str, code: str) -> bool:
        """
        정규식과 문자열 패턴을 이용하여 순수 주식이 아닌 종목을 엄격히 걸러냅니다.
        """
        # 스팩(SPAC), ETF, ETN, KODEX, TIGER, KBSTAR 등 시장 인덱스 제외
        invalid_keywords = r"(스팩|SPAC|ETF|ETN|KODEX|TIGER|KBSTAR|ARIRANG|KINDEX|KOSEF)"
        if re.search(invalid_keywords, name, re.IGNORECASE):
            return False

        # 우선주(우, 우B 등), 선박, 리츠 등 제외 패턴
        if re.search(r"(우$|우B$|우C$|리츠|인프라|선박)", name):
            return False

        # 종목코드 끝자리가 0이 아닌 경우(보통 우선주나 파생상품) 제외
        if not code.endswith("0"):
            return False

        return True

    @future_safe
    async def build_top_n_universe(self, access_token: str, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        거래소에서 전체 종목 리스트와 거래대금을 가져와 필터링 후 Top N 종목을 선정합니다.
        (현재는 구조적 예시를 위해 Mock API 흐름으로 구현합니다)
        """
        headers = {
            "Authorization": f"Bearer {access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "OPT10030" # 당일거래대금상위요청 (가상 TR)
        }

        self.logger.info("거래대금 상위 종목 리스트 수집 및 필터링 시작...")

        # 실제 환경에서는 Kiwoom REST API를 호출하여 시장(KOSPI/KOSDAQ)의
        # 당일 또는 최근 5일 평균 거래대금 상위 리스트를 가져옵니다.

        # [Mock Data Generation] (실제 환경에서는 aiohttp를 통해 API 호출 후 처리)
        import random
        mock_raw_market = []

        # 비동기 블로킹 방지를 위한 가상의 네트워크 지연
        await asyncio.sleep(1.0)

        for i in range(1, 2000): # 약 2000개의 전 종목을 가정
            code = f"{i:05d}0"
            is_spac = random.random() < 0.05
            is_etf = random.random() < 0.05

            name = f"Stock_Company_{i}"
            if is_spac: name = f"대신스팩{i}호"
            elif is_etf: name = f"KODEX_레버리지{i}"

            mock_raw_market.append({
                "code": code,
                "name": name,
                "trading_value": random.randint(100, 100000) * 1000000 # 거래대금 모의
            })

        # 1. 노이즈 필터링
        filtered_universe = [
            stock for stock in mock_raw_market
            if self._is_valid_scalping_symbol(stock["name"], stock["code"])
        ]

        # 필터링 중 연산 지연 시뮬레이션
        await asyncio.sleep(0.5)

        # 2. 거래대금(Trading Value) 기준 내림차순 정렬
        sorted_universe = sorted(filtered_universe, key=lambda x: x["trading_value"], reverse=True)

        # 3. Top N 선정
        top_universe = sorted_universe[:top_n]

        self.logger.info(f"유니버스 필터링 완료: 원본 {len(mock_raw_market)}개 -> 필터링 {len(filtered_universe)}개 -> 최종 Top {len(top_universe)}개")

        return top_universe
