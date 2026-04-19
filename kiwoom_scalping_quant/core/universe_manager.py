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
    def __init__(self, config_manager=None):
        self.config_manager = config_manager

        self.base_url = "https://openapi.kiwoom.com"
        if self.config_manager and hasattr(self.config_manager, "get_rest_url"):
            self.base_url = self.config_manager.get_rest_url()
        else:
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

        endpoint = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
        self.logger.info(f"거래대금 상위 종목 리스트 수집 및 필터링 시작... (Target URL: {endpoint})")

        # 실제 환경에서는 Kiwoom REST API를 호출하여 시장(KOSPI/KOSDAQ)의
        # 당일 또는 최근 5일 평균 거래대금 상위 리스트를 가져옵니다.

        raw_market = []
        try:
            async with aiohttp.ClientSession() as session:
                # payload may be required for Kiwoom API depending on the spec, usually GET for inquiry
                # Adjust method (GET/POST) and parameters according to the exact Kiwoom OpenAPI spec
                async with session.get(endpoint, headers=headers, timeout=10) as response:
                    if response.status != 200:
                        err_text = await response.text()
                        self.logger.error(f"API Error ({response.status}): {err_text}")
                        # Return empty list or we could raise an Exception to be caught by @future_safe
                        raise RuntimeError(f"Kiwoom API 연동 실패: {response.status} - {err_text}")

                    data = await response.json()

                    # Kiwoom API returns a list of items typically in "output" or "output1"
                    # We will parse out standard keys
                    items = data.get("output", [])
                    if not items and "output1" in data:
                        items = data["output1"]

                    for item in items:
                        code = item.get("stck_shrn_iscd") or item.get("code") or ""
                        name = item.get("hts_kor_isnm") or item.get("name") or f"Unknown_{code}"

                        # Handle string representation of trading value
                        try:
                            tval_str = item.get("acml_tr_pbmn") or item.get("trading_value") or "0"
                            trading_value = float(tval_str)
                        except (ValueError, TypeError):
                            trading_value = 0.0

                        raw_market.append({
                            "code": code,
                            "name": name,
                            "trading_value": trading_value
                        })
        except asyncio.TimeoutError:
            self.logger.error("API 요청 시간 초과 (Timeout).")
            raise RuntimeError("API 연동 시간 초과")
        except Exception as e:
            self.logger.error(f"유니버스 데이터 수집 중 에러 발생: {str(e)}")
            raise e

        # 1. 노이즈 필터링
        filtered_universe = []
        for stock in raw_market:
            # 방어 코드: 딕셔너리가 아닌 경우 스킵
            if isinstance(stock, dict):
                if self._is_valid_scalping_symbol(stock.get("name", ""), stock.get("code", "")):
                    filtered_universe.append(stock)

        # 2. 거래대금(Trading Value) 기준 내림차순 정렬
        sorted_universe = sorted(filtered_universe, key=lambda x: x["trading_value"], reverse=True)

        # 3. Top N 선정
        top_universe = sorted_universe[:top_n]

        self.logger.info(f"유니버스 필터링 완료: 원본 {len(raw_market)}개 -> 필터링 {len(filtered_universe)}개 -> 최종 Top {len(top_universe)}개")

        return top_universe
