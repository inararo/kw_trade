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

        self.base_url = "https://api.kiwoom.com"
        if self.config_manager and hasattr(self.config_manager, "get_rest_url"):
            self.base_url = self.config_manager.get_rest_url()
        else:
            self.base_url = os.getenv("KIWOOM_BASE_URL", "https://api.kiwoom.com")

        self.app_key = os.getenv("KIWOOM_APP_KEY")
        self.app_secret = os.getenv("KIWOOM_APP_SECRET")
        self.logger = logging.getLogger("UniverseManager")

    def _is_valid_scalping_symbol(self, name: str, code: str) -> bool:
        """
        정규식과 문자열 패턴을 이용하여 순수 주식이 아닌 종목을 엄격히 걸러냅니다.
        """
        # 1. 키워드 기반 필터링 (ETF, ETN, 스팩 등)
        invalid_keywords = r"(스팩|SPAC|ETF|ETN|KODEX|TIGER|KBSTAR|ARIRANG|KINDEX|KOSEF)"
        if re.search(invalid_keywords, name, re.IGNORECASE):
            self.logger.debug(f"필터링 제외: {name}({code}) - 키워드 매칭")
            return False

        # 2. 우선주/리츠 등 명어 기반 필터링
        if re.search(r"(우$|우B$|우C$|리츠|인프라|선박|ETN)", name):
            self.logger.debug(f"필터링 제외: {name}({code}) - 우선주/리츠 등 명칭")
            return False

        # 3. 종목코드 기반 필터링 (우선주 등 체크)
        # 종목코드에 '_AL' 등 접미사가 붙어있을 수 있으므로 전처리 후 마지막 자리 체크
        clean_code = code.split('_')[0]
        if not clean_code.endswith("0"):
            self.logger.debug(f"필터링 제외: {name}({code}) - 우선주/파생상품 코드({clean_code})")
            return False

        return True

    @future_safe
    async def build_top_n_universe(self, access_token: str, top_n: int = 40) -> List[Dict[str, Any]]:
        """
        거래소에서 전체 종목 리스트와 거래대금을 가져와 필터링 후 Top N 종목을 선정합니다.
        (현재는 구조적 예시를 위해 Mock API 흐름으로 구현합니다)
        """
        # 1. 요청할 API URL
        # host = 'https://mockapi.kiwoom.com' # 모의투자
        # host = 'https://api.kiwoom.com'  # 실전투자
        # endpoint = '/api/dostk/rkinfo'
        # url = host + endpoint

        self.logger.error(f"JYJ 222 access_token : {access_token}")

        endpoint = f"{self.base_url}/api/dostk/rkinfo"
        self.logger.info(f"거래대금 상위 종목 리스트 수집 및 필터링 시작... (Target URL: {endpoint})")

        # 2. header 데이터
        headers = {
            'Content-Type': 'application/json;charset=UTF-8',  # 컨텐츠타입
            "authorization": f"Bearer {access_token}",
            'cont-yn': 'N',  # 연속조회여부
            'next-key': '',  # 연속조회키
            "api-id": "ka10030" # 당일거래대금상위요청 (가상 TR)
        }

        # 2. 요청 데이터
        params = {
            'mrkt_tp': '000',  # 시장구분 000:전체, 001:코스피, 101:코스닥
            'sort_tp': '1',  # 정렬구분 1:거래량, 2:거래회전율, 3:거래대금
            'mang_stk_incls': '0',
            # 관리종목포함 0:관리종목 포함, 1:관리종목 미포함, 3:우선주제외, 11:정리매매종목제외, 4:관리종목, 우선주제외, 5:증100제외, 6:증100마나보기, 13:증60만보기, 12:증50만보기, 7:증40만보기, 8:증30만보기, 9:증20만보기, 14:ETF제외, 15:스팩제외, 16:ETF+ETN제외
            'crd_tp': '0',  # 신용구분 0:전체조회, 9:신용융자전체, 1:신용융자A군, 2:신용융자B군, 3:신용융자C군, 4:신용융자D군, 8:신용대주
            'trde_qty_tp': '0',
            # 거래량구분 0:전체조회, 5:5천주이상, 10:1만주이상, 50:5만주이상, 100:10만주이상, 200:20만주이상, 300:30만주이상, 500:500만주이상, 1000:백만주이상
            'pric_tp': '0',
            # 가격구분 0:전체조회, 1:1천원미만, 2:1천원이상, 3:1천원~2천원, 4:2천원~5천원, 5:5천원이상, 6:5천원~1만원, 10:1만원미만, 7:1만원이상, 8:5만원이상, 9:10만원이상
            'trde_prica_tp': '0',
            # 거래대금구분 0:전체조회, 1:1천만원이상, 3:3천만원이상, 4:5천만원이상, 10:1억원이상, 30:3억원이상, 50:5억원이상, 100:10억원이상, 300:30억원이상, 500:50억원이상, 1000:100억원이상, 3000:300억원이상, 5000:500억원이상
            'mrkt_open_tp': '0',  # 장운영구분 0:전체조회, 1:장중, 2:장전시간외, 3:장후시간외
            'stex_tp': '3',  # 거래소구분 1:KRX, 2:NXT 3.통합
        }

        # 실제 환경에서는 Kiwoom REST API를 호출하여 시장(KOSPI/KOSDAQ)의
        # 당일 또는 최근 5일 평균 거래대금 상위 리스트를 가져옵니다.

        raw_market = []
        try:
            async with aiohttp.ClientSession() as session:
                # 공식 샘플 가이드에 따라 조회성 TR인 ka10030도 POST 방식을 사용합니다.
                async with session.post(endpoint, headers=headers, json=params, timeout=10) as response:
                    if response.status != 200:
                        err_text = await response.text()
                        self.logger.error(f"API Error ({response.status}): {err_text}")
                        # Return empty list or we could raise an Exception to be caught by @future_safe
                        raise RuntimeError(f"Kiwoom API 연동 실패: {response.status} - {err_text}")

                    data = await response.json()

                    # Kiwoom API returns a list of items typically in "output" or "output1"
                    # We will parse out standard keys
                    items = data.get("tdy_trde_qty_upper", [])
                    if not items and "output1" in data:
                        items = data["output1"]

                    self.logger.error(f"API 수신 데이터 확인: 총 {len(items)}개의 종목 수신됨.")

                    for item in items:
                        # 제공된 명세(stk_cd, stk_nm, trde_amt)를 최우선으로 적용합니다.
                        code = item.get("stk_cd") or item.get("stck_shrn_iscd") or item.get("code") or ""
                        name = item.get("stk_nm") or item.get("hts_kor_isnm") or item.get("name") or f"Unknown_{code}"

                        # Handle string representation of trading value
                        try:
                            # 명세상 '거래금액'은 trde_amt 필드입니다.
                            tval_str = item.get("trde_amt") or item.get("acml_tr_pbmn") or item.get("trading_value") or "0"
                            trading_value = float(tval_str)
                        except (ValueError, TypeError):
                            trading_value = 0.0

                        parsed_stock = {
                            "code": code,
                            "name": name,
                            "trading_value": trading_value
                        }
                        raw_market.append(parsed_stock)

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

        # 최종 선정된 유니버스 종목들 로그 출력
        top_symbols = [f"{s.get('name')}({s.get('code')})" for s in top_universe]
        self.logger.error(f"최종 선정된 유니버스 Top {len(top_universe)}: {', '.join(top_symbols)}")

        self.logger.error(f"유니버스 필터링 완료: 원본 {len(raw_market)}개 -> 필터링 {len(filtered_universe)}개 -> 최종 Top {len(top_universe)}개")

        return top_universe
