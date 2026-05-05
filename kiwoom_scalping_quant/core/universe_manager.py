import os
import re
import asyncio
import logging
import aiohttp
import socket
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
        
        # 종목명 로컬 캐시 설정
        self.cache_dir = "data"
        self.cache_path = os.path.join(self.cache_dir, "stock_names.json")
        self._name_cache = {}
        self._load_name_cache()

    def _load_name_cache(self):
        """로컬 JSON 파일에서 종목명 캐시를 불러옵니다."""
        import json
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._name_cache = json.load(f)
                self.logger.info(f"로컬 종목명 캐시 로드 완료: {len(self._name_cache)}건")
        except Exception as e:
            self.logger.error(f"종목명 캐시 로드 에러: {e}")

    def _save_name_cache(self):
        """메모리상의 종목명 캐시를 로컬 JSON 파일로 저장합니다."""
        import json
        try:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._name_cache, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.error(f"종목명 캐시 저장 에러: {e}")

    async def get_stock_name(self, access_token: str, code: str) -> Optional[str]:
        """키움 API 마스터 정보를 활용하여 종목코드에 해당하는 한글명을 반환합니다."""
        clean_code = code.split('_')[0].strip()
        
        # 1. 캐시 확인
        if clean_code in self._name_cache:
            return self._name_cache[clean_code]

        # 2. API 조회 (ka10001: 주식 기본정보 요청)
        endpoint = f"{self.base_url}/api/dostk/stkitem"
        headers = {
            "authorization": f"Bearer {access_token}",
            "api-id": "ka10001",
            "Content-Type": "application/json"
        }
        params = {"stk_cd": clean_code}

        try:
            connector = aiohttp.TCPConnector(family=socket.AF_INET, ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(endpoint, headers=headers, json=params, timeout=5) as response:
                    status = response.status
                    if status == 200:
                        data = await response.json()
                        # 응답 구조 내에서 한글 명칭 추출 (stkitem, opt10001, output 등 유연하게 대응)
                        output = data.get("stkitem", {}) or data.get("opt10001", {}) or data.get("output", {})
                        name = output.get("stk_nm") or output.get("hts_kor_isnm")
                        
                        if name:
                            self.logger.debug(f"종목명 매핑 성공: {clean_code} -> {name}")
                            self._name_cache[clean_code] = name
                            self._save_name_cache() # [신규] 로컬 파일에 즉시 저장
                            return name
                        else:
                            self.logger.warning(f"종목명 매핑 실패 (데이터 없음): {clean_code} | Keys: {list(data.keys())}")
                    else:
                        err_text = await response.text()
                        self.logger.error(f"종목명 조회 API 에러 (Status {status}): {err_text}")
        except Exception as e:
            self.logger.warning(f"종목명 조회 예외 발생 ({clean_code}): {e}")
            
        return None

    def get_stock_name_from_cache(self, code: str) -> Optional[str]:
        """서버 호출 없이 로컬 캐시에서만 이름을 즉시 반환합니다."""
        clean_code = code.split('_')[0].strip()
        return self._name_cache.get(clean_code)

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

        # 4. 사용자 요청 기반 특정 종목 제외 (삼성전자, SK하이닉스 등 대형주)
        blacklisted_codes = {"005930", "000660"}
        if clean_code in blacklisted_codes:
            self.logger.info(f"필터링 제외: {name}({code}) - 사용자 요청 블랙리스트 종목")
            return False

        return True

    @future_safe
    async def build_top_n_universe(self, access_token: str, top_n: int = 40) -> List[Dict[str, Any]]:
        """
        거래소에서 전체 종목 리스트와 거래대금을 가져와 필터링 후 Top N 종목을 선정합니다.
        09:00 이전에는 거래대금이 집계되지 않으므로 우회(Bypass)합니다.
        """
        import datetime
        now = datetime.datetime.now()
        # 09:00 이전 시간 방어 로직
        if now.hour < 9:
            self.logger.info(f"현재 시간 {now.strftime('%H:%M')} (09:00 이전). 당일 거래대금이 없으므로 유니버스 스캔을 생략합니다.")
            return []

        endpoint = f"{self.base_url}/api/dostk/rkinfo"
        self.logger.info(f"거래대금 상위 종목 리스트 수집 및 필터링 시작... (Target URL: {endpoint})")

        # 2. header 데이터
        headers = {
            'Content-Type': 'application/json;charset=UTF-8',  # 컨텐츠타입
            "authorization": f"Bearer {access_token}",
            'cont-yn': 'N',  # 연속조회여부
            'next-key': '',  # 연속조회키
            "api-id": "ka10030" # 당일거래량상위요청 (Volume Top)
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
            connector = aiohttp.TCPConnector(family=socket.AF_INET, ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                # 공식 샘플 가이드에 따라 조회성 TR인 ka10030도 POST 방식을 사용합니다.
                async with session.post(endpoint, headers=headers, json=params, timeout=10) as response:
                    if response.status != 200:
                        err_text = await response.text()
                        self.logger.error(f"API Error ({response.status}): {err_text}")
                        # Return empty list or we could raise an Exception to be caught by @future_safe
                        raise RuntimeError(f"Kiwoom API 연동 실패: {response.status} - {err_text}")

                    data = await response.json()

                    # Kiwoom API는 TR ID, "output", "output1" 등 다양한 키로 데이터가 올 수 있음
                    items = data.get("ka10030", []) or data.get("ka10032", [])
                    if not items and "output" in data:
                        items = data["output"]
                    if not items and "output1" in data:
                        items = data["output1"]
                    if not items and "tdy_trde_qty_upper" in data: # 기존 폴백
                        items = data["tdy_trde_qty_upper"]
                    
                    # 만약 여전히 비어있다면 전체 키 중 리스트인 것을 찾아보는 최후의 수단
                    if not items:
                        for key, val in data.items():
                            if isinstance(val, list) and len(val) > 0:
                                items = val
                                break

                    self.logger.error(f"API 수신 데이터 확인: 총 {len(items)}개의 종목 수신됨.")
                    if items:
                        self.logger.debug(f"ITEM KEYS: {list(items[0].keys())}")

                    for item in items:
                        # 제공된 명세(stk_cd, stk_nm, trde_amt)를 최우선으로 적용합니다.
                        raw_code = item.get("stk_cd") or item.get("stck_shrn_iscd") or item.get("code") or ""
                        # [수정] 접미사(_AL 등) 제거하여 순수 종목 코드만 사용
                        code = raw_code.split('_')[0].strip()
                        name = item.get("stk_nm") or item.get("hts_kor_isnm") or item.get("name") or f"Unknown_{code}"

                        # [혁신] 발견된 종목명 정보를 로컬 캐시에 즉시 업데이트 (DB 로드 시 한글 이름 복원용)
                        if code and name and "Unknown" not in name:
                            clean_code = code.split('_')[0].strip()
                            self._name_cache[clean_code] = name

                        # Handle string representation of numerical values
                        try:
                            # 현재가 파싱 후보군 자동 탐색 (사용자 로그에서 cur_prc 확인됨)
                            price_candidates = ["cur_prc", "stck_prpr", "stk_prpr", "prpr", "curr_pric", "stk_prc", "stck_prc", "curr"]
                            price_val = "0"
                            for cand in price_candidates:
                                if item.get(cand):
                                    price_val = item.get(cand)
                                    break
                            
                            current_price = abs(float(str(price_val).replace(',', '')))

                            # 거래금액 후보군 (trde_amt, acml_tr_pbmn 등)
                            tval_candidates = ["trde_amt", "acml_tr_pbmn", "trading_value", "trde_amt_val"]
                            tval_str = "0"
                            for cand in tval_candidates:
                                if item.get(cand):
                                    tval_str = item.get(cand)
                                    break
                            trading_value = float(str(tval_str).replace(',', ''))

                            # 추가 필터링용 데이터 추출
                            sign = item.get("pred_pre_sig") or item.get("prdy_vrss_sign") or "3"

                            # 등락률
                            flu_rt_str = item.get("flu_rt") or item.get("prdy_ctrt") or "0"
                            flu_rt = float(flu_rt_str)

                            # 거래량 후보군 (trde_qty, acml_tr_qty 등)
                            vol_candidates = ["trde_qty", "acml_tr_qty", "stck_vol", "vol"]
                            vol_val = "0"
                            for cand in vol_candidates:
                                if item.get(cand):
                                    vol_val = item.get(cand)
                                    break
                            current_volume = float(str(vol_val).replace(',', ''))

                        except (ValueError, TypeError):
                            current_price = 0.0
                            trading_value = 0.0
                            sign = "3"
                            flu_rt = 0.0
                            current_volume = 0.0

                        parsed_stock = {
                            "code": code,
                            "name": name,
                            "price": current_price,
                            "trading_value": trading_value,
                            "sign": str(sign),
                            "flu_rt": flu_rt,
                            "volume": current_volume
                        }
                        raw_market.append(parsed_stock)

                    # [혁신] 루프 종료 후 한글 종목명 캐시를 파일로 한번에 저장
                    if items:
                        self._save_name_cache()

        except asyncio.TimeoutError:
            self.logger.error("API 요청 시간 초과 (Timeout).")
            raise RuntimeError("API 연동 시간 초과")
        except Exception as e:
            self.logger.error(f"유니버스 데이터 수집 중 에러 발생: {str(e)}")
            raise e

        # 투자 한도 설정 가져오기 (RiskManager와 동일한 설정 키 사용)
        max_invest_limit = 5000000.0
        if self.config_manager:
            max_invest_limit = float(self.config_manager.get("max_invest_per_symbol", 5000000))

        # 1. 노이즈 및 현재 강세 기준(상태, 등락률, 가격 한도) 필터링
        filtered_universe = []
        for stock in raw_market:
            if not isinstance(stock, dict):
                continue

            code = stock.get("code", "")
            name = stock.get("name", "")

            # 기본 이름/종목코드 검증
            if not self._is_valid_scalping_symbol(name, code):
                continue

            # 투자 한도 초과 종목 제외 필터링 (1주 가격이 한도보다 비싸면 매수 불가하므로 제외)
            price = stock.get("price", 0.0)
            if price > 0 and price > max_invest_limit:
                self.logger.info(f"필터링 제외: {name}({code}) - 투자 한도 초과 (현재가: {price:,.0f} / 한도: {max_invest_limit:,.0f})")
                continue
            
            if price == 0:
                # 가격 정보를 읽어오지 못했을 경우, 유니버스 소멸을 막기 위해 제외하지 않음
                self.logger.debug(f"필터링 통과: {name}({code}) - 가격 데이터 부재(0원)로 필터 스킵")

            # 등락률 필터링: 1(상한가)나 4,5(하한가, 하락) 등 극단적 호가잠김 방지 (스캘핑 불가)
            sign = stock.get("sign", "3")
            if sign in ["1", "4"]:
                self.logger.debug(f"필터링 제외: {name}({code}) - 상/하한가(호가 잠김)")
                continue

            # 당일 시가 갭상승 필터링 (예: 2% 이상 상승 출발)
            opn_prc = stock.get("opn_prc", 0.0)
            prdy_clprc = stock.get("prdy_clprc", 0.0)

            # Note: API 응답에 0이 들어올 수 있으므로 방어 로직 필수
            if prdy_clprc > 0 and opn_prc > 0:
                gap_ratio = ((opn_prc - prdy_clprc) / prdy_clprc) * 100
                if gap_ratio < 2.0:
                    self.logger.debug(f"필터링 제외: {name}({code}) - 갭상승 미달 ({gap_ratio:.2f}%)")
                    # Note: 장 초반에 데이터가 0개로 나오는 것을 방지하기 위해 필터링을 한시적으로 완화하거나 스킵할 수 있음
                    # 현재는 유규한 필터링 정책을 유지하되, 데이터가 아예 없을 때만 통과시킴
                    continue
            elif prdy_clprc == 0 or opn_prc == 0:
                # 데이터가 아직 안 들어온 경우(09:00 직후)에는 일단 필터를 통과시켜 유니버스 0개를 방지함
                self.logger.debug(f"필터링 통과: {name}({code}) - 가격 데이터 부재로 필터 스킵")

            filtered_universe.append(stock)

        # 2. 거래대금(Trading Value) 기준 내림차순 정렬
        sorted_universe = sorted(filtered_universe, key=lambda x: x["trading_value"], reverse=True)

        # 3. Top N 선정
        top_universe = sorted_universe[:top_n]

        # 최종 선정된 유니버스 종목들 로그 출력
        top_symbols = [f"{s.get('name')}({s.get('code')})" for s in top_universe]
        self.logger.error(f"최종 선정된 유니버스 Top {len(top_universe)}: {', '.join(top_symbols)}")

        self.logger.info(f"유니버스 필터링 완료: 원본 {len(raw_market)}개 -> 필터링 {len(filtered_universe)}개 -> 최종 Top {len(top_universe)}개")

        return top_universe

    @future_safe
    async def fetch_top_30_volume_symbols(self, access_token: str) -> List[Dict[str, Any]]:
        """
        [NEW] 증권사 API를 호출하여 거래량 상위 30개 종목을 가져옵니다.
        관리종목, 우선주, ETF/ETN, SPAC은 필터링하여 순수 주식 리스트만 반환합니다.
        """
        endpoint = f"{self.base_url}/api/dostk/rkinfo"
        self.logger.info("거래량 상위 30개 종목 스캔 시작...")

        headers = {
            'Content-Type': 'application/json;charset=UTF-8',
            "authorization": f"Bearer {access_token}",
            'cont-yn': 'N',
            'next-key': '',
            "api-id": "ka10030"
        }

        # mang_stk_incls: 4 (관리종목, 우선주제외) 
        # sort_tp: 1 (거래량)
        params = {
            'mrkt_tp': '000',      # 000: 전체
            'sort_tp': '1',      # 1: 거래량
            'mang_stk_incls': '4', # 4: 관리종목, 우선주제외
            'crd_tp': '0',
            'trde_qty_tp': '0',
            'pric_tp': '0',
            'trde_prica_tp': '0',
            'mrkt_open_tp': '0',
            'stex_tp': '3',
        }

        top_30 = []
        try:
            connector = aiohttp.TCPConnector(family=socket.AF_INET, ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(endpoint, headers=headers, json=params, timeout=10) as response:
                    if response.status != 200:
                        err_text = await response.text()
                        raise RuntimeError(f"Kiwoom API 연동 실패: {response.status} - {err_text}")

                    data = await response.json()
                    items = data.get("ka10030", []) or data.get("output", []) or data.get("output1", [])
                    
                    if not items:
                        self.logger.warning(f"Kiwoom API 응답에 예상된 데이터 키가 없습니다. 수신된 키: {list(data.keys())}")
                        # [안정화] 일부 TR은 'output' 대신 다른 키를 사용할 수 있으므로 전체 탐색 시도
                        for k, v in data.items():
                            if isinstance(v, list) and len(v) > 0:
                                items = v
                                self.logger.info(f"대체 데이터 키 발견: '{k}' (종목 수: {len(v)})")
                                break
                    
                    for item in items:
                        if len(top_30) >= 30:
                            break
                            
                        raw_code = item.get("stk_cd") or item.get("stck_shrn_iscd") or ""
                        # [수정] 접미사(_AL 등) 제거하여 순수 종목 코드만 사용
                        code = raw_code.split('_')[0].strip()
                        name = item.get("stk_nm") or item.get("hts_kor_isnm") or f"Unknown_{code}"

                        # 추가 필터링 (ETF, SPAC 등)
                        if not self._is_valid_scalping_symbol(name, code):
                            continue

                        top_30.append({
                            "code": code,
                            "name": name,
                            "price": float(str(item.get("cur_prc", 0)).replace(',', '')),
                            "volume": float(str(item.get("trde_qty", 0)).replace(',', ''))
                        })
                        
            self.logger.info(f"거래량 상위 30개 종목 추출 완료 (필터링 후 {len(top_30)}개)")
            return top_30

        except Exception as e:
            self.logger.error(f"Top 30 스캔 중 에러: {e}")
            raise e
