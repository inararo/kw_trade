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
        # 종목명 로컬 캐시 설정
        self.cache_dir = "data"
        self.cache_path = os.path.join(self.cache_dir, "stock_names.json")
        self._name_cache = {}
        self._load_name_cache()

    async def get_condition_list(self, access_token: str) -> Dict[str, str]:
        """서버에 저장된 조건검색식 목록 조회 (가상 TR: ka10050)"""
        endpoint = f"{self.base_url}/api/dostk/rkinfo"
        headers = {
            "authorization": f"Bearer {access_token}",
            "api-id": "ka10050",
            "Content-Type": "application/json"
        }
        
        try:
            connector = aiohttp.TCPConnector(family=socket.AF_INET, ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(endpoint, headers=headers, json={}, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get("ka10050", []) or data.get("output", [])
                        conditions = {item.get("cond_idx"): item.get("cond_nm") for item in items if item.get("cond_idx")}
                        self.logger.info(f"✅ 조건식 목록 수신 완료 ({len(conditions)}개 항목)")
                        return conditions
                    else:
                        self.logger.error(f"❌ 조건식 목록 조회 실패 (Status {resp.status})")
                        return {}
        except Exception as e:
            self.logger.error(f"❌ 조건식 목록 조회 통신 에러: {e}")
            return {}

    async def get_condition_symbols(self, access_token: str, cond_idx: str, cond_nm: str) -> List[Dict[str, Any]]:
        """특정 조건식에 해당하는 실시간 종목 리스트 조회 (가상 TR: ka10051)"""
        endpoint = f"{self.base_url}/api/dostk/rkinfo"
        headers = {
            "authorization": f"Bearer {access_token}",
            "api-id": "ka10051",
            "Content-Type": "application/json"
        }
        params = {
            "cond_idx": cond_idx,
            "cond_nm": cond_nm
        }
        
        symbols = []
        try:
            connector = aiohttp.TCPConnector(family=socket.AF_INET, ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(endpoint, headers=headers, json=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get("ka10051", []) or data.get("output", [])
                        for item in items:
                            code = item.get("stk_cd") or item.get("code", "")
                            name = item.get("stk_nm") or item.get("name", f"Unknown_{code}")
                            symbols.append({
                                "code": code.split('_')[0].strip(),
                                "name": name,
                                "price": float(str(item.get("cur_prc", 0)).replace(',', '')),
                                "flu_rt": float(str(item.get("flu_rt", 0)).replace(',', '')),
                                "volume": float(str(item.get("trde_qty", 0)).replace(',', ''))
                            })
                        self.logger.info(f"✅ 조건검색 결과 수신 완료: {cond_nm} ({len(symbols)}개 종목)")
                    else:
                        self.logger.error(f"❌ 조건검색 종목 조회 실패 (Status {resp.status})")
        except Exception as e:
            self.logger.error(f"❌ 조건검색 종목 조회 통신 에러: {e}")
        
        return symbols

    @future_safe
    async def build_condition_universe(self, access_token: str, target_cond_nm: str = "AI스캘핑주도주") -> List[Dict[str, Any]]:
        """서버 조건식을 검색하여 실전 매매 유니버스를 구성하는 통합 메서드"""
        conditions = await self.get_condition_list(access_token)
        if not conditions:
            return []

        # 대상 조건식 찾기
        cond_idx = next((idx for idx, nm in conditions.items() if target_cond_nm in nm), None)
        if not cond_idx:
            # Fallback: 만약 대상이 없으면 첫 번째 조건식 사용
            cond_idx, cond_nm = list(conditions.items())[0]
            self.logger.warning(f"⚠️ '{target_cond_nm}' 조건식을 찾을 수 없어 '{cond_nm}'을 대신 사용합니다.")
        else:
            cond_nm = conditions[cond_idx]

        symbols = await self.get_condition_symbols(access_token, cond_idx, cond_nm)
        
        # 필터링 적용
        valid_symbols = [s for s in symbols if self._is_valid_scalping_symbol(s["name"], s["code"])]
        return valid_symbols

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
        invalid_keywords = r"(스팩|SPAC|ETF|ETN|KODEX|TIGER|KBSTAR|ARIRANG|KINDEX|KOSEF|RISE|SOL|HANARO)"
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
    async def build_top_n_universe(self, access_token: str, top_n: int = 40, sort_by: str = "volume") -> List[Dict[str, Any]]:
        """
        거래소에서 전체 종목 리스트를 가져와 필터링 후 Top N 종목을 선정합니다. (연속 조회 지원)
        - sort_by: 'volume' (거래량), 'value' (거래대금), 'flu_rt' (등락률)
        """
        import datetime
        now = datetime.datetime.now()
        if now.hour < 9:
            self.logger.info(f"현재 시간 {now.strftime('%H:%M')} (09:00 이전). 유니버스 스캔을 생략합니다.")
            return []

        endpoint = f"{self.base_url}/api/dostk/rkinfo"
        
        # 정렬 기준명 매핑
        sort_nm = {"volume": "거래량", "value": "거래대금", "flu_rt": "등락률"}.get(sort_by, "거래량")
        self.logger.info(f"🚀 [{sort_nm}] 상위 유니버스 수집 시작 (Target: {top_n}개)...")

        # TR 및 파라미터 설정
        api_id = "ka10030" # 기본: 거래량
        sort_tp = "1"      # 1:거래량, 2:거래회전율, 3:거래대금
        
        if sort_by == "value":
            api_id = "ka10032" # 거래대금 상위 TR
            sort_tp = ""
        elif sort_by == "flu_rt":
            api_id = "ka10027" # [변경] 전일대비등락률상위 TR
            sort_tp = "1"      # 1:상승률

        raw_market = []
        
        # [혁신] 100개 이상의 종목을 받기 위해 코스피(001)와 코스닥(101)을 각각 호출하여 병합
        markets = ["001", "101"] if top_n > 100 else ["000"]
        
        try:
            connector = aiohttp.TCPConnector(family=socket.AF_INET, ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                
                for mrkt_tp in markets:
                    next_key = ""
                    page_cnt = 0
                    m_name = "코스피" if mrkt_tp == "001" else ("코스닥" if mrkt_tp == "101" else "전체")
                    
                    while page_cnt < 5: # 각 시장당 최대 5페이지(500개) 시도 (필터링 고려하여 확대)
                        page_cnt += 1
                        headers = {
                            'Content-Type': 'application/json;charset=UTF-8',
                            "authorization": f"Bearer {access_token}",
                            'cont-yn': 'Y' if next_key else 'N',
                            'next-key': next_key,
                            "api-id": api_id
                        }

                        params = {
                            'mrkt_tp': mrkt_tp,
                            'mang_stk_incls': '0',
                            'stex_tp': '3',
                        }
                        if sort_tp:
                            params['sort_tp'] = sort_tp
                            # ka10030 전용 파라미터들
                            if api_id == "ka10030":
                                params.update({
                                    'crd_tp': '0',
                                    'trde_qty_tp': '0',
                                    'pric_tp': '0',
                                    'trde_prica_tp': '0',
                                    'mrkt_open_tp': '0',
                                })
                            # ka10027 전용 파라미터들
                            elif api_id == "ka10027":
                                params.update({
                                    'trde_qty_cnd': '0000',
                                    'stk_cnd': '0',
                                    'crd_cnd': '0',
                                    'updown_incls': '1',
                                    'pric_cnd': '0',
                                    'trde_prica_cnd': '0',
                                })

                        async with session.post(endpoint, headers=headers, json=params, timeout=10) as response:
                            if response.status != 200:
                                break

                            data = await response.json()
                            header = data.get("header", {})
                            next_key = data.get("next_key") or header.get("next_key") or header.get("next")
                            
                            # 데이터 추출 (각 TR 전용 키 및 폴백 대응)
                            items = data.get(api_id, []) or \
                                    data.get("trde_prica_upper", []) or \
                                    data.get("pred_pre_flu_rt_upper", []) or \
                                    data.get("output", []) or \
                                    data.get("output1", [])
                            if not items:
                                lists = [v for v in data.values() if isinstance(v, list)]
                                if lists: items = max(lists, key=len)
                            
                            if not items: break

                            self.logger.info(f"[{m_name}] {page_cnt}페이지: {len(items)}개 종목 수신 (누적: {len(raw_market) + len(items)})")

                            for item in items:
                                raw_code = item.get("stk_cd") or item.get("stck_shrn_iscd") or item.get("code") or ""
                                code = raw_code.split('_')[0].strip()
                                name = item.get("stk_nm") or item.get("hts_kor_isnm") or f"Unknown_{code}"

                                if code and name and "Unknown" not in name:
                                    self._name_cache[code] = name

                                try:
                                    price_val = item.get("cur_prc") or item.get("stck_prpr") or item.get("prpr") or "0"
                                    current_price = abs(float(str(price_val).replace(',', '')))
                                    tval_val = item.get("trde_amt") or item.get("acml_tr_pbmn") or "0"
                                    trading_value = float(str(tval_val).replace(',', ''))
                                    vol_val = item.get("trde_qty") or item.get("acml_tr_qty") or "0"
                                    current_volume = float(str(vol_val).replace(',', ''))
                                    flu_rt_val = item.get("flu_rt") or item.get("prdy_ctrt") or "0"
                                    flu_rt = float(flu_rt_val)
                                    sign = item.get("pred_pre_sig") or item.get("prdy_vrss_sign") or "3"
                                except:
                                    current_price = trading_value = current_volume = flu_rt = 0.0
                                    sign = "3"

                                parsed_stock = {
                                    "code": code, "name": name, "price": current_price,
                                    "trading_value": trading_value, "sign": str(sign),
                                    "flu_rt": flu_rt, "volume": current_volume
                                }
                                if not any(x["code"] == code for x in raw_market):
                                    raw_market.append(parsed_stock)

                            # [개선] 필터링 후에도 충분한 종목을 확보하기 위해 수집 버퍼를 대폭 확대 (top_n의 4배 또는 최소 300개)
                            if not next_key or len(raw_market) >= max(300, top_n * 4): 
                                break
                
                if raw_market:
                    self._save_name_cache()

        except Exception as e:
            self.logger.error(f"유니버스 데이터 수집 중 에러 발생: {str(e)}")
            raise e

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

            # [삭제] 투자 한도 초과 종목 제외 필터링 (사용자 요청으로 제거)
            price = stock.get("price", 0.0)
            
            if price == 0:
                # 가격 정보를 읽어오지 못했을 경우, 유니버스 소멸을 막기 위해 제외하지 않음
                self.logger.debug(f"필터링 통과: {name}({code}) - 가격 데이터 부재(0원)로 필터 스킵")

            # [삭제] 등락률 필터링 (사용자 요청으로 상/하한가 포함 허용)
            # sign = stock.get("sign", "3")
            # if sign in ["1", "4"]:
            #     continue

            # 당일 시가 갭상승 필터링 (예: 2% 이상 상승 출발)
            opn_prc = stock.get("opn_prc", 0.0)
            prdy_clprc = stock.get("prdy_clprc", 0.0)

            # Note: API 응답에 0이 들어올 수 있으므로 방어 로직 필수
            if prdy_clprc > 0 and opn_prc > 0:
                gap_ratio = ((opn_prc - prdy_clprc) / prdy_clprc) * 100
                if gap_ratio < 2.0:
                    self.logger.debug(f"필터링 제외: {name}({code}) - 갭상승 미달 ({gap_ratio:.2f}%)")
                    continue
            elif prdy_clprc == 0 or opn_prc == 0:
                # 데이터가 아직 안 들어온 경우(09:00 직후)에는 일단 필터를 통과시켜 유니버스 0개를 방지함
                self.logger.debug(f"필터링 통과: {name}({code}) - 가격 데이터 부재로 필터 스킵")

            filtered_universe.append(stock)

        # [추가] 필터링 결과 로그
        excluded_count = len(raw_market) - len(filtered_universe)
        if excluded_count > 0:
            self.logger.info(f"💡 필터링 완료: {len(raw_market)}개 중 {excluded_count}개 종목 제외 (ETF/ETN/우선주/갭미달 등)")

        # 2. 정렬 (사용자 선택 기준)
        sort_key_map = {
            "volume": "volume",
            "value": "trading_value",
            "flu_rt": "flu_rt"
        }
        key = sort_key_map.get(sort_by, "volume")
        sorted_universe = sorted(filtered_universe, key=lambda x: x.get(key, 0), reverse=True)

        # 3. Top N 선정
        top_universe = sorted_universe[:top_n]

        # 최종 선정된 유니버스 종목들 로그 출력
        top_symbols = [f"{s.get('name')}({s.get('code')})" for s in top_universe]
        self.logger.info(f"✅ 최종 유니버스 선정 완료 (Top {len(top_universe)}): {', '.join(top_symbols[:10])}...")
        self.logger.info(f"결과 요약: 원본 {len(raw_market)}개 -> 필터링 {len(filtered_universe)}개 -> 최종 {len(top_universe)}개")

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

        # mang_stk_incls: 0 (전체 - 필터링은 내부 _is_valid_scalping_symbol에서 수행)
        params = {
            'mrkt_tp': '000',      # 000: 전체
            'sort_tp': '1',      # 1: 거래량
            'mang_stk_incls': '0', # 0: 전체 (기존 4에서 변경하여 호환성 강화)
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
                    r_code = str(data.get("return_code", ""))
                    r_msg = data.get("return_msg", "")
                    
                    items = data.get("ka10030", []) or data.get("output", []) or data.get("output1", [])
                    
                    if not items:
                        self.logger.warning(f"Kiwoom API 응답에 예상된 데이터 키가 없습니다. Code: {r_code}, Msg: {r_msg} | Keys: {list(data.keys())}")
                        
                        # [특수] 토큰 만료 처리
                        if r_code == "3" or "Token이 유효하지 않습니다" in r_msg:
                            raise RuntimeError("TOKEN_EXPIRED: API 접근 토큰이 만료되었습니다.")

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
