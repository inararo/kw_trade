import os
import json
import time
import asyncio
import logging
import aiohttp
from typing import Dict, Any, List, Optional, Callable
from returns.result import Result, Success, Failure
from returns.future import future_safe

class HistoricalFetcher:
    """
    키움 REST API를 통해 과거 분봉/틱 데이터를 비동기적으로 수집.
    엄격한 시간당 호출 제한(Throttling) 회피 및 로컬 Checkpoint 기반 Resume를 지원합니다.
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
        self.logger = logging.getLogger("HistoricalFetcher")

        # API 제약 회피용 세마포어 (초당 최대 N회, 시간당 최대 M회)
        self.throttle_limit_per_sec = 5
        self.semaphore = asyncio.Semaphore(self.throttle_limit_per_sec)
        self.request_timestamps = []

        self.checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    async def _throttle(self):
        """초당 호출 횟수 제한 회피 큐잉"""
        now = time.time()
        self.request_timestamps = [t for t in self.request_timestamps if now - t < 1.0]

        if len(self.request_timestamps) >= self.throttle_limit_per_sec:
            wait_time = 1.0 - (now - self.request_timestamps[0])
            if wait_time > 0:
                self.logger.debug(f"API Rate Limit 방어 대기 ({wait_time:.2f}s)")
                await asyncio.sleep(wait_time)
        self.request_timestamps.append(time.time())

    def _load_checkpoint(self, symbol: str) -> str:
        """이전 수집 이력(마지막 수집된 기준일 또는 next_token)을 로드"""
        cp_file = os.path.join(self.checkpoint_dir, f"{symbol}.ckpt")
        if os.path.exists(cp_file):
            with open(cp_file, "r") as f:
                data = json.load(f)
                return data.get("last_fetched_date", "")
        return ""

    def _save_checkpoint(self, symbol: str, last_date: str):
        """수집 상태 저장 (실패 시 이어받기 위함)"""
        cp_file = os.path.join(self.checkpoint_dir, f"{symbol}.ckpt")
        with open(cp_file, "w") as f:
            json.dump({"last_fetched_date": last_date}, f)

    @future_safe
    async def fetch_historical_data(self,
                                    symbol: str,
                                    start_date: str,
                                    access_token: str,
                                    progress_callback: Optional[Callable[[int, str], None]] = None,
                                    stop_timestamp: str = None) -> List[Dict[str, Any]]:
        """
        특정 종목의 과거 데이터를 키움 REST API (ka10080) 명세에 맞춰 수집합니다.
        """
        all_data = []
        next_key = ""
        cont_yn = "N"

        # 1. Resume Check (마지막 수집 시점보다 과거 데이터를 더 받고 싶을 때 사용)
        last_fetched = self._load_checkpoint(symbol)
        
        # 증분 수집을 위해 target_start_date는 항상 start_date(오늘 등)를 기준으로 하되
        # 과거에 어디까지 받았었는지는 stop_timestamp로 판단합니다.
        target_start_date = start_date

        endpoint = f"{self.base_url}/api/dostk/chart"
        total_pages = 100  # 수집 페이지 제한
        current_page = 0

        # 2. 시도할 종목코드 형식 목록 생성 (SOR 데이터 수집 최적화)
        symbol_only = symbol.split("_")[0] if "_" in symbol else symbol
        if symbol.endswith("_AL"):
            formats_to_try = [f"SOR:{symbol}", symbol, f"SOR:{symbol_only}", f"KRX:{symbol_only}"]
        elif symbol.endswith("_NX"):
            formats_to_try = [f"NXT:{symbol}", symbol, f"NXT:{symbol_only}", f"KRX:{symbol_only}"]
        else:
            formats_to_try = [f"KRX:{symbol}", symbol]

        final_data = []
        
        async with aiohttp.ClientSession() as session:
            for formatted_symbol in formats_to_try:
                all_data = []
                next_key = ""
                cont_yn = "N"
                current_page = 0
                
                self.logger.error(f"[{symbol}] 수집 시도 (Format: {formatted_symbol}, API-ID: ka10080)")

                while current_page < total_pages:
                    # 3. Throttling 방지
                    await self._throttle()

                    headers = {
                        'Content-Type': 'application/json;charset=UTF-8',
                        "authorization": f"Bearer {access_token}",
                        "api-id": "ka10080",
                        "cont-yn": cont_yn,
                        "next-key": next_key
                    }

                    # 입력 파라미터 (명세 준수 + qry_tp 추가)
                    base_dt_param = target_start_date.replace("-", "")[:8]
                    params = {
                        "stk_cd": formatted_symbol,
                        "tic_scope": "1",
                        "upd_stkpc_tp": "1",
                        "base_dt": base_dt_param,
                        "qry_tp": "0"
                    }

                    try:
                        async with session.post(endpoint, headers=headers, json=params, timeout=15) as response:
                            resp_headers = response.headers
                            cont_yn = resp_headers.get("cont-yn", "N")
                            next_key = resp_headers.get("next-key", "")
                            
                            if response.status != 200:
                                break

                            data = await response.json()
                            
                            if data.get("return_code") == 3 or "Token이 유효하지 않습니다" in data.get("return_msg", ""):
                                return Failure("TOKEN_EXPIRED")

                            # 리스트 추출
                            items = data.get("stk_min_pole_chart_qry")
                            if items is None: items = data.get("output2")
                            if items is None: items = data.get("grid")
                            if items is None: items = data.get("output")
                            
                            if not items or not isinstance(items, list):
                                break

                            batch_data = []
                            last_timestamp_in_batch = ""
                            valid_item_found = False
                            
                            stop_reached = False
                            for item in items:
                                # 유효성 검사: 핵심 필드가 모두 빈 값인지 확인
                                raw_time = item.get("cntr_tm") or item.get("stck_cntg_hour") or ""
                                cur_prc = item.get("cur_prc") or item.get("stck_prpr") or ""
                                
                                if not raw_time or not cur_prc:
                                    continue # 빈 데이터 스킵
                                
                                valid_item_found = True
                                
                                # 시간 포맷팅
                                if len(raw_time) >= 14:
                                    formatted_ts = f"{raw_time[:4]}-{raw_time[4:6]}-{raw_time[6:8]} {raw_time[8:10]}:{raw_time[10:12]}:{raw_time[12:14]}"
                                elif len(raw_time) == 12: 
                                    formatted_ts = f"20{raw_time[:2]}-{raw_time[2:4]}-{raw_time[4:6]} {raw_time[6:8]}:{raw_time[8:10]}:{raw_time[10:12]}"
                                else:
                                    formatted_ts = raw_time

                                # 증분 수집 중단 체크: 이미 DB에 있는 시점에 도달함
                                # [버그 수정] T 문자와 공백 혼용으로 인한 문자열 비교 오류 방지 (정규화)
                                norm_target = formatted_ts.replace("T", " ")
                                norm_stop = stop_timestamp.replace("T", " ") if stop_timestamp else ""
                                
                                if stop_timestamp and norm_target <= norm_stop:
                                    self.logger.error(f"[{symbol}] 증분 수집 중단 시점 도달: {norm_target} <= {norm_stop}")
                                    stop_reached = True
                                    break

                                last_timestamp_in_batch = formatted_ts

                                try:
                                    def _to_float(v): 
                                        if v is None or v == "": return 0.0
                                        return float(str(v).lstrip('+-'))
                                    
                                    o = _to_float(item.get("open_pric") or item.get("stck_oprc"))
                                    h = _to_float(item.get("high_pric") or item.get("stck_hgpr"))
                                    l = _to_float(item.get("low_pric") or item.get("stck_lwpr"))
                                    c = _to_float(cur_prc)
                                    v = _to_float(item.get("trde_qty") or item.get("cntg_vol"))
                                except (ValueError, TypeError):
                                    continue

                                batch_data.append({
                                    "timestamp": formatted_ts,
                                    "symbol": symbol,
                                    "open": o,
                                    "high": h,
                                    "low": l,
                                    "price": c,
                                    "volume": v
                                })

                            if not valid_item_found and current_page == 0:
                                # 첫 페이지인데 유효한 데이터가 하나도 없으면 이 형식은 실패로 간주
                                break

                            all_data.extend(batch_data)
                            current_page += 1

                            if progress_callback:
                                progress_callback(int((current_page / total_pages) * 100), 
                                                f"[{symbol}] {current_page}페이지 수집됨 ({len(batch_data)}건)")

                            if cont_yn == "N" or not next_key or stop_reached:
                                break

                    except Exception as e:
                        self.logger.error(f"[{symbol}] 형식 {formatted_symbol} 시도 중 오류: {e}")
                        break
                
                if all_data:
                    final_data = all_data
                    self.logger.error(f"[{symbol}] 유효한 데이터 수집 성공 (형식: {formatted_symbol}, 건수: {len(all_data)})")
                    break
                else:
                    self.logger.error(f"[{symbol}] 형식 {formatted_symbol} 결과가 유효하지 않음. 다음 형식 시도...")

        if final_data:
            self.logger.error(f"[{symbol}] 최종 수집 완료: 총 {len(final_data)}건")
            self._save_checkpoint(symbol, "COMPLETED")
            return final_data
        
        self.logger.error(f"[{symbol}] 모든 가능한 형식으로 시도했으나 유효한 데이터를 찾지 못했습니다.")
        return []
