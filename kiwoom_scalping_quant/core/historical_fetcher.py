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
                                    progress_callback: Optional[Callable[[int, str], None]] = None) -> List[Dict[str, Any]]:
        """
        특정 종목의 과거 데이터를 연속 조회합니다. (Pagination & Resume)
        """
        all_data = []

        # 1. Resume Check
        last_fetched = self._load_checkpoint(symbol)
        if last_fetched and last_fetched > start_date:
            self.logger.info(f"[{symbol}] 체크포인트 발견. {last_fetched} 이후 데이터만 수집합니다.")
            target_start_date = last_fetched
        else:
            target_start_date = start_date

        # 가상의 TR 반복 호출 세팅
        next_token = ""
        total_pages = 20 # 1년치 분봉의 가상 페이지 수
        current_page = 0

        headers = {
            'Content-Type': 'application/json;charset=UTF-8',  # 컨텐츠타입
            "Authorization": f"Bearer {access_token}",
            "api-id": "OPT10080" # 주식분봉차트조회요청
        }

        async with aiohttp.ClientSession() as session:
            while current_page < total_pages:
                # 2. 강제 딜레이 큐 대기
                await self._throttle()

                params = {"symbol": symbol, "start_date": target_start_date, "next": next_token}
                url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-time-item"

                if current_page == 0:
                    self.logger.info(f"[{symbol}] 과거 데이터 조회 시작 (Target URL: {url})")

                # Mock API 호출 (서버 과부하 회피를 위한 시간 추가)
                await asyncio.sleep(0.5)

                # 수집된 가상 데이터 (역순 수집 가정)
                current_date = f"2023-{12 - (current_page // 3):02d}-01"
                fake_batch = [{"timestamp": current_date, "symbol": symbol, "price": 50000 + current_page * 10}]
                all_data.extend(fake_batch)

                current_page += 1
                progress = int((current_page / total_pages) * 100)

                if progress_callback:
                    progress_callback(progress, f"[{symbol}] {current_page}/{total_pages} 페이지 수집 중...")

                # 3. 매 N페이지마다 상태 저장 (정전 및 API 제한 대비)
                if current_page % 5 == 0:
                    self._save_checkpoint(symbol, current_date)

                # next_token 갱신 로직 (data.get("next_token"))
                # if not next_token: break

        # 수집 완료 후 최신 상태로 체크포인트 갱신
        self._save_checkpoint(symbol, "COMPLETED")
        return all_data
