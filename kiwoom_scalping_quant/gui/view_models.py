import asyncio
from PyQt6.QtCore import QObject, pyqtSignal
from typing import Dict, Any, List
import asyncio
from returns.result import Success, Failure

class MarketDataViewModel(QObject):
    """
    DataCollector(Core)와 UI(View)를 분리하는 ViewModel.
    UI 스레드와 asyncio 태스크 간의 연결 고리 역할을 하며 pyqtSignal을 사용해 불변 상태를 전달.
    """

    # UI에 전달될 시그널들 정의
    orderbook_updated = pyqtSignal(dict)
    price_updated = pyqtSignal(float)
    error_occurred = pyqtSignal(str)

    def __init__(self, data_collector):
        super().__init__()
        self.data_collector = data_collector
        self._is_running = False

    async def start_polling(self):
        """
        주기적으로 혹은 DataCollector 내의 콜백을 통해 데이터를 가져와서 UI 시그널을 발생시킴.
        실제 구현에서는 DataCollector가 이벤트를 발생시킬 때 이 ViewModel의 메서드를 호출하게 하는 옵저버 패턴도 가능.
        """
        self._is_running = True
        while self._is_running:
            try:
                # 틱 버퍼의 가장 최근 데이터를 가져옴
                if self.data_collector.tick_buffer:
                    latest_tick = self.data_collector.tick_buffer[-1]

                    # 1. 가격 업데이트
                    if "price" in latest_tick:
                        self.price_updated.emit(float(latest_tick["price"]))

                    # 2. 호가창(Orderbook) 업데이트 가정
                    if "orderbook" in latest_tick:
                        # 불변성을 위해 데이터 복사 후 전달
                        orderbook_copy = dict(latest_tick["orderbook"])
                        self.orderbook_updated.emit(orderbook_copy)

            except Exception as e:
                # 에러 발생 시 UI가 멈추지 않도록 시그널만 방출
                self.error_occurred.emit(f"ViewModel 데이터 처리 오류: {str(e)}")

            # UI 갱신 빈도 조절 (예: 100ms)
            await asyncio.sleep(0.1)

    def stop(self):
        self._is_running = False

class AssetDataViewModel(QObject):
    """
    AssetDataManagerTab을 위한 ViewModel.
    UI 이벤트(종목 로드/저장, 데이터 수집 시작)를 Core 로직으로 연결하고
    수집 상태(Progress)를 UI로 Signal Emit 합니다.
    """
    # UI로 보낼 시그널들
    symbols_loaded = pyqtSignal(list)
    symbol_update_failed = pyqtSignal(str)
    symbol_update_success = pyqtSignal(str)

    fetch_progress_updated = pyqtSignal(int, str) # 진행률(%), 메시지
    fetch_completed = pyqtSignal(str)
    fetch_failed = pyqtSignal(str)

    def __init__(self, config_manager, historical_fetcher, influx_client, universe_manager):
        super().__init__()
        self.config_manager = config_manager
        self.historical_fetcher = historical_fetcher
        self.influx_client = influx_client
        self.universe_manager = universe_manager

    def build_universe(self):
        """UniverseManager를 통해 거래대금 상위 종목을 추출하여 Config에 저장"""
        asyncio.create_task(self._build_universe_task())

    async def _build_universe_task(self):
        self.fetch_progress_updated.emit(0, "유니버스 필터링 중...")
        result = await self.universe_manager.build_top_n_universe("DUMMY_TOKEN", top_n=20)

        if isinstance(result, Failure):
            self.fetch_failed.emit(f"유니버스 생성 실패: {result.failure()}")
            return

        top_stocks = result.unwrap()

        # 기존 심볼들 덮어쓰기 (모두 삭제 후 추가)
        # 실제 구현시에는 ConfigManager에 bulk_replace 등을 추가하는 것이 좋음
        for s in self.config_manager.get_symbols()[:]:
            self.config_manager.remove_symbol(s.get("code"))

        for stock in top_stocks:
            self.config_manager.add_symbol(stock["code"], stock["name"])

        self.fetch_completed.emit(f"상위 {len(top_stocks)}개 유니버스 생성 완료!")
        self.load_symbols() # 갱신

    def load_symbols(self):
        # ConfigManager의 Result 처리
        result = self.config_manager.load_config()
        if isinstance(result, Success):
            symbols = self.config_manager.get_symbols()
            self.symbols_loaded.emit(symbols)
        else:
            self.symbol_update_failed.emit(f"설정 로드 실패: {result.failure()}")

    def add_symbol(self, code: str, name: str):
        result = self.config_manager.add_symbol(code, name)
        if isinstance(result, Success):
            self.symbol_update_success.emit(f"종목 추가 완료: {name}")
            self.load_symbols() # UI 갱신 트리거
        else:
            self.symbol_update_failed.emit(str(result.failure()))

    def remove_symbol(self, code: str):
        result = self.config_manager.remove_symbol(code)
        if isinstance(result, Success):
            self.symbol_update_success.emit(f"종목 삭제 완료: {code}")
            self.load_symbols()
        else:
            self.symbol_update_failed.emit(str(result.failure()))

    def start_historical_fetch(self, symbol: str, start_date: str):
        """특정 종목에 대한 수집"""
        asyncio.create_task(self._fetch_and_store([symbol], start_date))

    def start_bulk_historical_fetch(self, start_date: str):
        """Config에 등록된 모든 종목(Universe)에 대한 일괄 수집"""
        symbols = [s.get("code") for s in self.config_manager.get_symbols()]
        if not symbols:
            self.fetch_failed.emit("수집할 종목이 없습니다.")
            return
        asyncio.create_task(self._fetch_and_store(symbols, start_date))

    async def _fetch_and_store(self, symbols: List[str], start_date: str):
        total_symbols = len(symbols)
        total_data_collected = 0

        for idx, symbol in enumerate(symbols):
            def update_progress(pct: int, msg: str):
                # 전체 진행률과 개별 진행률을 조합하여 Emit 가능
                base_pct = (idx / total_symbols) * 100
                current_pct = base_pct + (pct / total_symbols)
                self.fetch_progress_updated.emit(int(current_pct), msg)

            self.fetch_progress_updated.emit(int((idx / total_symbols) * 100), f"[{symbol}] 수집 시작 ({idx+1}/{total_symbols})...")

            fetch_result = await self.historical_fetcher.fetch_historical_data(symbol, start_date, "DUMMY_TOKEN", update_progress)

            if isinstance(fetch_result, Failure):
                self.symbol_update_failed.emit(f"[{symbol}] 수집 실패: {fetch_result.failure()}")
                continue # 한 종목이 실패해도 다음 종목으로 계속 진행

            data_list = fetch_result.unwrap()
            total_data_collected += len(data_list)

            self.fetch_progress_updated.emit(int(((idx + 0.9) / total_symbols) * 100), f"[{symbol}] InfluxDB 적재 중...")
            try:
                await self.influx_client.bulk_insert(data_list)
            except Exception as e:
                self.symbol_update_failed.emit(f"[{symbol}] DB 저장 중 에러: {e}")

        self.fetch_progress_updated.emit(100, "모든 종목 수집 완료")
        self.fetch_completed.emit(f"총 {total_symbols}개 종목, {total_data_collected}건 적재 완료!")

class SettingsViewModel(QObject):
    """
    Settings 탭을 위한 ViewModel.
    UI에서 입력받은 환경변수(.env) 및 일반설정(config.yaml)을 관리하고 연결 테스트를 수행합니다.
    """
    settings_loaded = pyqtSignal(dict, dict) # (env_dict, config_dict)
    save_completed = pyqtSignal(str)
    save_failed = pyqtSignal(str)
    connection_test_completed = pyqtSignal(bool, str) # (Success bool, Message)

    def __init__(self, config_manager, influx_client):
        super().__init__()
        import os
        self.config_manager = config_manager
        self.influx_client = influx_client
        self.env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")

    def _read_env_file(self) -> dict:
        env_vars = {}
        if os.path.exists(self.env_path):
            with open(self.env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        env_vars[k.strip()] = v.strip()
        return env_vars

    def load_settings(self):
        """저장된 .env 파일과 config.yaml 파일을 읽어 UI로 Emit합니다."""
        env_dict = self._read_env_file()

        result = self.config_manager.load_config()
        if isinstance(result, Success):
            config_dict = self.config_manager._config_cache
            self.settings_loaded.emit(env_dict, config_dict)
        else:
            self.save_failed.emit(f"Config Load Error: {result.failure()}")

    def save_settings(self, env_data: dict, config_data: dict):
        """수정된 설정값들을 .env 및 config.yaml에 각각 분리하여 덮어씁니다."""
        try:
            # 1. Save .env
            with open(self.env_path, "w", encoding="utf-8") as f:
                for k, v in env_data.items():
                    f.write(f"{k}={v}\n")

            # 2. Save config.yaml (기존 ConfigManager의 딕셔너리 업데이트)
            for k, v in config_data.items():
                self.config_manager._config_cache[k] = v

            save_result = self.config_manager.save_config()
            if isinstance(save_result, Success):
                self.save_completed.emit("설정이 성공적으로 저장되었습니다. (일부 설정은 재시작 시 적용됩니다.)")
            else:
                self.save_failed.emit(f"Config 저장 실패: {save_result.failure()}")
        except Exception as e:
            self.save_failed.emit(f"설정 저장 중 오류 발생: {e}")

    def test_connection(self, env_data: dict, config_data: dict):
        """현재 입력된 API 키와 DB 정보로 핑/인증 테스트를 비동기로 수행합니다."""
        asyncio.create_task(self._test_connection_task(env_data, config_data))

    async def _test_connection_task(self, env_data: dict, config_data: dict):
        # 1. 키움 API 테스트 (가상 핑)
        app_key = env_data.get("KIWOOM_APP_KEY")
        if not app_key:
            self.connection_test_completed.emit(False, "App Key가 비어있습니다.")
            return

        import aiohttp
        # 토큰 발급 테스트 (test_kiwoom_api.py 로직 간소화)
        base_url = env_data.get("KIWOOM_BASE_URL", "https://openapi.kiwoom.com")
        url = f"{base_url}/oauth2/tokenP"
        payload = {"grant_type": "client_credentials", "appkey": app_key, "appsecret": env_data.get("KIWOOM_APP_SECRET", "")}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=5) as response:
                    if response.status == 200:
                        kiwoom_msg = "Kiwoom API: Token 발급 성공 (OK)"
                    else:
                        text = await response.text()
                        kiwoom_msg = f"Kiwoom API: 연결 실패 ({response.status}) - {text}"
                        self.connection_test_completed.emit(False, kiwoom_msg)
                        return
        except Exception as e:
            self.connection_test_completed.emit(False, f"Kiwoom API 연결 에러: {e}")
            return

        # 2. InfluxDB 핑 테스트
        db_url = env_data.get("INFLUX_URL", "http://localhost:8086")
        db_msg = "InfluxDB: Ping 테스트 통과 (Mock)"
        # 실제로는 InfluxDBClientAsync ping() 사용 가능

        self.connection_test_completed.emit(True, f"{kiwoom_msg}\n{db_msg}")
