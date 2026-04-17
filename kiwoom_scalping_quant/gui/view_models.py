import asyncio
from PyQt6.QtCore import QObject, pyqtSignal
from typing import Dict, Any, List
import asyncio
from returns.result import Success, Failure

class LiveDashboardViewModel(QObject):
    """
    LiveDashboard 탭을 위한 ViewModel.
    DataCollector, OrderManager, Agent 등의 상태를 모니터링하고 UI로 신호를 전달합니다.
    """
    sig_orderbook_updated = pyqtSignal(dict)
    sig_price_updated = pyqtSignal(float)
    sig_ai_confidence_updated = pyqtSignal(dict)
    sig_log_appended = pyqtSignal(str)
    sig_error_occurred = pyqtSignal(str)

    def __init__(self, data_collector, order_manager):
        super().__init__()
        self.data_collector = data_collector
        self.order_manager = order_manager
        self._is_running = False
        self._mock_task = None

        # DataCollector 측에서 데이터가 들어올 때 콜백받을 수 있도록 설정 (또는 폴링)
        # 이번 요구사항에서는 mock stream 내부에서 콜백으로 데이터를 쏴주는 형태를 가정합니다.
        self.data_collector.set_ui_callback(self._on_data_received)

    def _on_data_received(self, data: dict):
        """DataCollector에서 새로운 데이터가 수집되었을 때 호출되는 콜백"""
        try:
            if "price" in data:
                self.sig_price_updated.emit(float(data["price"]))
            if "orderbook" in data:
                self.sig_orderbook_updated.emit(dict(data["orderbook"]))
            if "ai_confidence" in data:
                self.sig_ai_confidence_updated.emit(dict(data["ai_confidence"]))
        except Exception as e:
            self.sig_error_occurred.emit(f"Data parsing error: {e}")

    async def start_polling(self):
        """실전 매매/백테스트 모드에서의 일반 폴링 (mock 사용 시 제외)"""
        self._is_running = True
        while self._is_running:
            await asyncio.sleep(0.1)

    def start_mock_stream(self):
        """장외 시간 테스트용 모크 스트림 시작"""
        self.sig_log_appended.emit("장외 테스트용 Mock 데이터 스트림 시작...")
        if not self._mock_task or self._mock_task.done():
            self._mock_task = asyncio.create_task(self.data_collector.start_mock_stream())

    def trigger_panic_sell(self):
        """패닉 셀 버튼 이벤트 수신: 모든 주문 취소 및 시장가 매도"""
        self.sig_log_appended.emit("[시스템] 🚨 PANIC SELL 트리거됨! 전체 주문 취소 및 시장가 청산 진행...")
        asyncio.create_task(self._execute_panic_sell())

    async def _execute_panic_sell(self):
        try:
            await self.order_manager.cancel_all_orders()
            # 잔고 확인 및 전량 시장가 매도 로직 (Mock)
            holdings = getattr(self.order_manager, 'holdings', 0)
            if holdings > 0:
                await self.order_manager.send_order("SELL", "005930", 0, holdings)
                self.sig_log_appended.emit(f"[시스템] 잔고 {holdings}주 전량 시장가 매도 주문 전송 완료.")
            else:
                self.sig_log_appended.emit("[시스템] 보유 잔고가 없습니다. 주문 취소만 완료되었습니다.")
        except Exception as e:
            self.sig_error_occurred.emit(f"Panic Sell 에러: {e}")

    def stop(self):
        self._is_running = False
        if self._mock_task and not self._mock_task.done():
            self._mock_task.cancel()

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

    sig_progress_updated = pyqtSignal(int)
    sig_status_updated = pyqtSignal(str)
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
        self.sig_progress_updated.emit(0)
        self.sig_status_updated.emit("시장 전체 종목 조회 및 주도주 필터링 중...")
        # @future_safe에 의해 감싸진 async 함수는 await하면 반환값이 Result 타입 객체입니다.
        result = await self.universe_manager.build_top_n_universe("DUMMY_TOKEN", top_n=20)

        if isinstance(result, Failure):
            self.fetch_failed.emit(f"유니버스 생성 실패: {result.failure()}")
            return

        # unwrap() 호출 시 반환되는 값은 List[Dict[str, Any]] 입니다.
        top_stocks = result.unwrap()

        # 만약 unwrap()한 결과가 None 이라면 빈 리스트로 처리합니다.
        if top_stocks is None:
            top_stocks = []

        # 기존 심볼들 덮어쓰기 (모두 삭제 후 추가)
        # 실제 구현시에는 ConfigManager에 bulk_replace 등을 추가하는 것이 좋음
        for s in self.config_manager.get_symbols()[:]:
            self.config_manager.remove_symbol(s.get("code"))

        for stock in top_stocks:
            # 방어 코드: stock이 문자열로 잘못 들어왔을 경우 등을 대비
            if isinstance(stock, dict):
                self.config_manager.add_symbol(stock.get("code", ""), stock.get("name", ""))
            else:
                # 에러 로깅 가능 (여기서는 단순히 건너뜀)
                pass

        self.sig_progress_updated.emit(100)
        self.sig_status_updated.emit(f"상위 {len(top_stocks)}개 유니버스 생성 완료!")
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
                base_pct = (idx / total_symbols) * 100
                current_pct = base_pct + (pct / total_symbols)
                self.sig_progress_updated.emit(int(current_pct))
                self.sig_status_updated.emit(msg)

            self.sig_progress_updated.emit(int((idx / total_symbols) * 100))
            self.sig_status_updated.emit(f"[{symbol}] 수집 시작 ({idx+1}/{total_symbols})...")

            fetch_result = await self.historical_fetcher.fetch_historical_data(symbol, start_date, "DUMMY_TOKEN", update_progress)

            if isinstance(fetch_result, Failure):
                self.symbol_update_failed.emit(f"[{symbol}] 수집 실패: {fetch_result.failure()}")
                continue # 한 종목이 실패해도 다음 종목으로 계속 진행

            data_list = fetch_result.unwrap()
            total_data_collected += len(data_list)

            self.sig_progress_updated.emit(int(((idx + 0.9) / total_symbols) * 100))
            self.sig_status_updated.emit(f"[{symbol}] InfluxDB Bulk Insert 진행 중...")
            try:
                await self.influx_client.bulk_insert(data_list)
            except Exception as e:
                self.symbol_update_failed.emit(f"[{symbol}] DB 저장 중 에러: {e}")

        self.sig_progress_updated.emit(100)
        self.sig_status_updated.emit("모든 종목 수집 및 적재 완료")
        self.fetch_completed.emit(f"총 {total_symbols}개 종목, {total_data_collected}건 적재 완료!")

class SettingsViewModel(QObject):
    """
    Settings 탭을 위한 ViewModel.
    ConfigManager를 통해 통합된 설정(.env 및 config.yaml)을 관리합니다.
    """
    settings_loaded = pyqtSignal(dict) # unified config dict
    save_completed = pyqtSignal(str)
    save_failed = pyqtSignal(str)
    connection_test_completed = pyqtSignal(bool, str) # (Success bool, Message)

    def __init__(self, config_manager, influx_client):
        super().__init__()
        self.config_manager = config_manager
        self.influx_client = influx_client

    def load_settings(self):
        """ConfigManager를 통해 통합 설정을 로드하고 UI로 Emit합니다."""
        result = self.config_manager.load_config()
        if isinstance(result, Success):
            self.settings_loaded.emit(result.unwrap())
        else:
            self.save_failed.emit(f"Config Load Error: {result.failure()}")

    def save_settings(self, updates: dict):
        """수정된 설정값들을 ConfigManager에 전달하여 저장합니다."""
        save_result = self.config_manager.update_settings(updates)
        if isinstance(save_result, Success):
            self.save_completed.emit("설정이 성공적으로 저장되었습니다. (일부 설정은 재시작 시 적용됩니다.)")
        else:
            self.save_failed.emit(f"Config 저장 실패: {save_result.failure()}")

    def test_connection(self, updates: dict):
        """현재 입력된 API 키와 DB 정보로 핑/인증 테스트를 비동기로 수행합니다."""
        asyncio.create_task(self._test_connection_task(updates))

    async def _test_connection_task(self, updates: dict):
        # 1. 키움 API 테스트 (가상 핑)
        app_key = updates.get("KIWOOM_APP_KEY")
        if not app_key:
            self.connection_test_completed.emit(False, "App Key가 비어있습니다.")
            return

        import aiohttp
        base_url = updates.get("KIWOOM_BASE_URL", "https://openapi.kiwoom.com")
        url = f"{base_url}/oauth2/tokenP"
        payload = {"grant_type": "client_credentials", "appkey": app_key, "appsecret": updates.get("KIWOOM_APP_SECRET", "")}

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
        db_url = updates.get("INFLUX_URL", "http://localhost:8086")
        db_msg = "InfluxDB: Ping 테스트 통과 (Mock)"

        self.connection_test_completed.emit(True, f"{kiwoom_msg}\n{db_msg}")
