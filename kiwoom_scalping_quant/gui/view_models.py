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

    def __init__(self, config_manager, historical_fetcher, influx_client):
        super().__init__()
        self.config_manager = config_manager
        self.historical_fetcher = historical_fetcher
        self.influx_client = influx_client

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
        # QThread/QTimer 대신 asyncio.create_task를 통해 비동기 실행을 위임
        # qasync를 사용하므로 안전함
        asyncio.create_task(self._fetch_and_store(symbol, start_date))

    async def _fetch_and_store(self, symbol: str, start_date: str):
        # 1. API 수집 (진행률 콜백을 시그널 Emit으로 연결)
        def update_progress(pct: int, msg: str):
            self.fetch_progress_updated.emit(pct, msg)

        self.fetch_progress_updated.emit(0, "데이터 수집 시작...")

        # future_safe Result 반환 확인
        fetch_result = await self.historical_fetcher.fetch_historical_data(symbol, start_date, "DUMMY_TOKEN", update_progress)

        if isinstance(fetch_result, Failure):
            self.fetch_failed.emit(f"수집 실패: {fetch_result.failure()}")
            return

        data_list = fetch_result.unwrap()

        # 2. InfluxDB Bulk Insert
        self.fetch_progress_updated.emit(90, "InfluxDB 적재 중...")
        try:
            await self.influx_client.bulk_insert(data_list)
            self.fetch_progress_updated.emit(100, "모든 작업 완료")
            self.fetch_completed.emit(f"총 {len(data_list)}건 적재 완료!")
        except Exception as e:
            self.fetch_failed.emit(f"DB 저장 중 에러: {e}")
