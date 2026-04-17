import asyncio
from PyQt6.QtCore import QObject, pyqtSignal
from typing import Dict, Any

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
