import sys
import asyncio
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel
from qasync import QEventLoop, asyncSlot

from core.container import Container
from core.data_collector import DataCollector
from core.order_manager import OrderManager

class MainWindow(QMainWindow):
    def __init__(self, data_collector, order_manager):
        super().__init__()
        self.setWindowTitle("Kiwoom Scalping Quant")
        self.setGeometry(100, 100, 400, 300)
        label = QLabel("스캘핑 퀀트 시스템 실행 중...", self)
        label.setGeometry(50, 50, 300, 50)

class QuantSystem:
    def __init__(self):
        # DI 컨테이너 초기화 및 설정 바인딩
        self.container = Container()
        self.container.config.from_dict({
            "symbol": "005930",
            "ws_url": "ws://localhost:8080/kiwoom",
            "max_buffer_size": 10000,
            "db_batch_size": 500
        })
        # 의존성 와이어링 (필요시)
        self.container.wire(modules=[__name__])

        # 컨테이너를 통해 객체 생성
        self.order_manager = self.container.order_manager()
        self.data_collector = self.container.data_collector()

        # GUI 초기화
        self.main_window = MainWindow(self.data_collector, self.order_manager)

    async def start(self):
        self.main_window.show()
        collector_task = asyncio.create_task(self.data_collector.start())
        await collector_task

def main():
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    system = QuantSystem()

    with loop:
        loop.run_until_complete(system.start())

if __name__ == "__main__":
    main()
