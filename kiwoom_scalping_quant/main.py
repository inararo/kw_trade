import sys
import asyncio
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel
from qasync import QEventLoop, asyncSlot

from core.container import Container
from core.data_collector import DataCollector
from core.order_manager import OrderManager

class MainWindow(QMainWindow):
    def __init__(self, data_collector, order_manager, system):
        super().__init__()
        self.system = system
        self.setWindowTitle("Kiwoom Scalping Quant")
        self.setGeometry(100, 100, 400, 300)
        label = QLabel("스캘핑 퀀트 시스템 실행 중...", self)
        label.setGeometry(50, 50, 300, 50)

    def closeEvent(self, event):
        """GUI 창 닫기 버튼 클릭 시 안전한 종료 트리거"""
        self.system.stop()
        event.accept()

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
        self.main_window = MainWindow(self.data_collector, self.order_manager, self)
        self.is_running = False

    async def start(self):
        self.is_running = True
        self.main_window.show()
        self.collector_task = asyncio.create_task(self.data_collector.start())

        # 무한 루프로 유지하되, GUI가 종료되면 빠져나옴
        while self.is_running:
            await asyncio.sleep(0.1)

    def stop(self):
        """시스템 종료 로직"""
        self.is_running = False
        if hasattr(self, 'collector_task'):
            self.collector_task.cancel()
        asyncio.create_task(self.data_collector.stop())

def main():
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    system = QuantSystem()

    with loop:
        loop.run_until_complete(system.start())

if __name__ == "__main__":
    main()
