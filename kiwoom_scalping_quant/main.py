import sys
import asyncio
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel
from qasync import QEventLoop, asyncSlot

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
        self.config = {"symbol": "005930"}

        self.order_manager = OrderManager(self.config, auth_manager=None)
        self.data_collector = DataCollector(self.config)

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
