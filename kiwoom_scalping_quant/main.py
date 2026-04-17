import sys
import os
import asyncio
import yaml
from dotenv import load_dotenv
from PyQt6.QtWidgets import QApplication
from qasync import QEventLoop

from core.container import Container
from gui.main_window import MainWindow

class QuantSystem:
    def __init__(self):
        # 환경 변수 로드 (.env)
        load_dotenv()

        # DI 컨테이너 초기화
        self.container = Container()

        # config.yaml에서 설정 로드
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        except Exception as e:
            print(f"설정 파일 로드 실패, 기본값 사용: {e}")
            config_dict = {
                "symbol": "005930",
                "ws_url": "ws://localhost:8080/kiwoom",
                "max_buffer_size": 10000,
                "db_batch_size": 500
            }

        self.container.config.from_dict(config_dict)
        # 의존성 와이어링 (필요시)
        self.container.wire(modules=[__name__])

        # 컨테이너를 통해 코어 객체 생성
        self.influx_client = self.container.influx_client()
        self.order_manager = self.container.order_manager()
        self.data_collector = self.container.data_collector()

        # ViewModel 생성
        self.view_model = self.container.market_data_view_model()

        # GUI 초기화: ViewModel만 주입
        self.main_window = MainWindow(self.view_model, self)
        self.is_running = False

    async def start(self):
        self.is_running = True
        self.main_window.show()

        # 백그라운드 태스크 시작
        self.influx_task = asyncio.create_task(self.influx_client.start())
        self.collector_task = asyncio.create_task(self.data_collector.start())
        self.view_model_task = asyncio.create_task(self.view_model.start_polling())

        try:
            # 무한 루프로 유지하되, GUI가 종료되면 빠져나옴
            while self.is_running:
                await asyncio.sleep(0.1)
        finally:
            # 루프를 빠져나오면 안전하게 데이터 수집기를 종료
            await self.data_collector.stop()

    def stop(self):
        """시스템 종료 로직 (GUI closeEvent에서 호출됨)"""
        self.is_running = False
        self.view_model.stop()
        if hasattr(self, 'view_model_task') and not self.view_model_task.done():
            self.view_model_task.cancel()
        if hasattr(self, 'collector_task') and not self.collector_task.done():
            self.collector_task.cancel()
        asyncio.create_task(self.influx_client.close())

def main():
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    system = QuantSystem()

    with loop:
        loop.run_until_complete(system.start())

if __name__ == "__main__":
    main()
