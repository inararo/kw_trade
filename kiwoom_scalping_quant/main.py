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

        # 대표 ViewModel 생성 (LiveDashboardViewModel)
        self.live_vm = self.container.live_dashboard_view_model()

        # GUI 초기화: ViewModel 주입
        self.main_window = MainWindow(self.live_vm, self)
        self.shutdown_event = asyncio.Event()

    async def start(self):
        self.main_window.show()

        # 백그라운드 태스크 시작
        self.influx_task = asyncio.create_task(self.influx_client.start())
        self.collector_task = asyncio.create_task(self.data_collector.start())
        self.view_model_task = asyncio.create_task(self.live_vm.start_polling())

        try:
            # 종료 시그널이 올 때까지 이벤트 루프 유지
            await self.shutdown_event.wait()
        finally:
            # 루프를 빠져나올 때 수행될 정리
            print("시스템: 메인 루프 종료됨.")

    async def stop(self):
        """비동기 파이프라인 안전 종료 로직 (Graceful Shutdown)"""
        print("시스템: 종료 파이프라인 가동...")

        # 1. 뷰모델 갱신 중지
        self.live_vm.stop()

        # 2. 미체결 주문 일괄 취소 (에이전트 종료 처리)
        print("시스템: 미체결 주문 전체 취소 중...")
        await self.order_manager.cancel_all_orders()

        # 3. 데이터 수집 루프 완전 정지 (WebSocket 및 Watchdog 취소됨)
        print("시스템: DataCollector 및 통신 종료 중...")
        await self.data_collector.stop()

        # 4. 백그라운드 태스크 Cancel
        if hasattr(self, 'view_model_task') and not self.view_model_task.done():
            self.view_model_task.cancel()
        if hasattr(self, 'collector_task') and not self.collector_task.done():
            self.collector_task.cancel()

        # 5. InfluxDB 등 DB 커넥션 종료 및 잔여 버퍼 Flush
        print("시스템: InfluxDB 연결 닫기 및 데이터 Flush...")
        await self.influx_client.close()

        # 6. 최종 윈도우/앱 정리 및 종료
        print("시스템: 모든 정리가 완료되었습니다. 프로그램을 종료합니다.")
        from PyQt6.QtWidgets import QApplication

        # 메인 루프를 끝내기 위해 이벤트 세트
        self.shutdown_event.set()
        QApplication.quit()

def main():
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    system = QuantSystem()

    with loop:
        loop.run_until_complete(system.start())

if __name__ == "__main__":
    main()
