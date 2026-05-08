import logging
import asyncio
from PyQt6.QtCore import QThread, pyqtSignal
from core.condition_service import ConditionService

class ConditionWorkerThread(QThread):
    """
    [UI Bridge] ConditionService를 QThread 내에서 구동하여 UI 블로킹을 방지합니다.
    - 발생한 이벤트는 pyqtSignal을 통해 ViewModel로 전달됩니다.
    """
    sig_snapshot_received = pyqtSignal(list)
    sig_symbol_inserted = pyqtSignal(str, dict)
    sig_symbol_deleted = pyqtSignal(str, dict)
    sig_error = pyqtSignal(str)

    def __init__(self, condition_service: ConditionService, ws_client=None):
        super().__init__()
        self.service = condition_service
        self.ws_client = ws_client # 웹소켓 연결 객체 (주입 필요)
        self.logger = logging.getLogger("ConditionWorkerThread")
        self._is_running = True

        # 서비스 콜백을 시그널로 연결
        self.service.register_callbacks(
            on_insert=self.sig_symbol_inserted.emit,
            on_delete=self.sig_symbol_deleted.emit,
            on_snapshot=self.sig_snapshot_received.emit
        )

    def run(self):
        """스레드 실행 루프 (웹소켓 수신 대기)"""
        self.logger.info("🧵 ConditionWorkerThread 가동 시작")
        try:
            # asyncio 루프를 스레드 내에서 별도로 가동하거나, 
            # 주입된 ws_client가 제공하는 수신 루프를 사용합니다.
            if self.ws_client:
                # 예시: ws_client.listen()이 메시지를 받을 때마다 service.handle_websocket_message 호출
                self.ws_client.on_message = self.service.handle_websocket_message
                self.ws_client.run_forever()
            else:
                self.logger.warning("⚠️ 웹소켓 클라이언트가 주입되지 않았습니다.")
        except Exception as e:
            self.sig_error.emit(str(e))
            self.logger.error(f"WorkerThread Error: {e}")

    def stop(self):
        self._is_running = False
        if self.ws_client:
            self.ws_client.stop()
        self.quit()
        self.wait()
