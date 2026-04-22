import sys
import os
import asyncio
import yaml
from dotenv import load_dotenv
from PyQt6.QtWidgets import QApplication
from qasync import QEventLoop

from core.container import Container
from gui.main_window import MainWindow

# import logging
#
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s [%(levelname)s] %(message)s"
# )

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
        self.strategy_manager = self.container.strategy_manager()
        self.token_manager = self.container.token_manager()
        self.market_scheduler = self.container.market_scheduler()
        self.risk_manager = self.container.risk_manager()

        # Risk Manager injection loop closing
        self.order_manager.risk_manager = self.risk_manager

        # [안정화] DataCollector에 TokenManager 참조 주입 → LOGIN 실패 시 토큰 자동 갱신 가능
        self.data_collector.config._token_manager = self.token_manager

        # Connect Daily Stop-Loss Signal
        self.risk_manager.signals.daily_stop_loss_hit.connect(self._on_stop_loss_hit)

        # 대표 ViewModel 생성 (LiveDashboardViewModel)
        self.live_vm = self.container.live_dashboard_view_model()
        self.asset_vm = self.container.asset_data_view_model()

        # Inject references for background managers safely
        config_mgr = self.container.config_manager()
        config_mgr._injected_scheduler = self.market_scheduler
        config_mgr._injected_strategy_manager = self.strategy_manager
        config_mgr._injected_live_vm = self.live_vm
        config_mgr._injected_asset_data_vm = self.asset_vm

        # GUI 초기화: ViewModel 주입
        self.main_window = MainWindow(self.live_vm, self)
        self.shutdown_event = asyncio.Event()

        # Boot sequence events
        self.token_ready_event = asyncio.Event()
        self.universe_ready_event = asyncio.Event()

        # Connect TokenManager signals to UI
        self.token_manager.signals.token_updated.connect(self._on_token_updated)
        self.token_manager.signals.token_error.connect(self._on_token_error)

        # Connect Universe ready signal
        self.asset_vm.symbols_loaded.connect(self._on_universe_ready)

    def _on_token_updated(self, msg: str):
        self.token_ready_event.set()
        if hasattr(self.main_window, 'statusBar'):
            self.main_window.statusBar().showMessage(f"[알림] {msg}", 5000)

    def _on_universe_ready(self, symbols: list):
        self.universe_ready_event.set()

    def _on_token_error(self, msg: str):
        if hasattr(self.main_window, 'statusBar'):
            self.main_window.statusBar().showMessage(f"[에러] {msg}", 5000)

    def _on_stop_loss_hit(self, loss_amount: float):
        # Notify UI and trigger Scheduler panic
        msg = f"🚨 [CRITICAL] 당일 최대 손실 도달 ({loss_amount:,.0f}원): 거래 강제 종료"
        self.live_vm.sig_log_appended.emit(msg)
        self.market_scheduler.trigger_daily_stop_loss()

        # Send telegram bot message
        import requests
        tg_token = self.container.config_manager().get("TELEGRAM_BOT_TOKEN")
        chat_id = self.container.config_manager().get("telegram_chat_id")

        if tg_token and chat_id:
            try:
                url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
                payload = {"chat_id": chat_id, "text": msg}
                # Fire and forget request
                requests.post(url, json=payload, timeout=2.0)
            except Exception as e:
                print(f"시스템: 텔레그램 발송 실패: {e}")

    async def start(self):
        self.main_window.show()
        print("시스템: 부팅 시퀀스를 시작합니다.")

        # Step 1: Token 발급 완료 대기
        self.token_task = asyncio.create_task(self.token_manager.start())
        print("시스템: [Step 1] 토큰 발급 대기 중...")
        try:
            await asyncio.wait_for(self.token_ready_event.wait(), timeout=10.0)
            print("시스템: [Step 1] 토큰 발급 완료.")
        except asyncio.TimeoutError:
            print("시스템: [Step 1] 토큰 발급 타임아웃! (인터넷 연결 및 앱 키를 확인하세요)")

        # Step 2: Universe 및 Scheduler 시작
        print("시스템: [Step 2] 스케줄러 가동 및 기존 유니버스 로드...")
        self.scheduler_task = asyncio.create_task(self.market_scheduler.start())

        # [버그 수정/기능 개선] 부팅 시마다 현재가와 투자 한도를 대조하여 필터링하고 새로운 종목으로 채웁니다.
        # 기존의 단순 로딩(load_symbols) 대신 자동 갱신(build_universe) 태스크를 비동기로 실행합니다.
        asyncio.create_task(self.asset_vm._build_universe_task(is_auto=True))

        try:
            # 유니버스 로드가 완료될 때까지 잠시 대기
            await asyncio.wait_for(self.universe_ready_event.wait(), timeout=10.0)
            print(f"시스템: [Step 2] 유니버스 로드 완료 (총 {len(self.asset_vm.config_manager.get_symbols())}개 종목).")
        except asyncio.TimeoutError:
            print("시스템: [Step 2] 유니버스 로드 타임아웃! 기본 설정으로 진행합니다.")

        # Step 3: DataCollector 시작 및 웹소켓 연결 대기
        self.influx_task = asyncio.create_task(self.influx_client.start())
        print("시스템: [Step 3] DataCollector 가동 및 웹소켓 구독 대기...")
        self.collector_task = asyncio.create_task(self.data_collector.start())

        try:
            if hasattr(self.data_collector, 'first_data_received_event'):
                await asyncio.wait_for(self.data_collector.first_data_received_event.wait(), timeout=15.0)
                print("시스템: [Step 3] 최초 웹소켓 틱 데이터 수신 확인 완료.")
        except asyncio.TimeoutError:
            print("시스템: [Step 3] 웹소켓 데이터 수신 타임아웃! (장이 닫혔거나 구독 실패일 수 있습니다)")

        # Step 4: saved_models/ 에서 최신 학습 모델을 자동 탐색하여 로드
        print("시스템: [Step 4] Agent 루프(StrategyManager) 및 Watchdog 가동 시작.")
        import glob as _glob
        _save_dir = "./saved_models/"
        _model_files = sorted(_glob.glob(f"{_save_dir}model_*.zip"))
        if _model_files:
            _latest = _model_files[-1].replace(".zip", "")
            print(f"시스템: [Step 4] 학습된 모델 발견 → 로드: {_model_files[-1]}")
            self.strategy_manager.load_model(_latest)
        else:
            print("시스템: [Step 4] 저장된 모델이 없습니다. 랜덤 초기 가중치로 실행합니다. (AI 학습 스튜디오에서 학습을 먼저 실행하세요)")
            self.strategy_manager.load_model("")
        self.strategy_task = asyncio.create_task(self.strategy_manager.start())
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

        # 1. 뷰모델 갱신 즉시 중지
        self.live_vm.stop()

        # 2. 미체결 주문 일괄 취소 (최대 3초 대기)
        try:
            print("시스템: 미체결 주문 전체 취소 중...")
            await asyncio.wait_for(self.order_manager.cancel_all_orders(), timeout=3.0)
        except Exception as e:
            print(f"시스템: 주문 취소 중 오류 또는 타임아웃 발생: {e}")

        # 3 & 4. 매매 및 수집 정지
        try:
            print("시스템: Strategy 및 DataCollector 정지 중...")
            # 동시에 정지 프로세스 가동 (시간 절약)
            await asyncio.wait_for(
                asyncio.gather(
                    self.strategy_manager.stop(),
                    self.data_collector.stop(),
                    return_exceptions=True
                ),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            print("시스템: 정지 프로세스 타임아웃 - 강제 다음 단계 진행")

        # 5. 백그라운드 태스크 Cancel 및 정리
        tasks = [
            ('view_model', getattr(self, 'view_model_task', None)),
            ('collector', getattr(self, 'collector_task', None)),
            ('strategy', getattr(self, 'strategy_task', None)),
            ('scheduler', getattr(self, 'scheduler_task', None)),
            ('token_manager', getattr(self, 'token_task', None))
        ]

        # Stop background helpers gracefully first if they have stop methods
        if hasattr(self, 'token_manager'):
            await self.token_manager.stop()
        if hasattr(self, 'market_scheduler'):
            await self.market_scheduler.stop()

        for name, task in tasks:
            if task and not task.done():
                print(f"시스템: {name} 태스크 취소 중...")
                task.cancel()
                try:
                    # 짧게 대기하며 정리 기회 부여
                    await asyncio.wait_for(task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        # 6. DB 연결 닫기 (가장 마지막에 수행)
        print("시스템: InfluxDB 연결 닫기 및 데이터 Flush...")
        if hasattr(self, 'influx_client'):
            try:
                await asyncio.wait_for(self.influx_client.close(), timeout=3.0)
            except:
                pass

        # [안정화] 7. Windows IOCP 잔여 처리 유예
        # 소켓이 닫힌 후 프로액터가 완료 이벤트를 인지할 수 있는 최소 1틱의 시간을 제공
        await asyncio.sleep(0.2)

        # 8. 종료 이벤트 세트 (main 함수의 loop가 이를 인지하고 탈출하도록 함)
        print("시스템: 모든 정리가 완료되었습니다.")
        self.shutdown_event.set()

def main():
    app = QApplication(sys.argv)

    if not hasattr(app, "exec_"):
        app.exec_ = app.exec

    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    system = QuantSystem()

    try:
        # 1. Main Loop Execution
        loop.run_until_complete(system.start())
    except KeyboardInterrupt:
        print("\nSystem: Force stopped by user (KeyboardInterrupt).")
        try:
            loop.run_until_complete(system.stop())
        except Exception as stop_e:
            print(f"System: Error during stop: {stop_e}")
    except RuntimeError as e:
        if "Event loop stopped before Future completed" in str(e):
            print("System: Async loop finished normally.")
        else:
            raise e
    finally:
        # 2. Final cleanup and grace period
        print("System: Performing final cleanup sequence...")
        try:
            # Cancel all remaining tasks before closing loop
            tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
            if tasks:
                for task in tasks:
                    task.cancel()
                # Give tasks a chance to finalize (0.5s)
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            
            # Final delay for Proactor (IOCP) handles
            loop.run_until_complete(asyncio.sleep(0.1))
            
            if not loop.is_closed():
                loop.stop()
                loop.close()
                print("System: Async loop closed safely.")
        except Exception as cleanup_e:
            print(f"System: Cleanup error (ignored): {cleanup_e}")
        
        print("System: Program fully terminated.")
        sys.exit(0)

if __name__ == "__main__":
    main()
