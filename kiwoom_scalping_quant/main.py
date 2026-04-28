import sys
import os
import asyncio
import yaml
from dotenv import load_dotenv
from PyQt6.QtWidgets import QApplication
from qasync import QEventLoop

from core.container import Container
from core.scheduler import MarketState
from gui.main_window import MainWindow

import logging

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s"
)

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
        
        # [추가] 로그 레벨 동적 적용
        log_level_str = config_dict.get("log_level", "INFO").upper()
        logging.getLogger().setLevel(getattr(logging, log_level_str, logging.INFO))
        print(f"시스템: 로그 레벨이 {log_level_str}로 설정되었습니다.")

        # 의존성 와이어링 (필요시)
        self.container.wire(modules=[__name__])

        self.main_window = None
        self.shutdown_event = asyncio.Event()
        self.token_ready_event = asyncio.Event()
        self.universe_ready_event = asyncio.Event()

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
        msg_title = "당일 최대 손실 도달"
        msg_content = f"현재 손실액 ₩{loss_amount:,.0f}이 설정된 한도를 초과했습니다. 거래를 강제 종료하고 모든 포지션을 청산합니다."
        
        self.live_vm.sig_log_appended.emit(f"🚨 [CRITICAL] {msg_title}: {msg_content}")
        self.market_scheduler.trigger_daily_stop_loss()

        # Send telegram bot message via centralized notifier
        notifier = self.container.telegram_notifier()
        asyncio.create_task(notifier.notify_critical(msg_title, msg_content))

    async def start(self):
        # [안정화] 비동기 루프가 실행 중인 상태에서 코어 객체 및 ViewModel 생성 (InfluxDB 에러 방지)
        self.influx_client = self.container.influx_client()
        self.order_manager = self.container.order_manager()
        # =========================================================
        # 🔄 [MOCK 스위치] 환경 변수에 따라 DataCollector를 갈아끼움
        # =========================================================
        import os
        if os.getenv("USE_MOCK_DATA") == "True":
            from core.mock_data_collector import MockDataCollector
            # 배속(speed_multiplier)을 20으로 주면 하루 치 장을 20분 만에 돌려볼 수 있습니다.
            self.data_collector = MockDataCollector(self.container.config_manager(), data_file="mock_data.csv", speed_multiplier=20.0)
            print("🚨 시스템: [주의] MOCK_MODE가 켜져 있습니다. 가상 데이터를 재생합니다.")
        else:
            self.data_collector = self.container.data_collector()
            print("🌐 시스템: [REAL] 실제 증권사 데이터 수집기를 가동합니다.")
        # =========================================================
        self.strategy_manager = self.container.strategy_manager()
        # [🚨 100% 해결 핵심 패치]
        # 전략 매니저가 엉뚱한(새로 생성된) 수집기를 바라보지 못하도록,
        # 현재 웹소켓을 담당할 '진짜' 수집기를 강제로 주입합니다.
        self.strategy_manager.data_collector = self.data_collector
        self.token_manager = self.container.token_manager()
        self.market_scheduler = self.container.market_scheduler()
        self.risk_manager = self.container.risk_manager()
        self.live_vm = self.container.live_dashboard_view_model()
        self.asset_vm = self.container.asset_data_view_model()

        # Risk Manager injection loop closing
        self.order_manager.risk_manager = self.risk_manager
        
        # [안정화] DataCollector에 TokenManager 참조 주입
        self.data_collector.config._token_manager = self.token_manager

        # Inject references for background managers safely
        config_mgr = self.container.config_manager()
        config_mgr._injected_scheduler = self.market_scheduler
        config_mgr._injected_strategy_manager = self.strategy_manager
        config_mgr._injected_live_vm = self.live_vm
        config_mgr._injected_asset_data_vm = self.asset_vm

        # GUI 초기화: ViewModel 주입 및 MainWindow 생성
        self.main_window = MainWindow(self.live_vm, self)
        
        # Connect Signals
        self.risk_manager.signals.daily_stop_loss_hit.connect(self._on_stop_loss_hit)
        self.token_manager.signals.token_updated.connect(self._on_token_updated)
        self.token_manager.signals.token_error.connect(self._on_token_error)
        self.asset_vm.symbols_loaded.connect(self._on_universe_ready)

        # Step 0: DB 연결 확인
        self.main_window.show()
        print("시스템: 부팅 시퀀스를 시작합니다. (모델 기반 에이전트 모드)")
        
        # 텔레그램 부팅 알림 전송
        notifier = self.container.telegram_notifier()
        await notifier.notify_app_start()

        # Step 1: 에이전트 무기 장착 (모델 로드) - Fail-Safe 전략
        # (모델 로드 중 오류가 발생해도 random 모델로 자가 복구하도록 StrategyManager에 구현됨)
        print("시스템: [Step 1] Config 기반 에이전트 모델 로드 시작...")
        self.strategy_manager.load_model_from_config()

        # Step 2: Token 발급 및 유니버스 준비
        print("시스템: [Step 2] 토큰 발급 및 스케줄러 가동 준비...")
        self.token_task = asyncio.create_task(self.token_manager.start())
        
        try:
            # 토큰 발급 대기 (최대 10초)
            await asyncio.wait_for(self.token_ready_event.wait(), timeout=10.0)
            print("시스템: [Step 2] 토큰 발급 완료.")
        except asyncio.TimeoutError:
            print("시스템: [Step 2] 토큰 발급 타임아웃! (인터넷 연결 확인 필요)")
            # [안정화] 타임아웃 시 잠시 유예를 두어 루프 스트레스 분산
            await asyncio.sleep(1.0)

        # 스케줄러 및 유니버스 세팅
        self.scheduler_task = asyncio.create_task(self.market_scheduler.start())

        # [기능 개선] 장시간(매매 가능 시간)에 부팅할 경우에만 유니버스를 자동으로 갱신합니다.
        # 장시간 외(야간, 주말 등) 부팅 시에는 불필요한 API 호출을 방지하기 위해 수동 수집만 허용합니다.
        current_state = self.market_scheduler.determine_state(self.market_scheduler.get_current_time())
        prepare_states = [MarketState.PREPARE, MarketState.TRADING, MarketState.CUTOFF, MarketState.LIQUIDATING]
        
        if current_state in prepare_states:
            print("시스템: 장시간 부팅 - 로컬 로드를 생략하고 서버에서 실시간 유니버스를 수집합니다.")
            try:
                # 서버에서 최신 주도주 수집 (자동으로 config 저장 및 UI 갱신 시그널 발생)
                await asyncio.wait_for(self.asset_vm._build_universe_task(is_auto=True), timeout=15.0)
            except Exception as e:
                print(f"시스템: [Step 2] 유니버스 자동 갱신 중 오류 발생: {e}")
                # 서버 수집 실패 시 폴백으로 로컬 로드 시도
                self.asset_vm.load_symbols()
        else:
            print("시스템: 장외시간 부팅 - 서버 통신을 생략하고 저장된 로컬 유니버스를 로드합니다.")
            self.asset_vm.load_symbols()
            await asyncio.sleep(0.5)

        try:
            # 유니버스 로드가 완료될 때까지 잠시 대기
            await asyncio.wait_for(self.universe_ready_event.wait(), timeout=10.0)
            universe_list = self.asset_vm.config_manager.get_symbols()
            universe_len = len(universe_list)
            print(f"시스템: [Step 2] 유니버스 로드 완료 (총 {universe_len}개 종목).")

            # [Step 2.5] 확정된 유니버스를 바탕으로 매매 엔진(LiveTradingEngine) 초기화 실행
            print("시스템: [Step 2.5] 확정된 유니버스에 대해 전용 매매 엔진 초기화 시작...")
            self.strategy_manager.init_engines(universe_list)
            
            # [안정화] 텔레그램 준비 완료 알림 전송 (네트워크 에러 시 무시하고 진행)
            try:
                await notifier.notify_app_ready(universe_len)
            except:
                pass
        except asyncio.TimeoutError:
            print("시스템: [Step 2] 유니버스 로드 지연 - 기본 설정 리스트로 지연 초기화를 진행합니다.")
            self.strategy_manager.init_engines(self.asset_vm.config_manager.get_symbols())

        # Step 3: 데이터 수집 및 매매 엔진 가동 (병목 차단)
        self.influx_task = asyncio.create_task(self.influx_client.start())
        print("시스템: [Step 3] DataCollector 가동 및 웹소켓 데이터 스트림 연결...")

        # =====================================================================
        # 🚀 [MASTER BRIDGE] DI 컨테이너 다 무시하고 여기서 직접 혈관을 뚫습니다.
        # =====================================================================
        self.data_collector.on_state_updated_callbacks.clear()
        self.data_collector.on_state_updated_callbacks.append(self.strategy_manager._on_tick_event)
        print("시스템: [SUCCESS] 마스터 브릿지 연결 완료! (DataCollector -> StrategyManager)")
        # =====================================================================

        self.collector_task = asyncio.create_task(self.data_collector.start())

        # 매매 로직 정식 구동
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
    
    # [프리미엄 다크 테마 적용]
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #1a1a1a;
            color: #e0e0e0;
            font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
            font-size: 10pt;
        }
        
        QTabWidget::pane {
            border: 1px solid #333;
            background: #1a1a1a;
        }
        
        QTabBar::tab {
            background: #2b2b2b;
            padding: 10px 20px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        
        QTabBar::tab:selected {
            background: #3d3d3d;
            border-bottom: 2px solid #007acc;
            font-weight: bold;
        }
        
        QGroupBox {
            border: 1px solid #333;
            border-radius: 8px;
            margin-top: 15px;
            font-weight: bold;
            padding-top: 20px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: #007acc;
        }
        
        QPushButton {
            background-color: #333;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 8px 15px;
            min-height: 25px;
        }
        
        QPushButton:hover {
            background-color: #444;
            border-color: #007acc;
        }
        
        QPushButton:pressed {
            background-color: #222;
        }
        
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateEdit, QTimeEdit {
            background-color: #2b2b2b;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 5px;
            selection-background-color: #007acc;
        }
        
        QLineEdit:focus, QSpinBox:focus {
            border: 1px solid #007acc;
        }
        
        QTableWidget {
            background-color: #1a1a1a;
            border: 1px solid #333;
            gridline-color: #2b2b2b;
            selection-background-color: #004c80;
        }
        
        QHeaderView::section {
            background-color: #2b2b2b;
            color: #aaa;
            padding: 5px;
            border: 0px;
            border-bottom: 1px solid #333;
        }
        
        QProgressBar {
            border: 1px solid #333;
            border-radius: 2px;
            text-align: center;
        }
        
        QProgressBar::chunk {
            background-color: #007acc;
        }
    """)

    if not hasattr(app, "exec_"):
        app.exec_ = app.exec

    # [안정화] qasync 루프가 가비지 컬렉션되는 것을 방지하기 위해 app 객체에 강한 참조로 고정
    app.loop = QEventLoop(app)
    asyncio.set_event_loop(app.loop)
    loop = app.loop
    
    # [안정화] Windows Proactor (IOCP)가 안정화될 시간을 아주 짧게 부여
    loop.run_until_complete(asyncio.sleep(0.1))
    
    # 글로벌 예외 처리기 등록 (루프 크래시 방지)
    def handle_exception(loop, context):
        msg = context.get("exception", context["message"])
        logging.error(f"Global Async Error: {msg}")
        if "QMutex" in str(msg) or "deleted" in str(msg).lower():
            # 이미 삭제된 자원 접근 시 무시
            return
            
    loop.set_exception_handler(handle_exception)

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
            logging.error(f"Critical RuntimeError: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        logging.error(f"Critical Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 2. 종료 후 잔여 태스크 정리 및 I/O 캐시 플러시를 위한 최종 유예
        print("System: Performing final cleanup sequence...")
        try:
            # 루프가 닫히기 전 모든 태스크 취소
            tasks = [t for t in asyncio.all_workers(loop) if not t.done()] if hasattr(asyncio, 'all_workers') else [t for t in asyncio.all_tasks(loop) if not t.done()]
            
            if tasks:
                for task in tasks:
                    task.cancel()
                
                # 취소된 태스크들이 정리될 기회를 주기 위해 루프를 잠깐 더 돌림
                try:
                    # gather를 통해 모든 태스크 취소를 기다림 (타임아웃 1초)
                    loop.run_until_complete(asyncio.wait(tasks, timeout=1.0))
                except Exception:
                    pass
            
            # Windows Proactor (IOCP) 핸들이 완전히 닫힐 시간을 주기 위한 짧은 유예
            try:
                loop.run_until_complete(asyncio.sleep(0.2))
            except Exception:
                pass

            if not loop.is_closed():
                loop.stop()
                # [중요] qasync/proactor 환경에서 이미 Mutex가 삭제되었을 수 있으므로 예외 무시
                try:
                    loop.close()
                    print("System: Async loop closed safely.")
                except RuntimeError as e:
                    if "QMutex" in str(e) or "deleted" in str(e).lower():
                        print("System: Async loop terminated (resource already cleaned by OS/Qt).")
                    else:
                        raise e
        except Exception as cleanup_e:
            print(f"System: Cleanup error (ignored): {cleanup_e}")
        
        print("System: Program fully terminated.")
        sys.exit(0)

if __name__ == "__main__":
    main()
