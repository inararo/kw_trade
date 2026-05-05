import sys
import os
import asyncio
import yaml
from dotenv import load_dotenv
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, QEvent
from qasync import QEventLoop

from core.container import Container
from core.scheduler import MarketState
from gui.main_window import MainWindow

import logging

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class WheelEventFilter(QObject):
    """
    마우스 휠로 인한 QSpinBox, QComboBox 등의 값 변경을 방지하는 이벤트 필터
    """
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            from PyQt6.QtWidgets import QAbstractSpinBox, QComboBox
            if isinstance(obj, (QAbstractSpinBox, QComboBox)):
                # 휠 이벤트를 무시하여 값 변경 방지
                return True
        return super().eventFilter(obj, event)

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
        
        # [신규] 장중 수동 유니버스 갱신 대응
        # 시스템이 이미 실행 중(is_running)이라면 전략 매니저에게 즉각적인 엔진 및 구독 교체를 요청합니다.
        if hasattr(self, 'strategy_manager') and self.strategy_manager.is_running:
            print(f"시스템: 장중 유니버스 수동 갱신 감지 (총 {len(symbols)}개 종목) - 실시간 구독 및 엔진 교체 시작...")
            asyncio.create_task(self.strategy_manager.update_universe(symbols))

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

        # [핵심] 실시간 체결(Chejan) 데이터 연동 콜백 등록
        if hasattr(self.data_collector, 'on_execution_callbacks'):
            self.data_collector.on_execution_callbacks.append(self.order_manager.on_receive_chejan_data)

        # Inject references for background managers safely
        config_mgr = self.container.config_manager()
        config_mgr._injected_scheduler = self.market_scheduler
        config_mgr._injected_strategy_manager = self.strategy_manager
        config_mgr._injected_live_vm = self.live_vm
        config_mgr._injected_asset_data_vm = self.asset_vm

        # GUI 초기화: ViewModel 주입 및 MainWindow 생성
        self.main_window = MainWindow(self.live_vm, self)
        
        # [신규] 대시보드 자산/현금 폴링 루프 시작 (1초 주기 UI 갱신)
        asyncio.create_task(self.live_vm.start_polling())
        
        # Connect Signals
        self.risk_manager.signals.daily_stop_loss_hit.connect(self._on_stop_loss_hit)
        self.token_manager.signals.token_updated.connect(self._on_token_updated)
        self.token_manager.signals.token_error.connect(self._on_token_error)
        self.asset_vm.symbols_loaded.connect(self._on_universe_ready)
        self.market_scheduler.signals.state_changed.connect(self._on_market_state_changed)

        # Step 0: DB 연결 확인
        self.main_window.show()
        print("시스템: 부팅 시퀀스를 시작합니다. (모델 기반 에이전트 모드)")

        # [Firebase] Firebase 매니저 초기화 및 부팅 상태 전송
        self.firebase_manager = self.container.firebase_manager()
        # [역방향 동기화 활성화] ConfigManager에 FirebaseManager 주입
        config_mgr.firebase_manager = self.firebase_manager
        
        asyncio.create_task(self.firebase_manager.update_system_status("BOOTING"))
        # 초기 제어 상태도 함께 보고 (기본값: Monitoring=False, AI=True)
        asyncio.create_task(self.firebase_manager.update_control_status(
            is_monitoring_active=True,
            is_ai_trading_active=True
        ))
        asyncio.create_task(self.firebase_manager.update_engine_status("RUNNING"))
        self._heartbeat_task = asyncio.create_task(self.firebase_manager.start_heartbeat())
        
        print("시스템: [Firebase] 부팅 상태(BOOTING) 및 가동 상태(RUNNING)를 Firestore에 전송합니다.")

        # [Firebase] settings/core 기본값 업로드 (모바일 앱 설정 화면 초기화)
        # config.yaml 실제 값을 읽어 업로드하되, 보안·내부 항목은 제외합니다.
        # merge=True 적용: 이미 변경된 값은 보존, 새 키만 추가
        _SETTINGS_EXCLUDED_KEYS = {
            # ── 보안 (인증 / 접속정보) ──────────────────────────────
            "account_number",
            "KIWOOM_APP_KEY", "KIWOOM_APP_SECRET", "KIWOOM_ACCESS_TOKEN",
            "INFLUX_URL", "INFLUX_TOKEN", "INFLUX_ORG",
            "influx_bucket", "INFLUX_BUCKET",
            "TELEGRAM_BOT_TOKEN", "telegram_chat_id",
            "FIREBASE_KEY_PATH",
            # ── 내부 시스템 설정 (모바일 앱에서 수정 불필요) ──────────
            "active_model_path",
            "kiwoom",          # 중첩 딕셔너리 (API URL, trading_mode 포함)
            "ws_url",
            "max_buffer_size", "db_batch_size",
            # ── 복합 타입 (리스트/딕셔너리 — Firestore 별도 관리) ────
            "symbols", "universe", "protected_symbols", "global_max_loss",
            "slippage", "seq_len", "initial_balance", "live_trading_model_type",
            "last_updated_by_engine", # 시스템 관리용 타임스탬프 (yaml 저장 제외)
        }
        # config_mgr에서 스칼라(int/float/str/bool) 값만 추려 업로드
        _default_settings = {
            key: value
            for key, value in config_mgr._config_cache.items()
            if key not in _SETTINGS_EXCLUDED_KEYS
            and isinstance(value, (int, float, str, bool))
        }
        # [제어 플래그 기본값] 첫 설치 시 앱에 기본 ON 상태가 표시되도록 추가
        # merge=True 적용: 앱에서 이미 변경해 둔 값은 보존됨
        _default_settings.setdefault("is_monitoring_active", True)
        _default_settings.setdefault("is_ai_trading_active", True)

        asyncio.create_task(
            self.firebase_manager.initialize_default_settings(_default_settings)
        )

        # 텔레그램 부팅 알림 전송
        notifier = self.container.telegram_notifier()
        await notifier.notify_app_start()

        # [Firebase] 원격 설정/명령 리스너 활성화 (백그라운드 스레드 기반)
        self._setup_firebase_listeners()

        # [Firebase] 부팅 시 제어 플래그 초기 상태 동기화 ─────────────────────
        # 리스너 연결 전 앱이 설정해 둔 is_monitoring_active / is_ai_trading_active 값을
        # 1회 읽어와 엔진 내부 상태를 원격 설정에 맞게 초기화합니다.
        try:
            boot_settings = await self.firebase_manager.get_current_settings()
            if boot_settings:
                # 종목 감시 초기 상태
                if boot_settings.get("is_monitoring_active") is False:
                    logging.warning("[Firebase] 부팅 시 원격 설정: 종목 감시가 OFF 상태입니다. 웹소켓 연결을 건너뜁니다.")
                    # collector_task는 Step 3에서 생성되므로 플래그만 기록
                    self._remote_monitoring_off_at_boot = True

                # AI 매매 초기 상태
                if boot_settings.get("is_ai_trading_active") is False:
                    sm = getattr(config_mgr, "_injected_strategy_manager", None)
                    if sm:
                        sm.set_ai_paused(True)
                        logging.warning("[Firebase] 부팅 시 원격 설정: AI 매매가 일시정지 상태로 시작됩니다.")
        except Exception as e:
            logging.error(f"[Firebase] 부팅 시 제어 플래그 초기화 실패 (무시): {e}")

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

            # [Step 3] 초기 계좌 잔고 동기화 (실전 모드 대응)
            print("시스템: [Step 3] 초기 계좌 잔고 동기화 시도...")
            await self.order_manager.sync_balance(force=True)
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
        # [스마트 소켓 관리] 장시간 상태에 따라서만 웹소켓 가동
        active_ws_states = [MarketState.PREPARE, MarketState.TRADING, MarketState.CUTOFF, MarketState.LIQUIDATING]

        # [원격 제어 반영] 부팅 시 is_monitoring_active=False 였다면 장시간이라도 연결을 건너뜁니다.
        _remote_monitoring_off = getattr(self, "_remote_monitoring_off_at_boot", False)

        if _remote_monitoring_off:
            print("[Firebase] 원격 제어: 종목 감시 OFF 상태로 부팅합니다. 웹소켓 연결을 보류합니다.")
            self.collector_task = None
        elif current_state in active_ws_states:
            print(f"시스템: [Step 3] 장시간({current_state}) 확인 - DataCollector 가동 및 웹소켓 연결 시작...")
            self.collector_task = asyncio.create_task(self.data_collector.start())
        else:
            print(f"시스템: [Step 3] 장외 시간({current_state})입니다. 불필요한 웹소켓 연결을 생략하고 수면 모드로 대기합니다.")
            self.collector_task = None

        # =====================================================================
        # 🚀 [MASTER BRIDGE] DI 컨테이너 다 무시하고 여기서 직접 혈관을 뚫습니다.
        # =====================================================================
        self.data_collector.on_state_updated_callbacks.clear()
        self.data_collector.on_state_updated_callbacks.append(self.strategy_manager._on_tick_event)
        print("시스템: [SUCCESS] 마스터 브릿지 연결 완료! (DataCollector -> StrategyManager)")
        # =====================================================================
        # 매매 로직 정식 구동
        self.strategy_task = asyncio.create_task(self.strategy_manager.start())
        self.view_model_task = asyncio.create_task(self.live_vm.start_polling())

        try:
            # 종료 시그널이 올 때까지 이벤트 루프 유지
            await self.shutdown_event.wait()
        finally:
            # [Firebase] 종료 상태 보고 및 하트비트 정지
            if hasattr(self, 'firebase_manager'):
                # 동기적으로 실행되는 것이 아니므로 create_task 후 잠시 대기하거나 direct 호출 고려
                # 여기서는 루프 종료 직전이므로 마지막 인사를 건넵니다.
                await self.firebase_manager.update_engine_status("OFFLINE")
                if hasattr(self, '_heartbeat_task'):
                    self._heartbeat_task.cancel()
            
            # 루프를 빠져나올 때 수행될 정리
            print("시스템: 메인 루프 종료됨. (Firebase: OFFLINE 보고 완료)")

    def _setup_firebase_listeners(self):
        """
        [Firebase] 실시간 리스너를 설정합니다.
        - settings/core 변경 감지 → ConfigManager 메모리 캐시 즉시 반영
        - commands PENDING 감지 → PANIC_SELL 실행 후 COMPLETED 보고

        [Thread-Safe 설계]
        on_snapshot은 Firebase 백그라운드 스레드에서 실행됩니다.
        - 단순 설정값 변경: call_soon_threadsafe로 메인 루프에 동기화
        - 비동기 코루틴 실행: run_coroutine_threadsafe로 메인 루프에서 실행
        """
        if not hasattr(self, 'firebase_manager') or not self.firebase_manager:
            return

        loop = asyncio.get_running_loop()

        # ── 1. 설정 변경 리스너 ──────────────────────────────────────
        def on_settings_changed(data: dict):
            """백그라운드 스레드에서 호출됨 → call_soon_threadsafe로 메인 루프에서 안전하게 실행"""
            def _apply():
                # ── [DEBUG] _apply() 실행 확인 ──────────────────────────────
                logging.info(f"[DEBUG] _apply() 진입 확인 - 수신 키 목록: {list(data.keys())}")
                for _k, _v in data.items():
                    logging.info(f"[DEBUG] _apply() 수신 데이터: {_k} -> {_v}")

                # ── [제어 플래그 처리] ──────────────────────────────────────────
                # is_monitoring_active / is_ai_trading_active 는 yaml에 저장하지 않는
                # 순수 원격 제어 필드이므로, 설정값 동기화보다 먼저 처리하고 제거합니다.
                _CONTROL_KEYS = {
                    "is_monitoring_active", "is_ai_trading_active",
                    "last_updated_by_engine", "last_heartbeat", "engine_status",
                    "current_state", "updated_at"
                }

                # 종목 감시 원격 제어
                if "is_monitoring_active" in data:
                    monitoring_active = data["is_monitoring_active"]
                    logging.info(f"[DEBUG] is_monitoring_active 감지됨: {monitoring_active}")
                    if monitoring_active:
                        asyncio.create_task(self.data_collector.start())
                        logging.info("[Firebase] 📡 원격 명령: 실시간 종목 감시를 재개합니다. (재연결 시도 중...)")
                    else:
                        asyncio.create_task(self.data_collector.stop())
                        logging.warning("[Firebase] 📡 원격 명령: 실시간 종목 감시가 중단되었습니다. (웹소켓 해제)")
                    # [UI 동기화] 버튼 상태 갱신 — stopped=True가 감시 중단(active=False)
                    self.live_vm.sig_monitoring_stopped.emit(not monitoring_active)

                # AI 매매 원격 제어
                if "is_ai_trading_active" in data:
                    ai_active = data["is_ai_trading_active"]
                    logging.info(f"[DEBUG] is_ai_trading_active 감지됨: {ai_active}")
                    sm = getattr(self.container.config_manager(), "_injected_strategy_manager", None)
                    if sm:
                        sm.set_ai_paused(not ai_active)
                        status = "재개" if ai_active else "일시정지"
                        logging.info(f"[Firebase] 🤖 원격 명령: AI 매매 의사결정이 {status}되었습니다.")
                    # [UI 동기화] 버튼 상태 갱신 — paused=True가 AI 정지(active=False)
                    self.live_vm.sig_trading_paused.emit(not ai_active)

                # ── [설정값 동기화] ──────────────────────────────────────────────
                # 제어 필드 및 시스템 관리 필드를 제거한 뒤 일반 설정값만 처리합니다.
                filtered_data = {
                    k: v for k, v in data.items()
                    if k not in _CONTROL_KEYS
                }

                if not filtered_data:
                    return

                # 1. 메모리 반영 및 파일 저장 (실제 변경이 있을 때만 True 반환)
                applied = self.container.config_manager().hot_reload_settings(filtered_data)

                # 2. Firebase에 최종 반영 상태 보고 (실제 변경 시에만 피드백 전송)
                if applied:
                    asyncio.create_task(self.firebase_manager.report_settings_applied())

                    # 3. GUI 설정 탭 화면 실시간 갱신
                    settings_vm = self.container.settings_view_model()
                    settings_vm.on_remote_settings_changed(filtered_data)
            loop.call_soon_threadsafe(_apply)

        self.firebase_manager.listen_to_settings(on_settings_changed)

        # ── 2. 긴급 명령 리스너 ─────────────────────────────────────
        def on_command_received(doc_id: str, data: dict):
            """백그라운드 스레드에서 호출됨 → run_coroutine_threadsafe로 코루틴 실행"""
            action = data.get("action", "")
            if action != "PANIC_SELL":
                logging.warning(f"[Firebase] 알 수 없는 명령 무시: {action}")
                return

            logging.warning(f"[Firebase] PANIC_SELL 명령 수신 (ID: {doc_id})")

            async def _execute():
                try:
                    await self.order_manager.emergency_liquidate()
                    await self.firebase_manager.update_command_status(doc_id, "COMPLETED")
                    logging.critical(f"[Firebase] 긴급 청산 완료 보고 (ID: {doc_id} → COMPLETED)")
                except Exception as e:
                    logging.error(f"[Firebase] 긴급 청산 중 오류: {e}")

            asyncio.run_coroutine_threadsafe(_execute(), loop)

        self.firebase_manager.listen_to_commands(on_command_received)
        logging.info("[Firebase] 실시간 리스너 설정 완료 ✅")

    def _on_market_state_changed(self, old_state: str, new_state: str):
        """스케줄러 상태 변경에 따른 웹소켓 자동 토글 (Event-Driven)"""
        active_ws_states = [MarketState.PREPARE, MarketState.TRADING, MarketState.CUTOFF, MarketState.LIQUIDATING]
        
        # 장 개시 (Inactive -> Active)
        if old_state not in active_ws_states and new_state in active_ws_states:
            print(f"시스템: [알림] 장 개시 타이머 작동({new_state})! 웹소켓 연결을 자동으로 재개합니다.")
            if not self.data_collector.is_running:
                self.collector_task = asyncio.create_task(self.data_collector.start())
                # [Firebase] 제어 상태 동기화
                asyncio.create_task(self.firebase_manager.update_control_status(
                    is_monitoring_active=True,
                    is_ai_trading_active=not self.strategy_manager.is_ai_paused
                ))
        
        # 장 마감 (Active -> Inactive)
        elif old_state in active_ws_states and new_state not in active_ws_states:
            print(f"시스템: [알림] 장 마감 타이머 작동({new_state})! 웹소켓 연결을 안전하게 해제합니다.")
            if self.data_collector.is_running:
                asyncio.create_task(self.data_collector.stop())
                self.collector_task = None
                # [Firebase] 제어 상태 동기화
                asyncio.create_task(self.firebase_manager.update_control_status(
                    is_monitoring_active=False,
                    is_ai_trading_active=not self.strategy_manager.is_ai_paused
                ))

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

        # [Firebase] 종료 상태 전송 (await로 동기 처리하여 루프 종료 전 확실히 전송)
        if hasattr(self, 'firebase_manager') and self.firebase_manager:
            try:
                await asyncio.wait_for(
                    self.firebase_manager.update_system_status("STOPPED"),
                    timeout=3.0
                )
                print("시스템: [Firebase] 종료 상태(STOPPED)를 Firestore에 전송했습니다.")
            except Exception:
                pass

        self.shutdown_event.set()

def main():
    app = QApplication(sys.argv)
    
    # [추가] 마우스 휠 값 변경 방지 필터 설치
    wheel_filter = WheelEventFilter()
    app.installEventFilter(wheel_filter)
    
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
            padding-right: 30px; /* 에디트 박스와 버튼 간 확실한 유격 확보 */
            selection-background-color: #007acc;
        }
        
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid #007acc;
        }

        /* SpinBox 버튼 스타일 개선: 겹침 방지 및 클릭 영역 최적화 */
        QSpinBox::up-button, QDoubleSpinBox::up-button, QTimeEdit::up-button {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 25px;
            background-color: #454545;
            border-left: 1px solid #333;
            border-bottom: 0.5px solid #333;
            border-top-right-radius: 3px;
            margin-right: 1px;
            margin-top: 1px;
        }
        
        QSpinBox::down-button, QDoubleSpinBox::down-button, QTimeEdit::down-button {
            subcontrol-origin: padding;
            subcontrol-position: bottom right;
            width: 25px;
            background-color: #454545;
            border-left: 1px solid #333;
            border-top: 0.5px solid #333;
            border-bottom-right-radius: 3px;
            margin-right: 1px;
            margin-bottom: 1px;
        }

        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QTimeEdit::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover, QTimeEdit::down-button:hover {
            background-color: #606060;
        }

        QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed, QTimeEdit::up-button:pressed,
        QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed, QTimeEdit::down-button:pressed {
            background-color: #007acc;
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
