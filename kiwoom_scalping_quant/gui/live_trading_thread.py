"""
gui/live_trading_thread.py

비동기 매매 백엔드(KiwoomBrokerWrapper + StrategyManager)를 PyQt6 UI와
완전히 격리하기 위한 QThread 래퍼입니다.

Firebase 연동 기능 포함:
  - 부팅 시 엔진 상태(RUNNING) 보고 및 기본 설정 업로드
  - 1분 주기 하트비트(Heartbeat) 전송
  - settings/core 실시간 리스너 → ConfigManager 핫 리로드
  - system_status/engine 리스너 → 종목 감시 중지/재개, AI 매매 일시정지/재개
  - commands 리스너 → PANIC_SELL(긴급 전량 청산) 실행

아키텍처:
  [QThread: LiveTradingThread]
       ↓ 새 asyncio 루프 생성 (GUI 루프와 독립)
       ↓ KiwoomBrokerWrapper → WebSocket → StrategyManager
       ↓ FirebaseManager → Firestore 실시간 리스너 (백그라운드 스레드)
       ↓ pyqtSignal (Thread-Safe)
  [Main Thread: PyQt6 GUI]
       ↓ .connect(slot)
       ↓ LiveDashboardTab 로그창/보유현황 갱신
"""
import asyncio
import logging
from datetime import datetime

from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger("LiveTradingThread")


class LiveTradingThread(QThread):
    """
    실전 매매 비동기 백엔드를 독립된 OS 스레드에서 구동하는 QThread.

    이 스레드가 직접 UI 위젯에 접근하는 것은 절대 금지입니다.
    모든 UI 업데이트는 아래 pyqtSignal을 통해서만 수행해야 합니다.
    """

    # ──────────────────────────────────────────────
    # Thread-Safe 시그널 정의
    # ──────────────────────────────────────────────
    signal_condition_inserted = pyqtSignal(str, str)   # 조건검색 편입 (종목코드, 조건식명)
    signal_condition_deleted  = pyqtSignal(str)         # 조건검색 이탈 (종목코드)
    signal_snapshot_received  = pyqtSignal(list)        # 조건검색 스냅샷 (전체 종목 리스트)
    signal_order_executed     = pyqtSignal(str, str, float, int)  # 체결 (종목, 방향, 가격, 수량)
    signal_log_message        = pyqtSignal(str)         # 로그 메시지
    signal_engine_status      = pyqtSignal(bool)        # 엔진 상태 (True=실행, False=중단)
    # Firebase 원격 제어 시그널 (모바일 앱 → GUI 버튼 상태 동기화)
    signal_monitoring_toggled = pyqtSignal(bool)        # True=감시중지, False=감시중
    signal_ai_trading_toggled = pyqtSignal(bool)        # True=일시정지, False=정상

    def __init__(self, config_manager, order_manager, risk_manager,
                 strategy_manager, condition_manager, broker_api,
                 condition_service=None, firebase_manager=None, parent=None):
        super().__init__(parent)
        self.config_manager    = config_manager
        self.order_manager     = order_manager
        self.risk_manager      = risk_manager
        self.strategy_manager  = strategy_manager
        self.condition_manager = condition_manager
        self.broker_api        = broker_api
        self.condition_service = condition_service # [Shared Core] 서비스
        self.firebase_manager  = firebase_manager  # 선택적 (None이면 Firebase 비활성)

        self._loop: asyncio.AbstractEventLoop = None
        self._stop_event: asyncio.Event = None

    # ──────────────────────────────────────────────
    # 공개 API: GUI 스레드에서 호출
    # ──────────────────────────────────────────────
    def request_stop(self):
        """GUI 스레드에서 안전하게 종료 요청"""
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)

    def inject_firebase_manager(self, firebase_manager):
        """
        QThread 시작 전에 FirebaseManager를 늦게 주입할 때 사용합니다.
        (firebase_manager가 LiveTradingThread 생성 시점보다 늦게 초기화될 때 활용)
        """
        self.firebase_manager = firebase_manager

    # ──────────────────────────────────────────────
    # QThread 진입점
    # ──────────────────────────────────────────────
    def run(self):
        """
        새 asyncio 이벤트 루프를 이 OS 스레드 안에서 생성·실행합니다.
        GUI 루프(qasync)와 완전히 분리되므로 블로킹 없이 실행됩니다.
        """
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:
            logger.error(f"[LiveTradingThread] 치명적 오류로 종료: {e}", exc_info=True)
            self.signal_log_message.emit(f"❌ 매매 엔진 치명적 오류: {e}")
        finally:
            self._loop.close()
            self.signal_engine_status.emit(False)
            logger.info("[LiveTradingThread] 이벤트 루프 종료.")

    # ──────────────────────────────────────────────
    # 내부 비동기 진입점
    # ──────────────────────────────────────────────
    async def _main(self):
        self._stop_event = asyncio.Event()
        self.signal_engine_status.emit(True)
        self.signal_log_message.emit("🚀 실전 매매 엔진 부팅 시작...")

        # 1. REST API 로그인 및 토큰 취득
        is_logged_in = await self.broker_api.login()
        if not is_logged_in:
            self.signal_log_message.emit("❌ 로그인 실패 - 매매 엔진을 종료합니다.")
            return

        self.signal_log_message.emit("✅ 키움 API 로그인 완료!")

        # OrderManager에 토큰 공급기 주입
        class _SimpleAuth:
            def get_token(inner_self):
                return self.broker_api.access_token

        self.order_manager.auth_manager = _SimpleAuth()

        # 2. 잔고 동기화
        await self.order_manager.sync_balance()
        self.signal_log_message.emit("✅ 계좌 잔고 동기화 완료")

        # 3. Firebase 부팅 시 초기화 (선택적)
        await self._firebase_on_boot()

        # 4. Firebase 실시간 리스너 설정 (Firebase SDK 백그라운드 스레드에서 실행)
        self._setup_firebase_listeners()

        # 5. 부팅 시 Firebase 제어 플래그 초기 상태 반영
        await self._firebase_sync_initial_flags()

        # 6. [Shared Core] 조건검색 서비스 연동 (스냅샷 및 실시간 이벤트)
        if self.condition_service:
            def _handle_snapshot(symbols):
                self.signal_log_message.emit(f"📋 초기 조건 만족 종목 {len(symbols)}개 수신 (엔진 라우팅 시작)")
                self._loop.create_task(self.strategy_manager.handle_condition_snapshot(symbols))
                # UI 업데이트를 위해 시그널 발생
                self.signal_snapshot_received.emit(symbols)

            def _handle_insert(sym, data):
                self.signal_log_message.emit(f"🌟 [편입] {sym}")
                self.signal_condition_inserted.emit(sym, "AI스캘핑주도주")
                self._loop.create_task(self.strategy_manager.handle_condition_insert(sym))

            def _handle_delete(sym, data):
                self.signal_log_message.emit(f"🗑️ [이탈] {sym}")
                self.signal_condition_deleted.emit(sym)
                self._loop.create_task(self.strategy_manager.handle_condition_delete(sym))

            self.condition_service.register_callbacks(
                on_snapshot=_handle_snapshot,
                on_insert=_handle_insert,
                on_delete=_handle_delete
            )
            self.signal_log_message.emit("✅ [SharedCore] 조건검색 콜백 바인딩 완료")

        # 7. 조건식 인덱스(target_idx) 조회
        target_condition_name = "AI스캘핑주도주"
        condition_dict = await self.broker_api.get_condition_list()
        target_idx = next((idx for idx, name in condition_dict.items() if name == target_condition_name), "0")
        self.signal_log_message.emit(f"📡 감시 조건식: {target_condition_name} (Index: {target_idx})")

        # 8. 병렬 태스크 가동
        tasks = [
            asyncio.create_task(self.strategy_manager.start(),
                                name="strategy_loop"),
            asyncio.create_task(self._balance_sync_loop(),
                                name="balance_sync"),
            asyncio.create_task(self._stop_event.wait(),
                                name="stop_sentinel"),
        ]

        # [안정화] 웹소켓 리스너가 이미 (main.py 등에 의해) 실행 중인 경우 중복 가동 방지
        if not getattr(self.broker_api, 'ws_running', False):
            self.signal_log_message.emit("📡 웹소켓 리스너를 새로 가동합니다.")
            tasks.append(asyncio.create_task(
                self.broker_api.ws_listener_loop(target_idx), name="ws_listener"
            ))
        else:
            self.signal_log_message.emit("🌐 이미 가동 중인 웹소켓 리스너를 공유합니다.")

        # Firebase 하트비트 (Firebase 초기화된 경우에만)
        if self.firebase_manager and getattr(self.firebase_manager, '_initialized', False):
            tasks.append(asyncio.create_task(
                self.firebase_manager.start_heartbeat(), name="fb_heartbeat"
            ))

        self.signal_log_message.emit("⚙️ 메인 트레이딩 파이프라인 가동 중...")

        # stop_sentinel이 완료되면 나머지 태스크를 모두 취소
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

        # 8. 종료 시 Firebase에 오프라인 보고
        await self._firebase_on_shutdown()
        self.signal_log_message.emit("🛑 매매 엔진이 안전하게 종료되었습니다.")

    # ──────────────────────────────────────────────
    # Firebase 연동 메서드들
    # ──────────────────────────────────────────────
    async def _firebase_on_boot(self):
        """부팅 시 Firebase 상태 보고 및 기본 설정 업로드"""
        fb = self.firebase_manager
        if not fb or not getattr(fb, '_initialized', False):
            return
        try:
            await fb.update_system_status("RUNNING")
            await fb.update_engine_status("RUNNING")
            await fb.update_control_status(
                is_monitoring_active=True,
                is_ai_trading_active=True
            )
            # config 캐시에서 스칼라 설정만 추려 Firebase 기본값 업로드
            _EXCLUDED = {
                "account_number", "KIWOOM_APP_KEY", "KIWOOM_APP_SECRET",
                "KIWOOM_ACCESS_TOKEN", "INFLUX_URL", "INFLUX_TOKEN", "INFLUX_ORG",
                "influx_bucket", "INFLUX_BUCKET", "TELEGRAM_BOT_TOKEN",
                "telegram_chat_id", "FIREBASE_KEY_PATH", "active_model_path",
                "kiwoom", "ws_url", "max_buffer_size", "db_batch_size",
                "symbols", "universe", "protected_symbols", "global_max_loss",
                "slippage", "seq_len", "initial_balance", "live_trading_model_type",
                "is_monitoring_active", "is_ai_trading_active", "last_updated_by_engine",
                "BYPASS_MARKET_HOURS"
            }
            config_cache = getattr(self.config_manager, '_config_cache', {})
            default_settings = {
                k: v for k, v in config_cache.items()
                if k not in _EXCLUDED and isinstance(v, (int, float, str, bool))
            }
            await fb.initialize_default_settings(default_settings)
            if not self.config_manager.get("OFFLINE_MODE", False):
                self.signal_log_message.emit("✅ [Firebase] 부팅 상태(RUNNING) 및 기본 설정 업로드 완료")
        except Exception as e:
            logger.error(f"[Firebase] 부팅 초기화 실패 (무시): {e}")

    async def _firebase_on_shutdown(self):
        """종료 시 Firebase에 오프라인 상태 보고"""
        fb = self.firebase_manager
        if not fb or not getattr(fb, '_initialized', False):
            return
        try:
            await fb.update_engine_status("OFFLINE")
            logger.info("[Firebase] 엔진 종료 상태(OFFLINE) 보고 완료")
        except Exception as e:
            logger.error(f"[Firebase] 종료 보고 실패 (무시): {e}")

    async def _firebase_sync_initial_flags(self):
        """부팅 시 Firestore의 제어 플래그를 1회 읽어 엔진 내부 상태 초기화"""
        fb = self.firebase_manager
        if not fb or not getattr(fb, '_initialized', False):
            return
        try:
            engine_status = await fb.get_engine_status()
            if not engine_status:
                return
            if engine_status.get("is_ai_trading_active") is False:
                self.strategy_manager.set_ai_paused(True)
                self.signal_ai_trading_toggled.emit(True)
                self.signal_log_message.emit("⚠️ [Firebase] 부팅 시 원격 설정: AI 매매 일시정지 상태로 시작")
            if engine_status.get("is_monitoring_active") is False:
                self.signal_monitoring_toggled.emit(True)
                self.signal_log_message.emit("⚠️ [Firebase] 부팅 시 원격 설정: 종목 감시 OFF 상태로 시작")
        except Exception as e:
            logger.error(f"[Firebase] 초기 제어 플래그 동기화 실패 (무시): {e}")

    def _setup_firebase_listeners(self):
        """
        Firebase 실시간 리스너 3종 설정.

        Firebase SDK의 on_snapshot은 백그라운드 스레드에서 실행되므로,
        asyncio 코루틴은 run_coroutine_threadsafe로, 단순 처리는
        call_soon_threadsafe로 이 스레드의 이벤트 루프에 안전하게 전달합니다.
        """
        fb = self.firebase_manager
        if not fb or not getattr(fb, '_initialized', False):
            return

        loop = self._loop  # 이 QThread의 전용 asyncio 루프

        # ── 리스너 1: settings/core 변경 → ConfigManager 핫 리로드 ──
        def on_settings_changed(data: dict):
            def _apply():
                _CONTROL_KEYS = {
                    "last_updated_by_engine", "last_heartbeat",
                    "engine_status", "current_state", "updated_at"
                }
                filtered = {k: v for k, v in data.items() if k not in _CONTROL_KEYS}
                if not filtered:
                    return
                applied = self.config_manager.hot_reload_settings(filtered)
                if applied:
                    asyncio.run_coroutine_threadsafe(fb.report_settings_applied(), loop)
                    self.signal_log_message.emit(
                        f"🔧 [Firebase] 원격 설정 반영 완료: {list(filtered.keys())}"
                    )
            loop.call_soon_threadsafe(_apply)

        fb.listen_to_settings(on_settings_changed)

        # ── 리스너 2: system_status/engine → 종목 감시/AI 매매 원격 제어 ──
        def on_engine_status_changed(data: dict):
            def _apply():
                if "is_monitoring_active" in data:
                    monitoring_active = data["is_monitoring_active"]
                    status_str = "재개" if monitoring_active else "중단"
                    logger.info(f"[Firebase] 원격 명령: 종목 감시 {status_str}")
                    self.signal_monitoring_toggled.emit(not monitoring_active)
                    self.signal_log_message.emit(f"📡 [Firebase] 원격 명령: 종목 감시 {status_str}")

                if "is_ai_trading_active" in data:
                    ai_active = data["is_ai_trading_active"]
                    self.strategy_manager.set_ai_paused(not ai_active)
                    status_str = "재개" if ai_active else "일시정지"
                    logger.info(f"[Firebase] 원격 명령: AI 매매 {status_str}")
                    self.signal_ai_trading_toggled.emit(not ai_active)
                    self.signal_log_message.emit(f"🤖 [Firebase] 원격 명령: AI 매매 {status_str}")
            loop.call_soon_threadsafe(_apply)

        fb.listen_to_engine_status(on_engine_status_changed)

        # ── 리스너 3: commands → PANIC_SELL 긴급 전량 청산 ──
        def on_command_received(doc_id: str, data: dict):
            action = data.get("action", "")
            if action != "PANIC_SELL":
                logger.warning(f"[Firebase] 알 수 없는 명령 무시: {action}")
                return
            logger.warning(f"[Firebase] 🚨 PANIC_SELL 명령 수신 (ID: {doc_id})")
            self.signal_log_message.emit("🚨 [Firebase] 긴급 청산 명령 수신! 전량 매도를 실행합니다...")

            async def _execute():
                try:
                    await self.order_manager.emergency_liquidate()
                    await fb.update_command_status(doc_id, "COMPLETED")
                    self.signal_log_message.emit("✅ [Firebase] 긴급 전량 청산 완료!")
                    logger.critical(f"[Firebase] 긴급 청산 완료 보고 (ID: {doc_id} → COMPLETED)")
                except Exception as e:
                    await fb.update_command_status(doc_id, "FAILED")
                    logger.error(f"[Firebase] 긴급 청산 중 오류: {e}")
                    self.signal_log_message.emit(f"❌ [Firebase] 긴급 청산 실패: {e}")

            asyncio.run_coroutine_threadsafe(_execute(), loop)

        fb.listen_to_commands(on_command_received)
        if not self.config_manager.get("OFFLINE_MODE", False):
            self.signal_log_message.emit("✅ [Firebase] 실시간 리스너 3종 활성화 완료 (설정/제어/명령)")

    # ──────────────────────────────────────────────
    # 주기적 잔고 동기화 루프
    # ──────────────────────────────────────────────
    async def _balance_sync_loop(self):
        """60초마다 잔고를 REST로 동기화"""
        while True:
            await asyncio.sleep(60)
            try:
                await self.order_manager.sync_balance()
                balance = self.order_manager.get_balance()
                ts = datetime.now().strftime('%H:%M:%S')
                self.signal_log_message.emit(f"💰 잔고 동기화 완료: {balance:,.0f}원 ({ts})")
            except Exception as e:
                self.signal_log_message.emit(f"⚠️ 잔고 동기화 오류: {e}")


# ──────────────────────────────────────────────────────────────────────
# Signal-to-Log 브릿지 핸들러 (Python logging → pyqtSignal)
# ──────────────────────────────────────────────────────────────────────
class QtLogHandler(logging.Handler):
    """
    Python 표준 logging 메시지를 pyqtSignal로 UI 로그창에 전달합니다.
    LiveTradingThread 인스턴스를 생성한 뒤 이 핸들러를 루트 로거에 추가하세요.
    """
    def __init__(self, signal: pyqtSignal):
        super().__init__()
        self._signal = signal
        self.setFormatter(logging.Formatter("[%(name)s] %(message)s"))

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self._signal.emit(msg)
        except Exception:
            pass
