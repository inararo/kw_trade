import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Optional
from PyQt6.QtCore import QObject, pyqtSignal

class SchedulerSignals(QObject):
    state_changed = pyqtSignal(str, str) # old_state, new_state
    condition_switched = pyqtSignal(str) # new_condition_name

class MarketState:
    IDLE = "IDLE"           # 휴장 또는 야간
    PREPARE = "PREPARE"     # 08:50 ~ 09:00 - 유니버스 갱신 및 WS 연결 준비
    TRADING = "TRADING"     # 09:00 ~ 15:20 - 정상 매매 진행
    CUTOFF = "CUTOFF"       # 지정 시간 이후 신규 매수 금지 (모니터링 및 매도만 가능)
    LIQUIDATING = "LIQUIDATING" # 15:20 ~ 15:30 - 신규 진입 금지 및 청산 (Panic Sell)
    STOPPED = "STOPPED"     # 15:30 이후 - 데이터 Flush 및 연결 종료
    STOPPED_FOR_DAY = "STOPPED_FOR_DAY" # 당일 거래 강제 중지 (Stop-Loss 등)
    POST_MARKET = "POST_MARKET" # 16:00 - 장 종료 후 데이터 수집 스캔

class MarketScheduler:
    """
    한국 거래소(KRX) 시간에 맞춰 시스템 상태를 제어하는 스케줄러.
    주말/휴장일 처리 및 강제 시간 조절(디버깅) 기능을 지원합니다.
    """
    def __init__(self, data_collector=None, order_manager=None, universe_manager=None, telegram_bot=None, firebase_manager=None, config=None):
        self.logger = logging.getLogger("MarketScheduler")
        self.signals = SchedulerSignals()
        self.config_manager = config

        self.data_collector = data_collector
        self.order_manager = order_manager
        self.universe_manager = universe_manager
        self.telegram_bot = telegram_bot
        self.firebase_manager = firebase_manager  # [Firebase] 시스템 상태 업데이트용

        self.current_state = MarketState.IDLE
        self._is_running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._universe_lock = asyncio.Lock()

        # Debug / Time travel
        self._mock_time: Optional[datetime] = None

    def set_mock_time(self, mock_dt: datetime):
        """For testing/debugging to force a specific time."""
        self.logger.warning(f"Mock time set to: {mock_dt}")
        self._mock_time = mock_dt

    def get_current_time(self) -> datetime:
        if self._mock_time:
            # Advance mock time by a small tick or just return it if static
            # In a real mock simulation, we might advance it. For simplicity, we just return it.
            return self._mock_time
        return datetime.now()

    def is_holiday_or_weekend(self, dt: datetime) -> bool:
        """
        주말 여부 확인.
        실제 상용화 시에는 pykrx의 is_holiday() 등을 이용할 수 있습니다.
        """
        if dt.weekday() >= 5: # 5=Saturday, 6=Sunday
            return True
        return False

    def trigger_daily_stop_loss(self):
        """Force the system into STOPPED_FOR_DAY state."""
        self.logger.critical("🚨 Triggering Daily Stop-Loss. Force-stopping market scheduler.")
        asyncio.create_task(self._transition_state(self.current_state, MarketState.STOPPED_FOR_DAY))

    def determine_state(self, dt: datetime) -> str:
        if self.current_state == MarketState.STOPPED_FOR_DAY:
            return MarketState.STOPPED_FOR_DAY

        # [수정] 장외 테스트 모드(BYPASS_MARKET_HOURS)일 경우 주말/공휴일 체크를 건너뜁니다.
        bypass_on = False
        if self.config_manager and self.config_manager.get("BYPASS_MARKET_HOURS", False):
            bypass_on = True

        if not bypass_on and self.is_holiday_or_weekend(dt):
            return MarketState.IDLE

        current_time = dt.time()

        t_0850 = time(8, 50)
        t_0900 = time(9, 0)
        t_1520 = time(15, 20)
        t_1530 = time(15, 30)
        t_1600 = time(16, 0)

        # Check Custom Cutoff Time
        t_cutoff = None

        # Determine config manager reference
        config_mgr = None
        if self.universe_manager and hasattr(self.universe_manager, 'config_manager'):
            config_mgr = self.universe_manager.config_manager
        elif hasattr(self, 'config_manager'):
            config_mgr = self.config_manager
        elif self.order_manager and hasattr(self.order_manager, 'config'):
            config_mgr = self.order_manager.config

        if config_mgr and config_mgr.get("enable_cutoff", False):
            cutoff_str = config_mgr.get("cutoff_time", "13:00")
            try:
                h, m = map(int, cutoff_str.split(':'))
                t_cutoff = time(h, m)
            except Exception:
                pass

        if current_time < t_0850:
            return MarketState.IDLE
        elif t_0850 <= current_time < t_0900:
            return MarketState.PREPARE
        elif t_0900 <= current_time < t_1520:
            if t_cutoff and current_time >= t_cutoff:
                return MarketState.CUTOFF
            return MarketState.TRADING
        elif t_1520 <= current_time < t_1530:
            return MarketState.LIQUIDATING
        elif t_1530 <= current_time < t_1600:
            return MarketState.STOPPED
        else:
            return MarketState.POST_MARKET

    async def start(self):
        if self._is_running:
            return
        self._is_running = True

        # Initial evaluation
        initial_state = self.determine_state(self.get_current_time())
        await self._transition_state(MarketState.IDLE, initial_state)

        self._loop_task = asyncio.create_task(self._schedule_loop())

    async def stop(self):
        self._is_running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

    async def _transition_state(self, old_state: str, new_state: str):
        if old_state == new_state:
            return

        # [추가] config_manager 참조 획득
        config_mgr = None
        if self.universe_manager and hasattr(self.universe_manager, 'config_manager'):
            config_mgr = self.universe_manager.config_manager
        elif self.order_manager and hasattr(self.order_manager, 'config'):
            config_mgr = self.order_manager.config

        self.logger.info(f"State Transition: {old_state} -> {new_state}")
        self.current_state = new_state
        self.signals.state_changed.emit(old_state, new_state)

        # [Firebase] 시스템 상태를 Firestore에 실시간 업데이트
        if self.firebase_manager:
            asyncio.create_task(self.firebase_manager.update_system_status(new_state))

        # Execute actions based on the new state
        if new_state == MarketState.PREPARE:
            # [수정] 자동 갱신 설정 확인
            if config_mgr and not config_mgr.get("enable_universe_update", True):
                self.logger.info("Market Prepare: 유니버스 자동 갱신 설정이 꺼져 있어 갱신을 건너뜁니다.")
            else:
                self.logger.info("Market Prepare: Requesting dynamic universe generation...")
                # Use the injected AssetDataViewModel if available to trigger the UI-bound universe logic
                vm = getattr(self.config_manager, "_injected_asset_data_vm", None) if hasattr(self, 'config_manager') else None
                if vm and hasattr(vm, 'build_universe'):
                    vm.build_universe()

        elif new_state == MarketState.TRADING:
            self.logger.info("Market Open: Activating trading agents.")

        elif new_state == MarketState.CUTOFF:
            self.logger.warning("Market Cutoff Time Reached. New AI buys are blocked. Only monitoring and liquidating active.")
            # Scanner loop inherently halts because it checks `self.current_state == MarketState.TRADING`

        elif new_state == MarketState.LIQUIDATING:
            self.logger.warning("Market Closing Soon: Liquidating positions (Panic Sell).")
            if self.telegram_bot:
                asyncio.create_task(self.telegram_bot.notify_critical(
                    "장 마감 임박 (Panic Sell)", 
                    "정규장 종료가 임박하여 AI 관리 종목의 전량 청산을 시작합니다."
                ))

            if self.order_manager:
                # 보호 종목 리스트 가져오기 (Scheduler에는 config_manager 의존성이 직접 주입되지 않으므로, None 체크 필요)
                # 만약 config_manager가 없다면 직접 order_manager 등에서 가져와야 하지만,
                # 현재는 order_manager.bot_holdings를 우선 활용하고 환경 변수 fallback을 씁니다.
                import os

                # [보강] 보호 종목 리스트 가져오기
                protected_symbols = []
                if config_mgr:
                    protected_symbols = config_mgr.get("protected_symbols", [])

                bot_holdings = getattr(self.order_manager, 'bot_holdings', {})

                # Trigger panic sell for bot holdings
                for symbol, qty in self.order_manager.holdings.items():
                    if qty <= 0:
                        continue
                    
                    if symbol in protected_symbols:
                        self.logger.info(f"🛡️ 보호 종목 청산 제외 (LIQUIDATING): {symbol}")
                        continue

                    bot_qty = bot_holdings.get(symbol, 0)
                    if bot_qty <= 0:
                        self.logger.info(f"수동 매수 종목 청산 제외 (LIQUIDATING): {symbol}")
                        continue

                    target_qty = min(qty, bot_qty)
                    self.logger.info(f"Liquidating {target_qty} shares of {symbol} (Bot managed)")
                    # Execute market sell (03)
                    await self.order_manager.send_order("SELL", symbol, price=0, qty=target_qty)

        elif new_state == MarketState.STOPPED:
            self.logger.info("Market Closed: Flushing data and disconnecting.")
            if self.data_collector:
                await self.data_collector.stop()
            
            if self.telegram_bot:
                await self.telegram_bot.notify_app_stop()

        elif new_state == MarketState.STOPPED_FOR_DAY:
            self.logger.critical("🚨 Market Stopped For Day: Emergency Liquidation Triggered.")
            if self.telegram_bot:
                asyncio.create_task(self.telegram_bot.notify_critical(
                    "당일 거래 강제 중단",
                    "심각한 리스크가 감지되어 당일 모든 거래를 중단하고 포지션을 긴급 청산합니다."
                ))
            
            if self.order_manager:
                # Cancel all and market sell immediately
                await self.order_manager.cancel_all_orders()

                # [보강] 보호 종목 리스트
                protected_symbols = []
                if self.universe_manager and hasattr(self.universe_manager, 'config_manager'):
                    protected_symbols = self.universe_manager.config_manager.get("protected_symbols", [])

                bot_holdings = getattr(self.order_manager, 'bot_holdings', {})
                for symbol, qty in self.order_manager.holdings.items():
                    if qty <= 0:
                        continue

                    if symbol in protected_symbols:
                        self.logger.info(f"🛡️ 보호 종목 강제 청산 제외 (STOPPED_FOR_DAY): {symbol}")
                        continue

                    bot_qty = bot_holdings.get(symbol, 0)
                    if bot_qty <= 0:
                        self.logger.info(f"수동 매수 종목 강제 청산 제외 (STOPPED_FOR_DAY): {symbol}")
                        continue

                    target_qty = min(qty, bot_qty)
                    self.logger.critical(f"Emergency Liquidating {target_qty} shares of {symbol}")
                    await self.order_manager.send_order("SELL", symbol, price=0, qty=target_qty)

        elif new_state == MarketState.POST_MARKET:
            self.logger.info("Post-Market: 장 종료 및 정산 시점입니다. (자동 수집 생략)")

    async def _schedule_loop(self):
        # [신규] 조건식 스위칭 관리 플래그
        switched_today = False
        morning_reset_done = False
        last_date = None

        while self._is_running:
            dt = self.get_current_time()
            current_date = dt.date()
            
            # 날짜가 바뀌면 플래그 초기화
            if last_date != current_date:
                switched_today = False
                morning_reset_done = False
                last_date = current_date

            new_state = self.determine_state(dt)

            if new_state != self.current_state:
                await self._transition_state(self.current_state, new_state)

            # [신규] 조건식 스위칭 로직 (09:00 장 시작 / 09:30 전환)
            if new_state == MarketState.TRADING:
                config_mgr = self.config_manager
                if not config_mgr:
                    if self.universe_manager and hasattr(self.universe_manager, 'config_manager'):
                        config_mgr = self.universe_manager.config_manager
                    elif self.order_manager and hasattr(self.order_manager, 'config'):
                        config_mgr = self.order_manager.config

                if config_mgr:
                    switch_time_str = config_mgr.get("SWITCH_TIME", "09:30:00")
                    try:
                        h, m, s = map(int, switch_time_str.split(':'))
                        t_switch = time(h, m, s)
                        
                        # 1. 장 시작 시 초기화 (09:00 ~ 09:30 사이인 경우 장 시작 조건식으로)
                        if not morning_reset_done:
                            if dt.time() < t_switch:
                                morning_cond = config_mgr.get("COND_NAME_MORNING", "AI스캘핑주도주장시작")
                                self.logger.warning(f"⏰ [시스템] 장 시작 - 초기 조건식 설정 ({morning_cond})")
                                self.signals.condition_switched.emit(morning_cond)
                            morning_reset_done = True

                        # 2. 09:30 도달 시 전환
                        if not switched_today and dt.time() >= t_switch:
                            normal_cond = config_mgr.get("COND_NAME_NORMAL", "AI스캘핑주도주")
                            self.logger.warning(f"⏰ [시스템] {switch_time_str} 도달 - 조건식 전환 시도 ({normal_cond})")
                            self.signals.condition_switched.emit(normal_cond)
                            switched_today = True
                    except Exception as e:
                        self.logger.error(f"스위칭 로직 실행 중 에러: {e}")

            # Check every second
            await asyncio.sleep(1)
