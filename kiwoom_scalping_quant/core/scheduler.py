import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Optional
from PyQt6.QtCore import QObject, pyqtSignal

class SchedulerSignals(QObject):
    state_changed = pyqtSignal(str, str) # old_state, new_state

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
    def __init__(self, data_collector=None, order_manager=None, universe_manager=None, telegram_bot=None):
        self.logger = logging.getLogger("MarketScheduler")
        self.signals = SchedulerSignals()

        self.data_collector = data_collector
        self.order_manager = order_manager
        self.universe_manager = universe_manager
        self.telegram_bot = telegram_bot

        self.current_state = MarketState.IDLE
        self._is_running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._intraday_scanner_task: Optional[asyncio.Task] = None
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
            # Remain stopped for the rest of the day.
            # To reset, the system must be restarted the next day.
            return MarketState.STOPPED_FOR_DAY

        if self.is_holiday_or_weekend(dt):
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
        if self._intraday_scanner_task:
            self._intraday_scanner_task.cancel()
            try:
                await self._intraday_scanner_task
            except asyncio.CancelledError:
                pass

    async def _transition_state(self, old_state: str, new_state: str):
        if old_state == new_state:
            return

        self.logger.info(f"State Transition: {old_state} -> {new_state}")
        self.current_state = new_state
        self.signals.state_changed.emit(old_state, new_state)

        # Execute actions based on the new state
        if new_state == MarketState.PREPARE:
            self.logger.info("Market Prepare: Requesting dynamic universe generation...")
            # Use the injected AssetDataViewModel if available to trigger the UI-bound universe logic
            vm = getattr(self.config_manager, "_injected_asset_data_vm", None) if hasattr(self, 'config_manager') else None
            if vm and hasattr(vm, 'build_universe'):
                vm.build_universe()

        elif new_state == MarketState.TRADING:
            self.logger.info("Market Open: Activating trading agents.")
            if self.data_collector and not self.data_collector.is_running:
                # Assuming data collector starts listening
                pass

            # Start Intraday dynamic universe scanner
            if not self._intraday_scanner_task or self._intraday_scanner_task.done():
                self._intraday_scanner_task = asyncio.create_task(self._intraday_scanner_loop())

        elif new_state == MarketState.CUTOFF:
            self.logger.warning("Market Cutoff Time Reached. New AI buys are blocked. Only monitoring and liquidating active.")
            # Scanner loop inherently halts because it checks `self.current_state == MarketState.TRADING`

        elif new_state == MarketState.LIQUIDATING:
            self.logger.warning("Market Closing Soon: Liquidating positions (Panic Sell).")
            if self.order_manager:
                # 보호 종목 리스트 가져오기 (Scheduler에는 config_manager 의존성이 직접 주입되지 않으므로, None 체크 필요)
                # 만약 config_manager가 없다면 직접 order_manager 등에서 가져와야 하지만,
                # 현재는 order_manager.bot_holdings를 우선 활용하고 환경 변수 fallback을 씁니다.
                import os

                # We can inject or grab protected_symbols, but it's simpler to just check bot_holdings
                bot_holdings = getattr(self.order_manager, 'bot_holdings', {})

                # Trigger panic sell for bot holdings
                for symbol, qty in self.order_manager.holdings.items():
                    if qty <= 0:
                        continue

                    bot_qty = bot_holdings.get(symbol, 0)
                    if bot_qty <= 0:
                        self.logger.info(f"수동 매수 종목 청산 제외 (LIQUIDATING): {symbol}")
                        continue

                    target_qty = min(qty, bot_qty)
                    self.logger.info(f"Liquidating {target_qty} shares of {symbol} (Bot managed)")
                    # Execute market sell (03)
                    await self.order_manager.send_order("SELL", symbol, price=0, qty=target_qty, order_type="03")

        elif new_state == MarketState.STOPPED:
            self.logger.info("Market Closed: Flushing data and disconnecting.")
            if self.data_collector:
                await self.data_collector.stop()
            if self.telegram_bot:
                # Send daily summary
                pass

        elif new_state == MarketState.STOPPED_FOR_DAY:
            self.logger.critical("🚨 Market Stopped For Day: Emergency Liquidation Triggered.")
            if self.order_manager:
                # Cancel all and market sell immediately
                await self.order_manager.cancel_all_orders()

                bot_holdings = getattr(self.order_manager, 'bot_holdings', {})
                for symbol, qty in self.order_manager.holdings.items():
                    if qty <= 0:
                        continue

                    bot_qty = bot_holdings.get(symbol, 0)
                    if bot_qty <= 0:
                        self.logger.info(f"수동 매수 종목 강제 청산 제외 (STOPPED_FOR_DAY): {symbol}")
                        continue

                    target_qty = min(qty, bot_qty)
                    self.logger.critical(f"Emergency Liquidating {target_qty} shares of {symbol}")
                    await self.order_manager.send_order("SELL", symbol, price=0, qty=target_qty, order_type="03")

        elif new_state == MarketState.POST_MARKET:
            self.logger.info("Post-Market: Starting end-of-day data collection for final top 20 universe.")
            vm = getattr(self.universe_manager.config_manager, "_injected_asset_data_vm", None)
            if vm and hasattr(vm, 'auto_collect_after_market'):
                # Call view model UI flow properly asynchronously
                asyncio.create_task(vm.auto_collect_after_market())

    async def _intraday_scanner_loop(self):
        """
        장중 주기적 스캐너. TRADING 상태일 때만 동작합니다.
        30분 주기(1800초)로 실행하여 동적 주도주 유니버스를 업데이트합니다.
        """
        try:
            while self._is_running and self.current_state == MarketState.TRADING:
                # 09:00에 시작 시 09:01까지 대기하여 당일 첫 1분 거래대금이 집계될 수 있도록 함.
                await asyncio.sleep(60)

                while self._is_running and self.current_state == MarketState.TRADING:
                    async with self._universe_lock:
                        self.logger.info("Intraday Scanner: 장중 주도주 재검색 시작...")
                        if self.universe_manager and hasattr(self.universe_manager, 'config_manager'):
                            token = self.universe_manager.config_manager.get("KIWOOM_ACCESS_TOKEN")
                            if token:
                                from returns.io import IOFailure, IOSuccess
                                result = await self.universe_manager.build_top_n_universe(token, top_n=20)

                                if isinstance(result, IOFailure):
                                    self.logger.error("Intraday Scanner: 유니버스 업데이트 실패")
                                else:
                                    new_universe = result.unwrap()._inner_value
                                    if new_universe:
                                        # Safe Swap Logic Delegate
                                        await self._safe_swap_universe(new_universe)

                    # 30분 (1800초) 대기
                    await asyncio.sleep(1800)
        except asyncio.CancelledError:
            self.logger.info("Intraday Scanner 태스크가 종료되었습니다.")

    async def _safe_swap_universe(self, new_universe):
        """
        안전한 종목 교체 (Safe Swap Logic). StrategyManager에 위임하거나 직접 관리합니다.
        """
        self.logger.info(f"Intraday Scanner: {len(new_universe)}개의 새 유니버스가 발견되었습니다. (스왑 위임)")
        # container를 통해 strategy_manager에 직접 호출을 전달하는 로직이 필요합니다.
        # 이 메서드는 의존성 또는 Signal을 통해 StrategyManager의 update_universe()를 트리거합니다.

        # 임시로 Signal을 만들거나 hasattr로 직접 호출
        # GUI의 live_vm 등을 통해 signal_log에 이벤트를 띄웁니다.
        # 실제 교체 로직은 StrategyManager 안에서 수행하는 것이 안전합니다.
        strategy_manager = getattr(self.universe_manager.config_manager, "_injected_strategy_manager", None)
        if strategy_manager and hasattr(strategy_manager, 'update_universe'):
            await strategy_manager.update_universe(new_universe)

    async def _schedule_loop(self):
        while self._is_running:
            dt = self.get_current_time()
            new_state = self.determine_state(dt)

            if new_state != self.current_state:
                await self._transition_state(self.current_state, new_state)

            # Check every second
            await asyncio.sleep(1)
