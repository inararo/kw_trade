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
    LIQUIDATING = "LIQUIDATING" # 15:20 ~ 15:30 - 신규 진입 금지 및 청산 (Panic Sell)
    STOPPED = "STOPPED"     # 15:30 이후 - 데이터 Flush 및 연결 종료
    STOPPED_FOR_DAY = "STOPPED_FOR_DAY" # 당일 거래 강제 중지 (Stop-Loss 등)

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

        if current_time < t_0850:
            return MarketState.IDLE
        elif t_0850 <= current_time < t_0900:
            return MarketState.PREPARE
        elif t_0900 <= current_time < t_1520:
            return MarketState.TRADING
        elif t_1520 <= current_time < t_1530:
            return MarketState.LIQUIDATING
        else:
            # After 15:30
            return MarketState.STOPPED

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

        self.logger.info(f"State Transition: {old_state} -> {new_state}")
        self.current_state = new_state
        self.signals.state_changed.emit(old_state, new_state)

        # Execute actions based on the new state
        if new_state == MarketState.PREPARE:
            self.logger.info("Market Prepare: Universe update and pre-connection logic here.")
            # e.g., await self.universe_manager.build_top_n_universe(...)

        elif new_state == MarketState.TRADING:
            self.logger.info("Market Open: Activating trading agents.")
            if self.data_collector and not self.data_collector.is_running:
                # Assuming data collector starts listening
                pass

        elif new_state == MarketState.LIQUIDATING:
            self.logger.warning("Market Closing Soon: Liquidating positions (Panic Sell).")
            if self.order_manager:
                # Trigger panic sell for all holdings
                for symbol, qty in self.order_manager.holdings.items():
                    if qty > 0:
                        self.logger.info(f"Liquidating {qty} shares of {symbol}")
                        # Execute market sell (03)
                        await self.order_manager.send_order("SELL", symbol, price=0, qty=qty, order_type="03")

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
                for symbol, qty in self.order_manager.holdings.items():
                    if qty > 0:
                        self.logger.critical(f"Emergency Liquidating {qty} shares of {symbol}")
                        await self.order_manager.send_order("SELL", symbol, price=0, qty=qty, order_type="03")

    async def _schedule_loop(self):
        while self._is_running:
            dt = self.get_current_time()
            new_state = self.determine_state(dt)

            if new_state != self.current_state:
                await self._transition_state(self.current_state, new_state)

            # Check every second
            await asyncio.sleep(1)
