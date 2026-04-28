import logging
from PyQt6.QtCore import QObject, pyqtSignal

class RiskSignals(QObject):
    daily_stop_loss_hit = pyqtSignal(float)
    risk_warning = pyqtSignal(str)

class RiskManager:
    """
    글로벌 세이프티 가드 (Global Safety Guard)
    주문 집행 직전에 반드시 거쳐야 하는 게이트키퍼로서 AI 판단의 오류나
    급격한 시장 변동으로부터 계좌를 보호합니다.
    """
    def __init__(self, config_manager, order_manager):
        self.config_manager = config_manager
        self.order_manager = order_manager
        self.logger = logging.getLogger("RiskManager")
        self.signals = RiskSignals()

        self.daily_realized_pnl: float = 0.0
        self.is_stopped_for_day: bool = False

    def get_max_invest_per_symbol(self) -> float:
        # Default 5,000,000 KRW
        return float(self.config_manager.get("max_invest_per_symbol", 5000000))

    def get_max_position_pct(self) -> float:
        # Default 100% (All capital allocated to trading)
        return float(self.config_manager.get("max_position_pct", 100.0))

    def get_dynamic_max_invest(self) -> float:
        """
        [핵심 리스크 관리] 
        전체 자산 대비 설정된 비중(%)을 종목 수로 나누어 동적 한도를 계산합니다.
        공식: (현재 잔고 * 비중 / 100) / 최대 보유 종목 수
        """
        balance = getattr(self.order_manager, 'current_balance', 10000000)
        pct = self.get_max_position_pct()
        max_slots = self.get_max_open_positions()

        # 1. 비중 기반 계산 (예: 1000만 * 50% / 5종목 = 종목당 100만)
        ratio_based_limit = (balance * (pct / 100.0)) / max(1, max_slots)

        # 2. 고정 한도값과 비교하여 더 작은 값을 최종 한도로 채택 (보수적 운영)
        fixed_limit = self.get_max_invest_per_symbol()
        
        dynamic_limit = min(ratio_based_limit, fixed_limit)
        
        # 최소 10,000원(한 주 가격 고려) 보장
        return max(10000, dynamic_limit)

    def get_daily_stop_loss_limit(self) -> float:
        # Default -500,000 KRW
        return float(self.config_manager.get("daily_stop_loss_limit", -500000))

    def get_max_open_positions(self) -> int:
        # Default 3
        return int(self.config_manager.get("max_open_positions", 3))

    def update_pnl(self, realized_profit: float):
        """체결 시 손익을 누적합니다."""
        self.daily_realized_pnl += realized_profit
        self.logger.info(f"Daily Realized PnL Updated: {self.daily_realized_pnl:,.0f} KRW")

        limit = self.get_daily_stop_loss_limit()
        if not self.is_stopped_for_day and self.daily_realized_pnl <= limit:
            self.logger.error(f"🚨 [CRITICAL] Daily Stop-Loss Hit! Current: {self.daily_realized_pnl:,.0f} / Limit: {limit:,.0f}")
            self.is_stopped_for_day = True
            self.signals.daily_stop_loss_hit.emit(self.daily_realized_pnl)

    def can_order(self, symbol: str, amount: float, order_type: str) -> bool:
        """
        주문 전송 전 모든 조건을 검증합니다.
        """
        # [최우선] 보호 종목 체크 (매수/매도 모두 차단)
        protected_symbols = self.config_manager.get("protected_symbols", [])
        if symbol in protected_symbols:
            self.logger.warning(f"Risk Check Failed: [{symbol}]은 보호 종목으로 설정되어 있어 모든 자동 주문이 차단됩니다.")
            self.signals.risk_warning.emit(f"보호 종목({symbol})에 대한 자동 주문이 차단되었습니다.")
            return False

        if order_type.upper() in ["CANCEL"]:
            return True

        if self.is_stopped_for_day:
            self.logger.warning(f"Risk Check Failed: System is STOPPED_FOR_DAY due to Daily Stop-Loss.")
            return False

        # Check Cutoff Time through MarketScheduler
        scheduler = getattr(self.config_manager, "_injected_scheduler", None)
        if scheduler:
            from core.scheduler import MarketState
            if scheduler.current_state in [MarketState.CUTOFF, MarketState.LIQUIDATING, MarketState.STOPPED]:
                self.logger.info(f"Risk Check: 마감 시간 경과로 인한 신규 진입 생략 ({order_type} {symbol})")
                self.signals.risk_warning.emit("마감 시간 경과로 신규 매수 주문이 차단되었습니다.")
                return False

        # 1. Dynamic Max Invest Check (고정값 대신 동적 계산값 사용)
        current_holding_qty = self.order_manager.holdings.get(symbol, 0)
        avg_price = self.order_manager.avg_entry_prices.get(symbol, 0.0)
        current_invested = current_holding_qty * avg_price

        max_invest = self.get_dynamic_max_invest()
        if current_invested + amount > max_invest:
            self.logger.warning(f"Risk Check Failed [{symbol}]: Dynamic max invest exceeded. "
                                f"Current: {current_invested:,.0f}, Adding: {amount:,.0f}, Limit: {max_invest:,.0f} "
                                f"(Pct: {self.get_max_position_pct()}%)")
            self.signals.risk_warning.emit(f"[{symbol}] 자산 비중 기반 투자 한도({max_invest:,.0f}원) 초과로 매수 거부")
            return False

        # 2. Max Open Positions Check
        # Count symbols with holdings > 0. If this symbol is new, check against limit.
        open_positions = sum(1 for sym, qty in self.order_manager.holdings.items() if qty > 0)
        if current_holding_qty == 0 and open_positions >= self.get_max_open_positions():
            self.logger.warning(f"Risk Check Failed [{symbol}]: Max open positions ({self.get_max_open_positions()}) reached.")
            self.signals.risk_warning.emit(f"최대 동시 보유 종목 수({self.get_max_open_positions()}개) 초과로 신규 진입 거부")
            return False

        return True
