import logging
from typing import List, Set

class SymbolSubscriptionManager:
    """
    키움 API의 실시간 구독 제한(예: 100종목)을 준수하며 동적으로 종목 구독을 관리합니다.
    """
    def __init__(self, max_subscriptions: int = 100):
        self.max_subscriptions = max_subscriptions
        self.active_symbols: Set[str] = set()
        self.logger = logging.getLogger("SymbolSubscriptionManager")

    def add_symbol(self, symbol: str) -> bool:
        """
        새로운 종목을 구독 목록에 추가합니다.
        제한을 초과하면 False를 반환합니다.
        """
        if symbol in self.active_symbols:
            return True

        if len(self.active_symbols) >= self.max_subscriptions:
            self.logger.warning(f"구독 초과! 최대 {self.max_subscriptions} 종목까지만 실시간 데이터를 수신할 수 있습니다.")
            return False

        self.active_symbols.add(symbol)
        self.logger.info(f"종목 추가됨: {symbol} (현재 {len(self.active_symbols)}/{self.max_subscriptions})")
        return True

    def remove_symbol(self, symbol: str):
        """
        구독 목록에서 종목을 제거합니다.
        """
        if symbol in self.active_symbols:
            self.active_symbols.remove(symbol)
            self.logger.info(f"종목 제거됨: {symbol} (현재 {len(self.active_symbols)}/{self.max_subscriptions})")

    def get_symbols(self) -> List[str]:
        """
        현재 구독 중인 모든 종목 리스트를 반환합니다.
        """
        return list(self.active_symbols)

    def get_subscription_string(self) -> str:
        """
        키움증권 API 규격 등에 맞추어 콤마(,)로 구분된 문자열을 반환합니다.
        """
        return ",".join(self.active_symbols)
