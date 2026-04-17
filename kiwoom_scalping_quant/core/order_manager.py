import asyncio
import time
import logging
from typing import Dict, Any
from returns.result import Result, Success, Failure
from returns.future import FutureResult, future_safe

class OrderManager:
    def __init__(self, config: Dict[str, Any], auth_manager=None):
        self.config = config
        self.auth_manager = auth_manager
        self.logger = logging.getLogger("OrderManager")

        self.unexecuted_orders = {}
        self.holdings = 0

        self.rate_limit = 5
        self.order_semaphore = asyncio.Semaphore(self.rate_limit)
        self.order_timestamps = []

    async def _throttle_order(self):
        now = time.time()
        self.order_timestamps = [t for t in self.order_timestamps if now - t < 1.0]

        if len(self.order_timestamps) >= self.rate_limit:
            wait_time = 1.0 - (now - self.order_timestamps[0])
            if wait_time > 0:
                self.logger.warning(f"Throttling 활성화: {wait_time:.2f}초 대기")
                await asyncio.sleep(wait_time)

        self.order_timestamps.append(time.time())

    @future_safe
    async def send_order(self, order_type: str, symbol: str, price: int, qty: int) -> str:
        """
        REST API를 통한 주문 발송 (매수/매도)
        함수형 에러 처리(Result 패턴)를 사용하여 예외를 안전하게 감싸서 FutureResult로 반환.
        """
        await self._throttle_order()

        async with self.order_semaphore:
            # 1. API 요청 전송 (aiohttp 등 사용 로직 대체)
            # if request_fails: raise Exception("API 연결 에러")

            temp_order_id = f"ORD_{int(time.time() * 1000)}"

            # 2. 미체결 주문 등록
            self.unexecuted_orders[temp_order_id] = {
                'symbol': symbol,
                'type': order_type,
                'price': price,
                'qty': qty,
                'unexecuted_qty': qty,
                'timestamp': time.time()
            }

            self.logger.info(f"주문 접수 완료: {order_type} {qty}주 @ {price}원 (ID: {temp_order_id})")
            return temp_order_id

    # 사용 예시: (외부에서 호출할 때)
    # result: Result[str, Exception] = await order_manager.send_order("BUY", "005930", 50000, 10)
    # if isinstance(result, Success):
    #     order_id = result.unwrap()
    # else:
    #     error = result.failure()

    def update_execution_from_ws(self, execution_data: Dict[str, Any]):
        order_id = execution_data.get('order_id')
        executed_qty = execution_data.get('executed_qty', 0)

        if order_id in self.unexecuted_orders:
            order = self.unexecuted_orders[order_id]
            order['unexecuted_qty'] -= executed_qty

            if order['type'] == 'BUY':
                self.holdings += executed_qty
            elif order['type'] == 'SELL':
                self.holdings -= executed_qty

            if order['unexecuted_qty'] <= 0:
                del self.unexecuted_orders[order_id]
                self.logger.info(f"주문 전량 체결 완료 (ID: {order_id})")
            else:
                self.logger.info(f"주문 부분 체결 (ID: {order_id}, 잔여: {order['unexecuted_qty']})")

    def has_unexecuted_orders(self) -> bool:
        return len(self.unexecuted_orders) > 0

    async def cancel_all_orders(self):
        for order_id, order in list(self.unexecuted_orders.items()):
            self.logger.info(f"미체결 주문 취소 요청 (ID: {order_id})")
            del self.unexecuted_orders[order_id]
