import asyncio
import time
import logging
from typing import Dict, Any, Optional
from returns.result import Result, Success, Failure
from returns.future import FutureResult, future_safe

class OrderState:
    PENDING = "PENDING"        # 서버 전송 후 응답 대기
    ACCEPTED = "ACCEPTED"      # 서버 접수 완료 (주문번호 발급)
    PARTIAL = "PARTIAL_FILL"   # 부분 체결
    FILLED = "FILLED"          # 전량 체결
    CANCELLED = "CANCELLED"    # 취소 완료
    REPLACED = "REPLACED"      # 정정 완료
    FAILED = "FAILED"          # 거부/오류

class OrderManager:
    def __init__(self, config: Dict[str, Any], auth_manager=None):
        self.config = config
        self.auth_manager = auth_manager
        self.logger = logging.getLogger("OrderManager")

        # 고유 주문 ID(내부)를 키로, 상태 딕셔너리를 값으로 가지는 중앙 추적기
        self.active_orders: Dict[str, Dict[str, Any]] = {}

        # 키움증권 원주문번호(Broker ID)와 내부 ID 맵핑
        self.broker_id_map: Dict[str, str] = {}

        self.holdings = {sym: 0 for sym in [s.get('code') for s in config.get('universe', [{'code': '005930'}])]}
        self.avg_entry_prices = {sym: 0.0 for sym in [s.get('code') for s in config.get('universe', [{'code': '005930'}])]}

        # Global Risk Limits
        self.global_max_loss = config.get("global_max_loss", -500000) # e.g. Daily limit
        self.global_max_exposure = config.get("global_max_exposure", 50000000) # e.g. Total asset exposure

        self.daily_realized_pnl = 0.0

        self.rest_base_url = config.get_rest_url() if hasattr(config, 'get_rest_url') else "https://openapivts.kiwoom.com"

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

    def _check_global_risk(self, symbol: str, price: int, qty: int, order_type: str) -> bool:
        """새로운 주문(신규 진입) 시 글로벌 리스크를 점검합니다."""
        # 매도(청산), 정정, 취소 주문은 글로벌 리스크 한도와 무관하게 허용해야 함 (포지션 정리 목적)
        if order_type != "BUY":
            return True

        if self.daily_realized_pnl <= self.global_max_loss:
            self.logger.error(f"Global Risk: 일일 최대 손실({self.global_max_loss}) 초과. 신규 진입 차단.")
            return False

        # 총 노출 금액 = 모든 종목의 (보유 수량 * 현재가(또는 진입가))
        # 여기서는 단순화를 위해 현재 주문가격을 해당 종목의 현재가로 취급하여 노출 금액을 추산
        current_exposure = 0
        for sym, holding_qty in self.holdings.items():
            if sym == symbol:
                current_exposure += holding_qty * price
            else:
                current_exposure += holding_qty * self.avg_entry_prices.get(sym, 0)

        if current_exposure + (price * qty) > self.global_max_exposure:
            self.logger.error(f"Global Risk: 최대 노출 금액({self.global_max_exposure}) 초과. 신규 진입 차단.")
            return False

        return True

    @future_safe
    async def send_order(self, order_type: str, symbol: str, price: int, qty: int, orig_order_no: str = "") -> str:
        """
        REST API를 통한 주문 발송 (신규/정정/취소)
        orig_order_no가 있으면 정정/취소 주문으로 간주.
        """
        if not orig_order_no and not self._check_global_risk(symbol, price, qty, order_type):
            raise Exception("글로벌 리스크 점검 실패로 주문이 거부되었습니다.")

        await self._throttle_order()

        async with self.order_semaphore:
            internal_id = f"INT_{int(time.time() * 1000)}"

            # 상태 추적기 등록 (PENDING)
            self.active_orders[internal_id] = {
                'internal_id': internal_id,
                'broker_id': None,         # 접수 시 발급될 번호
                'orig_broker_id': orig_order_no,
                'symbol': symbol,
                'type': order_type,        # BUY, SELL, REPLACE, CANCEL
                'price': price,
                'qty': qty,
                'unexecuted_qty': qty,
                'status': OrderState.PENDING,
                'timestamp': time.time(),
                'ack_event': asyncio.Event() # 응답 대기용 이벤트
            }

            self.logger.info(f"주문 전송: {order_type} {qty}주 @ {price}원 (Internal ID: {internal_id})")

            # API 요청 전송 로직 (Mock)
            # 실제 구현에서는 aiohttp를 활용하여 self.rest_base_url 에 요청을 전송합니다.
            # endpoint = f"{self.rest_base_url}/uapi/domestic-stock/v1/trading/order-cash"
            # headers = {"authorization": f"Bearer {self.auth_manager.get_token()}", ...}
            # async with aiohttp.ClientSession() as session:
            #     async with session.post(endpoint, json=payload, headers=headers) as resp:
            #         ...

            # 백그라운드에서 3초 타임아웃 검사 실행
            asyncio.create_task(self._wait_for_ack(internal_id, timeout=3.0))

            # 백테스팅/Mock 환경을 위한 자동 접수 에뮬레이션
            asyncio.create_task(self._mock_broker_ack(internal_id))

            return internal_id

    async def _wait_for_ack(self, internal_id: str, timeout: float):
        """주문 접수 후 브로커 응답(접수 확인) 타임아웃 감시"""
        order = self.active_orders.get(internal_id)
        if not order: return

        try:
            await asyncio.wait_for(order['ack_event'].wait(), timeout=timeout)
        except asyncio.TimeoutError:
            if order['status'] == OrderState.PENDING:
                order['status'] = OrderState.FAILED
                self.logger.error(f"주문 응답 타임아웃 (3초 초과)! 실패 처리됨. ID: {internal_id}")

    async def _mock_broker_ack(self, internal_id: str):
        """Mock: 브로커가 0.1초 후 접수 확인(ACCEPTED)을 준다고 가정"""
        await asyncio.sleep(0.1)
        order = self.active_orders.get(internal_id)
        if order and order['status'] == OrderState.PENDING:
            broker_id = f"BRK_{internal_id.split('_')[1]}"

            # OnReceiveChejanData의 '접수' 이벤트 에뮬레이션
            mock_chejan = {
                'msg_type': '접수',
                'internal_id': internal_id,
                'broker_id': broker_id,
                'status': OrderState.ACCEPTED
            }
            self.on_receive_chejan_data(mock_chejan)

    def on_receive_chejan_data(self, data: Dict[str, Any]):
        """
        키움 웹소켓(또는 REST 폴링)에서 수신된 실시간 체결/잔고(t1301 등) 데이터 파싱.
        """
        internal_id = data.get('internal_id')
        broker_id = data.get('broker_id')
        msg_type = data.get('msg_type') # '접수', '체결', '취소확인' 등

        order = self.active_orders.get(internal_id)
        if not order:
            # broker_id로 역추적
            internal_id = self.broker_id_map.get(broker_id)
            order = self.active_orders.get(internal_id)
            if not order:
                return

        if msg_type == '접수':
            order['status'] = OrderState.ACCEPTED
            order['broker_id'] = broker_id
            self.broker_id_map[broker_id] = internal_id
            order['ack_event'].set() # 타임아웃 해제
            self.logger.info(f"브로커 접수 완료. (Broker ID: {broker_id})")

        elif msg_type == '체결':
            exec_qty = data.get('exec_qty', 0)
            exec_price = data.get('exec_price', order.get('price', 0))
            order['unexecuted_qty'] -= exec_qty
            symbol = order['symbol']

            if symbol not in self.holdings:
                self.holdings[symbol] = 0
                self.avg_entry_prices[symbol] = 0.0

            if order['type'] == 'BUY':
                # 평균 단가 갱신 (단순 이동 평균 형태)
                current_qty = self.holdings[symbol]
                total_value = (current_qty * self.avg_entry_prices[symbol]) + (exec_qty * exec_price)
                self.holdings[symbol] += exec_qty
                self.avg_entry_prices[symbol] = total_value / self.holdings[symbol]

            elif order['type'] == 'SELL':
                self.holdings[symbol] -= exec_qty
                # 체결가 기반으로 daily_realized_pnl 업데이트
                realized_profit = (exec_price - self.avg_entry_prices[symbol]) * exec_qty
                self.daily_realized_pnl += realized_profit
                self.logger.info(f"실현 손익 업데이트: {realized_profit:,.0f} (누적: {self.daily_realized_pnl:,.0f})")

                if self.holdings[symbol] <= 0:
                    self.holdings[symbol] = 0
                    self.avg_entry_prices[symbol] = 0.0

            if order['unexecuted_qty'] <= 0:
                order['status'] = OrderState.FILLED
                self.logger.info(f"주문 전량 체결 완료! (Broker ID: {broker_id})")
            else:
                order['status'] = OrderState.PARTIAL
                self.logger.info(f"주문 부분 체결 (Broker ID: {broker_id}, 잔여: {order['unexecuted_qty']})")
                self.handle_partial_fill(order)

        elif msg_type == '취소확인':
            order['status'] = OrderState.CANCELLED
            order['unexecuted_qty'] = 0
            self.logger.info(f"주문 취소 완료. (Broker ID: {broker_id})")

        elif msg_type == '정정확인':
            order['status'] = OrderState.REPLACED
            self.logger.info(f"주문 정정 완료. (Broker ID: {broker_id})")

    def handle_partial_fill(self, order: Dict[str, Any]):
        """
        부분 체결 발생 시 미체결 잔량을 즉시 시장가 취소(또는 정정)하여 포지션 꼬임 방지.
        """
        if order['unexecuted_qty'] > 0:
            self.logger.warning(f"부분 체결 감지! 잔여 수량({order['unexecuted_qty']}주) 긴급 취소 진행. (Broker ID: {order['broker_id']})")
            # 비동기로 취소 주문 전송
            asyncio.create_task(self.send_order(
                order_type="CANCEL",
                symbol=order['symbol'],
                price=0,
                qty=order['unexecuted_qty'],
                orig_order_no=order['broker_id']
            ))

    async def execute_smart_order(self, action: str, symbol: str, target_qty: int, data_collector):
        """
        동적 지정가 추적 매매 (Cancel & Replace 로직).
        시장가 대신 최우선 호가를 추적하며 유리한 가격에 체결되도록 유도합니다.
        """
        max_retries = 3
        timeout_sec = 5.0

        retries = 0
        start_time = time.time()

        # 최초 1호가 진입 (실제 가격 추출)
        best_price = int(data_collector.get_latest_price(symbol))
        if best_price <= 0:
            self.logger.error("스마트 주문 실패: 최신 가격 정보를 가져오지 못했습니다.")
            return

        self.logger.info(f"[Smart Order] 진입 시작: {action} {target_qty}주 @ {best_price}")

        from returns.io import IOFailure, IOSuccess

        # 1. 주문 발송
        result = await self.send_order(action, symbol, best_price, target_qty)
        if isinstance(result, IOFailure):
            self.logger.error("스마트 주문 전송 실패")
            return

        internal_id = result.unwrap()._inner_value

        while time.time() - start_time < timeout_sec and retries < max_retries:
            await asyncio.sleep(0.5) # 0.5초마다 호가 확인

            order = self.active_orders.get(internal_id)
            if not order or order['status'] in [OrderState.FILLED, OrderState.CANCELLED, OrderState.FAILED]:
                self.logger.info("[Smart Order] 추적 종료 (체결/취소/실패 완료).")
                return

            if order['status'] == OrderState.PENDING:
                continue # 아직 접수 안됨

            # 호가 변화 감지 (Mock 로직: 10% 확률로 가격이 도망갔다고 가정)
            import random
            if random.random() < 0.1:
                retries += 1
                new_price = best_price + (100 if action == "BUY" else -100)
                broker_id = order.get('broker_id')

                self.logger.warning(f"[Smart Order] 호가 이탈 감지! 정정 주문 발송 (Retries: {retries}/{max_retries}) | {best_price} -> {new_price}")

                # 기존 주문 정정 (Replace)
                replace_res = await self.send_order(
                    order_type="REPLACE",
                    symbol=symbol,
                    price=new_price,
                    qty=order['unexecuted_qty'],
                    orig_order_no=broker_id
                )

                if isinstance(replace_res, IOSuccess):
                    internal_id = replace_res.unwrap()._inner_value
                    best_price = new_price
                else:
                    self.logger.error("[Smart Order] 정정 주문 실패.")
                    break

        # 루프 종료 후에도 미체결 남아있으면 전량 취소
        order = self.active_orders.get(internal_id)
        if order and order['unexecuted_qty'] > 0 and order['status'] not in [OrderState.FILLED, OrderState.CANCELLED]:
            self.logger.error(f"[Smart Order] 시간 초과(5초) 또는 재시도 초과! 남은 {order['unexecuted_qty']}주 전량 취소.")
            await self.send_order("CANCEL", symbol, 0, order['unexecuted_qty'], orig_order_no=order.get('broker_id'))


    def has_unexecuted_orders(self, symbol: str = None) -> bool:
        for o in self.active_orders.values():
            if o['unexecuted_qty'] > 0 and o['status'] not in [OrderState.FILLED, OrderState.CANCELLED, OrderState.FAILED]:
                if symbol is None or o['symbol'] == symbol:
                    return True
        return False

    async def cancel_all_orders(self):
        for int_id, order in list(self.active_orders.items()):
            if order['unexecuted_qty'] > 0 and order['status'] not in [OrderState.CANCELLED, OrderState.FILLED]:
                self.logger.info(f"전체 미체결 취소 요청 (Internal ID: {int_id})")
                await self.send_order("CANCEL", order['symbol'], 0, order['unexecuted_qty'], orig_order_no=order.get('broker_id'))
