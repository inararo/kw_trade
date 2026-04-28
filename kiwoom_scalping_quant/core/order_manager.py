import asyncio
import time
import logging
from typing import Dict, Any, Optional
from returns.result import Result, Success, Failure
from returns.future import FutureResult, future_safe
from PyQt6.QtCore import QObject, pyqtSignal

class OrderSignals(QObject):
    signal_only_log = pyqtSignal(str)
    balance_synced = pyqtSignal(float) # [신규] 잔고 동기화 완료 시그널

class OrderState:
    PENDING = "PENDING"        # 서버 전송 후 응답 대기
    ACCEPTED = "ACCEPTED"      # 서버 접수 완료 (주문번호 발급)
    PARTIAL = "PARTIAL_FILL"   # 부분 체결
    FILLED = "FILLED"          # 전량 체결
    CANCELLED = "CANCELLED"    # 취소 완료
    REPLACED = "REPLACED"      # 정정 완료
    FAILED = "FAILED"          # 거부/오류

class OrderManager:
    def __init__(self, config: Dict[str, Any], auth_manager=None, telegram_notifier=None):
        self.config = config
        self.auth_manager = auth_manager
        self.notifier = telegram_notifier
        self.logger = logging.getLogger("OrderManager")
        self.signals = OrderSignals()

        # 고유 주문 ID(내부)를 키로, 상태 딕셔너리를 값으로 가지는 중앙 추적기
        self.active_orders: Dict[str, Dict[str, Any]] = {}

        # 실시간 자산 관리 (초기 자산에서 시작)
        self.current_balance = float(config.get("initial_balance", 10000000)) # 총 자산 (Equity)
        self._broker_orderable_cash = self.current_balance                    # 증권사 확인 현금
        self.pending_buy_amount = 0.0                                          # 주문 전송 후 체결 대기 중인 금액 (내부 예약)
        self.last_sync_time = 0

        # 키움증권 원주문번호(Broker ID)와 내부 ID 맵핑
        self.broker_id_map: Dict[str, str] = {}

        self.holdings = {sym: 0 for sym in [s.get('code') for s in config.get('universe', [{'code': '005930'}])]}
        self.avg_entry_prices = {sym: 0.0 for sym in [s.get('code') for s in config.get('universe', [{'code': '005930'}])]}

        # 봇(Agent) 전용 매수/보유 수량 트래킹 (수동 매수 종목과 구분 위함)
        self.bot_holdings = {sym: 0 for sym in [s.get('code') for s in config.get('universe', [{'code': '005930'}])]}

        # Safety Guard Risk Manager
        self.risk_manager = None # Will be injected

    @property
    def orderable_cash(self) -> float:
        """내부 예약 금액을 제외한 실제 가용 현금"""
        return max(0, self._broker_orderable_cash - self.pending_buy_amount)

        # Global Risk Limits (Deprecated in favor of RiskManager)
        self.global_max_loss = config.get("global_max_loss", -500000) # e.g. Daily limit
        self.global_max_exposure = config.get("global_max_exposure", 50000000) # e.g. Total asset exposure

        self.daily_realized_pnl = 0.0

        self.rest_base_url = config.get_rest_url() if hasattr(config, 'get_rest_url') else "https://mockapi.kiwoom.com"

        self.rate_limit = 5
        self.order_semaphore = asyncio.Semaphore(self.rate_limit)
        self.order_timestamps = []

        # [신규] 마지막 동기화 시간
        self.last_sync_time = 0

    def get_balance(self) -> float:
        """현재 가용 잔고를 반환합니다."""
        return self.current_balance

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
        if not orig_order_no:
            if self.risk_manager and not self.risk_manager.can_order(symbol, price * qty, order_type):
                raise Exception(f"RiskManager: 글로벌 세이프티 가드 제한으로 인해 신규 주문({order_type} {symbol})이 거부되었습니다.")
            elif not self.risk_manager and not self._check_global_risk(symbol, price, qty, order_type):
                raise Exception("글로벌 리스크 점검 실패로 주문이 거부되었습니다.")

        await self._throttle_order()

        async with self.order_semaphore:
            internal_id = f"INT_{int(time.time() * 1000)}"

            # --- Signal Only Bypass Logic ---
            signal_only = False
            # Check config manager if available, else standard config dict fallback
            if hasattr(self.config, 'get'):
                signal_only = self.config.get('signal_only_mode', False)

            if signal_only:
                msg = f"[SIGNAL ONLY] 🔴 {order_type}: {symbol} ({qty}주 @ {price}) - 실제 주문 생략됨"
                self.logger.error(msg)

                # EMIT SIGNAL
                self.signals.signal_only_log.emit(msg)

                # Fake success order tracking registration
                self.active_orders[internal_id] = {
                    'internal_id': internal_id,
                    'broker_id': f"SIG_{internal_id}",
                    'orig_broker_id': orig_order_no,
                    'symbol': symbol,
                    'type': order_type,
                    'price': price,
                    'qty': qty,
                    'unexecuted_qty': 0, # Immediately consider "filled" mentally or bypassed
                    'status': OrderState.FILLED,
                    'timestamp': time.time()
                }
                return internal_id

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
                'reserved_amt': price * qty if order_type == "BUY" else 0,
                'status': OrderState.PENDING,
                'timestamp': time.time(),
                'ack_event': asyncio.Event() # 응답 대기용 이벤트
            }

            # [내부 예약] 매수 주문 시 가용 현금에서 즉시 차감 (Race Condition 방지)
            if order_type == "BUY":
                order_amt = price * qty
                self.pending_buy_amount += order_amt
                self.logger.debug(f"주문 예약: +{order_amt:,.0f} (총 예약: {self.pending_buy_amount:,.0f})")

            self.logger.error(f"주문 전송: {order_type} {qty}주 @ {price}원 (Internal ID: {internal_id})")

            # ─────────────────────────────────────────────────────
            # 키움증권 REST API 실거래 주문 (api-id 헤더 방식)
            # 매수: kt10000 / 매도: kt10001 / 정정: kt10002 / 취소: kt10003
            # ─────────────────────────────────────────────────────
            import aiohttp

            # 1. 인증 정보 수집
            app_key    = self.config.get("KIWOOM_APP_KEY", "")    if hasattr(self.config, 'get') else ""
            app_secret = self.config.get("KIWOOM_APP_SECRET", "") if hasattr(self.config, 'get') else ""
            account_no = self.config.get("ACCOUNT_NO") or self.config.get("account_number") or ""
            token      = self.auth_manager.get_token()            if self.auth_manager else None

            if not token:
                self.logger.error("❌ 유효한 API 토큰이 없어 주문을 거절합니다. 토큰 갱신을 확인하세요.")
                self.active_orders[internal_id]['status'] = OrderState.FAILED
                raise Exception("API_TOKEN_MISSING")

            if not account_no:
                self.logger.error("❌ 계좌번호(ACCOUNT_NO)가 설정되지 않아 주문을 거절합니다.")
                self.active_orders[internal_id]['status'] = OrderState.FAILED
                raise Exception("ACCOUNT_NO_MISSING")

            # 2. 주문 종류별 api-id 및 엔드포인트 결정
            endpoint = f"{self.rest_base_url}/api/dostk/ordr"

            if order_type == "BUY":
                api_id = 'kt10000'
                # trde_tp: '0'=지정가, '3'=시장가
                trde_tp = '0' if price > 0 else '3'
                body = {
                    "dmst_stex_tp": 'KRX',  # 국내거래소 구분 필수, 예시로 KRX 고정
                    "stk_cd":       symbol,
                    "ord_qty":      str(qty),
                    "ord_uv":       str(price) if price > 0 else '',
                    "trde_tp":      trde_tp,
                }

            elif order_type == "SELL":
                api_id = 'kt10001'
                trde_tp = '0' if price > 0 else '3'
                body = {
                    "dmst_stex_tp": 'KRX',
                    "stk_cd":       symbol,
                    "ord_qty":      str(qty),
                    "ord_uv":       str(price) if price > 0 else '',
                    "trde_tp":      trde_tp,
                }

            elif order_type == "REPLACE":
                api_id = 'kt10002'
                body = {
					"dmst_stex_tp": 'KRX',
                    "orig_ord_no":   str(orig_order_no),
                    "stk_cd":        symbol,
                    "mdfy_qty":      str(qty),
                    "mdfy_uv":       str(price) if price > 0 else '',
                    "mdfy_cond_uv":  '', # 정정 조건 가격 (필요시 추가)
                }

            elif order_type == "CANCEL":
                api_id = 'kt10003'
                body = {
					"dmst_stex_tp": 'KRX',
                    "orig_ord_no": str(orig_order_no),
                    "stk_cd":      symbol,
                    "cncl_qty":    str(qty),
                }

            else:
                self.active_orders[internal_id]['status'] = OrderState.FAILED
                raise Exception(f"지원하지 않는 주문 타입: {order_type}")

            # 3. HTTP Headers (키움 REST API 주문 전용 규격)
            headers = {
                "Content-Type": "application/json;charset=UTF-8",
                "authorization": f"Bearer {token}",
                "api-id":       str(api_id),
            }

            self.logger.info(f"📤 주문 전송 [{api_id}] {order_type} {symbol} {qty}주 @ {price:,}원")
            self.logger.debug(f"   └ Body: {body}")

            # 4. 비동기 HTTP POST (키움 증권사 서버망)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        endpoint, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=5.0)
                    ) as resp:
                        res_data = await resp.json(content_type=None)

                        # return_code와 return_msg는 문자열 키로 정확히 접근
                        return_code = str(res_data.get('return_code', '-1'))
                        return_msg  = str(res_data.get('return_msg', ''))

                        if return_code == '0':
                            # 주문 접수 성공 – 원주문번호(ord_no) 파싱 (문자열 키 사용)
                            output = res_data.get('output', {}) or {}
                            broker_id = (
                                str(output.get('ord_no', ''))
                                or str(output.get('org_ord_no', ''))
                                or str(res_data.get('ord_no', ''))
                                or internal_id
                            )
                            self.logger.info(f"✅ 키움 접수 완료! 주문번호: {broker_id} | {return_msg}")

                            # 내부 상태 업데이트 시 문자열 키 사용 ('status', 'broker_id', 'ack_event')
                            self.active_orders[internal_id]['status']    = OrderState.ACCEPTED
                            self.active_orders[internal_id]['broker_id'] = broker_id
                            self.broker_id_map[broker_id]                = internal_id
                            self.active_orders[internal_id]['ack_event'].set()

                        else:
                            self.logger.error(f"❌ 키움 주문 거부: [{return_code}] {return_msg}")
                            self.active_orders[internal_id]['status'] = OrderState.FAILED
                            self.active_orders[internal_id]['ack_event'].set()
                            
                            # [내부 예약 해제]
                            if order_type == "BUY":
                                reserved = self.active_orders[internal_id].get('reserved_amt', 0)
                                self.pending_buy_amount = max(0, self.pending_buy_amount - reserved)
                                self.logger.debug(f"예약 해제(거부): -{reserved:,.0f} (남은 예약: {self.pending_buy_amount:,.0f})")
                            
                            raise Exception(f"KIWOOM_ORDER_REJECTED: {return_msg}")

            except asyncio.TimeoutError:
                self.logger.error(f"⏰ 키움 API 응답 타임아웃 (5초 초과)! ID: {internal_id}")
                self.active_orders[internal_id]['status'] = OrderState.FAILED
                self.active_orders[internal_id]['ack_event'].set()
                # [내부 예약 해제]
                if order_type == "BUY":
                    reserved = self.active_orders[internal_id].get('reserved_amt', 0)
                    self.pending_buy_amount = max(0, self.pending_buy_amount - reserved)
                raise Exception("KIWOOM_API_TIMEOUT")

            except Exception as e:
                # 이미 KIWOOM_ 접두어가 붙은 예외는 중복 처리 방지
                if "KIWOOM_" not in str(e):
                    self.logger.error(f"🔥 주문 전송 중 예외 발생: {e}")
                    self.active_orders[internal_id]['status'] = OrderState.FAILED
                    self.active_orders[internal_id]['ack_event'].set()
                    
                    # [내부 예약 해제]
                    if order_type == "BUY":
                        reserved = self.active_orders[internal_id].get('reserved_amt', 0)
                        self.pending_buy_amount = max(0, self.pending_buy_amount - reserved)
                raise

            return internal_id

    @future_safe
    async def cancel_order(self, internal_id: str) -> bool:
        """
        내부 주문 ID를 기반으로 미체결 잔량을 확인하여 취소 주문을 전송합니다.
        """
        order = self.active_orders.get(internal_id)
        if not order:
            self.logger.error(f"취소 실패: 내부 ID {internal_id}를 찾을 수 없습니다.")
            return False

        broker_id = order.get('broker_id')
        unexecuted_qty = order.get('unexecuted_qty', 0)
        status = order.get('status')

        # 취소 불가능한 상태 체크 (이미 체결됨, 이미 취소됨, 전송 실패 등)
        if status in [OrderState.FILLED, OrderState.CANCELLED, OrderState.FAILED] or unexecuted_qty <= 0:
            self.logger.info(f"취소 건너뜀: 주문 {internal_id}는 이미 종료되었거나 미체결 물량이 없습니다. (상태: {status})")
            return True

        if not broker_id:
            self.logger.warning(f"취소 지연: 주문 {internal_id}의 브로커 주문번호가 아직 없습니다. (PENDING 상태)")
            return False

        self.logger.error(f"🚫 미체결 취소 요청 시작: {order['symbol']} | 원주문번호: {broker_id} | 취소수량: {unexecuted_qty}")
        
        # kt10003 취소 주문 실행 (send_order의 CANCEL 타입 활용)
        try:
            # 취소 주문은 가격(price)이 의미가 없으므로 0으로 전송 (kt10003 규격 준수)
            result = await self.send_order("CANCEL", order['symbol'], 0, unexecuted_qty, orig_order_no=broker_id)
            return True
        except Exception as e:
            self.logger.error(f"취소 주문 전송 중 오류 발생: {e}")
            return False


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

            if symbol not in self.bot_holdings:
                self.bot_holdings[symbol] = 0

            if order['type'] == 'BUY':
                # 평균 단가 갱신 (단순 이동 평균 형태)
                current_qty = self.holdings[symbol]
                total_value = (current_qty * self.avg_entry_prices[symbol]) + (exec_qty * exec_price)
                self.holdings[symbol] += exec_qty
                self.avg_entry_prices[symbol] = total_value / self.holdings[symbol]

                # 봇이 진입한 수량 추가
                self.bot_holdings[symbol] += exec_qty

            elif order['type'] == 'SELL':
                self.holdings[symbol] -= exec_qty

                # 봇이 청산한 수량 감소 (0 미만으로 떨어지지 않게 방어)
                self.bot_holdings[symbol] = max(0, self.bot_holdings[symbol] - exec_qty)

                # 체결가 기반으로 daily_realized_pnl 업데이트
                realized_profit = (exec_price - self.avg_entry_prices[symbol]) * exec_qty
                self.daily_realized_pnl += realized_profit
                self.logger.info(f"실현 손익 업데이트: {realized_profit:,.0f} (누적: {self.daily_realized_pnl:,.0f})")

                # Update global safety guard PnL
                if self.risk_manager:
                    self.risk_manager.update_pnl(realized_profit)

                if self.holdings[symbol] <= 0:
                    self.holdings[symbol] = 0
                    self.avg_entry_prices[symbol] = 0.0

            if order['unexecuted_qty'] <= 0:
                order['status'] = OrderState.FILLED
                self.logger.info(f"주문 전량 체결 완료! (Broker ID: {broker_id})")
                
                # [내부 예약 해제]
                if order['type'] == 'BUY':
                    reserved = order.get('reserved_amt', 0)
                    self.pending_buy_amount = max(0, self.pending_buy_amount - reserved)
                    self.logger.debug(f"예약 해제(체결): -{reserved:,.0f} (남은 예약: {self.pending_buy_amount:,.0f})")
            else:
                order['status'] = OrderState.PARTIAL
                self.logger.info(f"주문 부분 체결 (Broker ID: {broker_id}, 잔여: {order['unexecuted_qty']})")
                self.handle_partial_fill(order)

            # 텔레그램 알림 발송 (체결 시)
            if self.notifier:
                # 종목명 찾기 (universe 설정에서)
                symbol_name = symbol
                universe = self.config.get_symbols() if hasattr(self.config, 'get_symbols') else []
                for s in universe:
                    if s.get('code') == symbol:
                        symbol_name = s.get('name', symbol)
                        break
                
                # 실현 손익은 SELL일 때만 의미가 있음 (위 로직에서 realized_profit 계산됨)
                pnl = (exec_price - self.avg_entry_prices[symbol]) * exec_qty if order['type'] == 'SELL' else 0
                
                asyncio.create_task(self.notifier.notify_trade(
                    action=order['type'],
                    symbol=symbol,
                    name=symbol_name,
                    price=exec_price,
                    qty=exec_qty,
                    pnl=pnl
                ))

        elif msg_type == '취소확인':
            order['status'] = OrderState.CANCELLED
            order['unexecuted_qty'] = 0
            
            # [내부 예약 해제]
            if order['type'] == 'BUY':
                reserved = order.get('reserved_amt', 0)
                self.pending_buy_amount = max(0, self.pending_buy_amount - reserved)
                self.logger.debug(f"예약 해제(취소): -{reserved:,.0f} (남은 예약: {self.pending_buy_amount:,.0f})")
                
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

    # ─────────────────────────────────────────────────────
    # [신규] 실전 잔고 동기화 로직 (Real Balance Sync)
    # ─────────────────────────────────────────────────────
    async def fetch_real_balance(self) -> Optional[float]:
        """
        키움 REST API를 통해 실제 계좌의 총 자산 및 보유 종목을 동기화합니다.
        """
        import aiohttp
        import socket
        
        token = self.auth_manager.get_token() if self.auth_manager else None
        # [수정] 다양한 계좌번호 키 이름 대응 (ACCOUNT_NO, account_number 등)
        account_no = self.config.get("ACCOUNT_NO") or self.config.get("account_number") or ""
        
        if not token or not account_no:
            missing = []
            if not token: missing.append("토큰")
            if not account_no: missing.append("계좌번호")
            self.logger.warning(f"잔고 동기화 건너뜀: {', '.join(missing)} 정보가 없습니다.")
            return None

        # [수정] 키움증권 kt00018 (계좌평가잔고내역요청) API 규격 적용
        endpoint = f"{self.rest_base_url}/api/dostk/acnt"
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {token}",
            "api-id": "kt00018" 
        }
        # qry_tp: 1(합산), dmst_stex_tp: KRX
        body = {
            "qry_tp": "1",
            "dmst_stex_tp": "KRX"
        }

        try:
            connector = aiohttp.TCPConnector(family=socket.AF_INET, ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(endpoint, json=body, headers=headers, timeout=5.0) as resp:
                    if resp.status == 200:
                        res_data = await resp.json(content_type=None)
                        self.logger.debug(f"잔고 조회 API 응답: {res_data}")
                        
                        if str(res_data.get('return_code')) == '0':
                            # output이 딕셔너리일 수도 있고, 바로 데이터가 있을 수도 있음
                            output = res_data.get('output', {})
                            if not output: output = res_data # 폴백
                            
                            # 1. 총 자산 (Equity) 파싱
                            candidates = ['prsm_dpst_aset_amt', 'estm_dpst_ast_amt', 'tot_evlt_amt']
                            balance = 0
                            for key in candidates:
                                val = output.get(key)
                                if val is not None:
                                    try:
                                        balance = float(val)
                                        if balance > 0: break
                                    except: continue
                            
                            # 2. 실제 주문 가능 현금 계산 (Total Orderable Cash)
                            # 공식: 순자산 - (주식 평가액 - 신용 융자액)
                            tot_evlt_amt = float(output.get('tot_evlt_amt', 0))
                            tot_loan_amt = float(output.get('tot_crd_loan_amt', 0)) + float(output.get('tot_loan_amt', 0))
                            stock_equity = max(0, tot_evlt_amt - tot_loan_amt)
                            
                            # API에서 puse_amt를 직접 주면 그것을 사용, 없으면 계산
                            self._broker_orderable_cash = float(output.get('puse_amt') or (balance - stock_equity))
                            self.logger.debug(f"증권사 확인 현금: {self._broker_orderable_cash:,.0f} 원")

                            # [신규] 동기화 시점에 체결 완료된 건들에 대해 예약 금액 정산 가능하지만,
                            # 여기서는 단순하게 API 값을 기준점으로 잡고, pending_buy_amount는 
                            # 주문 시점과 체결/취소 시점에만 관리하여 오차를 최소화합니다.

                            # 3. 보유 종목 동기화 (acnt_evlt_remn_indv_tot)
                            holdings_list = res_data.get('acnt_evlt_remn_indv_tot', [])
                            if holdings_list:
                                new_holdings = {}
                                new_avg_prices = {}
                                for item in holdings_list:
                                    raw_code = item.get('stk_cd', '')
                                    # 'A323410' -> '323410'
                                    code = raw_code[1:] if raw_code.startswith('A') else raw_code
                                    qty = int(float(item.get('rmnd_qty', 0)))
                                    price = float(item.get('pur_pric', 0))
                                    
                                    if code:
                                        new_holdings[code] = new_holdings.get(code, 0) + qty
                                        new_avg_prices[code] = price # 가중 평균이 필요할 수 있으나 단순화
                                
                                # 기존 보유 정보 업데이트 (새 리스트에 없으면 0으로 처리)
                                all_symbols = set(list(self.holdings.keys()) + list(new_holdings.keys()))
                                for sym in all_symbols:
                                    qty = new_holdings.get(sym, 0)
                                    price = new_avg_prices.get(sym, 0.0)
                                    
                                    self.holdings[sym] = qty
                                    self.bot_holdings[sym] = qty
                                    if qty > 0:
                                        self.avg_entry_prices[sym] = price
                                    else:
                                        self.avg_entry_prices[sym] = 0.0
                                        
                                self.logger.info(f"📊 보유 종목 {len(new_holdings)}개 동기화 완료 (그 외 종목 0 처리)")

                            if balance > 0:
                                return balance
                            else:
                                self.logger.warning(f"잔고 필드를 찾을 수 없거나 값이 0입니다. 응답 키: {list(output.keys())}")
                        else:
                            self.logger.error(f"잔고 조회 API 오류: {res_data.get('return_msg')}")
                    else:
                        self.logger.error(f"잔고 조회 HTTP 오류: {resp.status}")
        except Exception as e:
            self.logger.error(f"잔고 조회 중 예외 발생: {e}")
            
        return None

    async def sync_balance(self, force: bool = False):
        """
        현재 매매 모드에 따라 잔고를 동기화합니다.
        MOCK_MODE(virtual)일 때는 내부 계산을 유지하고, REAL_MODE일 때만 API를 호출합니다.
        """
        kiwoom_config = self.config.get("kiwoom", {})
        mode = kiwoom_config.get("trading_mode", "virtual")
        self.logger.debug(f"sync_balance 호출됨 (mode={mode}, force={force})")
        
        if mode == "real":
            # 30초 주기로 동기화 (force가 아닐 경우)
            now = time.time()
            if not force and now - self.last_sync_time < 30:
                return

            real_balance = await self.fetch_real_balance()
            if real_balance is not None:
                diff = real_balance - self.current_balance
                if abs(diff) > 1: # 1원 이상의 차이가 있을 때만 로깅
                    self.logger.info(f"🔄 잔고 동기화 완료: {self.current_balance:,.0f} -> {real_balance:,.0f} (오차: {diff:,.0f})")
                
                self.current_balance = real_balance
                self.last_sync_time = now
                # 시그널 발생 (총자산과 주문가능현금 함께 전달)
                self.signals.balance_synced.emit(self.current_balance)
        else:
            # 가상 매매 모드에서는 동기화 시그널만 발생 (내부 계산 유지)
            self.signals.balance_synced.emit(self.current_balance)
