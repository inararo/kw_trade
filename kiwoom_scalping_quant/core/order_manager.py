import asyncio
import time
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from utils.math_jit import get_valid_tick_price
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
    def __init__(self, config: Dict[str, Any], auth_manager=None, telegram_notifier=None, firebase_manager=None, account_service=None):
        self.config = config
        self.auth_manager = auth_manager
        self.notifier = telegram_notifier
        self.firebase_manager = firebase_manager  # [Firebase] Firestore 연동 매니저
        self.account_service = account_service    # [Shared Core] 서비스
        self.logger = logging.getLogger("OrderManager")
        self.logger.info(f"🛠️ [OrderManager] 초기화 완료 (AccountService 주입 여부: {self.account_service is not None})")
        self.signals = OrderSignals()

        # 고유 주문 ID(내부)를 키로, 상태 딕셔너리를 값으로 가지는 중앙 추적기
        self.active_orders: Dict[str, Dict[str, Any]] = {}

        # 실시간 자산 관리 (초기 자산에서 시작)
        self.current_balance = float(config.get("initial_balance", 10000000)) # 총 자산 (Equity)
        self._broker_orderable_cash = self.current_balance                    # 증권사 확인 현금
        self.pending_buy_amount = 0.0                                          # 주문 전송 후 체결 대기 중인 금액 (내부 예약)
        self.last_sync_time = 0
        self.last_known_prices: Dict[str, float] = {}                          # [신규] 잔고 조회 시 확인된 마지막 현재가

        # 키움증권 원주문번호(Broker ID)와 내부 ID 맵핑
        self.broker_id_map: Dict[str, str] = {}

        # [영구 저장] 봇 관리 종목 데이터 경로
        self.data_dir = os.path.join(os.getcwd(), "data")
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.holdings_file = os.path.join(self.data_dir, "bot_holdings.json")
        
        # 1. 봇(Agent) 전용 매수/보유 수량 트래킹 (파일에서 로드)
        self.bot_holdings: Dict[str, int] = self._load_bot_holdings()

        # 2. 실시간 잔고 및 평균단가 맵 초기화 (유니버스 + 보유 종목 합산)
        universe_codes = [str(s.get('code')).split('_')[0] for s in config.get('universe', [])]
        initial_codes = set(universe_codes) | set(self.bot_holdings.keys())
        
        self.holdings = {code: self.bot_holdings.get(code, 0) for code in initial_codes if code}
        self.avg_entry_prices = {code: 0.0 for code in initial_codes if code}

        # Safety Guard Risk Manager
        self.risk_manager = None # Will be injected
        self._sync_lock = asyncio.Lock() # [추가] 잔고 동기화 레이스 컨디션 방지 락
        self._last_real_sync_time = 0.0  # [추가] 실전 잔고 동기화 쿨타임 체크용
        self.daily_realized_pnl = 0.0    # 당일 매도 확정 손익 (실현)
        self.daily_evaluation_pnl = 0.0  # 현재 보유 종목 평가 손익 (미실현)
        
        # [수정] 키움 REST API Base URL 설정 (config.yaml 기반)
        # 1순위: kiwoom.rest_base_url.real/virtual
        # 2순위: kiwoom.rest_url (레거시 지원)
        # 3순위: 하드코딩된 기본값
        kiwoom_cfg = config.get("kiwoom", {})
        trading_mode = kiwoom_cfg.get("trading_mode", "real")
        
        base_url_config = kiwoom_cfg.get("rest_base_url")
        if isinstance(base_url_config, dict):
            self.rest_base_url = base_url_config.get(trading_mode, "https://api.kiwoom.com")
        else:
            self.rest_base_url = base_url_config or kiwoom_cfg.get("rest_url", "https://api.kiwoom.com")
            
        self.logger.info(f"🚀 [REST API] 설정된 서버 주소 ({trading_mode} 모드): {self.rest_base_url}")
        
        # [추가] 실전 잔고 상태 및 쿨다운 관리 변수
        self._last_sell_fill_time: Dict[str, float] = {}  # 매도 직후 API 지연 방어용

        # [추가] 주문 제한(Throttling) 및 중복 취소 방지
        self.rate_limit = config.get("rate_limit", 5)
        self.order_semaphore = asyncio.Semaphore(self.rate_limit)
        self.order_timestamps = []
        self._cancelling_orders = set() # [추가] 중복 취소 전송 방지용

        # [추가] 글로벌 리스크 제한 (RiskManager 주입 전 기본값)
        self.global_max_loss = config.get("global_max_loss", -500000)
        self.global_max_exposure = config.get("global_max_exposure", 50000000)

    def has_unexecuted_orders(self, symbol: str) -> bool:
        """특정 종목에 대해 아직 체결/취소되지 않은 주문이 있는지 확인합니다."""
        for order in self.active_orders.values():
            if order.get('symbol') == symbol and order.get('status') in [OrderState.PENDING, OrderState.ACCEPTED, OrderState.PARTIAL]:
                if order.get('unexecuted_qty', 0) > 0:
                    return True
        return False

    @property
    def orderable_cash(self) -> float:
        """내부 예약 금액을 제외한 실제 가용 현금"""
        val = max(0, self._broker_orderable_cash - self.pending_buy_amount)
        if val <= 0 and self._broker_orderable_cash > 0:
            self.logger.warning(f"⚠️ 주문 가능 현금이 0입니다. (Broker: {self._broker_orderable_cash:,.0f}, Pending: {self.pending_buy_amount:,.0f})")
        return val

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
            print(f"Global Risk: 일일 최대 손실({self.global_max_loss}) 초과. 신규 진입 차단.")
            return False

        # 총 노출 금액 = 모든 종목의 (보유 수량 * 현재가(또는 진입가))
        # 여기서는 단순화를 위해 현재 주문가격을 해당 종목의 현재가로 취급하여 노출 금액을 추산
        current_exposure = 0
        for sym, holding_qty in self.holdings.items():
            if sym == symbol:
                current_exposure += holding_qty * price
            else:
                current_exposure += holding_qty * self.avg_entry_prices.get(sym, 0)

        limit = self.config.get("global_max_exposure", 50000000)
        if current_exposure + (price * qty) > limit:
            print(f"Global Risk: 최대 노출 금액({limit}) 초과. 신규 진입 차단.")
            return False

        return True

    @future_safe
    async def send_order(self, order_type: str, symbol: str, price: int, qty: int, orig_order_no: str = "") -> str:
        """
        REST API를 통한 주문 발송 (신규/정정/취소)
        orig_order_no가 있으면 정정/취소 주문으로 간주.
        """
        # [안전장치] 모든 주문 가격을 유효한 호가 단위(Tick Size)로 강제 보정
        original_price = price
        price = get_valid_tick_price(float(price), order_type)
        
        # [핵심] 키움 REST API는 순수 숫자 종목 코드만 허용함
        clean_symbol = symbol.split('_')[0].strip()

        if original_price != price and original_price != 0:
            self.logger.warning(f"⚠️ 주문 가격 보정 발생: {original_price} -> {price} (호가 단위 준수)")

        if not orig_order_no:
            # [🚨 신규] 매수 시작 시간 제한 검사 (BUY에만 적용)
            if order_type == "BUY":
                # 장외 시간 테스트 모드(BYPASS_MARKET_HOURS)가 참이 아니면 제한 검사 실행
                bypass_market = self.config.get("BYPASS_MARKET_HOURS", False) if hasattr(self.config, "get") else False
                if not bypass_market:
                    buy_start_str = self.config.get("buy_start_time") if hasattr(self.config, "get") else None
                    if buy_start_str:
                        try:
                            current_time = datetime.now().time()
                            if len(buy_start_str.split(":")) == 2:
                                buy_start_time = datetime.strptime(buy_start_str, "%H:%M").time()
                            else:
                                buy_start_time = datetime.strptime(buy_start_str, "%H:%M:%S").time()
                            
                            if current_time < buy_start_time:
                                msg = f"BUY_TIME_RESTRICTED: 현재 시각({current_time.strftime('%H:%M:%S')})이 매수 허용 최소 시간({buy_start_str}) 이전입니다. 매수 주문 전송을 생략(스킵)합니다."
                                self.logger.warning(f"🚫 [매수 주문 제한] {msg}")
                                raise Exception(msg)
                        except Exception as e:
                            if "BUY_TIME_RESTRICTED" in str(e):
                                raise
                            self.logger.warning(f"매수 시작 시간 검사 중 오류: {e}")

            if self.risk_manager and not self.risk_manager.can_order(symbol, price * qty, order_type):
                raise Exception(f"RiskManager: 글로벌 세이프티 가드 제한으로 인해 신규 주문({order_type} {symbol})이 거부되었습니다.")
            elif not self.risk_manager and not self._check_global_risk(symbol, price, qty, order_type):
                raise Exception("글로벌 리스크 점검 실패로 주문이 거부되었습니다.")

        await self._throttle_order()

        # [사전 필터] 매수 주문 시 주문 가능 금액(kt00010) 확인
        if order_type == "BUY" and self.account_service:
            try:
                # 주문 직전 최신 가용 자금 동기화
                await asyncio.wait_for(self.account_service.sync_all(), timeout=5.0)
                if not self.account_service.can_afford(price * qty):
                    order_amt = price * qty
                    available = self.account_service.orderable_cash
                    print(f"❌ [자금 부족] 주문 가능 금액 초과: 필요 {order_amt:,.0f}원 / 가용 {available:,.0f}원")
                    raise Exception(f"INSUFFICIENT_FUNDS: Required {order_amt:,.0f}, Available {available:,.0f}")
            except asyncio.TimeoutError:
                self.logger.warning("⚠️ 주문 가능 금액 동기화 타임아웃. 내부 잔고 기준으로 계속 진행합니다.")
            except Exception as e:
                if "INSUFFICIENT_FUNDS" in str(e): raise
                self.logger.warning(f"계좌 동기화 중 오류 발생: {e}")

        async with self.order_semaphore:
            internal_id = f"INT_{int(time.time() * 1000)}"

            # --- Signal Only Bypass Logic ---
            signal_only = False
            # Check config manager if available, else standard config dict fallback
            if hasattr(self.config, 'get'):
                signal_only = self.config.get('signal_only_mode', False)

            if signal_only:
                msg = f"[SIGNAL ONLY] 🔴 {order_type}: {symbol} ({qty}주 @ {price}) - 실제 주문 생략됨"
                self.logger.warning(msg)

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
                'symbol': clean_symbol,
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
                # 시장가(price=0)인 경우 현재가 또는 마지막 확인가로 예약금 추정
                calc_price = price if price > 0 else self.last_known_prices.get(clean_symbol, 0)
                if calc_price <= 0:
                    # 현재가 정보도 없는 경우(부팅 직후 등) 안전을 위해 0으로 두거나 에러 처리
                    # 여기서는 0으로 두되 로그 출력
                    self.logger.warning(f"⚠️ {clean_symbol} 시장가 주문 예약금 추정 실패 (현재가 정보 없음)")
                
                order_amt = calc_price * qty
                self.active_orders[internal_id]['reserved_amt'] = order_amt
                self.pending_buy_amount += order_amt
                self.logger.debug(f"주문 예약: +{order_amt:,.0f} (총 예약: {self.pending_buy_amount:,.0f})")

            self.logger.warning(f"📤 주문 전송: {order_type} {qty}주 @ {price}원 (Internal ID: {internal_id})")

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
                self.logger.warning("❌ 유효한 API 토큰이 없어 주문을 거절합니다. 토큰 갱신을 확인하세요.")
                self.active_orders[internal_id]['status'] = OrderState.FAILED
                raise Exception("API_TOKEN_MISSING")

            if not account_no:
                self.logger.error("❌ 계좌번호(ACCOUNT_NO)가 설정되지 않아 주문을 거절합니다.")
                self.active_orders[internal_id]['status'] = OrderState.FAILED
                raise Exception("ACCOUNT_NO_MISSING")

            # 2. 주문 종류별 api-id 및 엔드포인트 결정
            endpoint = f"{self.rest_base_url}/api/dostk/ordr"
            
            # [핵심] 키움 REST API는 순수 숫자 종목 코드만 허용함
            clean_symbol = symbol.split('_')[0]

            if order_type == "BUY":
                api_id = 'kt10000'
                # trde_tp: '0'=지정가, '3'=시장가
                trde_tp = '0' if price > 0 else '3'
                body = {
                    "cano":         str(account_no),
                    "dmst_stex_tp": 'KRX',  # 국내거래소 구분 필수, 예시로 KRX 고정
                    "stk_cd":       clean_symbol,
                    "ord_qty":      str(qty),
                    "ord_uv":       str(price) if price > 0 else '',
                    "trde_tp":      trde_tp,
                }

            elif order_type == "SELL":
                api_id = 'kt10001'
                trde_tp = '0' if price > 0 else '3'
                body = {
                    "cano":         str(account_no),
                    "dmst_stex_tp": 'KRX',
                    "stk_cd":       clean_symbol,
                    "ord_qty":      str(qty),
                    "ord_uv":       str(price) if price > 0 else '',
                    "trde_tp":      trde_tp,
                }

            elif order_type == "REPLACE":
                api_id = 'kt10002'
                body = {
                    "cano":          str(account_no),
                    "dmst_stex_tp": 'KRX',
                    "orig_ord_no":   int(orig_order_no) if str(orig_order_no).isdigit() else orig_order_no,
                    "stk_cd":        clean_symbol,
                    "mdfy_qty":      str(qty),
                    "mdfy_uv":       str(price) if price > 0 else '',
                    "mdfy_cond_uv":  '', # 정정 조건 가격 (필요시 추가)
                }

            elif order_type == "CANCEL":
                api_id = 'kt10003'
                body = {
                    "cano":        str(account_no),
                    "dmst_stex_tp": 'KRX',
                    "orig_ord_no": int(orig_order_no) if str(orig_order_no).isdigit() else orig_order_no,
                    "stk_cd":      clean_symbol,
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

            print(f"📤 주문 전송 [{api_id}] {order_type} {symbol} {qty}주 @ {price:,}원")
            self.logger.debug(f"   └ Body: {body}")

            # 4. 비동기 HTTP POST (키움 증권사 서버망)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        endpoint, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=10.0)
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
                            self.logger.warning(f"❌ 키움 주문 거부: [{return_code}] {return_msg}")
                            
                            # [핵심 패치] 매도가능수량 부족 시 잔고 강제 동기화 (무한 매도 시도 방지)
                            if "매도가능수량" in return_msg or "800033" in return_msg:
                                self.logger.warning(f"⚠️ [잔고 불일치] 브로커와 엔진의 수량이 다릅니다. {symbol} 보유 수량을 0으로 강제 초기화합니다.")
                                self.holdings[symbol] = 0
                                if symbol in self.bot_holdings:
                                    self.bot_holdings[symbol] = 0
                                # 실잔고 다시 불러오기 (백그라운드)
                                asyncio.create_task(self.fetch_real_balance())

                            self.active_orders[internal_id]['status'] = OrderState.FAILED
                            self.active_orders[internal_id]['ack_event'].set()
                            
                            # [내부 예약 해제]
                            if order_type == "BUY":
                                reserved = self.active_orders[internal_id].get('reserved_amt', 0)
                                self.pending_buy_amount = max(0, self.pending_buy_amount - reserved)
                                self.logger.debug(f"예약 해제(거부): -{reserved:,.0f} (남은 예약: {self.pending_buy_amount:,.0f})")
                            
                            raise Exception(f"KIWOOM_ORDER_REJECTED: {return_msg}")

            except asyncio.TimeoutError:
                self.logger.warning(f"⏰ 키움 API 응답 타임아웃 (10초 초과)! ID: {internal_id}")
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
                    self.logger.warning(f"🔥 주문 전송 중 예외 발생: {e}")
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
            self.logger.warning(f"취소 실패: 내부 ID {internal_id}를 찾을 수 없습니다.")
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

        print(f"🚫 미체결 취소 요청 시작: {order['symbol']} | 원주문번호: {broker_id} | 취소수량: {unexecuted_qty}")
        
        # kt10003 취소 주문 실행 (send_order의 CANCEL 타입 활용)
        try:
            # 취소 주문은 가격(price)이 의미가 없으므로 0으로 전송 (kt10003 규격 준수)
            result = await self.send_order("CANCEL", order['symbol'], 0, unexecuted_qty, orig_order_no=broker_id)
            return True
        except Exception as e:
            self.logger.warning(f"취소 주문 전송 중 오류 발생: {e}")
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
                self.logger.warning(f"주문 응답 타임아웃 (3초 초과)! 실패 처리됨. ID: {internal_id}")

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

    async def on_receive_chejan_data(self, data: Dict[str, Any]):
        """
        키움 웹소켓(또는 REST 폴링)에서 수신된 실시간 체결/잔고 데이터 파싱.
        """
        # [디버그] 수신 데이터 확인
        self.logger.warning(f"📡 [Chejan Raw] {data}")

        raw_symbol = data.get('symbol') or data.get('stk_cd') or ""
        clean_symbol = str(raw_symbol).split('_')[0].strip()
        internal_id = data.get('internal_id')
        broker_id = str(data.get('broker_id', '')) # 문자열 정규화
        msg_type = data.get('msg_type') # '접수', '체결', '취소확인' 등

        order = self.active_orders.get(internal_id)
        if not order and broker_id:
            # broker_id로 역추적
            internal_id = self.broker_id_map.get(broker_id)
            order = self.active_orders.get(internal_id)
            
        if not order:
            self.logger.warning(f"⚠️ [Chejan] 매칭되는 주문을 찾을 수 없습니다. (Internal ID: {internal_id}, Broker ID: {broker_id})")
            return

        # [수정] 모든 내부 처리에 정제된 심볼 사용
        symbol = order['symbol'].split('_')[0]

        if msg_type == '접수':
            order['status'] = OrderState.ACCEPTED
            order['broker_id'] = broker_id
            self.broker_id_map[broker_id] = internal_id
            order['ack_event'].set() # 타임아웃 해제
            self.logger.info(f"브로커 접수 완료. (Broker ID: {broker_id})")

            # [핵심 수정] 접수 완료 시 내부 예약금 즉시 해제 (브로커가 이미 차감함)
            if order['type'] == 'BUY':
                reserved = order.get('reserved_amt', 0)
                # 이미 해제되었을 수도 있으므로(recalculate 등) 체크 후 차감
                if reserved > 0:
                    self.pending_buy_amount = max(0, self.pending_buy_amount - reserved)
                    self.logger.info(f"💰 [예약 해제] 접수 확인으로 인한 예약금 해제: -{reserved:,.0f} (남은 예약: {self.pending_buy_amount:,.0f})")
                    order['reserved_amt'] = 0 # 중복 해제 방지

            # [Firebase] 주문 접수 로그 전송 (연동 확인용)
            if self.firebase_manager:
                asyncio.create_task(self.firebase_manager.add_trade_log({
                    "log_type":      order['type'] + "_RECEIPT",
                    "symbol":        symbol,
                    "symbol_name":   self.config.get_symbol_name(symbol) if hasattr(self.config, 'get_symbol_name') else symbol,
                    "price":         float(order.get('price', 0)),
                    "qty":           int(order.get('qty', 0)),
                    "profit_loss":   0.0,
                    "profit_rate":   0.0,
                    "timestamp_str": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "broker_id":     broker_id
                }))

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
                self.bot_holdings[symbol] = self.bot_holdings.get(symbol, 0) + exec_qty
                
                # [실시간 반영] 가용 현금 차감
                exec_amt = exec_qty * exec_price
                self._broker_orderable_cash -= exec_amt
                self.current_balance -= exec_amt
                
                # 데이터 영구 저장
                self._save_bot_holdings()

            elif order['type'] == 'SELL':
                self.holdings[symbol] -= exec_qty

                # 봇이 청산한 수량 감소 (0 미만으로 떨어지지 않게 방어)
                self.bot_holdings[symbol] = max(0, self.bot_holdings[symbol] - exec_qty)
                if self.bot_holdings[symbol] == 0:
                    del self.bot_holdings[symbol]

                # 데이터 영구 저장
                self._save_bot_holdings()

                # [실시간 반영] 가용 현금 가산 및 총자산 갱신
                exec_amt = exec_qty * exec_price
                self._broker_orderable_cash += exec_amt
                self.current_balance += exec_amt

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
                    self.logger.warning(f"🎯 [{symbol}] {order['type']} 체결 완료 ({exec_qty}주, 잔량: {order['unexecuted_qty']}주)")
                    
                    # [Firebase] 실시간 매매 로그 전송
                    symbol_name = symbol
                    universe = self.config.get_symbols() if hasattr(self.config, 'get_symbols') else []
                    for s in universe:
                        if s.get('code') == symbol:
                            symbol_name = s.get('name', symbol)
                            break
                    
                    avg_price = self.avg_entry_prices.get(symbol, 0)
                    pnl = (exec_price - avg_price) * exec_qty if order['type'] == 'SELL' and avg_price > 0 else 0
                    pnl_rate = ((exec_price - avg_price) / avg_price) * 100 if order['type'] == 'SELL' and avg_price > 0 else 0
                    
                    # [핵심] 실현 손익 즉시 업데이트 (로컬 누적) - 중복 방지 플래그 체크
                    if pnl != 0 and not order.get('pnl_processed'):
                        self.daily_realized_pnl += pnl
                        order['pnl_processed'] = True
                        if self.risk_manager:
                            self.risk_manager.update_pnl(pnl)
                        self.logger.warning(f"💰 실현 손익 업데이트(Chejan): {pnl:,.0f}원 (당일 누적: {self.daily_realized_pnl:,.0f}원)")
                    
                    trade_data = {
                        "log_type":      order['type'],
                        "symbol":        symbol,
                        "symbol_name":   symbol_name,
                        "price":         float(exec_price),
                        "qty":           int(exec_qty),
                        "profit_loss":   float(pnl),
                        "profit_rate":   float(pnl_rate),
                        "timestamp_str": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    
                    if self.firebase_manager:
                        self.logger.warning(f"📤 [Firebase 전송 시도] {symbol} {order['type']} {exec_qty}주 @ {exec_price}")
                        asyncio.create_task(self.firebase_manager.add_trade_log(trade_data))
                    else:
                        self.logger.warning(f"⚠️ [Firebase 전송 건너뜀] FirebaseManager가 주입되지 않았습니다. ({symbol})")
                    
                    # [신규] 매도 완료 시 시간 기록 및 실현 손익 동기화
                    if order['type'] == 'SELL':
                        self._last_sell_fill_time[symbol] = time.time()
                        # [핵심] 매도 완료 직후 실제 실현 손익 동기화 (ka10077)
                        if self.account_service:
                            async def deferred_sync():
                                await asyncio.sleep(1.0) # 체결 후 정산 반영 대기
                                await self.account_service.sync_all()
                                summary = self.account_service.get_summary()
                                self.daily_realized_pnl = summary.get("today_realized_profit", 0.0)
                                self.logger.info(f"🔄 [손익 동기화 완료] 실전 데이터 반영: {self.daily_realized_pnl:,.0f}원")
                            
                            asyncio.create_task(deferred_sync())
                else:
                    order['status'] = OrderState.PARTIAL
                    self.logger.info(f"주문 부분 체결 (Broker ID: {broker_id}, 잔여: {order['unexecuted_qty']})")
                    self.handle_partial_fill(order)
                
            # [내부 예약 해제] - 체결 시점 (접수 단계에서 이미 해제되지 않은 경우만)
            if order['type'] == 'BUY' and exec_qty > 0:
                total_qty = order.get('qty', 1)
                total_reserved = order.get('reserved_amt', 0)
                
                if total_reserved > 0:
                    release_amt = (exec_qty / total_qty) * total_reserved
                    self.pending_buy_amount = max(0, self.pending_buy_amount - release_amt)
                    order['reserved_amt'] = max(0, total_reserved - release_amt)
                    self.logger.debug(f"예약 해제(체결): -{release_amt:,.0f} (남은 예약: {self.pending_buy_amount:,.0f})")

            # 부분 체결 시 상태 업데이트 (SELL 외 타입 대응)
            if order['status'] not in [OrderState.FILLED, OrderState.CANCELLED, OrderState.FAILED] and order['unexecuted_qty'] > 0:
                order['status'] = OrderState.PARTIAL
                if order['type'] != 'SELL': # SELL은 위에서 이미 처리함
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

            # [Firebase] 체결 로그 Firestore 전송 (테스트 스키마와 통합)
            if self.firebase_manager:
                self.logger.warning(f"🚀 [Firebase] 전송 로직 진입 ({symbol})")
                # 1. 추가 정보 수집
                symbol_name = symbol
                universe = self.config.get_symbols() if hasattr(self.config, 'get_symbols') else []
                for s in universe:
                    if s.get('code') == symbol:
                        symbol_name = s.get('name', symbol)
                        break
                
                avg_price = self.avg_entry_prices.get(symbol, 0)
                pnl = (exec_price - avg_price) * exec_qty if order['type'] == 'SELL' and avg_price > 0 else 0
                
                # 2. 테스트 스키마에 맞춘 딕셔너리 구성
                trade_data = {
                    "log_type":      order['type'],      # "BUY" 또는 "SELL" (필드명 통합)
                    "symbol":        symbol,
                    "symbol_name":   symbol_name,        # 종목명 추가
                    "price":         float(exec_price),
                    "qty":           int(exec_qty),      # "quantity" -> "qty" (필드명 통합)
                    "profit_loss":   float(pnl),         # 실현 손익 유지
                    "timestamp_str": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), # 시각 문자열 추가
                }

                # 3. 비동기 업로드
                if self.firebase_manager:
                    self.logger.warning(f"📤 [Firebase 전송 시도] {symbol} {order['type']} {exec_qty}주 @ {exec_price}")
                    asyncio.create_task(self.firebase_manager.add_trade_log(trade_data))
                else:
                    self.logger.warning(f"⚠️ [Firebase 전송 건너뜀] FirebaseManager가 주입되지 않았습니다. ({symbol})")

        elif msg_type == '취소확인':
            order['status'] = OrderState.CANCELLED
            order['unexecuted_qty'] = 0
            
            # [내부 예약 해제]
            if order['type'] == 'BUY':
                reserved = order.get('reserved_amt', 0)
                if reserved > 0:
                    self.pending_buy_amount = max(0, self.pending_buy_amount - reserved)
                    self.logger.debug(f"예약 해제(취소): -{reserved:,.0f} (남은 예약: {self.pending_buy_amount:,.0f})")
                    order['reserved_amt'] = 0
                
            self.logger.info(f"주문 취소 완료. (Broker ID: {broker_id})")

        elif msg_type == '정정확인':
            order['status'] = OrderState.REPLACED
            self.logger.info(f"주문 정정 완료. (Broker ID: {broker_id})")

    def handle_partial_fill(self, order: Dict[str, Any]):
        """
        부분 체결 발생 시 미체결 잔량을 즉시 시장가 취소(또는 정정)하여 포지션 꼬임 방지.
        """
        internal_id = order.get('internal_id')
        if not internal_id: return

        if order['unexecuted_qty'] > 0:
            # [중복 방지] 이미 취소 주문이 전송 중이면 건너뜀
            if internal_id in self._cancelling_orders:
                return

            self.logger.warning(f"부분 체결 감지! 잔여 수량({order['unexecuted_qty']}주) 긴급 취소 진행. (Broker ID: {order['broker_id']})")
            
            # 취소 주문 전송 태스크 생성
            async def do_cancel():
                self._cancelling_orders.add(internal_id)
                try:
                    await self.send_order(
                        order_type="CANCEL",
                        symbol=order['symbol'],
                        price=0,
                        qty=order['unexecuted_qty'],
                        orig_order_no=order['broker_id']
                    )
                finally:
                    # 전송 후 일정 시간(예: 1초) 후에 셋에서 제거하여 과도한 재전송 방지
                    await asyncio.sleep(1.0)
                    if internal_id in self._cancelling_orders:
                        self._cancelling_orders.remove(internal_id)

            asyncio.create_task(do_cancel())

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

    async def emergency_liquidate(self):
        """
        [Firebase 원격 명령] 패닉 셀: 모든 미체결 주문을 즉시 취소 후
        보유 중인 모든 종목을 시장가로 전량 매도합니다.
        """
        self.logger.critical("🚨 [PANIC SELL] 긴급 청산 명령 수신! 즉시 모든 포지션을 정리합니다.")

        # 1단계: 미체결 주문 전량 취소
        await self.cancel_all_orders()

        # 2단계: 보유 수량이 0보다 큰 종목만 추출
        active_holdings = {sym: qty for sym, qty in self.holdings.items() if qty > 0}

        if not active_holdings:
            self.logger.info("[PANIC SELL] 정리할 보유 종목이 없습니다.")
            return

        # 3단계: 모든 보유 종목 시장가 매도 (price=0 → kt10001 trde_tp='3' 시장가)
        tasks = []
        for symbol, qty in active_holdings.items():
            self.logger.warning(f"[PANIC SELL] 시장가 매도: {symbol} {qty}주")
            tasks.append(self.send_order("SELL", symbol, 0, qty))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.logger.critical(f"[PANIC SELL] {len(tasks)}개 종목 청산 주문 전송 완료. 결과: {results}")

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
        # qry_tp: 2(개별), dmst_stex_tp: KRX(한국거래소)
        body = {
            "cano": str(account_no),
            "qry_tp": "2",
            "dmst_stex_tp": "KRX"
        }

        def safe_float(v):
            try:
                if v is None: return 0.0
                s = str(v).replace(',', '').strip()
                if not s or s == 'None': return 0.0
                return float(s)
            except: return 0.0

        try:
            endpoint = f"{self.rest_base_url}/api/dostk/acnt"
            self.logger.debug(f"[DEBUG] 잔고 조회 요청: {endpoint} | Body: {body}")
            
            connector = aiohttp.TCPConnector(family=socket.AF_INET, ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(endpoint, json=body, headers=headers, timeout=10.0) as resp:
                    # 응답 텍스트를 먼저 읽어 로깅 (JSON 파싱 에러 대비)
                    res_text = await resp.text()
                    # [디버깅] 잔고 조회 원본 응답을 에러 레벨로 출력하여 강제 확인
                    self.logger.debug(f"🔍 [잔고 조회 원본 응답] HTTP {resp.status}: {res_text}")

                    if resp.status == 200:
                        try:
                            res_data = json.loads(res_text)
                        except Exception as json_e:
                            self.logger.error(f"❌ 응답 JSON 파싱 실패: {json_e}")
                            return None
                        
                        if str(res_data.get('return_code')) == '0':
                            # [안정화] 응답 데이터의 계층 구조를 유연하게 탐색하기 위해 검색 함수 정의
                            def find_val(data, keys):
                                if isinstance(data, dict):
                                    for k in keys:
                                        if k in data and data[k] is not None:
                                            return safe_float(data[k])
                                    for v in data.values():
                                        res = find_val(v, keys)
                                        if res is not None: return res
                                elif isinstance(data, list):
                                    for item in data:
                                        res = find_val(item, keys)
                                        if res is not None: return res
                                return None

                            # 2. 총 자산 (Equity) 파싱
                            balance_candidates = ['prsm_dpst_aset_amt', 'tot_evlt_amt', 'tot_asst_amt', 'tot_evlt_amt_outpt', 'evlt_amt_tot']
                            balance = find_val(res_data, balance_candidates) or self.current_balance
                            
                            self.logger.warning(f"📊 [파싱 결과] 총 자산: {balance:,.0f}")
                            
                            # [신규] AccountService에 최신 총 자산 데이터 강제 주입 (Firebase 연동용)
                            if self.account_service:
                                self.account_service.total_assets = balance
                                # 백그라운드 태스크로 즉시 Firebase 동기화 트리거
                                asyncio.create_task(self.account_service._sync_to_firebase())

                            # 3. 당일 실현 손익 파싱
                            # [핵심] AccountService가 가져온 공식 실현 손익을 최우선 신뢰합니다.
                            if self.account_service:
                                summary = self.account_service.get_summary()
                                self.daily_realized_pnl = summary.get("today_realized_profit", 0.0)
                                self.logger.info(f"✅ [손익 동기화] AccountService [ID:{id(self.account_service)}] 데이터 수신: 실현손익={self.daily_realized_pnl:,.0f}원 | 가용현금={summary.get('orderable_cash', 0):,.0f}원")
                            else:
                                pnl_candidates = ['thdt_dbt_shrt_asst_amt', 'tdy_afr_pnl_amt', 'tot_pnl_amt', 'thst_exca_amt']
                                daily_pnl = find_val(res_data, pnl_candidates) or 0.0
                                if daily_pnl != 0:
                                    self.daily_realized_pnl = daily_pnl
                            
                            # 평가 손익 파싱
                            evlt_pnl = find_val(res_data, ['tot_evlt_pl', 'evlt_pnl_amt']) or 0.0
                            self.daily_evaluation_pnl = evlt_pnl

                             # [신규] 융자/대출금 파싱 (신용 계좌 대응)
                            loan_amt = find_val(res_data, ['tot_crd_loan_amt', 'tot_loan_amt', 'crd_loan_amt']) or 0.0

                            # 4. 실제 주문 가능 현금 (Orderable Cash)
                            # [핵심] AccountService가 kt00010으로 가져온 정확한 가용 현금을 우선 사용합니다.
                            if self.account_service:
                                summary = self.account_service.get_summary()
                                cash_val = summary.get("orderable_cash", 0.0)
                                if cash_val > 0:
                                    self._broker_orderable_cash = cash_val
                                    self.logger.info(f"💳 [자금 동기화] AccountService [ID:{id(self.account_service)}] 기반 가용현금 동기화 완료: {self._broker_orderable_cash:,.0f}원")
                                else:
                                    self.logger.warning(f"⚠️ [자금 동기화] AccountService [ID:{id(self.account_service)}]에 가용현금 데이터가 아직 없거나 0입니다. 폴백 파싱을 시도합니다.")
                            else:
                                cash_candidates = ['puse_amt', 'd2_dpst_amt', 'dpst_amt', 'ord_psbl_amt', 'n_ord_psbl_amt', 'dnca_tot_amt']
                                cash_val = find_val(res_data, cash_candidates)
                                
                                if cash_val is not None and cash_val > 0:
                                    self._broker_orderable_cash = cash_val
                                    self.logger.warning(f"💰 [파싱 결과] 주문 가능 현금 발견: {cash_val:,.0f}")
                                else:
                                    tot_evlt_amt = find_val(res_data, ['tot_evlt_amt', 'evlt_amt_tot']) or 0.0
                                    net_equity_in_stocks = max(0, tot_evlt_amt - loan_amt)
                                    calculated_cash = max(0, balance - net_equity_in_stocks)
                                    
                                    self._broker_orderable_cash = calculated_cash
                                    self.logger.warning(f"⚠️ [파싱 결과] 가용 현금 필드 미발견 -> 자동 계산 적용")
                                    self.logger.warning(f"   (총자산 {balance:,.0f} - (평가액 {tot_evlt_amt:,.0f} - 융자 {loan_amt:,.0f})) = {calculated_cash:,.0f}")

                            # 5. 보유 종목 (Holdings) 파싱
                            holdings_list = res_data.get('output2') or res_data.get('items') or res_data.get('acnt_evlt_remn_indv_tot') or []
                            
                            # 계층 구조가 더 깊은 경우 전체 탐색
                            if not holdings_list:
                                for out_key in ['output', 'output2', 'items']:
                                    sub = res_data.get(out_key)
                                    if isinstance(sub, dict):
                                        holdings_list = sub.get('output2') or sub.get('items') or []
                                        if holdings_list: break
                                    elif isinstance(sub, list):
                                        holdings_list = sub
                                        break
                            
                            new_holdings = {}
                            new_avg_prices = {}
                            new_current_prices = {}
                            
                            if holdings_list:
                                self.logger.info(f"보유 종목 리스트 수신: {len(holdings_list)}개")
                                if len(holdings_list) > 0:
                                    self.logger.debug(f"첫 번째 종목 필드 확인: {list(holdings_list[0].keys())}")
                                
                                for item in holdings_list:
                                    # 종목코드 후보
                                    raw_code = item.get('stk_cd') or item.get('symbol') or item.get('stk_code') or item.get('code') or ''
                                    code = str(raw_code).lstrip('A').split('_')[0].strip()
                                    
                                    # 보유수량 후보
                                    qty = int(safe_float(item.get('rmnd_qty') or item.get('hldg_qty') or item.get('quantity') or item.get('qty')))
                                    
                                    # 매입단가 후보
                                    avg_p = safe_float(item.get('pur_pric') or item.get('pchs_avg_pric') or item.get('buy_price') or item.get('avg_price'))
                                    
                                    # 현재가 후보
                                    cur_p = safe_float(item.get('evlt_prc') or item.get('cur_prc') or item.get('stck_prpr') or item.get('price'))
                                    
                                    if code and qty > 0:
                                        new_holdings[code] = new_holdings.get(code, 0) + qty
                                        new_avg_prices[code] = avg_p
                                        new_current_prices[code] = cur_p
                                        self.logger.debug(f"[DEBUG] 종목 파싱 성공: {code} | 수량: {qty} | 평단: {avg_p:,.0f} | 현재가: {cur_p:,.0f}")
                                    else:
                                        self.logger.debug(f"[DEBUG] 종목 스킵: code={code}, qty={qty}")
                            
                            # 기존 보유 정보 업데이트
                            all_symbols = set(list(self.holdings.keys()) + list(new_holdings.keys()))
                            for sym in all_symbols:
                                qty = new_holdings.get(sym, 0)
                                price = new_avg_prices.get(sym, 0.0)
                                cur_p = new_current_prices.get(sym, 0.0)
                                
                                last_sell = self._last_sell_fill_time.get(sym, 0)
                                if qty > 0 and self.holdings.get(sym, 0) == 0 and (time.time() - last_sell < 60):
                                    qty = 0
                                    price = 0.0
                                    cur_p = 0.0

                                self.holdings[sym] = qty
                                self.bot_holdings[sym] = qty
                                self.avg_entry_prices[sym] = price if qty > 0 else 0.0
                                if cur_p > 0:
                                    self.last_known_prices[sym] = cur_p
                                    
                            self.logger.info(f"📊 잔고 동기화 완료: 총자산 {balance:,.0f}원 | 실현손익 {self.daily_realized_pnl:,.0f}원 | 평가손익 {self.daily_evaluation_pnl:,.0f}원 | 가용현금 {self._broker_orderable_cash:,.0f}원")
                            
                            # [추가] 잔고 동기화 시 예약금 자동 교정
                            self.recalculate_pending_amount()
                            
                            return balance
                        else:
                            self.logger.error(f"잔고 조회 API 오류: {res_data.get('return_msg')}")
                    elif resp.status == 429:
                        self.logger.warning("⚠️ 잔고 조회 과부하(429): API 요청 제한에 도달했습니다. 잠시 후 재시도합니다.")
                        await asyncio.sleep(1.0)
                    else:
                        self.logger.error(f"잔고 조회 HTTP 오류: {resp.status}")
        except Exception as e:
            import traceback
            self.logger.error(f"잔고 조회 중 예외 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            
        return None

    def recalculate_pending_amount(self):
        """
        현재 활성 주문 상태를 기반으로 pending_buy_amount를 재계산하여 정합성을 맞춥니다.
        브로커에 아직 도달하지 않은(PENDING) 주문들의 금액만 합산합니다.
        """
        new_pending = 0.0
        pending_count = 0
        for internal_id, order in self.active_orders.items():
            if order['type'] == 'BUY' and order['status'] == OrderState.PENDING:
                new_pending += order.get('reserved_amt', 0)
                pending_count += 1
        
        old_pending = self.pending_buy_amount
        self.pending_buy_amount = new_pending
        
        if abs(old_pending - new_pending) > 1:
            self.logger.info(f"🔄 [예약금 교정] {old_pending:,.0f} -> {new_pending:,.0f} (미확정 주문: {pending_count}건)")

    async def sync_balance(self, force: bool = False):
        """
        현재 매매 모드에 따라 잔고를 동기화합니다.
        MOCK_MODE(virtual)일 때는 내부 계산을 유지하고, REAL_MODE일 때만 API를 호출합니다.
        """
        kiwoom_config = self.config.get("kiwoom", {})
        mode = kiwoom_config.get("trading_mode", "virtual")
        self.logger.debug(f"sync_balance 호출됨 (mode={mode}, force={force})")
        
        if mode == "real":
            async with self._sync_lock:
                now = time.time()
                # 1. 일반 주기(30초) 체크
                if not force and now - self.last_sync_time < 30:
                    return

                # 2. 강제 동기화(force=True) 보호: 최소 2초 간격 유지
                if force and now - self._last_real_sync_time < 2.0:
                    return

                real_balance = await self.fetch_real_balance()
                if real_balance is not None:
                    diff = real_balance - self.current_balance
                    if abs(diff) > 1: # 1원 이상의 차이가 있을 때만 로깅
                        self.logger.info(f"🔄 잔고 동기화 완료: {self.current_balance:,.0f} -> {real_balance:,.0f} (오차: {diff:,.0f})")
                    
                    self.current_balance = real_balance
                    self.last_sync_time = now
                    self._last_real_sync_time = now

        # [모드 공통] 실현손익 및 리스크 지표 동기화
        if self.risk_manager:
            self.risk_manager.daily_realized_pnl = self.daily_realized_pnl
        
        # 시그널 발생 (총자산과 주문가능현금 함께 전달)
        self.signals.balance_synced.emit(self.current_balance)

    def _save_bot_holdings(self):
        """현재 봇이 관리 중인 종목 수량을 파일에 저장합니다."""
        try:
            with open(self.holdings_file, 'w', encoding='utf-8') as f:
                json.dump(self.bot_holdings, f, indent=4, ensure_ascii=False)
            self.logger.debug(f"봇 관리 종목 데이터 저장 완료: {len(self.bot_holdings)}개 종목")
        except Exception as e:
            self.logger.error(f"봇 관리 종목 저장 실패: {e}")

    def _load_bot_holdings(self) -> Dict[str, int]:
        """파일에서 이전 봇 관리 종목 데이터를 불러옵니다."""
        if not os.path.exists(self.holdings_file):
            self.logger.info("이전 봇 관리 종목 데이터가 없습니다. 새로 시작합니다.")
            return {}
        
        try:
            with open(self.holdings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # [수정] 로드 시 종목 코드 키 정제하여 불일치 방지
                cleaned_data = {str(k).split('_')[0].strip(): int(v) for k, v in data.items() if k}
                self.logger.info(f"이전 봇 관리 종목 {len(cleaned_data)}개를 성공적으로 불러왔습니다.")
                return cleaned_data
        except Exception as e:
            self.logger.error(f"봇 관리 종목 로드 실패: {e}")
            return {}
