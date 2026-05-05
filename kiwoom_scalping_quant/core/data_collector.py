import asyncio
import time
import json
import random
import websockets
from collections import deque
import numpy as np
import logging
from typing import List
from core.feature_engineer import FeatureEngineer
from env.normalizer import OnlineRollingNormalizer
from core.subscription_manager import SymbolSubscriptionManager

class DataCollector:
    # [핵심 패치] 인스턴스 변수가 아닌 '클래스 변수'로 선언하여
    # 어떤 DataCollector 객체에서도 동일한 콜백 리스트를 공유하게 함
    on_state_updated_callbacks = []
    on_execution_callbacks = [] # [신규] 체결/주문 상태 업데이트 콜백

    def __init__(self, config):
        self.config = config

        self.ws_url = config.get_ws_url() if hasattr(config, 'get_ws_url') else config.get('ws_url', 'wss://mockapi.kiwoom.com:10000/api/dostk/websocket')
        self.max_buffer_size = config.get('max_buffer_size', 10000)

        # 구독 관리자
        self.subscription_manager = SymbolSubscriptionManager(max_subscriptions=100)

        # 피처 엔지니어링, 정규화, 롤링 버퍼를 종목별 딕셔너리로 동적 관리
        self.feature_engineers = {}
        self.normalizers = {}
        self.state_buffers = {}
        self.tick_buffers = {}
        self.min1_buffers = {}

        self.ws_connection = None
        self.is_running = False
        self.last_receive_time = time.time()
        self.latency_logs = deque(maxlen=1000)
        self.circuit_breaker_active = False

        self.logger = logging.getLogger("DataCollector")
        self._ui_callback = None
        self._watchdog_task = None

        # Connection and state events
        self.ws_connected_event = asyncio.Event()
        self.first_data_received_event = asyncio.Event()

        # Config에서 초기 심볼 등록 (start 시점에 구독하기 위해 저장만 함)
        self._initial_symbols = [s.get('code') for s in config.get('universe', [{'code': '005930'}])]
        if not self._initial_symbols:
            self._initial_symbols = ['005930']

        # 마지막 유효 현재가 및 등락률 저장용 (호가 패킷 등에 정보가 없을 때 사용)
        self.last_prices = {}
        self.last_change_rates = {}
        
        # [안정화] 관리되지 않는 비동기 태스크 추적용 (종료 시 정리)
        self._pending_tasks = set()
        
        # [최적화] 실시간 코드 매칭용 맵 (clean_code -> full_symbol)
        self._symbol_map = {}
        self._update_symbol_map()

        # 자신이 태어나면 전역 변수에 자신을 등록
        global global_data_collector_instance
        global_data_collector_instance = self

    def _update_symbol_map(self):
        """구독 관리자의 최신 심볼 리스트를 바탕으로 고속 조회용 맵 갱신"""
        new_map = {}
        for sym in self.subscription_manager.get_symbols():
            clean = sym.split('_')[0].strip()
            new_map[clean] = sym
        self._symbol_map = new_map

    def set_ui_callback(self, callback):
        self._ui_callback = callback

    async def subscribe_symbol(self, symbol: str):
        """새로운 종목을 구독하고 버퍼를 동적 할당합니다."""
        # [수정] 모든 내부 처리에 정제된 심볼 사용
        symbol = symbol.split('_')[0].strip()
        
        if not self.subscription_manager.add_symbol(symbol):
            return False
        
        self._update_symbol_map()

        # 종목 코드 정규화 (_AL 접미사 제거)
        clean_symbol = symbol.split('_')[0].strip()

        if symbol not in self.feature_engineers:
            self.feature_engineers[symbol] = FeatureEngineer(max_ticks=100)
            self.normalizers[symbol] = OnlineRollingNormalizer(window_size=1000, bypass_indices=[2])
            self.state_buffers[symbol] = deque(maxlen=self.max_buffer_size)
            self.tick_buffers[symbol] = deque(maxlen=self.max_buffer_size)
            self.min1_buffers[symbol] = deque(maxlen=self.max_buffer_size // 10)

        if self.is_running and self.ws_connection:
            # 키움 REST API 실전 규격 (type과 item 모두 배열 형식 필수)
            msg = json.dumps({
                "trnm": "REG",
                "grp_no": "1",
                "refresh": "0", # 0: 실시간 추가 구독, 1: 기존 구독 해제 후 신규 구독
                "data": [
                    {"type": ["0B"], "item": [clean_symbol]}, # 0B: 주식체결
                    {"type": ["0D"], "item": [clean_symbol]}  # 0D: 호가잔량
                ]
            })
            await self.ws_connection.send(msg)
            # 서버 부하 및 Windows 소켓 버퍼 오버플로우 방지 지연 (0.2 -> 0.3초)
            await asyncio.sleep(0.3)

        return True

    async def unsubscribe_symbol(self, symbol: str):
        """구독을 해제합니다."""
        symbol = symbol.split('_')[0].strip()
        self.subscription_manager.remove_symbol(symbol)
        self._update_symbol_map()
        clean_symbol = symbol

        # 백그라운드 웹소켓이 동작 중이면 실시간 구독 해제 메시지 발송
        if self.is_running and self.ws_connection:
            msg = json.dumps({
                "trnm": "UNREG",
                "data": [
                    {"type": ["0B"], "item": [clean_symbol]},
                    {"type": ["0D"], "item": [clean_symbol]}
                ]
            })
            await self.ws_connection.send(msg)
            await asyncio.sleep(0.1)

    async def update_subscriptions(self, to_add: List[str], to_remove: List[str]):
        """
        [최적화] 장중 유니버스 교체 시 다수의 종목을 일괄적으로 구독 해제 및 등록합니다.
        """
        if not self.is_running or not self.ws_connection:
            # 연결 전이면 관리자에게만 반영 (나중에 start 시점에 일괄 처리됨)
            for sym in to_remove: self.subscription_manager.remove_symbol(sym)
            for sym in to_add: 
                self.subscription_manager.add_symbol(sym)
                self._ensure_buffers(sym)
            self._update_symbol_map()
            return

        # 1. 일괄 해제 (UNREG)
        if to_remove:
            clean_removes = []
            for sym in to_remove:
                self.subscription_manager.remove_symbol(sym)
                clean_removes.append(sym.split('_')[0])
            
            msg = json.dumps({
                "trnm": "UNREG",
                "data": [
                    {"type": ["0B"], "item": clean_removes},
                    {"type": ["0D"], "item": clean_removes}
                ]
            })
            await self.ws_connection.send(msg)
            self.logger.info(f"DataCollector: {len(clean_removes)}개 종목 일괄 구독 해제 전송")
            await asyncio.sleep(0.5)

        # 2. 일괄 등록 (REG)
        if to_add:
            clean_adds = []
            for sym in to_add:
                if self.subscription_manager.add_symbol(sym):
                    self._ensure_buffers(sym)
                    clean_adds.append(sym.split('_')[0])
            
            if clean_adds:
                msg = json.dumps({
                    "trnm": "REG",
                    "grp_no": "1",
                    "refresh": "0",
                    "data": [
                        {"type": ["0B"], "item": clean_adds},
                        {"type": ["0D"], "item": clean_adds}
                    ]
                })
                await self.ws_connection.send(msg)
                self.logger.info(f"DataCollector: {len(clean_adds)}개 종목 일괄 구독 등록 전송")
                await asyncio.sleep(0.5)

        # 3. 맵 갱신 및 워치독 타이머 리셋 (재연결 방지)
        self._update_symbol_map()
        self.last_receive_time = time.time()

    def _ensure_buffers(self, symbol: str):
        """종목별 피처 엔진 및 버퍼가 없으면 생성합니다."""
        if symbol not in self.feature_engineers:
            self.feature_engineers[symbol] = FeatureEngineer(max_ticks=100)
            self.normalizers[symbol] = OnlineRollingNormalizer(window_size=1000, bypass_indices=[2])
            self.state_buffers[symbol] = deque(maxlen=self.max_buffer_size)
            self.tick_buffers[symbol] = deque(maxlen=self.max_buffer_size)
            self.min1_buffers[symbol] = deque(maxlen=self.max_buffer_size // 10)

    async def start(self):
        if self.is_running:
            self.logger.warning("DataCollector is already running.")
            return
        
        self.is_running = True
        self.logger.info(f"DataCollector: 수집을 시작합니다. (등록된 체결 콜백: {len(self.on_execution_callbacks)}개)")
        # 상태 이벤트 초기화
        self.ws_connected_event.clear()
        self.first_data_received_event.clear()
        
        # [동기화 수정] 부팅 시 Step 2에서 갱신된 최신 유니버스와 기존 등록된 종목(보유 종목 등)을 병합합니다.
        all_subs = set(self.subscription_manager.get_symbols())
        if hasattr(self.config, 'get_symbols'):
            for s in self.config.get_symbols():
                code = s.get('code')
                if code: all_subs.add(code)
        
        if all_subs:
            self._initial_symbols = list(all_subs)
            self.logger.info(f"동기화: 총 {len(all_subs)}개 종목(유니버스+보유)으로 구독 리스트를 확정했습니다.")

        # [안정화/복구] 초기 종목 구독 및 버퍼 초기화 (필수)
        # subscribe_symbol은 버퍼를 생성하고 관리자에 등록합니다.
        # 실제 웹소켓 전송은 내부의 if self.ws_connection 조건에 의해 연결 시점에만 수행됩니다.
        self.logger.info(f"초기 종목 {len(self._initial_symbols)}개에 대해 수집 준비 및 구독을 시도합니다.")
        for sym in self._initial_symbols:
            await self.subscribe_symbol(sym)

        # Watchdog 태스크 시작
        self._watchdog_task = asyncio.create_task(self._watchdog())

        retry_delay = 1
        try:
            while self.is_running:
                try:
                    await self._connect_and_listen()
                    # 연결이 한 번이라도 성공적으로 유지되었다가 끊기면 딜레이 초기화
                    retry_delay = 1
                except asyncio.CancelledError:
                    self.logger.info("DataCollector 루프 완전 취소됨.")
                    break
                except Exception as e:
                    self.logger.error(f"WebSocket 연결 오류: {e}")
                    if self.is_running:
                        self.logger.info(f"재연결 시도 중... ({retry_delay}초 대기)")
                        await asyncio.sleep(retry_delay)
                        # 지수 백오프 (최대 30초)
                        retry_delay = min(retry_delay * 2, 30)
        finally:
            self.is_running = False

    async def _connect_and_listen(self):
        # 인증 헤더 준비 (config 또는 환경변수에서 토큰 추출)
        token = self.config.get("KIWOOM_ACCESS_TOKEN") or getattr(self.config, "get", lambda x: None)("KIWOOM_ACCESS_TOKEN")
        
        headers = {}
        if token:
            headers["authorization"] = f"Bearer {token}"
            self.logger.info("웹소켓 인증 헤더(Bearer Token)를 포함하여 연결합니다.")

        async with websockets.connect(self.ws_url, extra_headers=headers) as websocket:
            self.ws_connection = websocket
            self.ws_connected_event.set()
            
            # [안정화] 연결 성공 시 Circuit Breaker 해제 및 타이머 초기화 (무한 재연결 방지)
            self.circuit_breaker_active = False
            self.last_receive_time = time.time()
            
            self.logger.info("WebSocket 연결 성공. 인증(LOGIN)을 시도합니다.")

            # [Step 1] 웹소켓 로그인 인증 요청
            await websocket.send(json.dumps({
                "trnm": "LOGIN",
                "token": token
            }))
            self.logger.info("LOGIN 요청 전송 완료. 서버 응답 대기 중...")
            
            # [Handshake] LOGIN 응답 수신 대기 (중요: 응답 확인 후 구독 진행)
            login_success = False
            try:
                first_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                login_res = json.loads(first_msg)
                if str(login_res.get("return_code")) == "0" or login_res.get("return_code") == 0:
                    self.logger.info(f"LOGIN 인증 성공: {login_res.get('return_msg', '정상')}")
                    login_success = True
                else:
                    self.logger.error(
                        f"LOGIN Auth Failed: {login_res.get('return_msg')} "
                        f"(Code: {login_res.get('return_code')}) "
                        f"-> Requesting token refresh and waiting for reconnect"
                    )
                    # [Token Auth Failure] Refresh token via config.token_manager
                    token_mgr = getattr(self.config, '_token_manager', None) or getattr(self.config, 'token_manager', None)
                    if token_mgr and hasattr(token_mgr, 'refresh_token'):
                        self.logger.error("LOGIN Failed: Attempting to refresh token...")
                        await token_mgr.refresh_token()
                        await asyncio.sleep(3.0)  # Wait for server processing
                    else:
                        await asyncio.sleep(10.0)  # Wait 10s if no manager
                    return  # Terminate ws context -> Auto reconnection loop
            except Exception as e:
                self.logger.error(f"LOGIN 응답 대기 중 오류: {e}")
                return

            # LOGIN 성공 시에만 구독 진행
            if not login_success:
                return

            symbols = self.subscription_manager.get_symbols()
            if symbols:
                self.logger.info(f"초기 종목 {len(symbols)}개에 대해 일괄 구독(Batch)을 시작합니다.")
                try:
                    clean_symbols = []
                    # 1. 내부 버퍼 먼저 일괄 생성
                    for sym in symbols:
                        clean = sym.split('_')[0].strip()
                        clean_symbols.append(clean)

                        if sym not in self.feature_engineers:
                            from core.feature_engineer import FeatureEngineer
                            from env.normalizer import OnlineRollingNormalizer
                            self.feature_engineers[sym] = FeatureEngineer(max_ticks=100)
                            self.normalizers[sym] = OnlineRollingNormalizer(window_size=1000, bypass_indices=[2])
                            self.state_buffers[sym] = deque(maxlen=self.max_buffer_size)
                            self.tick_buffers[sym] = deque(maxlen=self.max_buffer_size)
                            self.min1_buffers[sym] = deque(maxlen=self.max_buffer_size // 10)

                    # 2. 단 한 번의 웹소켓 요청으로 20개 종목 통째로 구독!
                    if self.is_running and self.ws_connection:
                        msg = json.dumps({
                            "trnm": "REG",
                            "grp_no": "1",
                            "refresh": "0",
                            "data": [
                                {"type": ["0B"], "item": clean_symbols},  # 배열 형태로 20개 한방에 전송
                                {"type": ["0D"], "item": clean_symbols}
                            ]
                        })
                        await self.ws_connection.send(msg)
                        await asyncio.sleep(0.5)

                    self.logger.info(f"일괄 구독 요청 완료: {clean_symbols[:5]} 등 {len(clean_symbols)}개 종목")
                except Exception as e:
                    self.logger.error(f"일괄 구독 프로세스 중 오류 발생: {e}")

            try:
                async for message in websocket:
                    recv_time = time.time()
                    self.last_receive_time = recv_time
                    
                    # [재적용] 키움증권 PING 메시지 처리 (Heartbeat 대응 및 R10002 방지)
                    if isinstance(message, str) and "PING" in message.upper():
                        self.logger.debug("📡 [키움 API] PING 수신 -> PONG 응답 전송")
                        # 서버 연결 유지를 위해 반드시 PONG을 보내야 합니다 (R10002 방지 핵심)
                        # 이 로직은 JSON 파싱 이전에 수행되므로 파싱 에러가 발생하지 않습니다.
                        await websocket.send(json.dumps({"trnm": "PONG"}))
                        continue

                    # 데이터 파싱
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        self.logger.error(f"JSON 파싱 에러 (비정상 메시지): {message}")
                        continue

                    # 진단 로그: 모든 루트 키 확인을 위해 로그 포맷 변경
                    root_keys = list(data.keys()) if isinstance(data, dict) else "Not Dict"
                    # self.logger.error(f"WS RECV (keys={root_keys}, len={len(message)})")
                    self.circuit_breaker_active = False

                    if not self.first_data_received_event.is_set():
                        self.first_data_received_event.set()

                    # 지연 시간(Latency) 프로파일링
                    exchange_time = data.get('timestamp', recv_time)
                    latency_ms = (recv_time - exchange_time) * 1000
                    self.latency_logs.append(latency_ms)

                    if latency_ms > 50:
                        self.logger.warning(f"High Latency 경고: {latency_ms:.2f}ms")

                    await self._process_tick(data)
            except asyncio.CancelledError:
                self.logger.info("DataCollector: WebSocket 메시지 수신 루프가 취소되었습니다.")
                raise
            finally:
                # [안정화] 연결이 끊기면 관련 상태를 명확히 초기화
                self.ws_connected_event.clear()
                self.first_data_received_event.clear()
                self.ws_connection = None
                self.logger.info("DataCollector: WebSocket 상태가 초기화되었습니다.")

    async def _process_tick(self, message_data):
        """수신된 실시간 데이터를 루프 돌며 파싱하여 피처 엔진 및 버퍼에 업데이트"""
        
        # 1. 결과 응답(REG, LOGIN 등) 처리
        if message_data.get("return_code") is not None:
            self.logger.info(f"WS API RESPONSE: {message_data.get('return_msg')} (Code: {message_data.get('return_code')})")
            return

        # 2. 실시간 데이터 프레임('data' 리스트) 처리
        entries = message_data.get("data", [])
        if not entries:
            # 루트 레벨에 데이터가 있는 경우 (Fallback)
            entries = [message_data]

        for entry in entries:
            # 2. 실시간 데이터 타입 식별
            msg_type = entry.get("type") or message_data.get("type") or message_data.get("tr_id") or message_data.get("trnm")
            
            # [디버그] 시장가 데이터가 아닌 모든 메시지 로깅
            if msg_type not in ["0B", "0D"]:
                self.logger.error(f"🔍 [WS Message] Type: {msg_type} | Content: {str(entry)[:200]}")
            
            # [신규] 주문/체결(Chejan) 데이터 처리
            # Kiwoom REST/WS API에서 주문/체결은 보통 trnm이 'ORDR' 또는 'CNTG'로 오거나, ord_no 필드가 포함됩니다.
            has_order_info = "ord_no" in entry or "ord_no" in message_data or msg_type in ['ORDR', 'CNTG', 'K1', 'H1', 'SC']
            
            if has_order_info:
                self.logger.error(f"🔔 [Chejan] 주문 관련 데이터 감지 (Type: {msg_type})")
                
                # 데이터 병합 (entry와 message_data에서 정보 추출)
                combined = {**message_data, **entry} if isinstance(message_data, dict) and isinstance(entry, dict) else entry
                
                broker_id = str(combined.get("ord_no") or combined.get("broker_id") or "")
                
                if broker_id:
                    chejan_data = {
                        "msg_type": "접수" if msg_type == "ORDR" or "접수" in str(combined.get("return_msg", "")) else "체결",
                        "broker_id": broker_id,
                        "exec_qty": int(float(combined.get("exec_qty") or combined.get("exec_qty", 0))),
                        "exec_price": float(combined.get("exec_prc") or combined.get("exec_price") or combined.get("exec_prc", 0)),
                        "symbol": (combined.get("stk_cd") or combined.get("symbol") or "").split('_')[0].strip(),
                        "timestamp": combined.get("timestamp") or combined.get("time") or combined.get("curr_time")
                    }
                    
                    self.logger.error(f"🚀 [Chejan Dispatch] {chejan_data}")

                    # 등록된 콜백(OrderManager 등)으로 전달
                    for callback in self.on_execution_callbacks:
                        if asyncio.iscoroutinefunction(callback):
                            asyncio.create_task(callback(chejan_data))
                        else:
                            try:
                                callback(chejan_data)
                            except Exception as e:
                                self.logger.error(f"Chejan 콜백 실행 에러: {e}")
                    
                    if msg_type in ["ORDR", "CNTG"]:
                        continue # 주문 데이터 처리를 마쳤으면 다음 엔트리로
            
            # [수정] raw_code 추출 로직 강화
            # Kiwoom Websocket entries usually have the stock code in 'item' or 'stk_cd'
            raw_code = entry.get("item") or entry.get("stk_cd") or entry.get("stk_code") or entry.get("symbol")
            
            # Fallback: Entry에 없으면 상위 message_data에서 찾되, 예약어(0B, 0D 등)는 제외
            if not raw_code:
                candidate = message_data.get("item") or message_data.get("stk_cd") or message_data.get("tr_key")
                reserved = ["0B", "0D", "REG", "LOGIN", "PING", "PONG", "SYSTEM"]
                if candidate and str(candidate).upper() not in reserved:
                    raw_code = candidate

            if not raw_code:
                continue

            # [최적화] O(1) 딕셔너리 기반 타겟 심볼 조회
            clean_code = str(raw_code).strip()
            
            # [RAW_DEBUG] 특정 종목 원시 데이터 필터링 출력
            # if clean_code == "010170":
            # self.logger.info(f"[RAW_SOCKET] {clean_code} tick received")

            target_symbol = self._symbol_map.get(clean_code)
            
            if not target_symbol:
                # 진단용 로그 (INFO 레벨로 일시적 격상)
                if random.random() < 0.001:
                    self.logger.info(f"매칭 대상 아님: {clean_code} (구독: {list(self._symbol_map.keys())[:5]}...)")
                continue

            # 3. FID 기반 정보 추출
            try:
                values = entry.get("values", entry)
                raw_price = values.get("10") or values.get("curr_pric") or values.get("cur_prc") or values.get("exec_prc") or "0"
                raw_vol = values.get("15") or values.get("cntg_vol") or values.get("trde_qty") or values.get("exec_qty") or "0"
                raw_chg = values.get("12") or values.get("flu_rt") or values.get("chg_rt") or "0"

                price = abs(float(str(raw_price).replace(',', '')))
                volume = abs(float(str(raw_vol).replace(',', '')))
                change_rate = float(str(raw_chg).replace(',', ''))

                # 등락률 캐시 갱신 (유효한 값이 들어올 때만)
                if change_rate != 0 or msg_type == "0B":
                    self.last_change_rates[target_symbol] = change_rate

                # ========================================================
                # [긴급 패치 1] UI 업데이트를 최상단으로 끌어올림 (방패 역할)
                # AI 콜백에서 에러가 터져도 화면은 무조건 갱신되도록 보장!
                # ========================================================
                if price > 0 or msg_type == "0D":
                    orderbook = {}
                    if msg_type == "0D":
                        asks, bids = [], []
                        for i in range(1, 11):
                            ask_p = values.get(str(40 + i))
                            ask_q = values.get(str(60 + i))
                            if ask_p and ask_q: asks.append({"price": abs(float(str(ask_p).replace(',', ''))),
                                                             "qty": abs(float(str(ask_q).replace(',', '')))})

                            bid_p = values.get(str(50 + i))
                            bid_q = values.get(str(70 + i))
                            if bid_p and bid_q: bids.append({"price": abs(float(str(bid_p).replace(',', ''))),
                                                             "qty": abs(float(str(bid_q).replace(',', '')))})
                        orderbook = {"asks": asks, "bids": bids}

                    current_display_price = price if price > 0 else self.last_prices.get(target_symbol, 0)
                    
                    current_change_rate = change_rate
                    if current_change_rate == 0 and target_symbol in self.last_change_rates:
                        current_change_rate = self.last_change_rates[target_symbol]

                    # [긴급 패치 2] volume, change_rate 필드 명시적 추가
                    ui_data = {
                        "symbol": target_symbol,
                        "price": current_display_price,
                        "change_rate": current_change_rate,
                        "volume": volume,
                        "orderbook": orderbook,
                    }

                    if self._ui_callback:
                        self._ui_callback(ui_data)

                # ========================================================
                # [긴급 패치 3] UI 갱신 후 AI 콜백 실행 (에러 발생 가능 구간)
                # ========================================================
                if price > 0:
                    features = self.feature_engineers[target_symbol].update_tick(price, volume)
                    raw_state = np.array(
                        [price, volume, features["OIR"], features["Volatility"], features["Aggressiveness"]],
                        dtype=np.float32)
                    normalized_state = self.normalizers[target_symbol].update_and_normalize(raw_state)
                    self.last_prices[target_symbol] = price

                    from datetime import datetime
                    now_time = datetime.now()

                    # 여기가 에러(TypeError)가 터지는 핵심 용의자입니다!
                    # ========================================================
                    # [비동기 에러 탐지기 & 콜백 미아 방지]
                    # ========================================================
                    if not self.on_state_updated_callbacks:
                        # 콜백이 비어있다면 1% 확률로 경고 (로그 도배 방지)
                        import random
                        if random.random() < 0.01:
                            self.logger.warning(
                                f"⚠️ [{target_symbol}] 틱 수신 완료... 그러나 연결된 AI 콜백(StrategyManager)이 없습니다!")

                    for callback in self.on_state_updated_callbacks:
                        if asyncio.iscoroutinefunction(callback):
                            task = asyncio.create_task(
                                callback(target_symbol, normalized_state, price=price, volume=volume,
                                         timestamp=now_time))
                            self._pending_tasks.add(task)

                            # [핵심] 조용히 죽는 비동기 에러를 끄집어내는 사냥꾼 함수
                            def _handle_task_result(t):
                                self._pending_tasks.discard(t)
                                try:
                                    exc = t.exception()
                                    if exc:
                                        import traceback
                                        err_msg = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                                        self.logger.error(f"🚨 [AI 콜백 붕괴] 틱 전달 중 치명적 에러 발생:\n{err_msg}")
                                except asyncio.CancelledError:
                                    pass

                            task.add_done_callback(_handle_task_result)
                        else:
                            try:
                                callback(target_symbol, normalized_state, price=price, volume=volume,
                                         timestamp=now_time)
                            except Exception as e:
                                import traceback
                                self.logger.error(f"🚨 [동기 콜백 붕괴] {e}\n{traceback.format_exc()}")

            except (ValueError, TypeError, Exception) as e:
                self.logger.error(f"[DC ERROR] Data 파싱 중 오류 ({raw_code}): {e}")
                import traceback
                self.logger.error(traceback.format_exc())  # [추가] 정확히 어디서 터졌는지 추적
                continue

        # [최적화] 개별 틱이 아닌 메시지 한 묶음 처리가 끝난 후 한 번만 양보하여 UI 기회 제공
        await asyncio.sleep(0)

    async def _watchdog(self):
        """3초 이상 데이터 수신이 없으면 Circuit Breaker 발동 및 재연결"""
        try:
            while self.is_running:
                await asyncio.sleep(1)

                # 방어 로직 1: 웹소켓 구독이 완료되고 최초 데이터가 들어온 이후에만 감시 시작
                if not self.ws_connected_event.is_set() or not self.first_data_received_event.is_set():
                    self.last_receive_time = time.time() # 억울하게 죽지 않도록 타이머 갱신
                    continue
                
                # [추가] 최초 데이터 수신 직후 10초간은 네트워크 안정화를 위해 감시 유예
                if time.time() - self.last_receive_time < 10.0:
                    continue

                # 방어 로직 2: Market Scheduler 상태 확인 (장이 열려있을 때만)
                scheduler = getattr(self.config, "_injected_scheduler", None)
                if scheduler:
                    from core.scheduler import MarketState
                    if scheduler.current_state not in [MarketState.TRADING, MarketState.LIQUIDATING]:
                        # Log debug info periodically (every ~30s to avoid spam)
                        if int(time.time()) % 30 == 0:
                            self.logger.info("장외 시간: Watchdog 대기 중")

                        # Reset last receive time to prevent immediate breaker when market opens
                        self.last_receive_time = time.time()
                        continue

                idle_time = time.time() - self.last_receive_time

                # 방어 로직 3: 구독 중인 종목이 아예 없으면 데이터가 안 오는 것이 정상이므로 건너뜀
                if not self.subscription_manager.get_symbols():
                    self.last_receive_time = time.time()
                    continue

                # [수정] 5.0초는 너무 짧아 30.0초로 연장 (장외 시간/저변동성 대응)
                if idle_time > 30.0 and not self.circuit_breaker_active:
                    self.logger.error(f"Watchdog: {idle_time:.1f}초간 시세 미수신! Circuit Breaker 발동 (재연결 시도).")
                    self.circuit_breaker_active = True

                    if self.ws_connection:
                        await self.ws_connection.close()
        except asyncio.CancelledError:
            self.logger.info("Watchdog 태스크가 취소되어 안전하게 종료됩니다.")

    async def stop(self):
        self.is_running = False
        
        # 1. Stop Watchdog
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
            try:
                await asyncio.wait_for(self._watchdog_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._watchdog_task = None

        # 2. Cancel all pending callback tasks
        if self._pending_tasks:
            self.logger.info(f"DataCollector: {len(self._pending_tasks)}개의 대기 중인 콜백 태스크 취소 중...")
            for task in list(self._pending_tasks):
                if not task.done():
                    task.cancel()
            
            try:
                await asyncio.wait_for(asyncio.gather(*self._pending_tasks, return_exceptions=True), timeout=2.0)
            except asyncio.TimeoutError:
                self.logger.warning("DataCollector: 태스크 취소 타임아웃 발생")
            self._pending_tasks.clear()

        # 3. Close Websocket
        if self.ws_connection:
            try:
                await asyncio.wait_for(self.ws_connection.close(), timeout=2.0)
            except Exception as e:
                self.logger.warning(f"WS 강제 종료 중 예외 (무시됨): {e}")
            finally:
                self.ws_connection = None
                self.ws_connected_event.clear()
                self.first_data_received_event.clear()

        self.logger.info("DataCollector: 모든 연결이 해제되고 리소스가 정리되었습니다.")

# 전역 싱글톤 인스턴스 저장소
global_data_collector_instance = None