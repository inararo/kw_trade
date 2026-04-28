import asyncio
import time
import json
import random
import websockets
from collections import deque
import numpy as np
import logging
from core.feature_engineer import FeatureEngineer
from env.normalizer import OnlineRollingNormalizer
from core.subscription_manager import SymbolSubscriptionManager

class DataCollector:
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

        self.on_state_updated_callbacks = []

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

        # 마지막 유효 현재가 저장용 (호가 패킷 등에 현재가가 없을 때 사용)
        self.last_prices = {}
        
        # [안정화] 관리되지 않는 비동기 태스크 추적용 (종료 시 정리)
        self._pending_tasks = set()
        
        # [최적화] 실시간 코드 매칭용 맵 (clean_code -> full_symbol)
        self._symbol_map = {}
        self._update_symbol_map()

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
        self.subscription_manager.remove_symbol(symbol)
        self._update_symbol_map()
        clean_symbol = symbol.split('_')[0]

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

    async def start(self):
        self.is_running = True
        
        # [동기화 수정] 부팅 시 Step 2에서 갱신된 최신 유니버스를 다시 읽어옵니다.
        if hasattr(self.config, 'get_symbols'):
            latest_symbols = [s.get('code') for s in self.config.get_symbols()]
            if latest_symbols:
                self._initial_symbols = latest_symbols
                self.logger.info(f"동기화: 최신 유니버스 {len(latest_symbols)}개 종목으로 구독 리스트를 갱신했습니다.")

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
                    
                    # 데이터 파싱
                    data = json.loads(message)

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
            # 2. 실시간 데이터 타입 식별 (0B: 체결, 0D: 호가)
            msg_type = entry.get("type") or message_data.get("type") or message_data.get("tr_id") or message_data.get("trnm")
            
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

                price = abs(float(str(raw_price).replace(',', '')))
                volume = abs(float(str(raw_vol).replace(',', '')))

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

                    # [긴급 패치 2] volume 필드 명시적 추가
                    ui_data = {
                        "symbol": target_symbol,
                        "price": current_display_price,
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
                    for callback in self.on_state_updated_callbacks:
                        if asyncio.iscoroutinefunction(callback):
                            task = asyncio.create_task(
                                callback(target_symbol, normalized_state, price=price, volume=volume,
                                         timestamp=now_time))
                            self._pending_tasks.add(task)
                            task.add_done_callback(self._pending_tasks.discard)
                        else:
                            callback(target_symbol, normalized_state, price=price, volume=volume,
                                     timestamp=now_time)

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
        """Safely stops the data collector."""
        self.is_running = False
        
        # 1. Stop Watchdog
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
            try:
                await asyncio.wait_for(self._watchdog_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # 2. Cancel all pending callback tasks
        if self._pending_tasks:
            self.logger.info(f"DataCollector: Cancelling {len(self._pending_tasks)} pending tasks...")
            for task in list(self._pending_tasks):
                task.cancel()
            
            try:
                await asyncio.wait_for(asyncio.gather(*self._pending_tasks, return_exceptions=True), timeout=2.0)
            except asyncio.TimeoutError:
                self.logger.warning("DataCollector: Task cancellation timeout")
            self._pending_tasks.clear()

        # 3. Close Websocket
        if self.ws_connection:
            try:
                # [Windows Stability] Wait briefly after setting stop flag to let recv loop exit naturally
                await asyncio.wait_for(self.ws_connection.close(), timeout=2.0)
            except Exception as e:
                self.logger.warning(f"WS force close exception (ignored): {e}")

        self.logger.info("DataCollector: All connections closed and resources cleaned up.")
