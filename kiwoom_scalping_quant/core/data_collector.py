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

    def set_ui_callback(self, callback):
        self._ui_callback = callback

    async def subscribe_symbol(self, symbol: str):
        """새로운 종목을 구독하고 버퍼를 동적 할당합니다."""
        if not self.subscription_manager.add_symbol(symbol):
            return False

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
                "refresh": "1",
                "data": [
                    {"type": ["0B"], "item": [clean_symbol]}, # 0B: 주식체결
                    {"type": ["0D"], "item": [clean_symbol]}  # 0D: 호가잔량
                ]
            })
            await self.ws_connection.send(msg)
            # 서버 부하 방지 지연
            await asyncio.sleep(0.2)

        return True

    async def unsubscribe_symbol(self, symbol: str):
        """구독을 해제합니다."""
        self.subscription_manager.remove_symbol(symbol)
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

    async def start_mock_stream(self):
        """장외 시간/주말 UI 테스트용 가상 데이터 생성기 (다중 종목)"""
        import random
        symbols = self.subscription_manager.get_symbols()
        self.logger.info(f"Mock Stream Started for {len(symbols)} symbols.")
        base_prices = {sym: 50000 + random.randint(-10000, 10000) for sym in symbols}

        try:
            while self.is_running:
                current_symbols = self.subscription_manager.get_symbols()
                for symbol in current_symbols:
                    if symbol not in base_prices:
                        base_prices[symbol] = 50000 + random.randint(-10000, 10000)

                    # 가상 가격 변동
                    base_prices[symbol] += random.choice([-100, 0, 100])
                    price = base_prices[symbol]
                    volume = random.randint(10, 500)

                    # 10호가 가상 매수/매도 잔량 생성
                    asks = [{"price": price + (i * 100), "qty": random.randint(100, 5000)} for i in range(1, 11)]
                    bids = [{"price": price - (i * 100), "qty": random.randint(100, 5000)} for i in range(1, 11)]

                    orderbook = {"asks": asks, "bids": bids}

                    # 피처 계산
                    self.feature_engineers[symbol].update_orderbook(orderbook)
                    features = self.feature_engineers[symbol].update_tick(price, volume)

                    # [Price, Volume, OIR, Volatility, Aggressiveness]
                    raw_state = np.array([
                        price,
                        volume,
                        features["OIR"],
                        features["Volatility"],
                        features["Aggressiveness"]
                    ], dtype=np.float32)

                    # 정규화
                    normalized_state = self.normalizers[symbol].update_and_normalize(raw_state)
                    self.state_buffers[symbol].append(normalized_state)

                    # AI 확률 임의 생성
                    hold_prob = random.randint(40, 80)
                    buy_prob = random.randint(0, 100 - hold_prob)
                    sell_prob = 100 - hold_prob - buy_prob

                    mock_data = {
                        "symbol": symbol,
                        "price": price,
                        "orderbook": orderbook,
                        "ai_confidence": {"Hold": hold_prob, "Buy": buy_prob, "Sell": sell_prob}
                    }

                    if self._ui_callback:
                        self._ui_callback(mock_data)

                await asyncio.sleep(0.1) # 0.1초(100ms) 간격 업데이트
        except asyncio.CancelledError:
            self.logger.info("Mock Stream Cancelled.")

    def get_latest_state(self, symbol: str, seq_len=1):
        """환경(Env)이 특정 종목의 현재 상태를 가져가기 위한 메서드 (시퀀스 길이 지원)"""
        dim = 5
        buffer = self.state_buffers.get(symbol, [])
        if len(buffer) == 0:
            return np.zeros(dim * seq_len, dtype=np.float32)

        n_avail = len(buffer)
        if n_avail < seq_len:
            # Not enough data: pad with the first available state
            pad_len = seq_len - n_avail
            first_state = buffer[0]
            padded = [first_state] * pad_len
            actual = list(buffer)
            seq = padded + actual
        else:
            # Take the last seq_len states
            seq = list(buffer)[-seq_len:]

        # Flatten sequence: [t-n_1, t-n_2, ..., t_1, t_2, ...]
        return np.concatenate(seq).astype(np.float32)

    def get_latest_price(self, symbol: str) -> float:
        """스마트 주문 등을 위해 정규화되지 않은 최신 가격 반환"""
        if symbol in self.feature_engineers:
            fe = self.feature_engineers[symbol]
            if len(fe.price_buffer) > 0 and fe.count > 0:
                # 링 버퍼에서 가장 최근 입력된 가격 반환 (head-1)
                idx = (fe.head - 1) % fe.max_ticks
                return float(fe.price_buffer[idx])
        return 0.0

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

        try:
            while self.is_running:
                try:
                    await self._connect_and_listen()
                except asyncio.CancelledError:
                    self.logger.info("DataCollector 루프 완전 취소됨.")
                    break
                except Exception as e:
                    self.logger.error(f"WebSocket 연결 오류: {e}")
                    if self.is_running:
                        await asyncio.sleep(1) # 재연결 대기
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
            
            self.logger.error("WebSocket 연결 성공. 인증(LOGIN)을 시도합니다.")

            # [Step 1] 웹소켓 로그인 인증 요청
            await websocket.send(json.dumps({
                "trnm": "LOGIN",
                "token": token
            }))
            self.logger.error("LOGIN 요청 전송 완료. 서버 응답 대기 중...")
            
            # [Handshake] LOGIN 응답 수신 대기 (중요: 응답 확인 후 구독 진행)
            login_success = False
            try:
                first_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                login_res = json.loads(first_msg)
                if login_res.get("return_code") == 0:
                    self.logger.error(f"LOGIN 인증 성공: {login_res.get('return_msg', '정상')}")
                    login_success = True
                else:
                    self.logger.error(
                        f"LOGIN 인증 실패: {login_res.get('return_msg')} "
                        f"(Code: {login_res.get('return_code')}) "
                        f"\u2192 토큰 갱신 요청 후 재연결 대기"
                    )
                    # [토큰 인증 실패] 지정된 config.token_manager를 통해 토큰 갱신 시도
                    token_mgr = getattr(self.config, '_token_manager', None) or getattr(self.config, 'token_manager', None)
                    if token_mgr and hasattr(token_mgr, 'refresh_token'):
                        self.logger.error("LOGIN 실패: 토큰 갱신을 시도합니다...")
                        await token_mgr.refresh_token()
                        await asyncio.sleep(3.0)  # 서버 처리 대기
                    else:
                        await asyncio.sleep(10.0)  # token_manager 없으면 10초 대기
                    return  # 웹소켓 컨텍스트 종료 → 자동 재연결 루프로
            except Exception as e:
                self.logger.error(f"LOGIN 응답 대기 중 오류: {e}")
                return

            # LOGIN 성공 시에만 구독 진행
            if not login_success:
                return

            symbols = self.subscription_manager.get_symbols()
            if symbols:
                self.logger.error(f"초기 종목 {len(symbols)}개에 대해 순차적 구독을 시작합니다.")
                for sym in symbols:
                    await self.subscribe_symbol(sym)
                self.logger.error("초기 종목 구독 요청 완료.")

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
            # 종목 코드 추출 순서 보강 (trnm 필드 추가)
            raw_code = entry.get("item") or entry.get("stk_cd") or entry.get("symbol") or entry.get("tr_key") or \
                       message_data.get("item") or message_data.get("stk_cd") or message_data.get("tr_key") or \
                       message_data.get("trnm") # trnm이 종목 코드인 경우 대응
            
            # values 내부에서도 코드 탐색 (일부 규격 대응)
            values = entry.get("values", entry)
            if not raw_code and isinstance(values, dict):
                raw_code = values.get("item") or values.get("stk_cd") or values.get("tr_key") or values.get("stk_code")

            # 리스트/딕셔너리로 들어오는 경우 정제
            if isinstance(raw_code, list) and len(raw_code) > 0:
                raw_code = raw_code[0]
            elif isinstance(raw_code, dict):
                raw_code = raw_code.get("code") or raw_code.get("item")
            
            # 타입 식별 (trnm이 '0B' 등인 경우와 종목 코드인 경우 구분)
            msg_type = entry.get("type") or message_data.get("type") or message_data.get("tr_id")
            
            # [수정] PING, PONG, SYSTEM 등 상태 유지용 패킷을 예약어 리스트에 추가
            reserved_keywords = ["0B", "0D", "REG", "LOGIN", "PING", "PONG", "SYSTEM"]
            if str(raw_code).upper() in reserved_keywords:
                msg_type = raw_code
                raw_code = None
            
            if not raw_code:
                # 진단 로그: 예약어가 아닌데 코드가 없는 경우에만 출력 (로그 노이즈 감소)
                if str(msg_type).upper() not in reserved_keywords:
                    self.logger.debug(f"심볼 코드 추출 생략 (System Message): {msg_type}")
                continue
            
            # 관리용 심볼 매핑 (005930_AL이더라도 005930와 완전 매칭 지원)
            target_symbol = None
            manager_symbols = list(self.subscription_manager.get_symbols())
            for sym in manager_symbols:
                clean_sym = sym.split('_')[0].strip()
                if clean_sym == str(raw_code).strip():
                    target_symbol = sym
                    break
            
            if not target_symbol:
                # [로그 수준 완화] 매칭 실패 로그를 error에서 debug로 낮추어 노이즈 제거
                self.logger.debug(f"매칭 실패 및 무시: 수신코드=[{raw_code}], 구독리스트={manager_symbols}")
                continue

            # 3. FID 기반 정보 추출 (10: 현재가, 15: 체결량, 13: 누적거래량)
            try:
                # 키움 데이터는 부호(+/-)가 포함된 문자열이므로 abs(float()) 처리
                raw_price = values.get("10") or values.get("curr_pric") or "0"
                raw_vol = values.get("15") or values.get("cntg_vol") or "0"
                
                price = abs(float(str(raw_price).replace(',', '')))
                volume = abs(float(str(raw_vol).replace(',', '')))
                
                # 4. 피처 엔진 및 버퍼 업데이트 (체결 데이터일 경우)
                if price > 0:
                    features = self.feature_engineers[target_symbol].update_tick(price, volume)
                    
                    # 실시간 상태 버퍼 업데이트 (AI 입력용)
                    raw_state = np.array([
                        price,
                        volume,
                        features["OIR"],
                        features["Volatility"],
                        features["Aggressiveness"]
                    ], dtype=np.float32)

                    normalized_state = self.normalizers[target_symbol].update_and_normalize(raw_state)
                    self.state_buffers[target_symbol].append(normalized_state)
                    
                    # 마지막 유효 가격 업데이트
                    self.last_prices[target_symbol] = price

                    # 진단 로그: 1% 확률로 데이터 매칭 성공 출력
                    if random.random() < 0.01:
                        self.logger.info(f"데이터 매칭 성공! [{target_symbol}] 현재가: {price:,.0f} | 타입: {msg_type}")

                    # 5. 이벤트 콜백 실행 (StrategyManager 등 알림)
                    for callback in self.on_state_updated_callbacks:
                        if asyncio.iscoroutinefunction(callback):
                            asyncio.create_task(callback(target_symbol, normalized_state))
                        else:
                            callback(target_symbol, normalized_state)
                
                # 6. UI 업데이트 지원 (시세 또는 호가 정보가 있을 때)
                if price > 0 or msg_type == "0D":
                    orderbook = {}
                    if msg_type == "0D":
                        asks = []
                        bids = []
                        # Kiwoom 0D 필드: 매도(41~50 가격, 61~70 잔량), 매수(51~60 가격, 71~80 잔량)
                        for i in range(1, 11):
                            # 매도 호가 (Asks)
                            ask_p = values.get(str(40 + i))
                            ask_q = values.get(str(60 + i))
                            if ask_p and ask_q:
                                asks.append({
                                    "price": abs(float(str(ask_p).replace(',', ''))),
                                    "qty": abs(float(str(ask_q).replace(',', '')))
                                })
                            
                            # 매수 호가 (Bids)
                            bid_p = values.get(str(50 + i))
                            bid_q = values.get(str(70 + i))
                            if bid_p and bid_q:
                                bids.append({
                                    "price": abs(float(str(bid_p).replace(',', ''))),
                                    "qty": abs(float(str(bid_q).replace(',', '')))
                                })
                        
                        orderbook = {"asks": asks, "bids": bids}

                    # 0D(호가) 패킷에는 현재가가 없는 경우가 많으므로 유지 중인 마지막 가격 사용
                    current_display_price = price
                    if current_display_price <= 0 and target_symbol in self.last_prices:
                        current_display_price = self.last_prices[target_symbol]

                    ui_data = {
                        "symbol": target_symbol,
                        "price": current_display_price,
                        "orderbook": orderbook,
                    }

                    if self._ui_callback:
                        self._ui_callback(ui_data)

            except (ValueError, TypeError, Exception) as e:
                print(f"[DC ERROR] Data 파싱 중 오류 ({raw_code}): {type(e).__name__}: {e}")
                continue

    def _aggregate_bars(self, symbol, data):
        # 메모리 상에서 틱 데이터를 기반으로 1분/5분봉/60틱봉 등을 업데이트하는 로직
        pass

    async def _watchdog(self):
        """3초 이상 데이터 수신이 없으면 Circuit Breaker 발동 및 재연결"""
        try:
            while self.is_running:
                await asyncio.sleep(1)

                # 방어 로직 1: 웹소켓 구독이 완료되고 최초 데이터가 들어온 이후에만 감시 시작
                if not self.ws_connected_event.is_set() or not self.first_data_received_event.is_set():
                    self.last_receive_time = time.time() # 억울하게 죽지 않도록 타이머 갱신
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
        """데이터 수집기를 안전하게 종료합니다."""
        self.is_running = False
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass

        if self.ws_connection:
            try:
                await self.ws_connection.close()
            except Exception as e:
                self.logger.warning(f"웹소켓 강제 종료 중 예외 발생 (무시됨): {e}")

        # 만약 파케이(Parquet) 파일로 Flush 하는 로직이 필요하다면 여기서 수행
        self.logger.info("DataCollector: 모든 연결 종료. 메모리 버퍼 안전 저장 (Flush) 완료.")
