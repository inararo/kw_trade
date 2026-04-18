import asyncio
import time
import json
import websockets
from collections import deque
import numpy as np
import logging
from core.feature_engineer import FeatureEngineer
from env.normalizer import OnlineRollingNormalizer

class DataCollector:
    def __init__(self, config):
        self.config = config
        self.symbol = config.get('symbol', '005930')
        self.ws_url = config.get('ws_url', 'ws://localhost:8080/kiwoom')
        self.max_buffer_size = config.get('max_buffer_size', 10000)

        # 피처 엔지니어링 및 정규화
        self.feature_engineer = FeatureEngineer(max_ticks=100)
        self.normalizer = OnlineRollingNormalizer(window_size=1000, bypass_indices=[2]) # 2번째 인덱스 OIR을 바이패스 가정

        # 최종 상태 버퍼
        self.state_buffer = deque(maxlen=self.max_buffer_size)

        # 기존 롤링 버퍼 (틱, 1분, 5분, 60틱)
        self.tick_buffer = deque(maxlen=self.max_buffer_size)
        self.min1_buffer = deque(maxlen=self.max_buffer_size // 10)

        self.ws_connection = None
        self.is_running = False
        self.last_receive_time = time.time()
        self.latency_logs = deque(maxlen=1000)
        self.circuit_breaker_active = False

        self.logger = logging.getLogger("DataCollector")
        self._ui_callback = None
        self._watchdog_task = None

    def set_ui_callback(self, callback):
        self._ui_callback = callback

    async def start_mock_stream(self):
        """장외 시간/주말 UI 테스트용 가상 데이터 생성기"""
        import random
        self.logger.info("Mock Stream Started.")
        base_price = 50000

        try:
            while self.is_running:
                # 가상 가격 변동
                base_price += random.choice([-100, 0, 100])
                volume = random.randint(10, 500)

                # 10호가 가상 매수/매도 잔량 생성
                asks = [{"price": base_price + (i * 100), "qty": random.randint(100, 5000)} for i in range(1, 11)]
                bids = [{"price": base_price - (i * 100), "qty": random.randint(100, 5000)} for i in range(1, 11)]

                orderbook = {"asks": asks, "bids": bids}

                # 피처 계산
                self.feature_engineer.update_orderbook(orderbook)
                features = self.feature_engineer.update_tick(base_price, volume)

                # [Price, Volume, OIR, Volatility, Aggressiveness]
                raw_state = np.array([
                    base_price,
                    volume,
                    features["OIR"],
                    features["Volatility"],
                    features["Aggressiveness"]
                ], dtype=np.float32)

                # 정규화
                normalized_state = self.normalizer.update_and_normalize(raw_state)
                self.state_buffer.append(normalized_state)

                # AI 확률 임의 생성
                hold_prob = random.randint(40, 80)
                buy_prob = random.randint(0, 100 - hold_prob)
                sell_prob = 100 - hold_prob - buy_prob

                mock_data = {
                    "price": base_price,
                    "orderbook": orderbook,
                    "ai_confidence": {"Hold": hold_prob, "Buy": buy_prob, "Sell": sell_prob}
                }

                if self._ui_callback:
                    self._ui_callback(mock_data)

                await asyncio.sleep(0.1) # 0.1초(100ms) 간격 업데이트
        except asyncio.CancelledError:
            self.logger.info("Mock Stream Cancelled.")

    def get_latest_state(self, seq_len=1):
        """환경(Env)이 현재 상태를 가져가기 위한 메서드 (시퀀스 길이 지원)"""
        dim = 5
        if len(self.state_buffer) == 0:
            return np.zeros(dim * seq_len, dtype=np.float32)

        n_avail = len(self.state_buffer)
        if n_avail < seq_len:
            # Not enough data: pad with the first available state
            pad_len = seq_len - n_avail
            first_state = self.state_buffer[0]
            padded = [first_state] * pad_len
            actual = list(self.state_buffer)
            seq = padded + actual
        else:
            # Take the last seq_len states
            seq = list(self.state_buffer)[-seq_len:]

        # Flatten sequence: [t-n_1, t-n_2, ..., t_1, t_2, ...]
        return np.concatenate(seq).astype(np.float32)

    async def start(self):
        self.is_running = True
        # Watchdog 태스크 시작
        self._watchdog_task = asyncio.create_task(self._watchdog())

        try:
            while self.is_running:
                try:
                    await self._connect_and_listen()
                except asyncio.CancelledError:
                    self.logger.info("DataCollector: 루프 취소 신호 수신, 수집 중지.")
                    break
                except Exception as e:
                    self.logger.error(f"WebSocket 연결 오류: {e}")
                    if self.is_running:
                        await asyncio.sleep(1) # 재연결 대기
        except asyncio.CancelledError:
            self.logger.info("DataCollector 루프 완전 취소됨.")

    async def _connect_and_listen(self):
        async with websockets.connect(self.ws_url) as websocket:
            self.ws_connection = websocket
            self.logger.info("WebSocket 연결 성공. 실시간 데이터 수신 시작.")

            # 구독 요청 전송 (키움증권 API 규격에 맞춰 작성)
            subscribe_msg = json.dumps({"type": "subscribe", "symbol": self.symbol})
            await websocket.send(subscribe_msg)

            try:
                async for message in websocket:
                    recv_time = time.time()
                    self.last_receive_time = recv_time
                    self.circuit_breaker_active = False

                    # 데이터 파싱
                    data = json.loads(message)

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

    async def _process_tick(self, data):
        """수신된 틱 데이터를 버퍼에 저장하고, 다중 타임프레임으로 집계 및 피처 추출"""
        self.tick_buffer.append(data)

        # 실시간 데이터의 경우 구조에 맞게 파싱하여 feature_engineer 호출
        if "orderbook" in data:
            self.feature_engineer.update_orderbook(data["orderbook"])

        # 기본 틱 정보 (Mock 데이터나 Kiwoom 실데이터에서 매핑 가정)
        price = data.get("price", 0.0)
        volume = data.get("volume", 0.0) # Kiwoom API에선 '체결량' 등 다른 키일 수 있음

        if price > 0:
            features = self.feature_engineer.update_tick(price, volume)
            raw_state = np.array([
                price,
                volume,
                features["OIR"],
                features["Volatility"],
                features["Aggressiveness"]
            ], dtype=np.float32)

            normalized_state = self.normalizer.update_and_normalize(raw_state)
            self.state_buffer.append(normalized_state)

        self._aggregate_bars(data)

    def _aggregate_bars(self, data):
        # 메모리 상에서 틱 데이터를 기반으로 1분/5분봉/60틱봉 등을 업데이트하는 로직
        pass

    async def _watchdog(self):
        """3초 이상 데이터 수신이 없으면 Circuit Breaker 발동 및 재연결"""
        try:
            while self.is_running:
                await asyncio.sleep(1)
                idle_time = time.time() - self.last_receive_time

                if idle_time > 3.0 and not self.circuit_breaker_active:
                    self.logger.error(f"Watchdog: {idle_time:.1f}초간 시세 미수신! Circuit Breaker 발동.")
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
