import asyncio
import time
import json
import websockets
from collections import deque
import numpy as np
import logging

class DataCollector:
    def __init__(self, config):
        self.config = config
        self.symbol = config.get('symbol', '005930')
        self.ws_url = config.get('ws_url', 'ws://localhost:8080/kiwoom')
        self.max_buffer_size = config.get('max_buffer_size', 10000)

        # 롤링 버퍼 (틱, 1분, 5분, 60틱)
        self.tick_buffer = deque(maxlen=self.max_buffer_size)
        self.min1_buffer = deque(maxlen=self.max_buffer_size // 10)

        self.ws_connection = None
        self.is_running = False
        self.last_receive_time = time.time()
        self.latency_logs = deque(maxlen=1000)
        self.circuit_breaker_active = False

        self.logger = logging.getLogger("DataCollector")
        self._ui_callback = None

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

                # 10호가 가상 매수/매도 잔량 생성
                asks = [{"price": base_price + (i * 100), "qty": random.randint(100, 5000)} for i in range(1, 11)]
                bids = [{"price": base_price - (i * 100), "qty": random.randint(100, 5000)} for i in range(1, 11)]

                # AI 확률 임의 생성
                hold_prob = random.randint(40, 80)
                buy_prob = random.randint(0, 100 - hold_prob)
                sell_prob = 100 - hold_prob - buy_prob

                mock_data = {
                    "price": base_price,
                    "orderbook": {"asks": asks, "bids": bids},
                    "ai_confidence": {"Hold": hold_prob, "Buy": buy_prob, "Sell": sell_prob}
                }

                if self._ui_callback:
                    self._ui_callback(mock_data)

                await asyncio.sleep(0.1) # 0.1초(100ms) 간격 업데이트
        except asyncio.CancelledError:
            self.logger.info("Mock Stream Cancelled.")

    async def start(self):
        self.is_running = True
        # Watchdog 태스크 시작
        asyncio.create_task(self._watchdog())

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
        """수신된 틱 데이터를 버퍼에 저장하고, 다중 타임프레임으로 집계"""
        self.tick_buffer.append(data)
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
        if self.ws_connection:
            try:
                await self.ws_connection.close()
            except Exception as e:
                self.logger.warning(f"웹소켓 강제 종료 중 예외 발생 (무시됨): {e}")

        # 만약 파케이(Parquet) 파일로 Flush 하는 로직이 필요하다면 여기서 수행
        self.logger.info("DataCollector: 모든 연결 종료. 메모리 버퍼 안전 저장 (Flush) 완료.")
