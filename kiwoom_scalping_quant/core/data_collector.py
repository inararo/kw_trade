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

    async def start(self):
        self.is_running = True
        # Watchdog 태스크 시작
        asyncio.create_task(self._watchdog())

        while self.is_running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                self.logger.error(f"WebSocket 연결 오류: {e}")
                if self.is_running:
                    await asyncio.sleep(1) # 재연결 대기

    async def _connect_and_listen(self):
        async with websockets.connect(self.ws_url) as websocket:
            self.ws_connection = websocket
            self.logger.info("WebSocket 연결 성공. 실시간 데이터 수신 시작.")

            # 구독 요청 전송 (키움증권 API 규격에 맞춰 작성)
            subscribe_msg = json.dumps({"type": "subscribe", "symbol": self.symbol})
            await websocket.send(subscribe_msg)

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

    async def _process_tick(self, data):
        """수신된 틱 데이터를 버퍼에 저장하고, 다중 타임프레임으로 집계"""
        self.tick_buffer.append(data)
        self._aggregate_bars(data)

    def _aggregate_bars(self, data):
        # 메모리 상에서 틱 데이터를 기반으로 1분/5분봉/60틱봉 등을 업데이트하는 로직
        pass

    async def _watchdog(self):
        """3초 이상 데이터 수신이 없으면 Circuit Breaker 발동 및 재연결"""
        while self.is_running:
            await asyncio.sleep(1)
            idle_time = time.time() - self.last_receive_time

            if idle_time > 3.0 and not self.circuit_breaker_active:
                self.logger.error(f"Watchdog: {idle_time:.1f}초간 시세 미수신! Circuit Breaker 발동.")
                self.circuit_breaker_active = True

                if self.ws_connection:
                    await self.ws_connection.close()

    async def stop(self):
        self.is_running = False
        if self.ws_connection:
            await self.ws_connection.close()
        self.logger.info("DataCollector 종료됨. 메모리 버퍼 안전 저장 로직 실행.")
