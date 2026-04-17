import asyncio
import logging
from typing import Dict, Any, List
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from influxdb_client import Point

class AsyncInfluxDBClient:
    """비동기 배치 처리를 지원하는 InfluxDB 클라이언트"""
    def __init__(self, config: Dict[str, Any]):
        self.url = config.get("influx_url", "http://localhost:8086")
        self.token = config.get("influx_token", "YOUR_TOKEN")
        self.org = config.get("influx_org", "YOUR_ORG")
        self.bucket = config.get("influx_bucket", "kiwoom_data")

        self.client = InfluxDBClientAsync(url=self.url, token=self.token, org=self.org)
        self.write_api = self.client.write_api()

        self.batch_queue = []
        self.batch_size = config.get("db_batch_size", 500)
        self.logger = logging.getLogger("InfluxDBClient")
        self.is_running = False

    async def start(self):
        """이벤트 루프가 시작된 후 메인 태스크에서 호출되어야 함"""
        self.is_running = True
        asyncio.create_task(self._periodic_flush())

    async def _periodic_flush(self):
        """배치가 꽉 차지 않아도 N초마다 남은 데이터를 플러시"""
        while self.is_running:
            await asyncio.sleep(5.0) # 5초 대기
            if self.batch_queue:
                await self._flush_batch()

    async def write_tick(self, data: Dict[str, Any]):
        """틱 데이터를 포인트로 변환하여 큐에 적재"""
        try:
            point = Point("tick_data") \
                .tag("symbol", data.get("symbol")) \
                .field("price", float(data.get("price", 0))) \
                .field("volume", float(data.get("volume", 0))) \
                .time(data.get("timestamp")) # timestamp가 유효한 datetime/ns 형식이라고 가정

            self.batch_queue.append(point)

            if len(self.batch_queue) >= self.batch_size:
                await self._flush_batch()

        except Exception as e:
            self.logger.error(f"Point 변환 오류: {e}")

    async def _flush_batch(self):
        if not self.batch_queue:
            return

        points_to_write = self.batch_queue[:]
        self.batch_queue.clear()

        try:
            await self.write_api.write(bucket=self.bucket, record=points_to_write)
            self.logger.debug(f"InfluxDB Batch Write 완료: {len(points_to_write)}건")
        except Exception as e:
            self.logger.error(f"InfluxDB Write 실패: {e}")
            # 실패 시 다시 큐에 넣거나 로컬 파일 시스템에 Fallback 처리 가능

    async def close(self):
        self.is_running = False
        await self._flush_batch()
        await self.client.close()
