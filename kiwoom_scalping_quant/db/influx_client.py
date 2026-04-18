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
        self._flush_task = None

    async def start(self):
        """이벤트 루프가 시작된 후 메인 태스크에서 호출되어야 함"""
        self.is_running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())

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

    async def fetch_recent_data(self, symbol: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        학습용 데이터를 제공하기 위해 InfluxDB에서 특정 종목의 최근 데이터를 가져옵니다.
        본 프로젝트에서는 API 대신 Mock List를 반환하여 학습 파이프라인 구조를 증명합니다.
        """
        self.logger.info(f"InfluxDB: [{symbol}] 학습용 과거 데이터 {limit}건 조회 (Mock)")
        import asyncio
        import random
        await asyncio.sleep(0.5) # DB 조회 지연 모사

        mock_data = []
        base_price = 50000
        for i in range(limit):
            base_price += random.randint(-50, 50)
            mock_data.append({
                "timestamp": i,
                "price": base_price,
                "volume": random.randint(10, 500)
            })
        return mock_data

    async def bulk_insert(self, data_list: List[Dict[str, Any]], measurement: str = "historical_data"):
        """과거 데이터(리스트/데이터프레임 등)를 InfluxDB에 한 번에 Bulk Insert 합니다."""
        if not data_list:
            return

        points = []
        for data in data_list:
            try:
                point = Point(measurement) \
                    .tag("symbol", data.get("symbol", "UNKNOWN")) \
                    .field("price", float(data.get("price", 0))) \
                    .time(data.get("timestamp"))
                points.append(point)
            except Exception as e:
                self.logger.warning(f"Bulk Insert 포인트 변환 실패: {e}")

        if points:
            try:
                # InfluxDB의 write_api는 리스트를 받아 한 번에 전송 가능
                # Sandbox 환경에서 InfluxDB가 구동되어 있지 않으므로 모의 로깅으로 처리
                # await self.write_api.write(bucket=self.bucket, record=points)
                self.logger.info(f"Bulk Insert (Mocked) 완료: InfluxDB에 {len(points)}건 적재 요청 성공.")
            except Exception as e:
                self.logger.error(f"Bulk Insert DB 전송 실패: {e}")

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

    async def ping(self) -> bool:
        """InfluxDB 서버의 상태(Ping)를 비동기로 점검합니다."""
        try:
            return await self.client.ping()
        except Exception as e:
            self.logger.error(f"InfluxDB Ping 실패: {e}")
            return False

    async def close(self):
        self.is_running = False
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self._flush_batch()
        await self.client.close()
