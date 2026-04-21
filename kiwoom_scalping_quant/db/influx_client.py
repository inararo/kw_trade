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
            import dateutil.parser
            from datetime import datetime, timezone

            ts = data.get("timestamp")
            if isinstance(ts, str):
                try:
                    ts = dateutil.parser.parse(ts)
                except Exception:
                    ts = datetime.now(timezone.utc)
            elif ts is None:
                ts = datetime.now(timezone.utc)

            point = Point("tick_data") \
                .tag("symbol", str(data.get("symbol", "UNKNOWN"))) \
                .field("price", float(data.get("price", 0.0))) \
                .field("volume", float(data.get("volume", 0.0))) \
                .time(ts)

            self.batch_queue.append(point)

            if len(self.batch_queue) >= self.batch_size:
                await self._flush_batch()

        except Exception as e:
            self.logger.error(f"Point 변환 오류: {e}")

    async def fetch_recent_data(self, symbol: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        학습용 데이터를 제공하기 위해 InfluxDB에서 특정 종목의 최근 데이터를 가져옵니다.
        """
        self.logger.info(f"InfluxDB: [{symbol}] 과거 데이터 {limit}건 조회 시도")

        try:
            query_api = self.client.query_api()
            # Simple Flux query to get recent data
            query = f'''
                from(bucket: "{self.bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r["_measurement"] == "historical_data" or r["_measurement"] == "tick_data")
                |> filter(fn: (r) => r["symbol"] == "{symbol}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"], desc: true)
                |> limit(n: {limit})
            '''

            tables = await query_api.query(query, org=self.org)

            results = []
            for table in tables:
                for record in table.records:
                    results.append({
                        "timestamp": record.get_time(),
                        "price": float(record.values.get("price", 0.0)),
                        "volume": float(record.values.get("volume", 0.0))
                    })

            if not results:
                self.logger.warning(f"InfluxDB: [{symbol}] 조회된 데이터가 없습니다 (0건).")

            # 역순 정렬을 원래 시간순(오름차순)으로 뒤집어서 반환
            return list(reversed(results))

        except Exception as e:
            self.logger.error(f"InfluxDB 조회 실패: {str(e)}")
            return []

    async def bulk_insert(self, data_list: List[Dict[str, Any]], measurement: str = "historical_data"):
        """과거 데이터(리스트/데이터프레임 등)를 InfluxDB에 한 번에 Bulk Insert 합니다."""
        if not data_list:
            return

        points = []
        import dateutil.parser
        from datetime import datetime, timezone

        for data in data_list:
            try:
                # Parse timestamp safely
                ts = data.get("timestamp")
                if isinstance(ts, str):
                    try:
                        # Try to parse string timestamp (RFC3339 or basic)
                        ts = dateutil.parser.parse(ts)
                    except Exception:
                        # Fallback if parsing fails
                        ts = datetime.now(timezone.utc)
                elif ts is None:
                    ts = datetime.now(timezone.utc)

                point = Point(measurement) \
                    .tag("symbol", str(data.get("symbol", "UNKNOWN"))) \
                    .field("price", float(data.get("price", 0.0))) \
                    .field("volume", float(data.get("volume", 0.0))) \
                    .time(ts)
                points.append(point)
            except Exception as e:
                self.logger.warning(f"Bulk Insert 포인트 변환 실패 (Data: {data}): {e}")

        if points:
            try:
                # Execute actual DB write
                await self.write_api.write(bucket=self.bucket, record=points)
                self.logger.info(f"Bulk Insert 완료: InfluxDB에 {len(points)}건 적재 요청 성공.")
            except Exception as e:
                # Safely extract HTTP status and reason if available in InfluxDBError
                status = getattr(e, 'response', None)
                status_code = status.status if status else 'Unknown'
                reason = status.reason if status else str(e)
                self.logger.error(f"🚨 Bulk Insert DB 전송 실패! [Status: {status_code}] Reason: {reason}")

    async def _flush_batch(self):
        if not self.batch_queue:
            return

        points_to_write = self.batch_queue[:]
        self.batch_queue.clear()

        try:
            await self.write_api.write(bucket=self.bucket, record=points_to_write)
            self.logger.debug(f"InfluxDB Batch Write 완료: {len(points_to_write)}건")
        except Exception as e:
            status = getattr(e, 'response', None)
            status_code = status.status if status else 'Unknown'
            reason = status.reason if status else str(e)
            self.logger.error(f"🚨 InfluxDB Batch Write 실패! [Status: {status_code}] Reason: {reason}")
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
