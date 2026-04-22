import asyncio
import logging
from typing import Dict, Any, List
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from influxdb_client import Point

class AsyncInfluxDBClient:
    """비동기 배치 처리를 지원하는 InfluxDB 클라이언트"""
    def __init__(self, config: Any):
        self.url = config.get("INFLUX_URL", "http://localhost:8086")
        self.token = config.get("INFLUX_TOKEN", "YOUR_TOKEN")
        self.org = config.get("INFLUX_ORG", "my-trade")
        self.bucket = config.get("influx_bucket", "stock_data")
        self.logger = logging.getLogger("AsyncInfluxDBClient")
        
        # 설정 로드 확인을 위한 로그 추가
        self.logger.info(f"InfluxDB 클라이언트 초기화: URL={self.url}, ORG={self.org}, BUCKET={self.bucket}")
        if self.token == "YOUR_TOKEN" or not self.token:
            self.logger.warning("⚠️ InfluxDB 토큰이 기본값(YOUR_TOKEN)이거나 비어있습니다. .env 파일을 확인하세요.")

        # 타임아웃 설정을 300초(5분)로 연장 (대량의 과거 데이터 조회 대응)
        from aiohttp import ClientTimeout
        timeout = ClientTimeout(total=300)
        self.client = InfluxDBClientAsync(url=self.url, token=self.token, org=self.org, timeout=timeout)
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
                .field("open", float(data.get("open", 0.0))) \
                .field("high", float(data.get("high", 0.0))) \
                .field("low", float(data.get("low", 0.0))) \
                .field("price", float(data.get("price", 0.0))) \
                .field("volume", float(data.get("volume", 0.0))) \
                .time(ts)

            self.batch_queue.append(point)

            if len(self.batch_queue) >= self.batch_size:
                await self._flush_batch()

        except Exception as e:
            self.logger.error(f"Point 변환 오류: {e}")

    async def get_last_timestamp(self, symbol: str) -> str:
        """특정 종목의 가장 최신 데이터 타임스탬프를 가져옵니다. (증분 수집용)"""
        try:
            query_api = self.client.query_api()
            
            # 접미사 제거된 순수 심볼 추출
            clean_symbol = symbol.split('_')[0].strip()
            
            query = f'''
                from(bucket: "{self.bucket}")
                |> range(start: -1y)
                |> filter(fn: (r) => r["_measurement"] == "historical_data" or r["_measurement"] == "tick_data")
                |> filter(fn: (r) => r["symbol"] == "{symbol}" or r["symbol"] == "{clean_symbol}")
                |> last()
            '''
            tables = await query_api.query(query, org=self.org)
            for table in tables:
                for record in table.records:
                    # _time 필드를 ISO 형식 문자열로 변환
                    return record.get_time().isoformat().replace("+00:00", "").replace("Z", "").split(".")[0]
            return None
        except Exception as e:
            self.logger.error(f"InfluxDB 마지막 타임스탬프 조회 실패 (Symbol: {symbol}/{clean_symbol}): {e}")
            return None

    async def fetch_recent_data(self, symbol: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        학습용 데이터를 제공하기 위해 InfluxDB에서 특정 종목의 최근 데이터를 가져옵니다.
        """
        # 접미사 제거된 순수 심볼 추출
        clean_symbol = symbol.split('_')[0].strip()
        search_range = "-1y" # 최근 30일에서 1년으로 확장 (과거 수집분 포함)

        self.logger.error(f"InfluxDB: [{symbol}/{clean_symbol}] 과거 데이터 {limit}건 조회 시도 (범위: {search_range})")

        try:
            query_api = self.client.query_api()
            # 원본 심볼과 정규화된 심볼을 모두 검색 (데이터 수집 시점의 형식 차이 대응)
            query = f'''
                from(bucket: "{self.bucket}")
                |> range(start: {search_range})
                |> filter(fn: (r) => r["_measurement"] == "historical_data" or r["_measurement"] == "tick_data")
                |> filter(fn: (r) => r["symbol"] == "{symbol}" or r["symbol"] == "{clean_symbol}")
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
                        "open": float(record.values.get("open", 0.0)),
                        "high": float(record.values.get("high", 0.0)),
                        "low": float(record.values.get("low", 0.0)),
                        "price": float(record.values.get("price", 0.0)),
                        "volume": float(record.values.get("volume", 0.0))
                    })

            if not results:
                self.logger.error(
                    f"InfluxDB: [{symbol}] 조회 결과가 없습니다. "
                    f"(필터: symbol='{symbol}' OR '{clean_symbol}', 범위: {search_range})"
                )

            # 역순 정렬을 원래 시간순(오름차순)으로 뒤집어서 반환
            return list(reversed(results))

        except Exception as e:
            self.logger.error(f"   [에러] InfluxDB [{symbol}] 조회 실패: {type(e).__name__} - {str(e)}")
            return []

    async def fetch_data_by_range(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        특정 기간(시작일~종료일)의 데이터를 InfluxDB에서 조회합니다.
        start_date, end_date: "YYYYMMDD" 형식 문자열
        """
        clean_symbol = symbol.split('_')[0].strip()
        
        try:
            # 날짜 포맷팅 (YYYYMMDD -> RFC3339)
            import datetime
            s_dt = datetime.datetime.strptime(start_date, "%Y%m%d")
            e_dt = datetime.datetime.strptime(end_date, "%Y%m%d").replace(hour=23, minute=59, second=59)
            
            start_iso = s_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            stop_iso = e_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            self.logger.info(f"InfluxDB: [{symbol}] 기간 조회 ({start_iso} ~ {stop_iso})")

            query_api = self.client.query_api()
            query = f'''
                from(bucket: "{self.bucket}")
                |> range(start: {start_iso}, stop: {stop_iso})
                |> filter(fn: (r) => r["_measurement"] == "historical_data" or r["_measurement"] == "tick_data")
                |> filter(fn: (r) => r["symbol"] == "{symbol}" or r["symbol"] == "{clean_symbol}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"], desc: false)
            '''

            tables = await query_api.query(query, org=self.org)
            results = []
            for table in tables:
                for record in table.records:
                    results.append({
                        "timestamp": record.get_time(),
                        "open": float(record.values.get("open", 0.0)),
                        "high": float(record.values.get("high", 0.0)),
                        "low": float(record.values.get("low", 0.0)),
                        "price": float(record.values.get("price", 0.0)),
                        "volume": float(record.values.get("volume", 0.0))
                    })
            
            return results

        except Exception as e:
            self.logger.error(f"InfluxDB [{symbol}] 범위 조회 실패: {e}")
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
                    .field("open", float(data.get("open", 0.0))) \
                    .field("high", float(data.get("high", 0.0))) \
                    .field("low", float(data.get("low", 0.0))) \
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

    async def delete_data(self, measurement: str, symbol: str = None):
        """특정 측정 항목 또는 종목의 데이터를 삭제합니다."""
        try:
            delete_api = self.client.delete_api()
            start = "1970-01-01T00:00:00Z"
            import datetime
            stop = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            predicate = f'_measurement="{measurement}"'
            if symbol:
                predicate += f' AND symbol="{symbol}"'
            
            await delete_api.delete(start, stop, predicate, bucket=self.bucket, org=self.org)
            self.logger.info(f"InfluxDB 데이터 삭제 완료: measurement={measurement}, symbol={symbol}")
            return True
        except Exception as e:
            self.logger.error(f"InfluxDB 데이터 삭제 실패: {e}")
            return False

    async def close(self):
        """Safely closes the client and flushes remaining data."""
        self.is_running = False
        
        # 1. Stop periodic flush task
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await asyncio.wait_for(self._flush_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        # 2. Force flush remaining data
        if self.batch_queue:
            try:
                self.logger.info(f"InfluxDB: Attempting to flush remaining {len(self.batch_queue)} points before closing...")
                await asyncio.wait_for(self._flush_batch(), timeout=2.0)
            except Exception as e:
                self.logger.error(f"InfluxDB: Final flush failed: {e}")

        # 3. Close HTTP session and client
        try:
            # [Windows Stability] Apply timeout during session close
            await asyncio.wait_for(self.client.close(), timeout=2.0)
            self.logger.info("InfluxDB: Client closed successfully.")
        except Exception as e:
            self.logger.warning(f"InfluxDB: Exception during client close (ignored): {e}")
