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
                .tag("symbol", str(data.get("symbol", "UNKNOWN")).split('_')[0].strip()) \
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

        print(f"InfluxDB: [{clean_symbol}] 과거 데이터 {limit}건 조회 시도 (범위: {search_range})")

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

    async def delete_symbol_data(self, symbol_code: str):
        """
        특정 종목의 모든 데이터를 DB에서 영구 삭제합니다.
        [초강력 개선] DB에서 해당 코드가 포함된 모든 실제 태그 값을 먼저 조회한 뒤 정밀 타격 삭제합니다.
        """
        try:
            clean_code = symbol_code.split('_')[0].strip()
            self.logger.info(f"InfluxDB: Symbol [{clean_code}] searching for actual tags in DB...")
            
            # 1. DB에 실제 존재하는 관련 심볼 태그 모두 찾기
            query = f'import "influxdata/influxdb/schema" schema.tagValues(bucket: "{self.bucket}", tag: "symbol")'
            query_api = self.client.query_api()
            tables = await query_api.query(query, org=self.org)
            
            actual_db_tags = []
            for table in tables:
                for record in table.records:
                    tag_val = record.get_value()
                    if tag_val and clean_code in tag_val:
                        actual_db_tags.append(tag_val)
            
            # 검색된 태그가 없으면 요청받은 코드라도 포함
            if not actual_db_tags:
                actual_db_tags = [symbol_code, clean_code, f"{clean_code}_AL"]
            
            target_codes = list(set(actual_db_tags))
            measurements = ["historical_data", "tick_data"]
            
            self.logger.info(f"InfluxDB: Targeted deletion targets -> {target_codes}")
            
            total_deleted = 0
            for m in measurements:
                for code in target_codes:
                    if not code: continue
                    # [변경] delete_data 내부에서 복잡한 로직 대신 직접 호출
                    res = await self._execute_delete(m, code)
                    if res:
                        total_deleted += 1
            
            self.logger.info(f"InfluxDB: [{symbol_code}] related {total_deleted} data point groups deleted.")
            return True
        except Exception as e:
            self.logger.error(f"InfluxDB symbol data deletion failed ({symbol_code}): {e}")
            return False

    async def _execute_delete(self, measurement: str, symbol: str):
        """내부 삭제 실행 함수"""
        try:
            delete_api = self.client.delete_api()
            start = "1970-01-01T00:00:00Z"
            import datetime
            stop = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Predicate를 가장 단순하고 확실하게 작성
            predicate = f'_measurement="{measurement}" AND symbol="{symbol}"'
            
            await delete_api.delete(start, stop, predicate, bucket=self.bucket, org=self.org)
            return True
        except Exception:
            return False

    async def fetch_data_by_range(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        특정 기간(시작일~종료일)의 데이터를 InfluxDB에서 조회합니다.
        start_date, end_date: "YYYYMMDD" 또는 "YYYY-MM-DD" 형식 지원
        """
        if not start_date or not end_date:
            self.logger.warning(f"InfluxDB: 조회 날짜가 비어있습니다. (Start: {start_date}, End: {end_date})")
            return []

        clean_symbol = symbol.split('_')[0].strip()
        
        try:
            import datetime
            # 하이픈(-) 제거하여 YYYYMMDD 포맷으로 통일
            s_str = start_date.replace("-", "").replace("/", "").strip()
            e_str = end_date.replace("-", "").replace("/", "").strip()
            
            s_dt = datetime.datetime.strptime(s_str, "%Y%m%d")
            # 종료일은 해당 날짜를 완벽히 포함하기 위해 다음날 자정(00:00:00)으로 설정 (InfluxDB stop은 exclusive임)
            e_dt = datetime.datetime.strptime(e_str, "%Y%m%d") + datetime.timedelta(days=1)
            
            # [방어 로직] InfluxDB의 "cannot query an empty range" 에러 방지
            if s_dt >= e_dt:
                self.logger.warning(f"InfluxDB: 시작일({s_str})이 종료일({e_str})보다 늦거나 같습니다. 범위를 강제 조정합니다.")
                # 시작일과 종료일이 같으면 종료일을 +1초 하여 최소 범위 확보
                e_dt = s_dt + datetime.timedelta(seconds=1)
            
            start_iso = s_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            stop_iso = e_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            self.logger.info(f"InfluxDB: [{symbol}/{clean_symbol}] 기간 조회 ({start_iso} ~ {stop_iso})")

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
                    .tag("symbol", str(data.get("symbol", "UNKNOWN")).split('_')[0].strip()) \
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

    async def get_all_symbols(self) -> List[str]:
        """조직 내 버킷의 모든 고유 심볼(Tag value)을 조회합니다."""
        query = f'''
            import "influxdata/influxdb/schema"
            schema.tagValues(bucket: "{self.bucket}", tag: "symbol")
        '''
        try:
            query_api = self.client.query_api()
            tables = await query_api.query(query, org=self.org)
            symbols = []
            for table in tables:
                for record in table.records:
                    v = record.get_value()
                    if v and v != "UNKNOWN":
                        # [수정] 접미사 제거 후 중복 방지
                        symbols.append(v.split('_')[0].strip())
            return sorted(list(set(symbols)))
        except Exception as e:
            self.logger.error(f"InfluxDB 심볼 리스트 조회 실패: {e}")
            return []

    async def delete_data(self, measurement: str, symbol: str = None):
        """
        특정 측정 항목 또는 종목의 데이터를 삭제합니다.
        """
        try:
            delete_api = self.client.delete_api()
            # 1970년부터 현재+1일 후까지 넉넉하게 잡음
            start = "1970-01-01T00:00:00Z"
            import datetime
            stop = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            
            predicate = f'_measurement="{measurement}"'
            if symbol:
                predicate += f' AND symbol="{symbol}"'
            
            self.logger.debug(f"InfluxDB Delete 실행: Predicate=[{predicate}]")
            await delete_api.delete(start, stop, predicate, bucket=self.bucket, org=self.org)
            return True
        except Exception as e:
            self.logger.error(f"InfluxDB 데이터 삭제 중 오류 (M:{measurement}, S:{symbol}): {e}")
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
