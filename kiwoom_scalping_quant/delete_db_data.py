import os
from influxdb_client import InfluxDBClient
from datetime import datetime
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 1. InfluxDB 접속 정보 (환경 변수에서 로드)
url = os.getenv("INFLUX_URL", "http://localhost:8086")
token = os.getenv("INFLUX_TOKEN")
org = os.getenv("INFLUX_ORG", "my-trade")
bucket = "stock_data" # 기본 버킷명

if not token:
    print("Error: INFLUX_TOKEN not found in .env file.")
    exit(1)

client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()
delete_api = client.delete_api()

print(f"Searching for symbols in bucket '{bucket}'...")

# 2. historical_data에 있는 모든 symbol 태그 값 가져오기
query = f'''
import "influxdata/influxdb/schema"
schema.measurementTagValues(
  bucket: "{bucket}",
  measurement: "historical_data",
  tag: "symbol"
)
'''

try:
    tables = query_api.query(query)

    al_symbols = ['TEST999']
    for table in tables:
        for record in table.records:
            symbol_value = record.get_value()
            if isinstance(symbol_value, str) and symbol_value.endswith("_AL"):
                al_symbols.append(symbol_value)

    # 3. 삭제 진행
    if not al_symbols:
        print("No symbols found to delete.")
    else:
        print(f"Found {len(al_symbols)} symbols. Starting deletion...")

        start = "1970-01-01T00:00:00Z"
        stop = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

        for target_symbol in al_symbols:
            # Measurement와 symbol 태그를 AND 조건으로 묶어서 삭제
            predicate = f'_measurement="historical_data" AND symbol="{target_symbol}"'
            try:
                delete_api.delete(start, stop, predicate, bucket, org)
                print(f"  Deleted: {target_symbol}")
            except Exception as e:
                print(f"  Failed to delete {target_symbol}: {e}")

        print("Cleanup completed.")

except Exception as e:
    print(f"Error during database operation: {e}")
finally:
    client.close()