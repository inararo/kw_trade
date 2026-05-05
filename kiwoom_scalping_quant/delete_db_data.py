from influxdb_client import InfluxDBClient
from datetime import datetime

# 1. InfluxDB 접속 정보 세팅 (본인 환경에 맞게 수정)
url = "http://localhost:8086"
token = "사용자님의_인플럭스_토큰"
org = "사용자님의_조직(org)_이름"
bucket = "저장된_버킷_이름"

client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()
delete_api = client.delete_api()

print("🔍 InfluxDB에서 '_AL'로 끝나는 Measurement를 검색합니다...")

# 2. 해당 버킷에 있는 모든 Measurement(테이블) 이름 가져오기
query = f'''
import "influxdata/influxdb/schema"
schema.measurements(bucket: "{bucket}")
'''

try:
    tables = query_api.query(query)

    # '_AL'로 끝나는 이름만 필터링해서 리스트에 담기
    al_measurements = []
    for table in tables:
        for record in table.records:
            m_name = record.get_value()
            if m_name.endswith("_AL"):
                al_measurements.append(m_name)

    # 3. 삭제 진행
    if not al_measurements:
        print("💡 삭제할 '_AL' 데이터가 없습니다.")
    else:
        print(f"⚠️ 총 {len(al_measurements)}개의 '_AL' 종목을 찾았습니다. 삭제를 시작합니다.")

        start = "1970-01-01T00:00:00Z"
        stop = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

        for target_name in al_measurements:
            predicate = f'_measurement="{target_name}"'
            try:
                delete_api.delete(start, stop, predicate, bucket, org)
                print(f"  ✅ 삭제 완료: {target_name}")
            except Exception as e:
                print(f"  ❌ 삭제 실패 ({target_name}): {e}")

        print("🎉 모든 '_AL' 종목 데이터가 성공적으로 지워졌습니다!")

except Exception as e:
    print(f"❌ 데이터베이스 검색 중 에러 발생: {e}")