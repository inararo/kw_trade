import pytest
import asyncio
import time
from core.data_collector import DataCollector

@pytest.fixture
def data_collector():
    config = {
        'symbol': '005930',
        'ws_url': 'ws://dummy_url',
        'max_buffer_size': 100
    }
    return DataCollector(config)

@pytest.mark.asyncio
async def test_watchdog_circuit_breaker(data_collector):
    """
    WebSocket 연결이 끊겨 3초간 시세 미수신 시
    Watchdog이 Circuit Breaker를 정상적으로 발동하는지 테스트
    """
    data_collector.is_running = True
    data_collector.last_receive_time = time.time()

    # Watchdog 태스크 실행
    watchdog_task = asyncio.create_task(data_collector._watchdog())

    # 1. 초기 상태: Circuit Breaker 비활성
    assert not data_collector.circuit_breaker_active

    # 2. 방어 로직 우회 (웹소켓 연결 및 최초 데이터 수신 완료 처리)
    data_collector.ws_connected_event.set()
    data_collector.first_data_received_event.set()

    # 3. 시간을 인위적으로 4초 전으로 되돌려 타임아웃 상황 모사
    data_collector.last_receive_time = time.time() - 4.0

    # 4. Watchdog이 루프를 돌면서 상태를 감지하도록 잠시 대기
    await asyncio.sleep(1.2)

    # 5. 검증: Circuit Breaker가 발동되어야 함
    assert data_collector.circuit_breaker_active is True

    # 태스크 정리
    data_collector.is_running = False
    watchdog_task.cancel()
