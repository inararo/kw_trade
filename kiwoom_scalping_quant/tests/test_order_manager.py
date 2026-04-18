import pytest
from core.order_manager import OrderManager
from returns.result import Success, Failure
from returns.io import IOSuccess, IOFailure

@pytest.fixture
def order_manager():
    config = {}
    return OrderManager(config)

@pytest.mark.asyncio
async def test_send_order_success(order_manager, mocker):
    """
    정상적인 API 호출 성공 시 Success(temp_order_id) 반환 검증
    """
    # 임의로 time.time을 모킹하여 항상 같은 ID 반환 유도 가능
    result = await order_manager.send_order("BUY", "005930", 50000, 10)

    # future_safe 적용 시 result는 IOSuccess/IOFailure 인스턴스
    assert isinstance(result, IOSuccess)
    order_id = result.unwrap()._inner_value
    assert order_id.startswith("INT_")
    assert order_id in order_manager.active_orders

@pytest.mark.asyncio
async def test_send_order_failure(order_manager, mocker):
    """
    API 통신 실패(500 에러, 타임아웃 등) 시 예외가 밖으로 던져지지 않고
    Failure 객체로 감싸져서 반환되는지 검증
    """
    # order_semaphore 획득 이후에 강제로 에러를 발생시키는 로직 모킹
    # Mocking _throttle_order to bypass sleep in tests
    mocker.patch.object(order_manager, '_throttle_order', return_value=None)

    # 원래 코드는 안전하게 try-except가 없으므로 future_safe가 잡아냄
    # 이를 모사하기 위해 send_order 내부의 일부를 강제 에러 발생시키도록 모킹
    original_semaphore = order_manager.order_semaphore

    class MockExceptionSemaphore:
        async def __aenter__(self):
            raise ConnectionError("Network 500 Error")
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    order_manager.order_semaphore = MockExceptionSemaphore()

    result = await order_manager.send_order("BUY", "005930", 50000, 10)

    assert isinstance(result, IOFailure)
    assert isinstance(result.failure()._inner_value, ConnectionError)
