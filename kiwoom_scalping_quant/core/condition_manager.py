import asyncio
import logging
import random
from typing import Callable, Dict, Any, List

class ConditionManager:
    """
    증권사 API 실시간 조건검색 이벤트 구독 관리 모듈.
    - 실시간 조건검색 편입/이탈 이벤트를 수신하여 콜백 핸들러를 호출합니다.
    """
    def __init__(self, config_manager, data_collector):
        self.config_manager = config_manager
        self.data_collector = data_collector
        self.logger = logging.getLogger("ConditionManager")
        
        # 콜백 핸들러 리스트 (insert/delete)
        self.on_insert_callbacks: List[Callable] = []
        self.on_delete_callbacks: List[Callable] = []
        
        self.is_running = False
        self._mock_task = None
        
        # [TODO] 실제 API WebSocket EndPoint 또는 TR 연동 부분
        # 현재 아키텍처에서는 이벤트를 Dispatch 해주는 역할을 합니다.

    def register_insert_callback(self, callback: Callable):
        self.on_insert_callbacks.append(callback)

    def register_delete_callback(self, callback: Callable):
        self.on_delete_callbacks.append(callback)

    async def start_condition_monitoring(self, condition_name: str, condition_idx: str):
        """
        특정 조건검색식을 실시간 구독하고 감시를 시작합니다.
        (실제로는 래퍼 서버나 키움 Open API+ 에 'send_condition' TR 및 'OnReceiveRealCondition' 이벤트를 등록하는 로직이 들어갑니다.)
        """
        self.logger.info(f"🚀 실시간 조건검색 감시 시작: {condition_name} (Index: {condition_idx})")
        self.is_running = True
        
        # [Mock Simulator] API가 미연동된 상태에서 아키텍처 테스트를 위한 모의 이벤트 발생 루프
        # 실제 적용 시 이 부분을 웹소켓 리스너로 교체합니다.
        # self._mock_task = asyncio.create_task(self._mock_event_generator())

    async def handle_insert_event(self, symbol: str, event_data: Dict[str, Any] = None):
        """외부(또는 웹소켓 파서)에서 편입 이벤트를 수신했을 때 호출하는 핸들러"""
        symbol = symbol.lstrip("A")
        event_data = event_data or {}
        self.logger.info(f"🔔 [조건검색 편입 이벤트 발생] 종목: {symbol}")
        for cb in self.on_insert_callbacks:
            if asyncio.iscoroutinefunction(cb):
                asyncio.create_task(cb(symbol, event_data))
            else:
                try:
                    cb(symbol, event_data)
                except Exception as e:
                    self.logger.error(f"Insert callback error: {e}")

    async def handle_delete_event(self, symbol: str, event_data: Dict[str, Any] = None):
        """외부(또는 웹소켓 파서)에서 이탈 이벤트를 수신했을 때 호출하는 핸들러"""
        symbol = symbol.lstrip("A")
        event_data = event_data or {}
        self.logger.info(f"🔕 [조건검색 이탈 이벤트 발생] 종목: {symbol}")
        for cb in self.on_delete_callbacks:
            if asyncio.iscoroutinefunction(cb):
                asyncio.create_task(cb(symbol, event_data))
            else:
                try:
                    cb(symbol, event_data)
                except Exception as e:
                    self.logger.error(f"Delete callback error: {e}")
                
    async def _mock_event_generator(self):
        """아키텍처 동작 테스트용 모의 이벤트 제너레이터"""
        test_symbols = ["005930", "000660", "035420", "035720", "051910", "006400", "068270"]
        active = set()
        
        while self.is_running:
            await asyncio.sleep(random.randint(10, 30))  # 10~30초 랜덤 간격
            
            action = random.choice(["insert", "delete"])
            if action == "insert":
                candidate = random.choice([s for s in test_symbols if s not in active])
                active.add(candidate)
                await self.handle_insert_event(candidate, {"reason": "모의 편입"})
            elif action == "delete" and active:
                candidate = random.choice(list(active))
                active.remove(candidate)
                await self.handle_delete_event(candidate, {"reason": "모의 이탈"})

    def stop(self):
        self.is_running = False
        if self._mock_task:
            self._mock_task.cancel()
        self.logger.info("ConditionManager Stopped.")
