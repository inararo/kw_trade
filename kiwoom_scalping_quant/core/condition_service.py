import logging
import json
from typing import Dict, Any, List, Callable

class ConditionService:
    """
    [Shared Core] 실시간 조건검색 웹소켓 이벤트 처리 서비스.
    - CNSRREQ: 초기 조건검색 결과 스냅샷 파싱
    - I: 실시간 종목 편입 이벤트 처리
    - D: 실시간 종목 이탈 이벤트 처리
    """
    def __init__(self):
        self.logger = logging.getLogger("ConditionService")
        self.current_symbols = set()
        
        # 콜백 등록 (UI 업데이트 또는 매매 엔진 연동용)
        self.on_insert: List[Callable] = []
        self.on_delete: List[Callable] = []
        self.on_snapshot: List[Callable] = []

    def handle_websocket_message(self, message: str):
        """웹소켓 메시지 수신 및 유형별 분기 처리"""
        try:
            data = json.loads(message)
            msg_type = data.get("type") # I, D, CNSRREQ 등
            
            if msg_type == "CNSRREQ":
                self._handle_snapshot(data)
            elif msg_type == "I":
                self._handle_insert(data)
            elif msg_type == "D":
                self._handle_delete(data)
                
        except Exception as e:
            self.logger.error(f"ConditionService: 메시지 파싱 에러: {e}")

    def _handle_snapshot(self, data: Dict[str, Any]):
        """초기 조건검색 결과 스냅샷 처리"""
        symbols = data.get("symbols", []) # 예: ["005930", "000660"]
        self.logger.info(f"📋 조건검색 스냅샷 수신: {len(symbols)} 종목")
        self.current_symbols = set(symbols)
        for cb in self.on_snapshot:
            cb(list(self.current_symbols))

    def _handle_insert(self, data: Dict[str, Any]):
        """실시간 종목 편입 처리"""
        symbol = data.get("symbol", "").lstrip("A")
        if symbol and symbol not in self.current_symbols:
            self.logger.info(f"🔔 [편입] {symbol}")
            self.current_symbols.add(symbol)
            for cb in self.on_insert:
                cb(symbol, data)

    def _handle_delete(self, data: Dict[str, Any]):
        """실시간 종목 이탈 처리"""
        symbol = data.get("symbol", "").lstrip("A")
        if symbol in self.current_symbols:
            self.logger.info(f"🔕 [이탈] {symbol}")
            self.current_symbols.remove(symbol)
            for cb in self.on_delete:
                cb(symbol, data)

    def register_callbacks(self, on_insert=None, on_delete=None, on_snapshot=None):
        if on_insert: self.on_insert.append(on_insert)
        if on_delete: self.on_delete.append(on_delete)
        if on_snapshot: self.on_snapshot.append(on_snapshot)
