import logging
import json
import asyncio
import inspect
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
            trnm = data.get("trnm")
            msg_type = data.get("type") or trnm
            
            if trnm == "CNSRREQ":
                self._handle_snapshot(data)
            elif msg_type == "I" or (data.get("event") == "condition" and data.get("status") == "I"):
                self._handle_insert(data)
            elif msg_type == "D" or (data.get("event") == "condition" and data.get("status") == "D"):
                self._handle_delete(data)
            elif trnm in ["REAL", "COND"]:
                # REAL 또는 COND 메시지는 data 배열 내부에 실제 정보가 들어있을 수 있음
                entries = data.get("data", [])
                if not entries:
                    entries = [data]
                
                for entry in entries:
                    e_type = entry.get("type")
                    if e_type == "02" or entry.get("name") == "조건검색" or trnm == "COND" or entry.get("type") == "COND":
                        values = entry.get("values", {})
                        code = (values.get("9001") or entry.get("item") or entry.get("stk_cd") or entry.get("symbol") or entry.get("code") or "").lstrip("A").strip()
                        status_val = values.get("843") or entry.get("status") or entry.get("type") or "I"
                        status = "I" if str(status_val).upper() in ["I", "INSERT", "편입", "1"] else "D"
                        if code:
                            self.update_realtime_condition(code, status)
                
        except Exception as e:
            self.logger.error(f"ConditionService: 메시지 파싱 에러: {e}")

    def update_realtime_condition(self, code: str, status: str):
        """
        [공개 API] 외부(DataCollector 등)에서 직접 종목 편입/이탈을 호출할 때 사용합니다.
        code: 종목코드 (예: 005930)
        status: 'I' (편입) 또는 'D' (이탈)
        """
        self.logger.info(f"🔔 [REAL 02] 실시간 조건검색 이벤트 수신: {code} ({'편입' if status == 'I' else '이탈'})")
        
        data = {"symbol": code, "status": status}
        if status == "I":
            self._handle_insert(data)
        elif status == "D":
            self._handle_delete(data)

    def _run_callback(self, callbacks: List[Callable], *args, **kwargs):
        """
        [보완] 등록된 콜백들을 동기/비동기 여부에 관계없이 안전하게 실행합니다.
        """
        for cb in callbacks:
            try:
                # 1. 코루틴 함수인 경우 (async def)
                if inspect.iscoroutinefunction(cb):
                    self.logger.info(f"Scheduling async callback: {cb.__name__ if hasattr(cb, '__name__') else 'unknown'} for {args[0] if args else 'unknown'}")
                    asyncio.create_task(cb(*args, **kwargs))
                else:
                    # 2. 일반 함수인 경우 호출 후 결과 확인
                    self.logger.info(f"Calling callback: {cb.__name__ if hasattr(cb, '__name__') else 'unknown'} for {args[0] if args else 'unknown'}")
                    res = cb(*args, **kwargs)
                    # 만약 일반 함수가 코루틴 객체를 반환했다면 (예: partial 등)
                    if inspect.iscoroutine(res):
                        asyncio.create_task(res)
            except Exception as e:
                self.logger.error(f"❌ [ConditionService] 콜백 실행 오류 ({cb.__name__ if hasattr(cb, '__name__') else 'unknown'}): {e}")

    def _handle_snapshot(self, data: Dict[str, Any]):
        """초기 조건검색 결과 스냅샷 처리 (CNSRREQ)"""
        raw_items = data.get("data", [])
        symbols = []
        
        if isinstance(raw_items, list):
            for item in raw_items:
                code = ""
                if isinstance(item, dict):
                    code = item.get("jmcode", "")
                elif isinstance(item, str):
                    code = item
                
                if code:
                    symbols.append(code.lstrip("A"))

        self.logger.info(f"📋 조건검색 스냅샷 수신: {len(symbols)} 종목")
        self.current_symbols = set(symbols)
        self._run_callback(self.on_snapshot, list(self.current_symbols))

    def _handle_insert(self, data: Dict[str, Any]):
        """실시간 종목 편입 처리"""
        symbol = data.get("symbol", "").lstrip("A")
        if symbol and symbol not in self.current_symbols:
            self.logger.info(f"🔔 [편입] {symbol}")
            self.current_symbols.add(symbol)
            self._run_callback(self.on_insert, symbol, data)

    def _handle_delete(self, data: Dict[str, Any]):
        """실시간 종목 이탈 처리"""
        symbol = data.get("symbol", "").lstrip("A")
        if symbol in self.current_symbols:
            self.logger.info(f"🔕 [이탈] {symbol}")
            self.current_symbols.remove(symbol)
            self._run_callback(self.on_delete, symbol, data)

    def register_callbacks(self, on_insert=None, on_delete=None, on_snapshot=None):
        if on_insert: self.on_insert.append(on_insert)
        if on_delete: self.on_delete.append(on_delete)
        if on_snapshot: self.on_snapshot.append(on_snapshot)
