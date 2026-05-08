import asyncio
import json
import logging
import aiohttp
import socket
from PyQt6.QtCore import QThread, pyqtSignal

class ConditionWebSocketThread(QThread):
    """
    서버의 실시간 조건검색(편입/이탈) 이벤트를 수신하는 전용 스레드.
    이벤트 발생 시 메인 스레드로 시그널을 전송합니다.
    """
    # (이벤트타입 'I'/'D', 종목코드)
    signal_condition_event = pyqtSignal(str, str)
    signal_error = pyqtSignal(str)

    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.logger = logging.getLogger("ConditionWS")
        self._is_running = True
        self.base_url = self.config_manager.get_rest_url().replace("https://", "wss://") # 웹소켓용 URL로 변환 필요 시

    def stop(self):
        self._is_running = False
        self.quit()

    def run(self):
        """비동기 루프를 실행하는 스레드 엔트리 포인트"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._connect_websocket())
        except Exception as e:
            self.logger.error(f"WS Thread Run Error: {e}")
            self.signal_error.emit(str(e))
        finally:
            loop.close()

    async def _connect_websocket(self):
        ws_url = "wss://api.kiwoom.com:10000/api/dostk/websocket"
        
        while self._is_running:
            # [방어 코드] 장 상태 확인하여 장외 시간 연결 차단
            scheduler = getattr(self.config_manager, "_injected_scheduler", None)
            if scheduler:
                from core.scheduler import MarketState
                current_state = scheduler.current_state
                allowed_states = [MarketState.PREPARE, MarketState.TRADING, MarketState.CUTOFF, MarketState.LIQUIDATING]
                
                if current_state not in allowed_states:
                    self.logger.info(f"💤 현재 장 상태({current_state})가 장외 시간입니다. 60초 후 재확인합니다.")
                    await asyncio.sleep(60)
                    continue

            token = self.config_manager.get("KIWOOM_ACCESS_TOKEN")
            if not token:
                self.logger.warning("Access Token이 없어 연결을 대기합니다 (10초).")
                await asyncio.sleep(10)
                continue

            self.logger.info(f"📡 실시간 조건검색 웹소켓 연결 시도... {ws_url}")
            
            try:
                connector = aiohttp.TCPConnector(family=socket.AF_INET, ssl=False)
                async with aiohttp.ClientSession(connector=connector) as session:
                    async with session.ws_connect(ws_url, headers={"Authorization": f"Bearer {token}"}, heartbeat=30) as ws:
                        self.logger.info("✅ 실시간 조건검색 웹소켓 연결 성공")
                        
                        # 수신 루프
                        async for msg in ws:
                            if not self._is_running:
                                break
                            
                            # 장 상태가 바뀌어 장 종료가 되면 루프 탈출
                            if scheduler and scheduler.current_state not in allowed_states:
                                self.logger.warning("🔔 장 종료 감지로 인해 웹소켓 연결을 종료합니다.")
                                break

                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                    # 이벤트 포맷 예시: {"type": "I", "code": "005930"}
                                    event_type = data.get("type")
                                    symbol = data.get("code")
                                    
                                    if event_type in ['I', 'D'] and symbol:
                                        self.logger.info(f"🔔 조건검색 이벤트 수신: {event_type} | {symbol}")
                                        self.signal_condition_event.emit(event_type, symbol)
                                except json.JSONDecodeError:
                                    continue
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
            except Exception as e:
                self.logger.error(f"웹소켓 연결 오류 (30초 후 재시도): {e}")
                await asyncio.sleep(30)
