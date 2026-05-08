import asyncio
import logging
import os
import sys
import json
import time
import aiohttp
import websockets
from typing import Dict, Any

# [환경 설정] Python 경로 추가 및 로깅 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("MainLiveTrader")

from core.config_manager import ConfigManager
from core.data_collector import DataCollector
from core.order_manager import OrderManager
from core.risk_manager import RiskManager
from core.strategy_manager import StrategyManager
from core.condition_manager import ConditionManager

# =====================================================================
# 1. 키움증권 Open API (REST / WebSocket) 통신 래퍼
# =====================================================================
class KiwoomBrokerWrapper:
    """키움증권 REST API 및 WebSocket 규격에 맞춘 비동기 래퍼"""
    def __init__(self, app_key: str, app_secret: str, base_url: str, ws_url: str):
        self.app_key = app_key
        self.app_secret = app_secret
        self.base_url = base_url
        self.ws_url = ws_url
        self.access_token = None
        
        # 콜백 함수들
        self.on_condition_event = None
        self.on_tick_event = None
        self.on_execution_event = None

    # ------------------ REST API (aiohttp) ------------------
    async def login(self):
        """OAuth2 토큰 발급 (키움 규격)"""
        logger.info(f"🔑 키움증권 REST API 로그인 시도: {self.base_url}")
        endpoint = f"{self.base_url}/oauth2/token"
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }
        
        try:
            # 실제 API 서버와 통신하여 토큰 발급
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, json=payload, timeout=5) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    self.access_token = data.get("access_token")
                    
            if self.access_token:
                logger.info("✅ 키움 API 토큰 발급 및 로그인 완료!")
                return True
            else:
                logger.error("❌ 토큰 응답에 access_token이 없습니다.")
                return False
        except Exception as e:
            logger.error(f"❌ 로그인 통신 에러: {e}")
            return False

    async def get_condition_list(self) -> Dict[str, str]:
        """서버에 저장된 조건검색식 목록 조회 (키움 api-id: ka10050 등 가상 TR)"""
        endpoint = f"{self.base_url}/api/dostk/rkinfo" # 예시 엔드포인트
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, headers=headers, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # 서버 응답 구조에 맞게 파싱 필요, 임시로 데이터 반환
                        logger.info(f"✅ 조건식 목록 수신 완료 ({len(data)}개 항목)")
                        return data.get("conditions", {"001": "AI스캘핑주도주", "002": "수급단타"})
                    else:
                        logger.error(f"❌ 조건식 조회 HTTP 에러: {resp.status}")
                        return {"001": "AI스캘핑주도주", "002": "수급단타"} # Fallback
        except Exception as e:
            logger.error(f"❌ 조건식 조회 통신 에러: {e}")
            return {"001": "AI스캘핑주도주", "002": "수급단타"} # Fallback

    async def send_order(self, action: int, symbol: str, price: float, qty: int):
        """키움 주식주문 TR (KOA) 전송"""
        tick_size = 10 if price >= 10000 else 5
        
        if action in [1, 2]: # Buy
            order_price = price + tick_size
            order_type = "1" # 1: 신규매수
        elif action in [3, 4]: # Sell
            order_price = price - tick_size
            order_type = "2" # 2: 신규매도
        else:
            return False

        endpoint = f"{self.base_url}/api/dostk/order" # 주식주문 엔드포인트
        headers = {"Authorization": f"Bearer {self.access_token}"}
        payload = {
            "symbol": symbol,
            "order_type": order_type,
            "qty": qty,
            "price": order_price
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=payload, timeout=3) as resp:
                    if resp.status == 200:
                        logger.info(f"✅ 키움 주문 전송 성공 [구분:{order_type}] {symbol} | 수량: {qty} | 단가: {order_price}")
                        return True
                    else:
                        err_msg = await resp.text()
                        logger.error(f"❌ 주문 전송 실패 (HTTP {resp.status}): {err_msg}")
                        return False
        except Exception as e:
            logger.error(f"❌ 주문 전송 중 통신 에러: {e}")
            return False

    # ------------------ WebSocket Listener ------------------
    def _parse_ws_message(self, message: str) -> Dict[str, Any]:
        """키움증권 WebSocket JSON 파싱 (FID 기반)"""
        try:
            data = json.loads(message)
            
            # 1. PING-PONG 하트비트 처리
            if isinstance(data, str) and "PING" in data.upper():
                return {"event": "ping"}

            trnm = data.get("trnm", "")
            entries = data.get("data", [])
            if not entries and isinstance(data, dict):
                entries = [data]

            for entry in entries:
                msg_type = entry.get("type", "") or data.get("type", "")
                
                # 조건검색 실시간 이벤트 (가상 포맷)
                if trnm == "COND" or msg_type == "CONDITION":
                    # type_str: I(편입), D(이탈)
                    status_str = entry.get("status", "I")
                    return {
                        "event": "condition",
                        "code": entry.get("stk_cd", ""),
                        "status": "I" if status_str in ["I", "INSERT", "편입"] else "D",
                        "name": entry.get("cond_name", "")
                    }
                
                # 0B: 주식체결 (틱 데이터)
                elif msg_type == "0B":
                    values = entry.get("values", entry)
                    symbol = entry.get("item") or data.get("item", "")
                    
                    # 키움증권 FID: 10(현재가), 15(거래량), 12(등락율)
                    price_val = values.get("10") or values.get("curr_pric") or "0"
                    vol_val = values.get("15") or values.get("cntg_vol") or "0"
                    chg_val = values.get("12") or values.get("flu_rt") or "0"
                    
                    return {
                        "event": "tick",
                        "symbol": str(symbol).strip(),
                        "price": abs(float(str(price_val).replace(',', ''))),
                        "volume": abs(float(str(vol_val).replace(',', ''))),
                        "change_rate": float(str(chg_val).replace(',', ''))
                    }
                    
                # ORDR, CNTG 등 주문/체결/잔고(Chejan) 이벤트
                elif msg_type in ["ORDR", "CNTG", "K1", "H1"]:
                    symbol = entry.get("stk_cd") or data.get("symbol", "")
                    return {
                        "event": "execution",
                        "type": "체결",
                        "symbol": str(symbol).strip()
                    }
            
            return {"event": "unknown"}
        except Exception as e:
            logger.error(f"메시지 파싱 에러: {e}")
            return {"event": "error"}

    async def ws_listener_loop(self, target_condition_idx: str):
        """키움증권 WebSocket 인증 및 실시간 스트림 수신 루프"""
        logger.info(f"📡 키움 WebSocket 리스너 시작: {self.ws_url}")
        
        # 키움증권 LOGIN 페이로드
        login_payload = {
            "trnm": "LOGIN",
            "token": self.access_token
        }
        
        # 조건검색 실시간 등록 (REG)
        cond_sub_payload = {
            "trnm": "REG",
            "grp_no": "2",
            "refresh": "0",
            "data": [
                {"type": ["COND"], "item": [target_condition_idx]}
            ]
        }
        
        logger.info(f"✉️ 키움 WS LOGIN & 조건검색 구독 페이로드 전송 대기")

        while True:
            try:
                logger.info(f"🔗 웹소켓 서버 접속 시도: {self.ws_url}")
                async with websockets.connect(self.ws_url, ping_interval=None) as ws:
                    logger.info("✅ 웹소켓 서버 접속 성공!")
                    
                    # 1. 인증(LOGIN)
                    await ws.send(json.dumps(login_payload))
                    login_resp = await ws.recv()
                    logger.info(f"✉️ 웹소켓 로그인 응답 수신: {login_resp}")
                    
                    # 2. 실시간 조건검색 구독 (간헐적 딜레이 방지)
                    await asyncio.sleep(0.5)
                    await ws.send(json.dumps(cond_sub_payload))
                    logger.info("✉️ 실시간 조건검색 구독 페이로드 전송 완료")
                    
                    # 3. 메시지 수신 무한 루프
                    async for message in ws:
                        parsed = self._parse_ws_message(message)
                        
                        if parsed["event"] == "ping":
                            await ws.send(json.dumps({"trnm": "PONG"}))
                            logger.debug("❤️ PONG 하트비트 응답 전송")
                        elif parsed["event"] == "condition" and self.on_condition_event:
                            await self.on_condition_event(parsed["code"], parsed["status"], parsed["name"])
                        elif parsed["event"] == "tick" and self.on_tick_event:
                            await self.on_tick_event(parsed)
                        elif parsed["event"] == "execution" and self.on_execution_event:
                            await self.on_execution_event(parsed)
                            
            except websockets.exceptions.ConnectionClosed as e:
                logger.error(f"❌ 웹소켓 연결이 끊어졌습니다. ({e}) 5초 후 재접속을 시도합니다.")
                await asyncio.sleep(5.0)
            except Exception as e:
                logger.error(f"❌ 웹소켓 통신 중 오류 발생: {e}")
                await asyncio.sleep(5.0)


# =====================================================================
# 2. 메인 실행 함수 (순수 asyncio 기반)
# =====================================================================
async def main():
    logger.info("🚀 동적 유니버스 기반 AI 트레이딩 봇 부팅 시작...")

    # 1. 코어 모듈 초기화
    config_manager = ConfigManager()
    data_collector = DataCollector(config_manager)
    risk_manager = RiskManager(config_manager)
    order_manager = OrderManager(config_manager, risk_manager)
    
    strategy_manager = StrategyManager(config_manager, data_collector, order_manager, risk_manager)
    condition_manager = ConditionManager(config_manager, data_collector)

    # 2. 모델 로드 및 StrategyManager 초기 세팅
    strategy_manager.load_model_from_config()
    await strategy_manager.init_engines([]) # 초기 유니버스 빈 상태로 구동

    # =====================================================================
    # 3. 비동기 통신 래퍼 초기화 및 REST API 로그인 (키움 기준)
    # =====================================================================
    broker_api = KiwoomBrokerWrapper(
        app_key="KIWOOM_APP_KEY", 
        app_secret="KIWOOM_SECRET", 
        base_url="https://api.kiwoom.com",
        ws_url="wss://mockapi.kiwoom.com:10000/api/dostk/websocket"
    )

    is_logged_in = await broker_api.login()
    if not is_logged_in:
        logger.error("시스템 종료: 로그인에 실패했습니다.")
        return

    # 잔고 동기화 (REST API)
    await order_manager.sync_balance()

    # 조건식 고유 ID 조회 (REST API)
    target_condition_name = "AI스캘핑주도주"
    condition_dict = await broker_api.get_condition_list()
    target_idx = next((idx for idx, name in condition_dict.items() if name == target_condition_name), None)

    if not target_idx:
        logger.error(f"❌ '{target_condition_name}' 조건식을 찾을 수 없습니다. 시스템을 종료합니다.")
        return

    # =====================================================================
    # 4. WebSocket 라우팅 콜백 바인딩
    # =====================================================================
    
    # 4-1. 조건검색 이벤트 라우팅
    async def on_condition_ws_event(code: str, status: str, name: str):
        if status == 'I': # 편입
            await condition_manager.handle_insert_event(code, {"name": name})
        elif status == 'D': # 이탈
            await condition_manager.handle_delete_event(code, {"name": name})

    broker_api.on_condition_event = on_condition_ws_event
    condition_manager.register_insert_callback(strategy_manager.handle_condition_insert)
    condition_manager.register_delete_callback(strategy_manager.handle_condition_delete)

    # 4-2. 틱 데이터 라우팅 (DataCollector 우회 직접 주입)
    async def on_tick_ws_event(tick_data: dict):
        sym = tick_data["symbol"]
        engine = strategy_manager.envs.get(sym)
        if engine and not getattr(engine, 'is_condition_deleted', False):
            # LiveTradingEngine으로 틱 직접 푸시
            import datetime
            await engine.update_tick(
                symbol=sym,
                price=tick_data["price"],
                volume=tick_data["volume"],
                change_rate=tick_data["change_rate"],
                timestamp=datetime.datetime.now()
            )

    broker_api.on_tick_event = on_tick_ws_event

    # 4-3. 체결/잔고 이벤트 라우팅
    async def on_execution_ws_event(exec_data: dict):
        logger.info(f"💰 [체결/잔고 업데이트 수신] {exec_data}")
        # 체결 발생 시 즉각 잔고 동기화 (내부 219차원 포트폴리오 상태 오류 방지)
        await order_manager.sync_balance()

    broker_api.on_execution_event = on_execution_ws_event

    # =====================================================================
    # 5. 백그라운드 태스크 무한 루프 실행 (asyncio.gather)
    # =====================================================================
    logger.info("⚙️ 메인 트레이딩 파이프라인 및 웹소켓 리스너 가동...")
    
    # ConditionManager 내부 상태 업데이트
    await condition_manager.start_condition_monitoring(target_condition_name, target_idx)

    tasks = [
        asyncio.create_task(broker_api.ws_listener_loop(target_idx)),  # 웹소켓 펌프 루프
        asyncio.create_task(strategy_manager.start())                  # 매매 엔진 워치독 및 웜업 루프
    ]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("🛑 강제 종료 감지. 안전 종료 절차 시작...")
        await strategy_manager.stop()
        if hasattr(data_collector, 'stop'):
            await data_collector.stop()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
