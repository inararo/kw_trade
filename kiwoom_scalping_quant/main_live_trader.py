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
from infrastructure.firebase_manager import FirebaseManager

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
        if not self.app_key or not self.app_secret:
            logger.error("❌ 환경변수(.env) 또는 설정에서 API KEY(KIWOOM_APP_KEY, KIWOOM_APP_SECRET)를 불러오지 못했습니다. 값이 비어있습니다!")
            return False

        payload = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "secretkey": self.app_secret  # 일부 프록시/API 버전 호환성용
        }
        
        try:
            # 실제 API 서버와 통신하여 토큰 발급
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, json=payload, timeout=5) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    self.access_token = data.get("access_token") or data.get("token")
                    
            if self.access_token:
                logger.info("✅ 키움 API 토큰 발급 및 로그인 완료!")
                return True
            else:
                logger.error(f"❌ 토큰 응답에 access_token이 없습니다. API 서버 원본 응답: {data}")
                return False
        except Exception as e:
            logger.error(f"❌ 로그인 통신 에러: {e}")
            return False

    async def reissue_token(self):
        """토큰 만료 시 재발급 처리 (기존 login 재활용)"""
        logger.warning("🔄 토큰 만료 감지: 토큰 재발급(Login) 프로세스를 가동합니다.")
        success = await self.login()
        if success:
            return self.access_token
        return None

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

    async def request_tr(self, api_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """공통 TR 요청 메서드 (RESTBrokerWrapper 호환용)"""
        endpoint = f"{self.base_url}/api/dostk/acnt" # 계좌 관련 엔드포인트 통합
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "Authorization": f"Bearer {self.access_token}",
            "api-id": api_id
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=body, timeout=5) as resp:
                    return await resp.json()
        except Exception as e:
            logger.error(f"❌ [TR 요청 에러] {api_id}: {e}")
            return {"return_code": "99", "return_msg": str(e)}

    async def get_realized_profit_details(self) -> Dict[str, Any]:
        """당일 실현 손익 상세 조회 (ka10077)"""
        return await self.request_tr("ka10077", {"stk_cd": "000000"})

    async def get_orderable_cash(self, symbol: str = "005930", price: int = 0) -> Dict[str, Any]:
        """주문 인출 가능 금액 조회 (kt00010)"""
        body = {
            "io_amt": "", "stk_cd": symbol, "trde_tp": "2",
            "trde_qty": "", "uv": str(price) if price > 0 else "250000",
            "exp_buy_unp": ""
        }
        return await self.request_tr("kt00010", body)

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
                
                # 조건검색 실시간 이벤트 (키움 공식 API 규격: type="02", name="조건검색")
                if trnm == "COND" or msg_type == "CONDITION" or entry.get("name") == "조건검색" or msg_type == "02":
                    values = entry.get("values", {})
                    # 키움 실시간 조건검색 FID: 843(I/D 여부), 9001(종목코드), 20(발생시간)
                    status_str = values.get("843", "I")
                    code_str = self._clean_code(values.get("9001") or entry.get("item", ""))
                    
                    return {
                        "event": "condition",
                        "code": code_str,
                        "status": "I" if status_str in ["I", "INSERT", "편입", "1"] else "D",
                        "name": entry.get("cond_name", "AI스캘핑주도주")
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
                        "symbol": self._clean_code(symbol or entry.get("item")),
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
                        "symbol": self._clean_code(symbol)
                    }
            
            return {"event": "unknown"}
        except Exception as e:
            logger.error(f"메시지 파싱 에러: {e}")
            return {"event": "error"}

    def _clean_code(self, code: Any) -> str:
        """종목 코드에서 'A' 접두사를 제거하고 6자리 숫자로 정제"""
        if not code:
            return ""
        code_str = str(code).strip()
        if code_str.startswith('A') and len(code_str) >= 7:
            return code_str[1:]
        return code_str.lstrip("A")

    async def ws_listener_loop(self, target_condition_idx: str):
        """키움증권 WebSocket 인증 및 실시간 스트림 수신 루프 (공식 스펙 기반)"""
        logger.info(f"📡 키움 WebSocket 리스너 시작: {self.ws_url}")
        
        login_payload = {
            "trnm": "LOGIN",
            "token": self.access_token
        }
        
        while True:
            try:
                logger.info(f"🔗 웹소켓 서버 접속 시도: {self.ws_url}")
                async with websockets.connect(self.ws_url, ping_interval=None) as ws:
                    logger.info("✅ 웹소켓 서버 접속 성공!")
                    
                    # 1. 인증(LOGIN)
                    await ws.send(json.dumps(login_payload))
                    logger.warning(f"✉️ [WS SEND] LOGIN 요청 전송")
                    
                    # 2. 메시지 수신 무한 루프
                    async for message in ws:
                        logger.warning(f"📩 [WS RECV] {message}")
                        
                        # [Shared Core] ConditionService로 메시지 라우팅
                        if hasattr(self, 'on_condition_ws_message') and self.on_condition_ws_message:
                            self.on_condition_ws_message(message)
                        
                        try:
                            response = json.loads(message)
                        except json.JSONDecodeError:
                            continue

                        trnm = response.get("trnm")

                        if trnm == "LOGIN":
                            if str(response.get("return_code")) != "0":
                                logger.error(f"❌ 웹소켓 로그인 실패: {response.get('return_msg')}")
                                return
                            logger.info("✅ 웹소켓 로그인 성공! 조건검색식 목록을 요청합니다.")
                            # 로그인 성공 시 조건검색식 목록(CNSRLST) 요청
                            await ws.send(json.dumps({"trnm": "CNSRLST"}))
                            logger.warning("✉️ [WS SEND] CNSRLST 전송")

                        elif trnm == "CNSRLST":
                            data_list = response.get("data", [])
                            logger.info(f"✅ 조건검색식 목록 수신: {data_list}")
                            
                            # 대상 조건식 고유번호(seq) 찾기 (이름으로 매칭, 없으면 파라미터 값 사용)
                            target_seq = target_condition_idx
                            for item in data_list:
                                # 구조: [seq, name] 또는 {"seq": ..., "name": ...} 대비
                                if isinstance(item, list) and len(item) >= 2:
                                    if item[1] == "AI스캘핑주도주":
                                        target_seq = str(item[0])
                                elif isinstance(item, dict):
                                    if item.get("name") == "AI스캘핑주도주":
                                        target_seq = str(item.get("seq"))
                            
                            # 실시간 조건검색 등록(CNSRREQ)
                            req_payload = {
                                "trnm": "CNSRREQ",
                                "seq": target_seq,
                                "search_type": "1", # 1: 조건검색 + 실시간조건검색
                                "stex_tp": "K"      # KRX 거래소
                            }
                            await ws.send(json.dumps(req_payload))
                            logger.warning(f"✉️ [WS SEND] CNSRREQ 전송 (seq={target_seq})")

                        elif trnm == "CNSRREQ":
                            if str(response.get("return_code")) == "0":
                                logger.info("✅ 조건검색 실시간 감시 등록 완료!")
                                # [수정] 초기 조건 만족 종목 리스트(Snapshot) 추출 및 편입
                                jm_list = response.get("data", [])
                                if isinstance(jm_list, list) and jm_list:
                                    logger.info(f"📋 초기 조건 만족 종목 {len(jm_list)}개 편입 시퀀스 시작...")
                                    for item in jm_list:
                                        # item이 딕셔너리인 경우와 문자열인 경우 모두 대응
                                        raw_code = item.get("jmcode", "") if isinstance(item, dict) else str(item)
                                        clean_code = self._clean_code(raw_code)
                                        if clean_code and self.on_condition_event:
                                            # 편입('I') 이벤트로 처리하여 엔진에 주입
                                            await self.on_condition_event(clean_code, "I", "초기스냅샷")
                                else:
                                    logger.warning(f"⚠️ 초기 스냅샷 데이터가 비어있거나 올바르지 않습니다: {jm_list}")
                            else:
                                logger.error(f"❌ 조건검색 실시간 등록 실패: {response.get('return_msg')}")

                        elif trnm == "REAL":
                            # 실시간 시세 및 조건검색 이벤트 파싱
                            parsed = self._parse_ws_message(message)
                            if parsed["event"] == "condition" and self.on_condition_event:
                                await self.on_condition_event(parsed["code"], parsed["status"], parsed["name"])
                            elif parsed["event"] == "tick" and self.on_tick_event:
                                await self.on_tick_event(parsed)
                            elif parsed["event"] == "execution" and self.on_execution_event:
                                await self.on_execution_event(parsed)

                        elif trnm == "PING":
                            # 서버 PING 메시지 에코 응답
                            await ws.send(message)
                            logger.warning("❤️ [WS SEND] PING 하트비트 응답")

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

    from core.account_service import AccountService
    from core.condition_service import ConditionService

    # 1. 코어 모듈 초기화
    config_manager = ConfigManager(config_path="config.yaml")

    # [추가] 로그 레벨 동적 적용
    log_level_str = config_manager.get("log_level", "INFO").upper()
    logging.getLogger().setLevel(getattr(logging, log_level_str, logging.INFO))
    logger.info(f"시스템: 로그 레벨이 {log_level_str}로 설정되었습니다.")

    data_collector = DataCollector(config_manager)
    # [Shared Core] 서비스 초기화
    # 3. 비동기 통신 래퍼 초기화
    broker_api = KiwoomBrokerWrapper(
        app_key=config_manager.get("KIWOOM_APP_KEY", ""), 
        app_secret=config_manager.get("KIWOOM_APP_SECRET", ""), 
        base_url=config_manager.get_rest_url(),
        ws_url=config_manager.get_ws_url()
    )
    
    account_service = AccountService(broker_api, data_collector)
    condition_service = ConditionService()

    order_manager = OrderManager(config_manager, account_service=account_service)
    risk_manager = RiskManager(config_manager, order_manager)
    order_manager.risk_manager = risk_manager
    
    strategy_manager = StrategyManager(config_manager, data_collector, order_manager, risk_manager)
    condition_manager = ConditionManager(config_manager, data_collector)

    # 1-1. Firebase 초기화 및 리스너 설정
    firebase_manager = FirebaseManager(config_manager)
    config_manager.firebase_manager = firebase_manager
    
    # [Firebase] 부팅 시 초기화 (상태 보고 및 기본 설정 업로드)
    if getattr(firebase_manager, '_initialized', False):
        await firebase_manager.update_system_status("RUNNING")
        await firebase_manager.update_engine_status("RUNNING")
        await firebase_manager.update_control_status(
            is_monitoring_active=True,
            is_ai_trading_active=True
        )
        # [중요] OrderManager에 FirebaseManager 주입 (매매 로그 전송용)
        order_manager.firebase_manager = firebase_manager
        
        # 기본 설정 업로드 (보안 항목 제외)
        _EXCLUDED = {
            "account_number", "KIWOOM_APP_KEY", "KIWOOM_APP_SECRET",
            "KIWOOM_ACCESS_TOKEN", "INFLUX_URL", "INFLUX_TOKEN", "INFLUX_ORG",
            "influx_bucket", "INFLUX_BUCKET", "TELEGRAM_BOT_TOKEN",
            "telegram_chat_id", "FIREBASE_KEY_PATH", "active_model_path",
            "kiwoom", "ws_url", "symbols", "universe"
        }
        _default_settings = {
            k: v for k, v in config_manager._config_cache.items()
            if k not in _EXCLUDED and isinstance(v, (int, float, str, bool))
        }
        await firebase_manager.initialize_default_settings(_default_settings)
        logger.info("✅ [Firebase] 부팅 상태 보고 및 기본 설정 업로드 완료")

    # [Firebase] 실시간 리스너 설정 (클로저를 활용해 현재 루프 및 매니저들과 연동)
    loop = asyncio.get_running_loop()
    
    # 상태 변경 여부 확인을 위한 캐시 변수
    last_states = {
        "is_monitoring_active": None,
        "is_ai_trading_active": None
    }
    
    def setup_firebase_listeners():
        if not getattr(firebase_manager, '_initialized', False):
            return

        # 리스너 1: 설정 변경
        def on_settings_changed(data: dict):
            def _apply():
                _CONTROL_KEYS = {"last_updated_by_engine", "last_heartbeat", "engine_status", "current_state", "updated_at"}
                filtered = {k: v for k, v in data.items() if k not in _CONTROL_KEYS}
                if filtered:
                    applied = config_manager.hot_reload_settings(filtered)
                    if applied:
                        # [추가] 로그 레벨 실시간 변경 반영
                        if "log_level" in filtered:
                            new_level = filtered["log_level"].upper()
                            logging.getLogger().setLevel(getattr(logging, new_level, logging.INFO))
                            logger.info(f"🔧 [Firebase] 로그 레벨이 {new_level}로 변경되었습니다.")
                            
                        asyncio.run_coroutine_threadsafe(firebase_manager.report_settings_applied(), loop)
                        logger.info(f"🔧 [Firebase] 원격 설정 반영 완료: {list(filtered.keys())}")
            loop.call_soon_threadsafe(_apply)
        firebase_manager.listen_to_settings(on_settings_changed)

        # 리스너 2: 엔진 제어 (모니터링, AI 매매)
        def on_engine_status_changed(data: dict):
            def _apply():
                # 1. 종목 감시 상태 변경 확인
                if "is_monitoring_active" in data:
                    active = data["is_monitoring_active"]
                    if last_states["is_monitoring_active"] != active:
                        last_states["is_monitoring_active"] = active
                        logger.info(f"[Firebase] 원격 제어: 종목 감시 {'재개' if active else '중지'}")
                        # 실제 감시 중지 로직은 필요시 여기에 추가 (예: 웹소켓 재연결 또는 구독 해제)
                
                # 2. AI 매매 상태 변경 확인
                if "is_ai_trading_active" in data:
                    active = data["is_ai_trading_active"]
                    if last_states["is_ai_trading_active"] != active:
                        last_states["is_ai_trading_active"] = active
                        strategy_manager.set_ai_paused(not active)
                        logger.info(f"🤖 [Firebase] 원격 제어: AI 매매 {'재개' if active else '일시정지'}")
            loop.call_soon_threadsafe(_apply)
        firebase_manager.listen_to_engine_status(on_engine_status_changed)

        # 리스너 3: 긴급 명령 (전량 청산)
        def on_command_received(doc_id: str, data: dict):
            action = data.get("action", "")
            if action == "PANIC_SELL":
                logger.critical(f"🚨 [Firebase] 긴급 청산(PANIC_SELL) 명령 수신! (ID: {doc_id})")
                async def _execute():
                    try:
                        await order_manager.emergency_liquidate()
                        await firebase_manager.update_command_status(doc_id, "COMPLETED")
                        logger.info("✅ [Firebase] 긴급 청산 완료 보고 완료.")
                    except Exception as e:
                        await firebase_manager.update_command_status(doc_id, "FAILED")
                        logger.error(f"❌ [Firebase] 긴급 청산 실패: {e}")
                asyncio.run_coroutine_threadsafe(_execute(), loop)
        firebase_manager.listen_to_commands(on_command_received)

    setup_firebase_listeners()

    # 2. 모델 로드 및 StrategyManager 초기 세팅
    strategy_manager.load_model_from_config()
    await strategy_manager.init_engines([]) # 초기 유니버스 빈 상태로 구동

    # [수정] REST API 로그인 및 인증 관리자 주입
    is_logged_in = await broker_api.login()
    if not is_logged_in:
        logger.error("시스템 종료: 로그인에 실패했습니다.")
        return

    if hasattr(strategy_manager, 'broker_api'):
        strategy_manager.broker_api = broker_api

    class SimpleAuthManager:
        def get_token(self):
            return broker_api.access_token
    order_manager.auth_manager = SimpleAuthManager()

    # 잔고 동기화 (Shared Core Service 활용)
    summary = await account_service.sync_all()
    # OrderManager에 결과 반영 (호환성 유지)
    order_manager._broker_orderable_cash = summary["orderable_cash"]
    order_manager.daily_realized_pnl = summary["realized_profit"]
    logger.info(f"📊 [SharedCore] 잔고 동기화 완료: {summary}")

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
    
    # 4-1. 조건검색 서비스 연동 및 콜백 바인딩
    condition_service.register_callbacks(
        on_insert=strategy_manager.handle_condition_insert,
        on_delete=strategy_manager.handle_condition_delete
    )
    
    # 웹소켓 리스너에서 메시지를 ConditionService로 전달하도록 설정
    broker_api.on_condition_ws_message = condition_service.handle_websocket_message

    # 4-2. 틱 데이터 라우팅 (기존 유지)
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
    
    # [Firebase] 하트비트 태스크 추가
    if getattr(firebase_manager, '_initialized', False):
        tasks.append(asyncio.create_task(firebase_manager.start_heartbeat()))

    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"❌ 메인 루프 실행 중 에러 발생: {e}")
    finally:
        logger.info("🛑 안전 종료 절차 시작...")

        # 1. [Firebase] 종료 상태 전송 (가장 먼저 수행하여 루프 종료 전 전송 보장)
        # 335번 라인 근처에서 정의된 firebase_manager 변수를 안전하게 참조
        fb_mgr = locals().get('firebase_manager')
        if fb_mgr and getattr(fb_mgr, '_initialized', False):
            try:
                logger.info("📡 [Firebase] 종료 상태(STOPPED/OFFLINE) 전송 중...")
                # gather 대신 순차적으로 전송하여 확실성 제고, 타임아웃은 유지
                await asyncio.wait_for(fb_mgr.update_engine_status("OFFLINE"), timeout=2.0)
                await asyncio.wait_for(fb_mgr.update_system_status("STOPPED"), timeout=2.0)
                logger.info("👋 [Firebase] 엔진 종료 상태 보고 완료.")
            except Exception as e:
                logger.warning(f"⚠️ [Firebase] 종료 상태 전송 중 오류 (무시): {e}")
        
        # 2. 엔진 및 백그라운드 태스크 정지
        if 'strategy_manager' in locals():
            await strategy_manager.stop()

        # 3. 기타 리소스 정리
        if hasattr(data_collector, 'stop'):
            await data_collector.stop()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
