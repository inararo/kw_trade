import aiohttp
import logging
import asyncio
import time
from typing import Dict, Any, Optional

class RESTBrokerWrapper:
    """
    키움 REST API 전용 통신 래퍼.
    다양한 TR(ka10077, kt00010 등)을 호출하고 결과를 정제하여 반환합니다.
    """
    # TR 코드별 전용 엔드포인트 맵핑
    URI_MAP = {
        "ka10077": "/api/dostk/acnt",    # 당일실현손익상세조회
        "kt00010": "/api/dostk/acnt",    # 주문인출가능금액요청
        "ka10050": "/api/dostk/rkinfo",  # 조건검색식 목록 조회
        "ka10051": "/api/dostk/rkinfo",  # 조건검색 종목 조회
    }

    def __init__(self, config_manager, token_manager):
        self.config = config_manager
        self.token_manager = token_manager
        self.logger = logging.getLogger("RESTBrokerWrapper")
        
        # [신규] 계좌 번호 설정
        self.account_number = self.config.get("account_number", "")
        
        # 기본 URL 설정
        kiwoom_cfg = self.config.get("kiwoom", {})
        trading_mode = kiwoom_cfg.get("trading_mode", "real")
        base_url_config = kiwoom_cfg.get("rest_base_url")
        if isinstance(base_url_config, dict):
            self.base_url = base_url_config.get(trading_mode, "https://api.kiwoom.com")
        else:
            self.base_url = base_url_config or kiwoom_cfg.get("rest_url", "https://api.kiwoom.com")

    async def _request(self, api_id: str, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """공통 HTTP POST 요청 처리부 (Rate Limit 고려)"""
        # [오프라인 모드] 실시간 API 요청 차단
        if self.config.get("OFFLINE_MODE", False):
            self.logger.info(f"🚫 오프라인 모드: [{api_id}] REST API 요청을 차단합니다.")
            return {"return_code": "OFFLINE", "return_msg": "System is running in OFFLINE mode."}

        token = self.token_manager.get_token()
        if not token:
            self.logger.error(f"[{api_id}] 요청 실패: Access Token이 없습니다.")
            return {"return_code": "-1", "return_msg": "TOKEN_MISSING"}

        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {token}",
            "api-id": api_id
        }

        # ─────────────────────────────────────────────────────
        # [수정] 전달받은 endpoint를 그대로 사용
        # ─────────────────────────────────────────────────────
        url = f"{self.base_url}{endpoint}"
        
        try:
            await asyncio.sleep(0.2)
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=body, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        self.logger.debug(f"🔍 [{api_id}] RAW Response: {data}")
                        return data
                    else:
                        err_text = await resp.text()
                        self.logger.error(f"[{api_id}] HTTP 에러 ({resp.status}): {err_text}")
                        return {"return_code": str(resp.status), "return_msg": err_text}
        except Exception as e:
            self.logger.error(f"[{api_id}] 통신 예외 발생: {e}")
            return {"return_code": "-999", "return_msg": str(e)}

    async def request_tr(self, tr_code: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """TR 코드별 URI를 동적으로 찾아 요청을 전송합니다."""
        target_uri = self.URI_MAP.get(tr_code)
        
        if not target_uri:
            # 기본값 라우팅 (알 수 없는 TR은 rkinfo로 시도)
            self.logger.warning(f"⚠️ [{tr_code}] 매핑된 URI가 없습니다. 기본값(/api/dostk/rkinfo)으로 시도합니다.")
            target_uri = "/api/dostk/rkinfo"
            
        return await self._request(tr_code, target_uri, body)

    async def get_realized_profit_details(self) -> Dict[str, Any]:
        """당일 실현 손익 상세 조회 (ka10077)"""
        # [명세 반영] 필수 필드: stk_cd
        body = {
            "acc_no": self.account_number, # [추가] 계좌번호 필수
            "stk_cd": "000000", # 필수: 종목코드 (전체 조회를 위해 더미/기본값 설정)
        }
        return await self.request_tr("ka10077", body)

    async def get_orderable_cash(self, symbol: str = "005930", price: int = 0) -> Dict[str, Any]:
        """주문 인출 가능 금액 조회 (kt00010)"""
        # [명세 반영] 필수 필드: stk_cd, trde_tp, uv
        # [명세 반영] 선택 필드: io_amt, trde_qty, exp_buy_unp
        body = {
            "acc_no": self.account_number, # [추가] 계좌번호 필수
            "io_amt": "",          # 입출금액 (선택)
            "stk_cd": symbol,      # 종목코드 (필수)
            "trde_tp": "2",        # 매매구분 1:매도, 2:매수 (필수)
            "trde_qty": "",        # 매매수량 (선택)
            "uv": str(price) if price > 0 else "85000", # 매수가격 (필수) - 상/하한가 에러 방지를 위해 85,000원으로 조정
            "exp_buy_unp": "",     # 예상매수단가 (선택)
        }
        return await self.request_tr("kt00010", body)
