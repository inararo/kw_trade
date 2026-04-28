import asyncio
import logging
import aiohttp
import socket
from datetime import datetime, timedelta
from typing import Optional
from PyQt6.QtCore import QObject, pyqtSignal

class TokenSignals(QObject):
    token_updated = pyqtSignal(str)
    token_error = pyqtSignal(str)

class TokenManager:
    """
    Manages the lifecycle of the Kiwoom REST API Access Token.
    Automatically renews the token 10 minutes before expiration.
    """
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger("TokenManager")
        self.signals = TokenSignals()

        self.access_token: Optional[str] = None
        self.expires_at: Optional[datetime] = None
        self._renewal_task: Optional[asyncio.Task] = None
        self._is_running = False

    async def start(self):
        """Starts the token manager background task."""
        if self._is_running:
            return
        self._is_running = True

        # Initial token fetch
        await self.refresh_token()

        # Start background monitor loop
        self._renewal_task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stops the token manager background task."""
        self._is_running = False
        if self._renewal_task:
            self._renewal_task.cancel()
            try:
                await self._renewal_task
            except asyncio.CancelledError:
                pass

    async def refresh_token(self):
        """Calls the Kiwoom API to get a new access token."""
        app_key = self.config_manager.get("KIWOOM_APP_KEY")
        app_secret = self.config_manager.get("KIWOOM_APP_SECRET")

        if not app_key or not app_secret:
            msg = "Cannot refresh token: App Key or Secret is missing in configuration."
            self.logger.error(msg)
            self.signals.token_error.emit(msg)
            return

        base_url = self.config_manager.get_rest_url()
        url = f"{base_url}/oauth2/token"

        headers = {
            'Content-Type': 'application/json;charset=UTF-8',  # 컨텐츠타입
        }

        payload = {
            "grant_type": "client_credentials",
            "appkey": app_key,
            "secretkey": app_secret
        }

        try:
            # [안정화] Windows qasync 환경에서 DNS 이슈 방지를 위해 TCPConnector 설정 추가
            connector = aiohttp.TCPConnector(family=socket.AF_INET, ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(url, headers=headers, json=payload, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.access_token = data.get("token")
                        # Kiwoom tokens are usually valid for 24 hours (86400 seconds)
                        expires_in = int(data.get("expires_in", 86400))
                        self.expires_at = datetime.now() + timedelta(seconds=expires_in)

                        token_val = data.get("token", "")
                        expires = data.get("expires_dt", "")
                        t_type = data.get("token_type", "")
                        r_code = data.get("return_code", "")
                        r_msg = data.get("return_msg", "")

                        # f-string을 사용하면 None이나 숫자 데이터도 안전하게 문자열로 합쳐집니다.
                        token_info = f"{token_val}, {expires}, {t_type}, {r_code}, {r_msg}"
                        print(f"JYJ  token_info: {token_info}")
                        print(f"Token successfully refreshed. Expires at {self.expires_at}, access_token : {self.access_token}")

                        # Update globally
                        self.config_manager.update_settings({"KIWOOM_ACCESS_TOKEN": self.access_token})
                        self.signals.token_updated.emit("Kiwoom API 토큰 갱신 완료")
                    else:
                        err_text = await response.text()
                        msg = f"Failed to refresh token: HTTP {response.status} - {err_text}"
                        self.logger.error(msg)
                        self.signals.token_error.emit(f"토큰 갱신 실패 ({response.status})")
        except Exception as e:
            msg = f"Exception during token refresh: {e}"
            self.logger.error(msg)
            self.signals.token_error.emit("토큰 갱신 중 에러 발생")

    async def _monitor_loop(self):
        """Background loop to check token expiration."""
        while self._is_running:
            if self.expires_at:
                now = datetime.now()
                # Renew 10 minutes before expiration
                time_until_expiry = (self.expires_at - now).total_seconds()
                if time_until_expiry <= 600:
                    self.logger.info("Token expiring soon. Renewing...")
                    await self.refresh_token()

            # Check every minute
            await asyncio.sleep(60)

    def get_token(self) -> Optional[str]:
        return self.access_token
