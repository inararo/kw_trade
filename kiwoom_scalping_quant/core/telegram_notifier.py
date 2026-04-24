import aiohttp
import logging
import asyncio
from typing import Optional, Dict, Any

class TelegramNotifier:
    """
    텔레그램 봇을 통해 시스템 이벤트를 알리는 비동기 노티파이어.
    """
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger("TelegramNotifier")
        self._session: Optional[aiohttp.ClientSession] = None
        
        # 설정 캐싱
        self.token = config_manager.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = config_manager.get("telegram_chat_id")
        
        if not self.token or not self.chat_id:
            self.logger.warning("텔레그램 설정(TOKEN 또는 Chat ID)이 누락되었습니다. 알림 기능이 작동하지 않을 수 있습니다.")

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def send_message(self, text: str):
        """메시지를 비동기로 전송합니다."""
        if not self.token or not self.chat_id or "<TOKEN>" in self.token:
            self.logger.debug(f"텔레그램 발송 생략 (설정 미비): {text}")
            return

        session = await self._ensure_session()
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        
        try:
            async with session.post(url, json=payload, timeout=5) as resp:
                if resp.status != 200:
                    err_body = await resp.text()
                    self.logger.error(f"텔레그램 발송 실패 (Status {resp.status}): {err_body}")
                else:
                    self.logger.debug(f"텔레그램 발송 완료: {text[:20]}...")
        except Exception as e:
            self.logger.error(f"텔레그램 발송 중 에러 발생: {e}")

    async def notify_app_start(self):
        msg = "🚀 <b>[Antigravity Trading System]</b>\n시스템 부팅 시퀀스가 시작되었습니다."
        await self.send_message(msg)

    async def notify_app_ready(self, universe_count: int):
        msg = f"✅ <b>시스템 준비 완료</b>\n현재 {universe_count}개 종목에 대한 실시간 감시를 시작합니다."
        await self.send_message(msg)

    async def notify_app_stop(self):
        msg = "⚠️ <b>시스템 종료</b>\n거래 시스템이 안전하게 종료되었습니다."
        await self.send_message(msg)

    async def notify_trade(self, action: str, symbol: str, name: str, price: float, qty: int, pnl: float = 0):
        emoji = "🔵" if action == "BUY" else "🔴"
        msg = (
            f"{emoji} <b>{action} 체결 알림</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"• 종목: {name} ({symbol})\n"
            f"• 가격: ₩{price:,.0f}\n"
            f"• 수량: {qty}주\n"
            f"• 총액: ₩{price * qty:,.0f}\n"
        )
        if action == "SELL" and pnl != 0:
            pnl_emoji = "📈" if pnl > 0 else "📉"
            msg += f"• 실현손익: {pnl_emoji} ₩{pnl:,.0f}\n"
            
        await self.send_message(msg)

    async def notify_critical(self, title: str, content: str):
        msg = f"🚨 <b>[CRITICAL] {title}</b>\n━━━━━━━━━━━━━━━\n{content}"
        await self.send_message(msg)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
