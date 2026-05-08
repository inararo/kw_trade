import logging
import asyncio
from typing import Dict, Any, Optional

class AccountService:
    """
    [Shared Core] 계좌 자산 및 실현 손익 관리 서비스.
    GUI와 백엔드 트레이더에서 공통으로 사용됩니다.
    """
    def __init__(self, broker_wrapper, data_collector=None):
        self.broker = broker_wrapper
        self.data_collector = data_collector
        self.logger = logging.getLogger("AccountService")
        
        # 상태 저장소
        self.today_realized_profit = 0.0
        self.orderable_cash = 0.0
        self.total_yield_rate = 0.0
        self._lock = asyncio.Lock()

    async def sync_all(self):
        """실현손익 및 주문가능금액 통합 동기화"""
        async with self._lock:
            # 1. 당일 실현 손익 (ka10077)
            profit_data = await self.broker.get_realized_profit_details()
            self._parse_profit(profit_data)
            
            # 2. 주문 가능 금액 (kt00010)
            await asyncio.sleep(0.2) # API 부하 방지
            orderable_data = await self._fetch_orderable_cash()
            self._parse_orderable(orderable_data)
            
            return self.get_summary()

    async def _fetch_orderable_cash(self):
        """실시간 가격을 반영한 주문 가능 금액 요청"""
        # [안정화] 유니버스의 첫 번째 종목 또는 하이닉스를 조회 대상으로 우선 선정
        target_symbol = "000660" # 하이닉스 (고가주 테스트용 기본값)
        target_price = 1500000   # 하이닉스 폴백 가격
        
        if self.data_collector and hasattr(self.data_collector, 'config'):
            universe = self.data_collector.config.get('universe', [])
            if universe:
                target_symbol = str(universe[0].get('code', '005930')).split('_')[0]
                target_price = 250000 # 삼성전자급 폴백

        # 실시간 가격 참조
        if self.data_collector and hasattr(self.data_collector, 'last_prices'):
            price = self.data_collector.last_prices.get(target_symbol, 0)
            if price > 0: 
                target_price = price
            
        # [핵심] 고가 종목(하이닉스 등) 하한가 에러 방지 보정
        if target_symbol == "000660" and target_price < 1500000:
            target_price = 1500000
            
        self.logger.info(f"📡 계좌 동기화 요청 (kt00010) -> 종목: {target_symbol} | 가격: {target_price:,}원")
        return await self.broker.get_orderable_cash(symbol=target_symbol, price=target_price)

    def _parse_profit(self, data: dict):
        self.logger.info(f"🔍 [ka10077] RAW Response: {data}")
        if str(data.get("return_code")) == "0" or "tdy_rlzt_pl" in data:
            output = data.get("output", [{}])[0] if isinstance(data.get("output"), list) else (data.get("output") or {})
            self.today_realized_profit = float(data.get("tdy_rlzt_pl") or output.get("tdy_rlzt_pl", 0))
            self.total_yield_rate = float(output.get("sl_pfls_rt", 0))
        else:
            self.logger.error(f"❌ [ka10077] 실현손익 조회 실패: {data.get('return_msg')}")

    def _parse_orderable(self, data: dict):
        self.logger.info(f"🔍 [kt00010] RAW Response: {data}")
        if str(data.get("return_code")) == "0" or "ord_alowa" in data:
            output = data.get("output", [{}])[0] if isinstance(data.get("output"), list) else (data.get("output") or {})
            self.orderable_cash = float(data.get("ord_alowa") or output.get("ord_alowa", output.get("ord_psbl_amt", 0)))
        else:
            self.logger.error(f"❌ [kt00010] 주문가능금액 조회 실패: {data.get('return_msg')}")

    def get_summary(self):
        return {
            "realized_profit": self.today_realized_profit,
            "orderable_cash": self.orderable_cash,
            "yield_rate": self.total_yield_rate
        }
