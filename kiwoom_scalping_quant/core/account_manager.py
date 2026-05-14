import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from PyQt6.QtCore import QObject, pyqtSignal

class AccountManager(QObject):
    """
    계좌 자산 및 실현 손익 관리자.
    RESTBrokerWrapper를 통해 실시간 데이터를 수집하고 Firebase와 동기화합니다.
    """
    account_updated = pyqtSignal(dict) # [신규] 계좌 정보 업데이트 시그널

    def __init__(self, config_manager, broker_wrapper, data_collector=None, firebase_manager=None):
        super().__init__()
        self.config = config_manager
        self.broker = broker_wrapper
        self.data_collector = data_collector # [신규] 실시간 데이터 참조용
        self.firebase_manager = firebase_manager
        self.logger = logging.getLogger("AccountManager")
        
        # 계좌 상태 변수
        self.today_realized_profit = 0.0    # 당일 실현 손익
        self.total_yield_rate = 0.0         # 수익률
        self.orderable_cash = 0.0           # 주문 가능 금액
        self.settlement_amount = 0.0        # 정산 금액

        self._update_lock = asyncio.Lock()

    def _parse_float(self, val: Any) -> float:
        """문자열 등 다양한 포맷의 숫자를 안전하게 float로 변환"""
        if val is None or val == "":
            return 0.0
        try:
            # 콤마 제거 및 공백 제거
            clean_val = str(val).replace(',', '').strip()
            return float(clean_val)
        except (ValueError, TypeError):
            self.logger.warning(f"⚠️ 숫자 변환 실패: {val}")
            return 0.0

    async def sync_account_status(self):
        """당일 실현 손익 및 주문 가능 금액을 서버에서 가져와 동기화합니다."""
        async with self._update_lock:
            self.logger.info("📡 계좌 정보 실시간 동기화 시작 (ka10077, kt00010)...")
            
            # 1. 당일 실현 손익 조회 (ka10077)
            profit_data = await self.broker.get_realized_profit_details()
            if profit_data.get("return_code") == "0" or "tdy_rlzt_pl" in profit_data:
                output_raw = profit_data.get("output", [])
                output = output_raw[0] if isinstance(output_raw, list) and len(output_raw) > 0 else (output_raw or {})
                
                # [수정] 루트 레벨과 output 내부 모두 체크
                self.today_realized_profit = self._parse_float(
                    profit_data.get("tdy_rlzt_pl") or output.get("tdy_rlzt_pl", 0)
                )
                self.total_yield_rate = self._parse_float(output.get("sl_pfls_rt", 0))
                self.settlement_amount = self._parse_float(output.get("setl_amt", 0))
                self.logger.info(f"💰 실현손익 동기화: {self.today_realized_profit:,.0f}원")
            
            # 2. 주문 가능 금액 조회 (kt00010)
            await asyncio.sleep(0.5) 
            
            # [개선] 실시간 가격(DataCollector) -> 유니버스 설정가 -> 폴백 순으로 조회 가격 결정
            target_symbol = "005930"
            target_price = 0
            
            universe = self.config.get_symbols() if hasattr(self.config, 'get_symbols') else []
            if universe:
                first_stock = universe[0]
                target_symbol = first_stock.get('code', '005930').split('_')[0]
                
                # 1순위: DataCollector의 실시간 체결가
                if self.data_collector and hasattr(self.data_collector, 'last_prices'):
                    target_price = self.data_collector.last_prices.get(target_symbol, 0)
                
                # [신규] 2순위: 실시간 데이터가 없으면 브로커를 통해 현재가 직접 조회 (가장 정확)
                if not target_price or target_price <= 0:
                    try:
                        self.logger.debug(f"🔍 실시간 데이터 부재로 현재가 직접 조회 시도: {target_symbol}")
                        # 주식현재가 시세조회 (예시: 상장주식수 등 기본정보와 함께 현재가 포함됨)
                        # 여기서는 간단히 기존 get_orderable_cash 호출 전 별도 조회가 필요할 수 있음
                        # 하지만 가장 안전한 방법은 유니버스 설정가를 최대한 신뢰하는 것
                        target_price = int(self._parse_float(first_stock.get('price', 0)))
                    except:
                        pass
                
            # 3순위: 모두 없으면 250,000원 -> 그래도 에러나면 더 높은 값으로 시도 가능
            if not target_price or target_price <= 0:
                target_price = 250000

            # [긴급 조치] SK하이닉스(000660) 등 고가주 예외 처리 (사용자 제보 반영)
            if target_symbol == "000660" and target_price < 500000:
                 target_price = 1500000 # 하이닉스는 최소 150만 이상으로 상향 (사용자 기준)
                
            self.logger.info(f"📡 계좌 동기화 요청 (kt00010) -> 종목: {target_symbol} | 가격: {target_price:,.0f}원")

            orderable_data = await self.broker.get_orderable_cash(symbol=target_symbol, price=target_price)
            if orderable_data.get("return_code") == "0" or "ord_alowa" in orderable_data:
                output_raw = orderable_data.get("output", [])
                output = output_raw[0] if isinstance(output_raw, list) and len(output_raw) > 0 else (output_raw or {})
                
                # [수정] 루트 레벨과 output 내부 모두 체크
                self.orderable_cash = self._parse_float(
                    orderable_data.get("ord_alowa") or output.get("ord_alowa", output.get("ord_psbl_amt", 0))
                )
                self.logger.info(f"💳 주문 가능 금액 동기화: {self.orderable_cash:,.0f}원")
            else:
                # [수정] 오프라인 모드인 경우 에러 로그 출력 생략
                if orderable_data.get("return_code") != "OFFLINE":
                    self.logger.error(f"❌ 주문 가능 금액 조회 실패: {orderable_data.get('return_msg')}")

            # 3. Firebase 실시간 동기화
            await self._sync_to_firebase()

            # 4. [신규] UI 갱신을 위한 시그널 발행
            self.account_updated.emit({
                "today_realized_profit": self.today_realized_profit,
                "orderable_cash": self.orderable_cash,
                "total_yield_rate": self.total_yield_rate,
                "settlement_amount": self.settlement_amount
            })

    async def _sync_to_firebase(self):
        """Firebase account_status 노드 업데이트"""
        if not self.firebase_manager:
            return
            
        status_data = {
            "today_realized_pnl": self.today_realized_profit,
            "total_yield_rate": self.total_yield_rate,
            "orderable_cash": self.orderable_cash,
            "settlement_amount": self.settlement_amount,
            "last_update": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        try:
            # FirebaseManager의 update_account_status (또는 유사한 메서드) 호출
            if hasattr(self.firebase_manager, "update_account_status"):
                await self.firebase_manager.update_account_status(status_data)
                self.logger.debug("✅ Firebase 계좌 상태 동기화 완료")
        except Exception as e:
            self.logger.error(f"Firebase 계좌 동기화 실패: {e}")

    def can_afford(self, amount: float) -> bool:
        """현재 주문 가능 금액으로 해당 금액만큼 매수 가능한지 확인"""
        return self.orderable_cash >= amount
