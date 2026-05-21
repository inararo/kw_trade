import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

class AccountService:
    """
    [Shared Core] 계좌 자산 및 실현 손익 관리 서비스.
    GUI와 백엔드 트레이더에서 공통으로 사용됩니다.
    """
    def __init__(self, broker_wrapper, data_collector=None, firebase_manager=None):
        self.broker = broker_wrapper
        self.data_collector = data_collector
        self.firebase_manager = firebase_manager
        self.logger = logging.getLogger("AccountService")
        
        # 상태 저장소
        self.today_realized_profit = 0.0
        self.orderable_cash = 0.0
        self.total_yield_rate = 0.0
        self.settlement_amount = 0.0
        self.total_assets = 0.0
        self._lock = asyncio.Lock()
        
        # 콜백 등록 (UI 업데이트용)
        self.on_update = []

    def register_callback(self, callback):
        self.on_update.append(callback)

    async def sync_all(self):
        """실현손익 및 주문가능금액 통합 동기화"""
        async with self._lock:
            # 1. 주문 가능 금액 (kt00010)
            orderable_data = await self._fetch_orderable_cash()
            self._parse_orderable(orderable_data)
            
            # 2. 당일 실현 손익 (ka10077)
            await asyncio.sleep(0.2) # API 부하 방지
            profit_data = await self.broker.get_realized_profit_details()
            self._parse_profit(profit_data)
            
            # 3. Firebase 동기화
            await self._sync_to_firebase()
            
            # 4. 콜백 호출 (UI 등)
            summary = self.get_summary()
            for cb in self.on_update:
                try:
                    cb(summary)
                except Exception as e:
                    self.logger.error(f"Callback error: {e}")

            return summary

    def _is_offline(self) -> bool:
        """안전하게 오프라인 모드 상태를 반환합니다 (브로커 config 누락 대응 방어 로직)"""
        if hasattr(self.broker, 'config') and self.broker.config is not None:
            try:
                return self.broker.config.get("OFFLINE_MODE", False)
            except AttributeError:
                pass
        return False

    async def _fetch_orderable_cash(self):
        """실시간 가격을 반영한 주문 가능 금액 요청"""
        # [안정화] 유니버스의 첫 번째 종목 또는 맥쿼리를 조회 대상으로 우선 선정
        target_symbol = "088980" # 맥쿼리 (고가주 테스트용 기본값)
        target_price = 11200   # 맥쿼리 폴백 가격
        
        if self.data_collector and hasattr(self.data_collector, 'config'):
            universe = self.data_collector.config.get('universe', [])
            if universe:
                target_symbol = str(universe[0].get('code', '415640')).split('_')[0]
                target_price = 9900 # kb발해인프라 폴백
 
        # 실시간 가격 참조
        if self.data_collector and hasattr(self.data_collector, 'last_prices'):
            price = self.data_collector.last_prices.get(target_symbol, 0)
            if price > 0: 
                target_price = price
            
        # [수정] 오프라인 모드인 경우 동기화 요청 로그 출력 생략
        if not self._is_offline():
            self.logger.info(f"📡 계좌 동기화 요청 (kt00010) -> 종목: {target_symbol} | 가격: {target_price:,}원")
        return await self.broker.get_orderable_cash(symbol=target_symbol, price=target_price)
 
    def _parse_profit(self, data: dict):
        is_offline = self._is_offline()
        if not is_offline:
            self.logger.debug(f"🔍 [ID:{id(self)}] [ka10077] RAW Response: {data}")
 
        if str(data.get("return_code")) == "0" or "tdy_rlzt_pl" in data:
            output = data.get("output", [{}])[0] if isinstance(data.get("output"), list) else (data.get("output") or {})
            self.today_realized_profit = float(data.get("tdy_rlzt_pl") or output.get("tdy_rlzt_pl", 0))
            self.total_yield_rate = float(output.get("sl_pfls_rt", 0))
            self.settlement_amount = float(output.get("setl_amt", 0))
            # [신규] 총 자산 (평가금액 포함) - 응답에 없으면 현금+정산금액으로 추정
            new_assets = float(output.get("tot_evl_amt") or 0)
            if new_assets > 0:
                self.total_assets = new_assets
        else:
            # [수정] 오프라인 모드인 경우 모든 에러 로그 출력 생략
            is_offline = self._is_offline()
            if not is_offline and data.get("return_code") != "OFFLINE":
                self.logger.error(f"❌ [ID:{id(self)}] [ka10077] 실현손익 조회 실패: {data.get('return_msg')}")
 
    def _parse_orderable(self, data: dict):
        is_offline = self._is_offline()
        self.logger.info(f"DEBUG: AccountService check -> is_offline={is_offline}")
        # [진단] kt00010 RAW 응답을 항상 WARNING으로 출력 (필드 확인용)
        self.logger.warning(f"🔍 [ID:{id(self)}] [kt00010] RAW Response: {data}")
 
        if str(data.get("return_code")) == "0" or any(k in data for k in ["ord_alowa", "tdy_reu_alowa", "d2entra"]):
            output = data.get("output", [{}])[0] if isinstance(data.get("output"), list) else (data.get("output") or {})
            # [핑시등분석] 키움 kt00010 응답 필드 우선순위:
            # 1. tdy_reu_alowa : 당일재사용가능금액 (실질적 매수가능 현금)
            # 2. d2entra      : D+2 예수금 (정산 기준 현금)
            # 3. ord_alowa    : 주문가능금액 (신용 미사용 시 0으로 올 수 있음)
            # 4. profa_20ord_alow_amt : 증거금 20% 기준 (레버리지 포함으로 과대 표시)
            
            def _safe_int(v):
                try:
                    return int(str(v).replace(',', '').strip()) if v else 0
                except:
                    return 0
            
            cash = (
                _safe_int(data.get("profa_100ord_alow_amt"))   # 증거금 100% = 순수 보유 현금 ✅ 최우선
                or _safe_int(output.get("profa_100ord_alow_amt"))
                or _safe_int(data.get("d2entra"))               # D+2 예수금
                or _safe_int(output.get("d2entra"))
                or _safe_int(data.get("tdy_reu_alowa"))         # 당일재사용가능금액
                or _safe_int(output.get("tdy_reu_alowa"))
                or _safe_int(data.get("ord_alowa"))             # 주문가능금액 (신용 미사용 시 0)
                or _safe_int(output.get("ord_alowa"))
                or _safe_int(output.get("ord_psbl_amt"))
                or _safe_int(output.get("ord_able_amt"))
                or 0
            )
            self.orderable_cash = float(cash)
            self.logger.warning(f"💳 [ID:{id(self)}] [kt00010] 가용현금 파싱 결과: {self.orderable_cash:,.0f}원")
        else:
            is_offline = self._is_offline()
            if not is_offline and data.get("return_code") != "OFFLINE":
                self.logger.error(f"❌ [ID:{id(self)}] [kt00010] 주문가능금액 조회 실패 (is_offline={is_offline}): {data.get('return_msg')}")

    async def _sync_to_firebase(self):
        """Firebase system_status/account 문서 업데이트"""
        if not self.firebase_manager:
            return
            
        try:
            # [요구사항 반영] FirebaseManager의 신규 메서드 호출
            if hasattr(self.firebase_manager, "update_account_status"):
                await self.firebase_manager.update_account_status(
                    total_assets=int(self.total_assets),
                    realized_profit=int(self.today_realized_profit),
                    buying_power=int(self.orderable_cash)
                )
                self.logger.debug(f"✅ [ID:{id(self)}] Firebase 실시간 계좌 상태 동기화 완료")
        except Exception as e:
            self.logger.error(f"❌ [ID:{id(self)}] Firebase 계좌 동기화 실패: {e}")

    def can_afford(self, amount: float) -> bool:
        """현재 주문 가능 금액으로 해당 금액만큼 매수 가능한지 확인"""
        return self.orderable_cash >= amount

    def get_summary(self):
        return {
            "today_realized_profit": self.today_realized_profit,
            "orderable_cash": self.orderable_cash,
            "yield_rate": self.total_yield_rate,
            "settlement_amount": self.settlement_amount,
            "total_assets": self.total_assets
        }
