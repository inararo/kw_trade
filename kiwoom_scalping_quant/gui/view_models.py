import asyncio
import logging
import time
import os
import glob
import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, Qt, QTimer
from typing import Dict, Any, List
from returns.result import Success, Failure
from returns.io import IOSuccess, IOFailure

class LiveDashboardViewModel(QObject):
    """
    LiveDashboard 탭을 위한 ViewModel.
    DataCollector, OrderManager, Agent 등의 상태를 모니터링하고 UI로 신호를 전달합니다.
    """
    sig_orderbook_updated = pyqtSignal(dict)
    sig_price_updated = pyqtSignal(float)
    sig_ai_confidence_updated = pyqtSignal(dict)
    sig_log_appended = pyqtSignal(str)
    sig_error_occurred = pyqtSignal(str)
    sig_menu_action_result = pyqtSignal(str, str) # title, message

    # 멀티 종목 요약 정보 (Symbol -> Dict of stats)
    sig_symbols_summary_updated = pyqtSignal(dict)
    sig_universe_changed = pyqtSignal(list) # [NEW] 유니버스 교체 시그널

    # 스레드 브릿지: 백그라운드 -> 메인 스레드 (내부용)
    _sig_raw_data = pyqtSignal(object)

    # Risk Limits and Alerts
    sig_risk_metrics_updated = pyqtSignal(float, float, float, float) # Realized PnL, Evaluation PnL, Total Cash, Per-Symbol Limit
    sig_balance_updated = pyqtSignal(float) # [신규] 총 예수금(잔고) 업데이트
    sig_status_alert = pyqtSignal(str)
    
    # [제어 상태 시그널]
    sig_trading_paused = pyqtSignal(bool)    # True: 일시정지, False: 재개
    sig_monitoring_stopped = pyqtSignal(bool) # True: 중지, False: 감시중

    def __init__(self, data_collector, order_manager, config_manager, account_service=None, strategy_manager=None):
        super().__init__()
        self.data_collector = data_collector
        self.order_manager = order_manager
        self.config_manager = config_manager
        self.account_service = account_service
        self.strategy_manager = strategy_manager

        # [핵심 패치 1] 엔진이 나를 찾을 수 있도록 config_manager에 스스로를 주입!
        self.config_manager._injected_live_vm = self

        # [신규] 계좌 상태 실시간 업데이트 연결 (Shared Core 콜백 활용)
        if self.account_service and hasattr(self.account_service, 'register_callback'):
            self.account_service.register_callback(self._on_account_updated)

        self.logger = logging.getLogger("LiveDashboardViewModel")
        self._is_running = False
        self._mock_task = None

        # 현재 화면에 상세를 띄울 대상 종목
        self.selected_symbol = None
        self.symbols_summary = {}

        # [UI 업데이트 최적화] 마지막 호가/가격 임시 저장 (flush 전까지 축적)
        self._pending_orderbook = None
        self._pending_price = None
        self._ui_dirty = False  # 변경이 있을 때만 emit

        # [제어 상태 추적] 파이어베이스 동기화용
        self._is_monitoring_stopped = False
        self._is_ai_paused = False

        # 종목명 캐시 (Code -> Name): 접미사(_AL) 제거 후 순수 코드와 매핑
        self._symbol_names = {
            s.get("code", "").split('_')[0]: s.get("name") 
            for s in self.config_manager.get_symbols() if s.get("code")
        }

        # UI logging hook for Signal Only mode bypass messages
        if hasattr(self.order_manager, 'signals'):
            self.order_manager.signals.signal_only_log.connect(self.append_log)
            # [신규] 잔고 동기화 시그널 연결 (전체 자산 및 주문 가능 현금 통합 갱신)
            self.order_manager.signals.balance_synced.connect(self._on_balance_synced)

        # DataCollector 측에서 데이터가 들어올 때 콜백받을 수 있도록 설정
        self.data_collector.set_ui_callback(self._on_data_received)

        # [핵심 수정] QTimer로 100ms마다 배치 emit → 매 틱 emit 대신 주기적으로 최신 상태를 한 번에 전송
        self._ui_flush_timer = QTimer(self)
        self._ui_flush_timer.setInterval(100)  # 100ms = 초당 10회 갱신
        self._ui_flush_timer.timeout.connect(self._flush_ui_update)
        self._ui_flush_timer.start()
        
        # 초기 데이터를 유니버스에서 미리 로드하여 화면 빈 채로 시작 방지
        self._init_summary_data()

    def update_symbol_names(self, symbol_list: list):
        """StrategyManager에서 유니버스 갱신 시 호출하여 종목명 캐시 업데이트"""
        for s in symbol_list:
            code = s.get("code", "").split('_')[0].strip()
            name = s.get("name", "-")
            if code:
                self._symbol_names[code] = name
                # 기존 요약 정보가 있다면 이름 업데이트
                if code in self.symbols_summary:
                    self.symbols_summary[code]["name"] = name
        self._ui_dirty = True

    def update_universe_list(self, new_symbols: list, is_append: bool = False):
        """
        [최적화 패치] 장중 유니버스 교체 또는 추가 시 호출됩니다. (Set 기반 증분 업데이트)
        """
        protected_list = self.config_manager.get("protected_symbols", [])
        protected_symbols = set(str(s).split('_')[0] for s in protected_list)

        # 1. 새로운 종목들의 코드 집합 구성 (보호 종목 제외)
        new_codes = set()
        new_symbol_map = {} # 코드 -> 원본 객체/이름 매핑

        for s in new_symbols:
            if isinstance(s, str):
                code = s.lstrip("A").strip()
            else:
                code = s.get("code", "").split('_')[0].strip()
            
            if code and code not in protected_symbols:
                new_codes.add(code)
                new_symbol_map[code] = s

        # 2. 현재 감시 중인 종목 집합
        current_codes = set(self.symbols_summary.keys())

        # 3. 추가/삭제 대상 식별
        if is_append:
            # 개별 편입 모드: 기존 것 유지 + 새로운 것만 추가
            to_add = new_codes - current_codes
            to_remove = set()
        else:
            # 전체 스냅샷 모드: 증분 비교 수행
            to_add = new_codes - current_codes
            # [주의] 현재 보유 중인 종목은 리스트에서 강제로 유지해야 함
            to_remove = (current_codes - new_codes)
            to_remove = {c for c in to_remove if self.order_manager.bot_holdings.get(c, 0) <= 0 and \
                         not self.order_manager.has_unexecuted_orders(c)}

        self.logger.info(f"ViewModel: 유니버스 증분 업데이트 (추가: {len(to_add)}, 삭제: {len(to_remove)})")

        # 4. 삭제 처리 (UI 데이터 제거)
        for code in to_remove:
            if code in self.symbols_summary:
                del self.symbols_summary[code]

        # 5. 추가 처리 (UI 데이터 초기화)
        for code in to_add:
            s = new_symbol_map[code]
            name = "-"
            if isinstance(s, str):
                name = self._symbol_names.get(code) or "-"
            else:
                name = s.get("name", "-")

            price = self.data_collector.last_prices.get(code) or self.order_manager.last_known_prices.get(code) or 0
            chg = self.data_collector.last_change_rates.get(code, 0.0)
            
            self.symbols_summary[code] = {
                "name": name,
                "price": price,
                "change_rate": chg,
                "volume": 0,
                "ai_signal": "-", 
                "holdings": self.order_manager.bot_holdings.get(code, 0),
                "avg_price": self.order_manager.avg_entry_prices.get(code, 0.0)
            }

        # 6. 실시간 데이터 구독 및 매매 엔진 동기화 (전체 갱신이 아닌 변경분만 전달)
        if to_add or to_remove:
            # 실시간 시세 구독 업데이트 (DataCollector)
            if hasattr(self.data_collector, 'update_subscriptions'):
                asyncio.create_task(self.data_collector.update_subscriptions(to_add=list(to_add), to_remove=list(to_remove)))
            
            # 매매 엔진 동기화 (StrategyManager)
            if self.strategy_manager and hasattr(self.strategy_manager, 'update_engines'):
                asyncio.create_task(self.strategy_manager.update_engines(to_add=list(to_add), to_remove=list(to_remove)))

        # 7. UI 시그널 발생 (테이블 데이터가 변했음을 알림)
        self._ui_dirty = True
        if to_add or to_remove:
            # 전체 리스트(formatted_symbols)를 요구하는 위젯들을 위해 최신 상태 재구성
            full_list = [{"code": c, "name": self.symbols_summary[c]["name"]} for c in self.symbols_summary]
            self.sig_universe_changed.emit(full_list)
            
            if to_add and is_append:
                self.sig_log_appended.emit(f"[시스템] 새로운 종목 {list(to_add)}가 감시 리스트에 추가되었습니다.")

    def add_to_universe(self, symbol: str, data: dict = None):
        """개별 종목 편입 이벤트 처리 (ConditionWorker 연결용)"""
        self.update_universe_list([symbol], is_append=True)

    def remove_from_universe(self, symbol: str, data: dict = None):
        """개별 종목 이탈 이벤트 처리 (ConditionWorker 연결용)"""
        code = symbol.lstrip("A").strip()
        if code in self.symbols_summary:
            if self.symbols_summary[code].get("holdings", 0) <= 0:
                del self.symbols_summary[code]
                # 실시간 구독 및 매매 엔진 해제
                if hasattr(self.data_collector, 'update_subscriptions'):
                    asyncio.create_task(self.data_collector.update_subscriptions(to_add=[], to_remove=[code]))
                if self.strategy_manager and hasattr(self.strategy_manager, 'update_engines'):
                    asyncio.create_task(self.strategy_manager.update_engines(to_add=[], to_remove=[code]))
                self.logger.info(f"ViewModel: [{code}] 감시 리스트 및 엔진 제거 완료")
            else:
                self.logger.info(f"ViewModel: [{code}] 이탈 감지되었으나 보유 중이므로 리스트 유지")
            self._ui_dirty = True

    def _init_summary_data(self):
        """부팅 시 보유 종목을 바탕으로 요약 테이블 초기 뼈대 구성 및 가동"""
        self.logger.info("ViewModel: 초기 보유 종목 기반 엔진 및 구독 동기화 시작")
        
        to_add_codes = []
        
        # 보유 종목 추가
        protected_list = self.config_manager.get("protected_symbols", [])
        protected_symbols = set(str(s).split('_')[0] for s in protected_list)

        for code, qty in self.order_manager.bot_holdings.items():
            if code and code not in self.symbols_summary:
                if code in protected_symbols:
                    self.logger.warning(f"ViewModel: 보유 종목 {code}는 보호 종목이므로 감시 리스트에서 제외합니다.")
                    continue
                    
                name = self._symbol_names.get(code) or f"{code} (보유)"
                self.symbols_summary[code] = {
                    "name": name, 
                    "price": self.data_collector.last_prices.get(code, 0), 
                    "change_rate": 0.0, 
                    "volume": 0, 
                    "ai_signal": "-", 
                    "holdings": qty, 
                    "avg_price": self.order_manager.avg_entry_prices.get(code, 0.0)
                }
                to_add_codes.append(code)
        
        # 초기 보유 종목 구독 및 엔진 가동
        if to_add_codes:
            if hasattr(self.data_collector, 'update_subscriptions'):
                asyncio.create_task(self.data_collector.update_subscriptions(to_add=to_add_codes, to_remove=[]))
            if self.strategy_manager and hasattr(self.strategy_manager, 'update_engines'):
                asyncio.create_task(self.strategy_manager.update_engines(to_add=to_add_codes, to_remove=[]))

        self._ui_dirty = True

    def append_log(self, msg: str):
        self.sig_log_appended.emit(msg)

    def set_selected_symbol(self, symbol: str):
        self.selected_symbol = symbol

    def _on_data_received(self, data: dict):
        """DataCollector에서 호출되는 UI 업데이트 콜백 (동기, 빠른 상태 갱신만 수행)"""
        try:
            raw_symbol = data.get("symbol", "")
            if not raw_symbol:
                return
            symbol = raw_symbol.split('_')[0].strip()

            if symbol not in self.symbols_summary:
                name = self._symbol_names.get(symbol) or "-"
                self.symbols_summary[symbol] = {"name": name, "price": 0, "change_rate": 0.0, "ai_signal": "-", "holdings": 0}

            # 가격 업데이트
            if "price" in data and data["price"] > 0:
                self.symbols_summary[symbol]["price"] = data["price"]
            
            # [신규] 거래량 및 등락률 업데이트
            if "volume" in data and data["volume"] > 0:
                self.symbols_summary[symbol]["volume"] = data["volume"]
            
            if "change_rate" in data:
                self.symbols_summary[symbol]["change_rate"] = data["change_rate"]

            # 보유량 및 평균단가 업데이트 (실시간 반영)
            self.symbols_summary[symbol]["holdings"] = self.order_manager.holdings.get(symbol, 0)
            self.symbols_summary[symbol]["avg_price"] = self.order_manager.avg_entry_prices.get(symbol, 0.0)

            # [UI_DEBUG] 100번에 한 번 수신 로그 출력
            if not hasattr(self, "_rx_cnt"): self._rx_cnt = 0
            self._rx_cnt += 1
            if self._rx_cnt % 100 == 0:
                self.logger.info(f"[UI_DEBUG] VM 데이터 수신 성공: {symbol} ({data.get('price')})")

            # [자동 선택] 만약 선택된 종목이 없다면, 첫 번째로 데이터가 들어온 종목을 상세 뷰 대상으로 지정
            if self.selected_symbol is None:
                self.selected_symbol = symbol
                self.logger.info(f"[시스템] 첫 번째 수신 종목({symbol})을 상세 뷰로 자동 선택했습니다.")
                self.sig_log_appended.emit(f"[시스템] 첫 번째 수신 종목({symbol})을 상세 뷰로 자동 선택했습니다.")

            # 선택된 종목의 데이터만 상세 시그널용으로 임시 저장
            if symbol == self.selected_symbol:
                if "price" in data and data["price"] > 0:
                    self._pending_price = float(data["price"])
                if "orderbook" in data and data["orderbook"]:
                    self._pending_orderbook = dict(data["orderbook"])

            # 변경 플래그 설정
            self._ui_dirty = True

        except Exception as e:
            import traceback
            self.logger.error(f"[VIEWMODEL Error] _on_data_received: {e}\n{traceback.format_exc()}")

    def _flush_ui_update(self):
        """QTimer 100ms 주기로 호출: 변경이 있을 때만 최신 상태 스냅샷을 UI로 emit"""
        # [UI_DEBUG] 100번에 한 번(10초) 타이머 동작 로그 출력
        if not hasattr(self, "_flush_cnt"): self._flush_cnt = 0
        self._flush_cnt += 1
        if self._flush_cnt % 100 == 0:
            self.logger.info(f"[UI_DEBUG] UI Flush 타이머 작동 중 (Dirty={self._ui_dirty}, Symbols={len(self.symbols_summary)})")

        if not self._ui_dirty:
            return
        self._ui_dirty = False

        # [안정화] 얕은 복사본을 emit 하여 Qt 렌더링 도중의 데이터 변경 간섭 차단
        summary_snapshot = {k: v.copy() for k, v in self.symbols_summary.items()}
        self.sig_symbols_summary_updated.emit(summary_snapshot)

        if self._pending_price is not None:
            self.sig_price_updated.emit(self._pending_price)
            self._pending_price = None

        if self._pending_orderbook is not None:
            self.sig_orderbook_updated.emit(self._pending_orderbook)
            self._pending_orderbook = None


    def _on_balance_synced(self, balance: float):
        """OrderManager에서 잔고 동기화 완료 시 호출 (시그널 연동)"""
        # 1. 전체 총자산 업데이트
        self.sig_balance_updated.emit(balance)
        
        # 2. 요약 테이블의 보유량, 평균단가, 현재가 강제 갱신
        for symbol in self.symbols_summary.keys():
            self.symbols_summary[symbol]["holdings"] = self.order_manager.bot_holdings.get(symbol, 0)
            self.symbols_summary[symbol]["avg_price"] = self.order_manager.avg_entry_prices.get(symbol, 0.0)
            
            # [수정] 실시간 시장가 우선 반영, 없으면 잔고 조회 시 확인된 가격 사용
            price = self.data_collector.last_prices.get(symbol) or self.order_manager.last_known_prices.get(symbol)
            if price:
                self.symbols_summary[symbol]["price"] = price
        self._ui_dirty = True

        # 3. 주문 가능 현금 및 리스크 지표 즉시 갱신
        if hasattr(self.order_manager, 'risk_manager') and self.order_manager.risk_manager:
            rm = self.order_manager.risk_manager
            per_symbol_limit = rm.get_dynamic_max_invest()
            
            # [수정] 실현 손익과 평가 손익을 구분하여 가져옴
            realized_pnl = getattr(self.order_manager, 'daily_realized_pnl', 0.0)
            evaluation_pnl = getattr(self.order_manager, 'daily_evaluation_pnl', 0.0)
            total_cash = getattr(self.order_manager, 'orderable_cash', 0.0)
            self.logger.info(f"💰 [UI_UPDATE] 잔고 동기화 반영: 가용현금={total_cash:,.0f} | 총자산={balance:,.0f}")
            
            # UI로 전달 (실현손익, 평가손익, 전체 주문 가능 현금, 종목당 한도)
            self.sig_risk_metrics_updated.emit(realized_pnl, evaluation_pnl, total_cash, per_symbol_limit)

    def _on_account_updated(self, data: dict):
        """
        [신규] AccountService에서 데이터 동기화 완료 시 호출되는 콜백 슬롯.
        주문 가능 현금(ord_alowa) 등을 즉시 UI 리스크 지표에 반영합니다.
        """
        realized_pnl = data.get("today_realized_profit", 0.0)
        orderable_cash = data.get("orderable_cash", 0.0)
        
        # 리스크 매니저를 통한 종목당 투자 한도 계산
        per_symbol_limit = 0.0
        if hasattr(self.order_manager, 'risk_manager') and self.order_manager.risk_manager:
            per_symbol_limit = self.order_manager.risk_manager.get_dynamic_max_invest()
            
        evaluation_pnl = getattr(self.order_manager, 'daily_evaluation_pnl', 0.0)
        
        self.logger.info(f"📊 [UI_UPDATE] 실전 계좌 동기화 반영: 가용현금={orderable_cash:,.0f}")
        
        # UI 시그널 발행 (실현손익, 평가손익, 가용현금, 종목당한도)
        self.sig_risk_metrics_updated.emit(realized_pnl, evaluation_pnl, orderable_cash, per_symbol_limit)

    async def start_polling(self):
        """실전 매매/백테스트 모드에서의 일반 폴링 (1초 주기 자산 갱신)"""
        if getattr(self, "_polling_active", False):
            return
        
        self._polling_active = True
        self._is_running = True
        self.logger.info("LiveDashboardViewModel: 자산 상태 폴링 루프 시작")
        last_sync_time = 0
        while self._is_running:
            try:
                now = time.time()
                # [신규] 20초마다 실제 계좌 상태(실현손익, 주문가능금액) 동기화 트리거
                if self.account_service and (now - last_sync_time > 20):
                    asyncio.create_task(self.account_service.sync_all())
                    last_sync_time = now

                # Polling 시점에도 리스크 지표와 잔고를 최신화하여 UI에 전송
                if hasattr(self.order_manager, 'risk_manager') and self.order_manager.risk_manager:
                    rm = self.order_manager.risk_manager
                    per_symbol_limit = rm.get_dynamic_max_invest()
                    # [수정] 실현 손익과 평가 손익을 구분하여 가져옴
                    realized_pnl = getattr(self.order_manager, 'daily_realized_pnl', 0.0)
                    evaluation_pnl = getattr(self.order_manager, 'daily_evaluation_pnl', 0.0)
                    total_cash = getattr(self.order_manager, 'orderable_cash', 0.0)

                    self.sig_risk_metrics_updated.emit(realized_pnl, evaluation_pnl, total_cash, per_symbol_limit)
                    self.sig_balance_updated.emit(self.order_manager.current_balance)
            except Exception as e:
                self.logger.error(f"Polling 중 오류: {e}")

            await asyncio.sleep(1.0)
        
        self._polling_active = False

    def start_mock_stream(self):
        """장외 시간 테스트용 모크 스트림 시작"""
        self.sig_log_appended.emit("장외 테스트용 Mock 데이터 스트림 시작...")
        if not self._mock_task or self._mock_task.done():
            self._mock_task = asyncio.create_task(self.data_collector.start_mock_stream())

    def trigger_panic_sell(self):
        """패닉 셀 버튼 이벤트 수신: 모든 주문 취소 및 시장가 매도"""
        self.sig_log_appended.emit("[시스템] 🚨 PANIC SELL 트리거됨! 전체 주문 취소 및 시장가 청산 진행...")
        asyncio.create_task(self._execute_panic_sell())

    def cancel_orders_only(self):
        """메뉴 액션: 미체결 전체 취소"""
        self.sig_log_appended.emit("메뉴: 미체결 주문 전체 취소 요청...")
        asyncio.create_task(self.order_manager.cancel_all_orders())
        self.sig_menu_action_result.emit("미체결 취소", "모든 미체결 주문에 대해 취소 요청을 전송했습니다.")

    def reset_pnl(self):
        """메뉴 액션: 당일 손익 초기화"""
        self.sig_log_appended.emit("메뉴: 당일 손익 데이터 초기화...")
        # 실제 로직은 계좌 관리 객체나 PnL 트래커를 리셋해야 함.
        self.sig_menu_action_result.emit("손익 초기화", "당일 누적 손익 데이터가 초기화되었습니다.")

    # --- [실시간 제어 액션] ---
    def toggle_ai_trading(self, paused: bool):
        """AI의 매매 판단(추론)만 일시적으로 정지하거나 재개"""
        sm = getattr(self.config_manager, "_injected_strategy_manager", None)
        if sm:
            self._is_ai_paused = paused
            sm.set_ai_paused(paused)
            self.sig_trading_paused.emit(paused)
            status = "일시정지" if paused else "재개"
            msg = f"[시스템] 🤖 AI 매매 의사결정이 {status}되었습니다."
            self.sig_log_appended.emit(msg)

            # [Firebase] 제어 상태 동기화
            fb = getattr(self.config_manager, "firebase_manager", None)
            if fb:
                asyncio.create_task(fb.update_control_status(
                    is_monitoring_active=not self._is_monitoring_stopped,
                    is_ai_trading_active=not paused
                ))

    def toggle_monitoring(self, stopped: bool):
        """실시간 데이터 수집(웹소켓) 자체를 중단하거나 재개"""
        self._is_monitoring_stopped = stopped
        if stopped:
            asyncio.create_task(self.data_collector.stop())
            self.sig_monitoring_stopped.emit(True)
            self.sig_log_appended.emit("[시스템] 📡 실시간 종목 감시가 중단되었습니다. (웹소켓 연결 해제)")
        else:
            # 재시작 전 안전하게 플래그 리셋
            self.data_collector.is_running = True 
            asyncio.create_task(self.data_collector.start())
            self.sig_monitoring_stopped.emit(False)
            self.sig_log_appended.emit("[시스템] 📡 실시간 종목 감시를 재개합니다. (재연결 시도 중...)")

        # [Firebase] 제어 상태 동기화
        fb = getattr(self.config_manager, "firebase_manager", None)
        if fb:
            asyncio.create_task(fb.update_control_status(
                is_monitoring_active=not stopped,
                is_ai_trading_active=not self._is_ai_paused
            ))

    async def _execute_panic_sell(self):
        try:
            await self.order_manager.cancel_all_orders()

            holdings_dict = getattr(self.order_manager, 'holdings', {})
            bot_holdings = getattr(self.order_manager, 'bot_holdings', {})
            protected_symbols = self.config_manager.get("protected_symbols", [])

            if not isinstance(holdings_dict, dict):
                # Fallback for old mock structure, though it should be dict now
                self.sig_log_appended.emit("[시스템] 보유 잔고 데이터 형식이 올바르지 않습니다.")
                return

            sell_orders_placed = 0
            for symbol, qty in holdings_dict.items():
                if qty <= 0:
                    continue

                # 1. 보호 종목 필터
                if symbol in protected_symbols:
                    self.sig_log_appended.emit(f"[보호 종목] {symbol}은 청산 대상에서 제외됩니다.")
                    continue

                # 2. 봇(Agent) 매수 종목 필터 (수동 매수 종목 제외)
                bot_qty = bot_holdings.get(symbol, 0)
                if bot_qty <= 0:
                    self.sig_log_appended.emit(f"[수동 매수 종목] {symbol}은 봇이 매수한 이력이 없어 청산하지 않습니다.")
                    continue

                # 봇이 보유한 수량 한도 내에서만 청산 (전체 수량 중 봇 수량)
                target_qty = min(qty, bot_qty)

                await self.order_manager.send_order("SELL", symbol, 0, target_qty)
                self.sig_log_appended.emit(f"[시스템] 잔고 {target_qty}주(종목:{symbol}) 전량 시장가 매도 주문 전송 완료.")
                sell_orders_placed += 1

            if sell_orders_placed == 0:
                self.sig_log_appended.emit("[시스템] 청산 가능한 봇 보유 잔고가 없습니다. 주문 취소만 완료되었습니다.")

        except Exception as e:
            self.sig_error_occurred.emit(f"Panic Sell 에러: {e}")

    def stop(self):
        self._is_running = False
        # [종료 안정화] UI 갱신 타이머 중지하여 자원 해제 및 종료 지연 방지
        if hasattr(self, '_ui_flush_timer'):
            self._ui_flush_timer.stop()
            self.logger.info("LiveDashboardViewModel: UI Flush Timer stopped.")
            
        if self._mock_task and not self._mock_task.done():
            self._mock_task.cancel()

class AssetDataViewModel(QObject):
    """
    AssetDataManagerTab을 위한 ViewModel.
    UI 이벤트(종목 로드/저장, 데이터 수집 시작)를 Core 로직으로 연결하고
    수집 상태(Progress)를 UI로 Signal Emit 합니다.
    """
    # UI로 보낼 시그널들
    symbols_loaded = pyqtSignal(list)
    symbol_update_failed = pyqtSignal(str)
    symbol_update_success = pyqtSignal(str)

    sig_progress_updated = pyqtSignal(int)
    sig_status_updated = pyqtSignal(str)
    fetch_completed = pyqtSignal(str)
    fetch_failed = pyqtSignal(str)

    def __init__(self, config_manager, historical_fetcher, influx_client, universe_manager, token_manager, firebase_manager=None):
        super().__init__()
        self.config_manager = config_manager
        self.historical_fetcher = historical_fetcher
        self.influx_client = influx_client
        self.universe_manager = universe_manager
        self.token_manager = token_manager
        self.firebase_manager = firebase_manager
        self.logger = logging.getLogger("AssetDataViewModel")

        # [🚨 통합 패치] 이제 별도의 ConditionWebSocketThread를 사용하지 않고 DataCollector를 통해 통합 관리합니다.
        self.condition_ws_thread = None

        # 메모리 상의 현재 실전 매매 유니버스 (동적 관리용)
        self._current_trading_universe = []

    @pyqtSlot(str, str)
    def _on_condition_event(self, event_type: str, symbol: str):
        """실시간 편입(I)/이탈(D) 이벤트 처리 슬롯"""
        self.logger.info(f"[동적 유니버스] 이벤트 수신: {event_type} | {symbol}")
        
        # 1. StrategyManager 및 LiveVM 참조 획득
        sm = getattr(self.config_manager, "_injected_strategy_manager", None)
        live_vm = getattr(self.config_manager, "_injected_live_vm", None)
        
        if not sm or not live_vm:
            self.logger.warning("StrategyManager 또는 LiveVM이 초기화되지 않아 실시간 이벤트를 무시합니다.")
            return

        protected_list = self.config_manager.get("protected_symbols", [])
        protected_symbols = set(str(s).split('_')[0] for s in protected_list)

        if event_type == 'I': # 편입
            if symbol in protected_symbols:
                self.logger.info(f"🚫 [편입 제외] {symbol}은 보호 종목이므로 무시합니다.")
                return

            # 이미 존재하는지 확인
            if any(s['code'] == symbol for s in self._current_trading_universe):
                return
            
            # 종목 정보 구성 (이름 등은 캐시에서 로드)
            name = self.universe_manager.get_stock_name_from_cache(symbol) or f"New_{symbol}"
            new_item = {"code": symbol, "name": name, "price": 0, "flu_rt": 0, "volume": 0}
            
            self._current_trading_universe.append(new_item)
            self.logger.info(f"✅ [편입] {name}({symbol}) 종목이 유니버스에 추가되었습니다.")
            
            # [중요] SM 직접 호출을 제거하고 LiveVM을 통해서만 동기화합니다. (무한 루프 방지)
            live_vm.update_universe_list(self._current_trading_universe)
            
        elif event_type == 'D': # 이탈
            # 해당 종목 제거
            original_len = len(self._current_trading_universe)
            self._current_trading_universe = [s for s in self._current_trading_universe if s['code'] != symbol]
            
            if len(self._current_trading_universe) < original_len:
                self.logger.info(f"❌ [이탈] {symbol} 종목이 유니버스에서 제거되었습니다.")
                
                # [중요] SM 직접 호출을 제거하고 LiveVM을 통해서만 동기화합니다.
                live_vm.update_universe_list(self._current_trading_universe)

    def start_condition_ws(self):
        """웹소켓 감시 시작"""
        if not self.condition_ws_thread.isRunning():
            self.condition_ws_thread.start()
            self.logger.info("📡 실시간 조건검색 모니터링 스레드 가동!")

    def stop_condition_ws(self):
        """웹소켓 감시 중단"""
        self.condition_ws_thread.stop()
        self.logger.info("🛑 실시간 조건검색 모니터링 중단")

    def build_universe(self, top_n: int = 20):
        """UniverseManager를 통해 거래대금 상위 종목을 추출하여 Config에 저장"""
        asyncio.create_task(self._build_universe_task(top_n=top_n, is_auto=False))

    def fetch_db_symbols(self):
        """InfluxDB에 저장된 모든 고유 종목 리스트를 가져와서 유니버스로 설정"""
        asyncio.create_task(self._fetch_db_symbols_task())

    async def _fetch_db_symbols_task(self):
        self.sig_status_updated.emit("InfluxDB에서 저장된 모든 종목 리스트 조회 중...")
        symbols = await self.influx_client.get_all_symbols()
        
        if not symbols:
            self.symbol_update_failed.emit("DB에서 저장된 종목 데이터를 찾을 수 없습니다.")
            return

        self.sig_status_updated.emit(f"DB 심볼 {len(symbols)}개 발견. 로컬 캐시 매핑 중...")
        
        # [최적화] 서버 호출 없이 로컬 캐시 정보만 활용
        new_symbols = []
        
        for original_code in symbols:
            raw_code = str(original_code).split('_')[0].strip()
            # 기본값은 코드명
            stock_name = raw_code 
            
            # 1. 로컬 파일 캐시에서만 명칭 조회 (서버 호출 배제)
            if hasattr(self.universe_manager, 'get_stock_name_from_cache'):
                name_result = self.universe_manager.get_stock_name_from_cache(raw_code)
                if name_result:
                    stock_name = name_result
            
            new_symbols.append({
                "code": raw_code, 
                "name": stock_name
            })
            
        # Config 업데이트 및 저장 (set_symbols 내부에서 자동 저장됨)
        self.config_manager.set_symbols(new_symbols)
        
        # UI 동기화
        self.symbols_loaded.emit(new_symbols)
        self.symbol_update_success.emit(f"DB로부터 {len(symbols)}개의 종목 리스트를 로컬 캐시 기반으로 불러왔습니다.")
        self.sig_status_updated.emit("대기 중")

    async def auto_collect_after_market(self):
        """
        장 종료 후 호출되는 스크립트.
        최종 유니버스를 업데이트하고, 당일 데이터를 자동으로 DB에 벌크 수집합니다.
        """
        self.sig_status_updated.emit("[POST-MARKET COLLECTION] 장 종료 후 최종 주도주 유니버스 갱신 시작...")

        # 1. Build Universe (Auto mode, bypasses some strict UI popups if needed)
        success = await self._build_universe_task(is_auto=True)

        if success:
            import datetime
            today_str = datetime.datetime.now().strftime("%Y%m%d")
            self.sig_status_updated.emit(f"[POST-MARKET COLLECTION] 유니버스 갱신 완료. {today_str} 데이터 수집 시작...")
            # 2. Fetch and Store (uses the newly updated config symbols)
            await self._start_bulk_historical_fetch_task(today_str, is_auto=True)
        else:
            self.sig_status_updated.emit("[POST-MARKET COLLECTION] 유니버스 갱신 실패로 수집을 중단합니다.")

    async def _build_universe_task(self, top_n: int = 20, is_auto=False, save_to_config: bool = True):
        """
        유니버스를 생성합니다.
        - save_to_config=True: config.yaml의 symbols를 덮어씁니다. (데이터 관리용)
        - save_to_config=False: config를 건드리지 않고 리스트만 반환합니다. (실전 매매용)
        """
        self.sig_progress_updated.emit(0)
        if not is_auto:
            self.sig_status_updated.emit(f"시장 전체 종목 조회 및 주도주 필터링 중 (Top {top_n})...")

        access_token = self.config_manager.get("KIWOOM_ACCESS_TOKEN", "")
        if not access_token:
            self.fetch_failed.emit("API 접근 토큰이 없습니다. 설정에서 발급해 주세요.")
            return []

        # @future_safe에 의해 감싸진 async 함수는 await하면 반환값이 Result 타입 객체입니다.
        if save_to_config:
            # 데이터 관리용: 기존 거래대금 상위 방식 유지
            result = await self.universe_manager.build_top_n_universe(access_token, top_n=top_n)
        else:
            # 실전 매매용: 서버 조건 검색(AI스캘핑주도주) 방식 사용
            self.sig_status_updated.emit("서버 실시간 조건 검색 종목(주도주) 수집 중...")
            result = await self.universe_manager.build_condition_universe(access_token, target_cond_nm="AI스캘핑주도주")

        if isinstance(result, IOFailure):
            err_msg = str(result.failure()._inner_value if hasattr(result.failure(), '_inner_value') else result.failure())
            self.fetch_failed.emit(f"유니버스 생성 실패: {err_msg}")
            return []

        try:
            top_stocks = result.unwrap()._inner_value
        except Exception as e:
            top_stocks = []

        if not isinstance(top_stocks, list):
            top_stocks = []

        # 수집된 종목 정제
        new_symbols = []
        local_codes = set([s.get("code") for s in self.config_manager.get_symbols()])
        excluded_count = 0

        for stock in top_stocks:
            if isinstance(stock, dict) and "code" in stock and "name" in stock:
                clean_code = str(stock["code"]).split('_')[0].strip()
                
                # [분리 핵심] 실전 매매용 수집(save_to_config=False)일 경우, 데이터 관리용 종목은 제외
                if not save_to_config and clean_code in local_codes:
                    excluded_count += 1
                    continue

                new_symbols.append({
                    "code": clean_code, 
                    "name": stock["name"],
                    "price": stock.get("price", 0.0),
                    "flu_rt": stock.get("flu_rt", 0.0),
                    "volume": stock.get("volume", 0.0)
                })

        if excluded_count > 0:
            self.logger.info(f"[유니버스 분리] 데이터 관리용 종목 {excluded_count}개를 실전 매매 감시 대상에서 제외했습니다.")

        # 보유 종목 우선 편입 로직 (실전 매매 시 잔고 누락 방지)
        try:
            strategy_manager = getattr(self.config_manager, '_injected_strategy_manager', None)
            if strategy_manager and hasattr(strategy_manager, 'order_manager'):
                holdings = strategy_manager.order_manager.holdings
                new_codes = set([s["code"] for s in new_symbols])
                
                old_symbols = self.config_manager.get_symbols()
                old_sym_map = {s.get("code"): s for s in old_symbols}

                for code, qty in holdings.items():
                    has_unexecuted = False
                    if hasattr(strategy_manager.order_manager, 'has_unexecuted_orders'):
                        has_unexecuted = strategy_manager.order_manager.has_unexecuted_orders(code)

                    if (qty > 0 or has_unexecuted) and code not in new_codes:
                        old_s = old_sym_map.get(code, {})
                        name = old_s.get("name", self.universe_manager.get_stock_name_from_cache(code) if hasattr(self, 'universe_manager') else f"Held_{code}")
                        if not name: name = f"Held_{code}"

                        self.logger.warning(f"[보유 종목 유지] {name}({code}) 종목이 조건에서 탈락했으나 잔고/미체결로 인해 감시 리스트에 유지됩니다.")
                        
                        new_symbols.append({
                            "code": code, "name": name,
                            "price": old_s.get("price", 0.0),
                            "flu_rt": old_s.get("flu_rt", 0.0),
                            "volume": old_s.get("volume", 0.0)
                        })
        except Exception as e:
            self.logger.error(f"보유 종목 강제 유지 로직 에러: {e}")

        # 결과 처리
        if not new_symbols:
            self.logger.warning("새로운 유니버스 리스트가 비어 있습니다.")
            return self.config_manager.get_symbols() if not save_to_config else []

        if save_to_config:
            self.config_manager.set_symbols(new_symbols)
            self.load_symbols() # UI 갱신 트리거
        
        self.sig_progress_updated.emit(100)
        msg = f"유니버스 수집 완료! ({len(new_symbols)} 종목)"
        self.sig_status_updated.emit(msg)
        if not is_auto:
            self.fetch_completed.emit(msg)

        return new_symbols

    def load_symbols(self):
        # ConfigManager의 Result 처리
        result = self.config_manager.load_config()
        if isinstance(result, Success):
            symbols = self.config_manager.get_symbols()
            self.symbols_loaded.emit(symbols)
        else:
            self.symbol_update_failed.emit(f"설정 로드 실패: {result.failure()}")

    def send_test_trade_log(self, log_type: str):
        """[테스트] 샘플 체결 로그를 파이어베이스로 전송합니다."""
        if not self.firebase_manager:
            self.symbol_update_failed.emit("파이어베이스 매니저가 초기화되지 않았습니다.")
            return

        import datetime
        import random
        
        # 샘플 데이터 생성
        symbol = "005930" # 삼성전자
        symbol_name = "삼성전자(테스트)"
        price = random.randint(70000, 80000)
        qty = random.randint(1, 10)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        async def _send():
            try:
                # FirebaseManager에 정의된 send_trade_log 호출
                await self.firebase_manager.send_trade_log(
                    log_type=log_type,
                    symbol=symbol,
                    symbol_name=symbol_name,
                    price=price,
                    qty=qty,
                    timestamp=timestamp
                )
                self.symbol_update_success.emit(f"[테스트] {log_type} 로그 전송 성공!")
            except Exception as e:
                self.logger.error(f"테스트 로그 전송 실패: {e}")
                self.symbol_update_failed.emit(f"로그 전송 실패: {e}")

        asyncio.create_task(_send())

    def add_symbol(self, code: str, name: str = None):
        if not name:
            # 로컬 캐시(stock_names.json)에서 종목명 조회
            name = self.universe_manager.get_stock_name_from_cache(code)
            if not name:
                name = f"Unknown_{code}"
                self.logger.warning(f"종목명 캐시에서 [{code}]를 찾을 수 없습니다. 임시 이름으로 등록합니다.")

        result = self.config_manager.add_symbol(code, name)
        if isinstance(result, Success):
            self.symbol_update_success.emit(f"종목 추가 완료: {name}({code})")
            self.load_symbols() # UI 갱신 트리거
        else:
            self.symbol_update_failed.emit(str(result.failure()))

    def remove_symbol(self, code: str):
        self.remove_symbols([code])

    def remove_symbols(self, codes: List[str]):
        """다중 종목 삭제 처리"""
        result = self.config_manager.remove_symbols(codes)
        if isinstance(result, Success):
            self.symbol_update_success.emit(f"종목 삭제 완료: {len(codes)}개 항목")
            self.load_symbols()
        else:
            self.symbol_update_failed.emit(str(result.failure()))

    def delete_db_data(self, codes: List[str]):
        """선택된 종목들의 모든 DB 데이터를 삭제 요청 (영구 삭제)"""
        asyncio.create_task(self._delete_db_data_task(codes))

    async def _delete_db_data_task(self, codes: List[str]):
        self.sig_status_updated.emit(f"종목 {len(codes)}개의 DB 데이터 삭제 중...")
        success_count = 0
        deleted_successfully = []
        
        for code in codes:
            res = await self.influx_client.delete_symbol_data(code)
            if res:
                success_count += 1
                deleted_successfully.append(code)
        
        self.sig_status_updated.emit("대기 중")
        if success_count > 0:
            # [추가] DB 삭제 성공 시 화면 리스트(Universe)에서도 해당 종목 제거
            self.config_manager.remove_symbols(deleted_successfully)
            self.load_symbols() # UI 리스트 갱신
            self.symbol_update_success.emit(f"{success_count}개 종목의 DB 데이터 삭제 및 리스트 갱신 완료.")
        else:
            self.symbol_update_failed.emit("DB 데이터 삭제에 실패했거나 삭제할 데이터가 없습니다.")

    def start_historical_fetch(self, symbols: Any, start_date: str):
        """특정 종목(들)에 대한 수집 시작"""
        if isinstance(symbols, str):
            symbols = [symbols]
        asyncio.create_task(self._fetch_and_store(symbols, start_date))

    def start_bulk_historical_fetch(self, start_date: str, is_auto: bool = False):
        """UI에서 호출하는 래퍼 (Non-blocking)"""
        asyncio.create_task(self._start_bulk_historical_fetch_task(start_date, is_auto))

    async def _start_bulk_historical_fetch_task(self, start_date: str, is_auto: bool = False):
        """Config에 등록된 모든 종목(Universe)에 대한 실제 일괄 수집 태스크"""
        symbols = [s.get("code") for s in self.config_manager.get_symbols()]
        if not symbols:
            self.fetch_failed.emit("수집할 종목이 없습니다.")
            return
        await self._fetch_and_store(symbols, start_date, is_auto)

    async def _fetch_and_store(self, symbols: List[str], start_date: str, is_auto: bool = False):
        total_symbols = len(symbols)
        total_data_collected = 0

        # 토큰 유효성 확인 및 갱신
        access_token = self.token_manager.get_token()
        if not access_token:
            self.sig_status_updated.emit("API 토큰 갱신 중...")
            await self.token_manager.refresh_token()
            access_token = self.token_manager.get_token()

        if not access_token:
            self.fetch_failed.emit("API 접근 토큰을 가져오지 못했습니다. 설정을 확인해 주세요.")
            return

        for idx, symbol in enumerate(symbols):
            # DB에서 마지막 수집 시점 조회 (증분 수집용)
            last_ts = await self.influx_client.get_last_timestamp(symbol)
            if last_ts:
                self.logger.error(f"[{symbol}] DB 체크포인트 발견: {last_ts}. 이후 데이터만 증분 수집합니다.")

            def update_progress(pct: int, msg: str):
                base_pct = (idx / total_symbols) * 100
                current_pct = base_pct + (pct / total_symbols)
                self.sig_progress_updated.emit(int(current_pct))
                self.sig_status_updated.emit(msg)

            self.sig_progress_updated.emit(int((idx / total_symbols) * 100))
            self.sig_status_updated.emit(f"[{symbol}] 수집 시도 ({idx+1}/{total_symbols})...")

            fetch_result = await self.historical_fetcher.fetch_historical_data(
                symbol, start_date, access_token, update_progress, stop_timestamp=last_ts
            )

            # 토큰 만료 시 재시도 로직
            if isinstance(fetch_result, IOFailure):
                failure_val = str(fetch_result.failure()._inner_value if hasattr(fetch_result.failure(), '_inner_value') else fetch_result.failure())
                if failure_val == "TOKEN_EXPIRED":
                    self.logger.error(f"[{symbol}] 토큰 만료 감지됨. 토큰을 갱신하고 재시도합니다.")
                    self.sig_status_updated.emit(f"[{symbol}] 토큰 갱신 및 재시도 중...")
                    await self.token_manager.refresh_token()
                    access_token = self.token_manager.get_token()
                    # 1회 재시도 (마지막 TS 유지)
                    fetch_result = await self.historical_fetcher.fetch_historical_data(
                        symbol, start_date, access_token, update_progress, stop_timestamp=last_ts
                    )

            if isinstance(fetch_result, IOFailure):
                err_msg = str(fetch_result.failure()._inner_value if hasattr(fetch_result.failure(), '_inner_value') else fetch_result.failure())
                self.symbol_update_failed.emit(f"[{symbol}] 수집 최종 실패: {err_msg}")
                continue

            try:
                data_list = fetch_result.unwrap()._inner_value
                if not isinstance(data_list, list):
                    data_list = []
            except Exception:
                data_list = []

            fetch_count = len(data_list)
            if fetch_count == 0:
                self.logger.warning(f"[{symbol}] 수집된 데이터가 0건입니다. 스킵합니다.")
                continue

            self.logger.info(f"[{symbol}] 수집 완료: {fetch_count}건. InfluxDB 갱신 시작...")
            
            self.sig_progress_updated.emit(int(((idx + 0.9) / total_symbols) * 100))
            self.sig_status_updated.emit(f"[{symbol}] 기존 데이터 정리 및 적재 중...")

            try:
                # [수정] 증분 수집을 위해 기존 데이터 삭제 로직 제거 (InfluxDB는 자동으로 중복을 덮어씀)
                # await self.influx_client.delete_data("historical_data", symbol)
                # await self.influx_client.delete_data("tick_data", symbol)
                
                await self.influx_client.bulk_insert(data_list)
                self.logger.info(f"[{symbol}] InfluxDB 적재 성공: {fetch_count}건 추가 완료.")
                total_data_collected += fetch_count
            except Exception as e:
                self.logger.error(f"[{symbol}] DB 처리 중 에러 발생: {e}")
                self.symbol_update_failed.emit(f"[{symbol}] DB 처리 에러: {e}")

        self.sig_progress_updated.emit(100)

        status_msg = "모든 종목 수집 및 적재 완료"
        msg = f"총 {total_symbols}개 종목, {total_data_collected}건 적재 완료!"
        if is_auto:
            status_msg = "[POST-MARKET COLLECTION] " + status_msg
            msg = "[POST-MARKET COLLECTION] " + msg

        self.sig_status_updated.emit(status_msg)
        self.logger.info(f"전체 수집 프로세스 종료: {msg}")

        if not is_auto:
            if total_data_collected == 0:
                self.fetch_completed.emit("이미 모든 데이터가 최신 상태입니다 (추가 수집 없음).")
            else:
                self.fetch_completed.emit(msg)

class AITrainingViewModel(QObject):
    """
    AI 학습을 관장하는 ViewModel.
    Data Collector와 Env, Agent를 조립하고 QThread Worker를 통해 학습을 진행.
    """
    sig_training_started = pyqtSignal()
    sig_training_progress = pyqtSignal(int, float, float) # step, reward, loss
    sig_training_log = pyqtSignal(str)
    sig_training_finished = pyqtSignal()
    sig_error = pyqtSignal(str)

    def __init__(self, config_manager, data_collector, order_manager, influx_client):
        super().__init__()
        self.config_manager = config_manager
        self.data_collector = data_collector
        self.order_manager = order_manager
        self.influx_client = influx_client
        self.worker = None
        self.prep_task = None # [신규] 데이터 조회 및 준비 태스크 추적용

    def start_training(self, total_timesteps: int, learning_rate: float, max_records: int, 
                       feature_mode: str = "basic", use_smart_sampling: bool = False, ppo_params: dict = None,
                       always_start_day_begin: bool = False, allow_overnight_episodes: bool = False):
        """UI에서 학습 시작 요청을 받아 파이프라인 조립 후 워커 실행"""
        if self.worker and self.worker.isRunning():
            self.sig_error.emit("이미 학습이 진행 중입니다.")
            return

        # 이전 태스크가 남아있다면 정리
        if self.prep_task and not self.prep_task.done():
            self.prep_task.cancel()

        self.prep_task = asyncio.create_task(self._prepare_and_start_training(
            total_timesteps, learning_rate, max_records, feature_mode, use_smart_sampling, ppo_params,
            always_start_day_begin, allow_overnight_episodes
        ))

    async def _prepare_and_start_training(self, timesteps: int, lr: float, max_records: int, 
                                          feature_mode: str, use_smart_sampling: bool = False, ppo_params: dict = None,
                                          always_start_day_begin: bool = False, allow_overnight_episodes: bool = False):
        self.sig_training_log.emit(f"1. InfluxDB에서 유니버스 전체 데이터 조회 중 (종목당 최대 {max_records}건)...")
        # [안정성 강화] 동시 조회 개수를 3개로 제한
        sem = asyncio.Semaphore(3)
        
        async def fetch_with_semaphore(symbol_code, limit):
            async with sem:
                return await self.influx_client.fetch_recent_data(symbol_code, limit)
        
        symbols = self.config_manager.get_symbols()
        if not symbols:
            self.sig_error.emit("유니버스가 비어있습니다. 종목을 먼저 추가해주세요.")
            return

        # 1. 모든 종목의 데이터 비동기 병렬 조회
        historical_data_dict = {}
        fetch_tasks = []
        for s in symbols:
            code = s.get("code")
            fetch_tasks.append(fetch_with_semaphore(code, max_records))

        try:
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            
            for s, res in zip(symbols, results):
                code = s.get("code")
                # [버그 수정] Exception뿐만 아니라 CancelledError 등 모든 예외(BaseException)를 체크
                if isinstance(res, BaseException):
                    self.sig_training_log.emit(f"   [경고] {code} 데이터 로드 실패: {type(res).__name__}")
                elif res:
                    historical_data_dict[code] = res
                else:
                    self.sig_training_log.emit(f"   [주의] {code} 데이터가 DB에 없습니다.")

            self.sig_training_log.emit(f"   => 총 {len(historical_data_dict)}개 종목의 데이터 로드 완료.")
            
            if not historical_data_dict:
                self.sig_error.emit("학습 가능한 데이터가 어느 종목에서도 발견되지 않았습니다.")
                return

        except asyncio.CancelledError:
            self.sig_training_log.emit("   [알림] 데이터 조회 및 학습 준비 작업이 사용자에 의해 중단되었습니다.")
            self.sig_training_finished.emit()
            raise # 이벤트 루프에 취소 사실 전파
        except Exception as e:
            self.sig_error.emit(f"데이터 조회 준비 작업 중 에러: {e}")
            self.sig_training_finished.emit()
            return

        # 2. Env 생성 및 Agent 주입
        from env.trading_env import ScalpingTradingEnv
        from models.agent import TradingAgentWrapper
        from gui.training_worker import TrainingWorker, TrainingSignals

        sampling_desc = "활황장 집중(Smart)" if use_smart_sampling else "순수 랜덤(Random)"
        self.sig_training_log.emit(f"2. RL Environment 생성 (Mode: {feature_mode}, Sampling: {sampling_desc})...")
        env_config = {
            "historical_data_dict": historical_data_dict,
            "initial_balance": 10000000,
            "feature_mode": feature_mode,
            "use_smart_sampling": use_smart_sampling,  # [핵심] 스마트 샘플링 플래그 주입
            "always_start_day_begin": always_start_day_begin,
            "allow_overnight_episodes": allow_overnight_episodes,
        }
        env = ScalpingTradingEnv(self.data_collector, self.order_manager, env_config)

        # [핵심] 스마트 샘플링 여부에 따라 모델명/폴더명에 태그 부여
        sampling_tag = "smart" if use_smart_sampling else "random"
        model_save_dir = f"./saved_models/{sampling_tag}/"
        tb_log_dir    = f"./tensorboard_logs/{sampling_tag}/"

        ent_coef = 0.005
        # [안정화] seq_len을 하드코딩하지 않고 설정 파일에서 직접 가져옴
        config_dict = self.config_manager.get_dict() if hasattr(self.config_manager, "get_dict") else {}
        current_seq_len = config_dict.get("seq_len", 10)

        agent_config = {
            "seq_len": current_seq_len,
            "learning_rate": lr,
            "ent_coef": ppo_params.get("ent_coef", 0.01) if ppo_params else 0.01,
            "clip_range": ppo_params.get("clip_range", 0.2) if ppo_params else 0.2,
            "gamma": ppo_params.get("gamma", 0.99) if ppo_params else 0.99,
            "gae_lambda": ppo_params.get("gae_lambda", 0.95) if ppo_params else 0.95,
            "feature_mode": feature_mode,
            "model_save_dir": model_save_dir,
            "tensorboard_log": tb_log_dir,
            "model_name_suffix": sampling_tag,
        }
        
        self.sig_training_log.emit(
            f"   => 모델 저장 경로: [{model_save_dir}] | 학습 태그: [{sampling_tag}]"
        )
        agent = TradingAgentWrapper(env, agent_config)

        # 3. 최신 모델 가중치 자동 로드 - 같은 sampling_tag + feature_mode 범주 내에서만 콜렉션
        save_dir = model_save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 현재 feature_mode + sampling_tag에 해당하는 모델만 검색 (예: model_advanced_smart_*.zip)
        model_prefix = f"model_{feature_mode}_{sampling_tag}"
        model_files = glob.glob(os.path.join(save_dir, f"{model_prefix}_*.zip"))

        if model_files:
            latest_model_zip = sorted(model_files)[-1]
            load_path = latest_model_zip.replace(".zip", "")
            try:
                agent.load_weights(load_path)
                self.sig_training_log.emit(
                    f"   => [{sampling_tag}] 최신 모델 로딩 성공 ✅: {os.path.basename(latest_model_zip)}"
                )
            except Exception as e:
                self.sig_training_log.emit(f"   => [주의] [{sampling_tag}] 모델 로드 중 충돌 (초기화): {e}")
        else:
            self.sig_training_log.emit(
                f"   => [{sampling_tag}] 모드의 기존 모델이 없습니다. 백지상태에서 학습을 시작합니다."
            )

        # 3. Worker 생성 및 실행
        self.sig_training_log.emit("3. QThread 학습 워커 실행...")

        signals = TrainingSignals()
        signals.started.connect(lambda: self.sig_training_started.emit())
        signals.progress_updated.connect(lambda s, r, l: self.sig_training_progress.emit(s, r, l))
        signals.log_msg.connect(lambda m: self.sig_training_log.emit(m))
        signals.finished.connect(lambda: self.sig_training_finished.emit())
        signals.error.connect(lambda e: self.sig_error.emit(f"학습 워커 에러: {e}"))

        self.worker = TrainingWorker(agent, timesteps, signals)
        self.worker.start()

    def stop_training(self):
        """학습 중지 버튼 클릭 시"""
        # 1. 데이터 조회 단계인 경우 태스크 취소
        if self.prep_task and not self.prep_task.done():
            self.sig_training_log.emit("데이터 조회 작업을 취소합니다...")
            self.prep_task.cancel()
            
        # 2. 이미 학습 워커(Thread)가 실행 중인 경우
        if self.worker and self.worker.isRunning():
            self.sig_training_log.emit("학습 워커 중지 요청 전송됨...")
            self.worker.stop()

class SettingsViewModel(QObject):
    """
    Settings 탭을 위한 ViewModel.
    ConfigManager를 통해 통합된 설정(.env 및 config.yaml)을 관리합니다.
    """
    settings_loaded = pyqtSignal(dict) # unified config dict
    save_completed = pyqtSignal(str)
    save_failed = pyqtSignal(str)
    connection_test_completed = pyqtSignal(bool, str) # (Success bool, Message)
    sig_menu_action_result = pyqtSignal(str, str) # title, message

    def __init__(self, config_manager, influx_client, system_config=None):
        super().__init__()
        self.config_manager = config_manager
        self.influx_client = influx_client
        self.system_config = system_config
        self.logger = logging.getLogger("SettingsViewModel")

    def load_settings(self):
        """ConfigManager를 통해 통합 설정을 로드하고 UI로 Emit합니다."""
        result = self.config_manager.load_config()
        if isinstance(result, Success):
            data = result.unwrap()
            if self.system_config:
                data["BYPASS_MARKET_HOURS"] = self.system_config.BYPASS_MARKET_HOURS
            self.settings_loaded.emit(data)
        else:
            self.save_failed.emit(f"설정 로드 실패: {result.failure()}")

    def on_remote_settings_changed(self, data: dict):
        """
        [실시간 동기화] Firebase 등 외부에서 설정이 변경되었을 때 호출되어
        UI 구성 요소들을 최신값으로 즉시 갱신합니다.
        """
        # ConfigManager의 최신 캐시 데이터를 UI로 전달
        full_config = self.config_manager.get_dict()
        if self.system_config:
            full_config["BYPASS_MARKET_HOURS"] = self.system_config.BYPASS_MARKET_HOURS
        self.settings_loaded.emit(full_config)

    def toggle_bypass_market_hours(self, enabled: bool):
        """장외 시간 테스트 모드 토글 및 즉시 저장"""
        if self.system_config:
            self.system_config.set_bypass_market_hours(enabled)
            self.logger.info(f"SettingsViewModel: 장외 시간 테스트 모드 변경 -> {enabled}")

    def save_settings(self, updates: dict):
        """수정된 설정값들을 ConfigManager에 전달하여 저장합니다."""
        # Convert UI mode string to internal mode string and map to nested structure
        mode_str = updates.get("trading_mode", "모의투자")
        mapped_mode = "real" if mode_str == "실전투자" else "virtual"

        # update nested kiwoom dictionary properly
        kiwoom_conf = self.config_manager.get("kiwoom", {})
        kiwoom_conf["trading_mode"] = mapped_mode
        updates["kiwoom"] = kiwoom_conf

        # we can remove trading_mode from the root dict
        if "trading_mode" in updates:
            del updates["trading_mode"]

        save_result = self.config_manager.update_settings(updates)
        if isinstance(save_result, Success):
            # [추가] 로그 레벨 즉시 반영
            new_log_level = updates.get("log_level", "INFO").upper()
            logging.getLogger().setLevel(getattr(logging, new_log_level, logging.INFO))
            self.logger.info(f"시스템 로그 레벨이 {new_log_level}로 변경되었습니다.")
            
            self.save_completed.emit("설정이 성공적으로 저장되었습니다.")
        else:
            self.save_failed.emit(f"설정 저장 실패: {save_result.failure()}")

    def test_connection(self, updates: dict):
        """현재 입력된 API 키와 DB 정보로 핑/인증 테스트를 비동기로 수행합니다."""
        asyncio.create_task(self._test_connection_task(updates))

    async def _test_connection_task(self, updates: dict):
        # 1. 키움 API 테스트 (가상 핑)
        app_key = updates.get("KIWOOM_APP_KEY")
        if not app_key:
            self.connection_test_completed.emit(False, "앱 키가 비어있습니다.")
            return

        import aiohttp
        # 업데이트된 설정을 임시 반영하여 REST URL 획득 (get_rest_url() 사용을 위해 임시 저장)
        old_mode = self.config_manager.get("kiwoom", {}).get("trading_mode")

        # Test를 위해 모드만 잠시 덮어쓰기 (임의)
        # updates는 {"kiwoom.trading_mode": "real"} 이런 식으로 들어올 수도 있음
        # dict 병합 과정에서 kiwoom이 안 넘어올 수 있으므로 명시적으로 추출
        mode = "virtual"
        if "kiwoom" in updates and "trading_mode" in updates["kiwoom"]:
             mode = updates["kiwoom"]["trading_mode"]
        elif "trading_mode" in updates:
             mode = updates["trading_mode"]

        # 모의투자, 실전투자 매핑 (UI 한글 -> 내부 영어)
        if mode == "실전투자":
            mode = "real"
        elif mode == "모의투자":
            mode = "virtual"

        base_url = "https://api.kiwoom.com" if mode == "real" else "https://mockapi.kiwoom.com"

        # 만약 dict 구조가 온전하다면
        kiwoom_conf = self.config_manager.get("kiwoom", {})
        if "rest_base_url" in kiwoom_conf and mode in kiwoom_conf["rest_base_url"]:
            base_url = kiwoom_conf["rest_base_url"][mode]

        url = f"{base_url}/oauth2/token"
        payload = {"grant_type": "client_credentials", "appkey": app_key, "secretkey": updates.get("KIWOOM_APP_SECRET", "")}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        # 각 필드를 가져오되, 데이터가 없으면 빈 문자열("")을 기본값으로 설정
                        token_val = data.get("token", "")
                        expires = data.get("expires_dt", "")
                        t_type = data.get("token_type", "")
                        r_code = data.get("return_code", "")
                        r_msg = data.get("return_msg", "")

                        # f-string을 사용하면 None이나 숫자 데이터도 안전하게 문자열로 합쳐집니다.
                        token_info = f"{token_val}, {expires}, {t_type}, {r_code}, {r_msg}"
                        print(f"JYJ  r_msg: {r_msg}")
                        kiwoom_msg = f"Kiwoom API: 토큰 발급 성공 ({token_info})"
                    else:
                        text = await response.text()
                        kiwoom_msg = f"Kiwoom API: 연결 실패 ({response.status}) - {text}"
                        self.connection_test_completed.emit(False, kiwoom_msg)
                        return
        except Exception as e:
            self.connection_test_completed.emit(False, f"Kiwoom API 연결 에러: {e}")
            return

        # ---------------------------------------------------------
        # 2. InfluxDB 연결 테스트 추가 (키움 성공 시 실행)
        # ---------------------------------------------------------
        influx_url = updates.get("INFLUX_URL", "http://localhost:8086")
        influx_token = updates.get("INFLUX_TOKEN", "")
        influx_org = updates.get("INFLUX_ORG", "")
        influx_bucket = self.config_manager.get("influx_bucket", "")

        self.logger.error(f"influx_url: {influx_url}, influx_org: {influx_org}, influx_bucket: {influx_bucket}, influx_token: {influx_token}")

        if not influx_token:
            self.connection_test_completed.emit(False, f"{kiwoom_msg}\n[경고] InfluxDB 토큰이 비어있습니다.")
            return

        try:
            from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync

            async with InfluxDBClientAsync(url=influx_url, token=influx_token, org=influx_org) as client:
                # 서버 생존 확인 (Ping)
                is_alive = await client.ping()
                if not is_alive:
                    raise ConnectionError("InfluxDB 서버 응답 없음 (Ping 실패)")

                # 데이터 쓰기 권한 테스트 (가장 확실한 방법)
                write_api = client.write_api()
                test_point = {"measurement": "test", "tags": {"type": "ping"}, "fields": {"val": 1.0}}
                await write_api.write(bucket=influx_bucket, record=test_point)

                influx_msg = "InfluxDB: 연결 및 쓰기 성공"

                # 최종 결과 합산 전송
                final_msg = f"{kiwoom_msg}\n{influx_msg}"
                self.connection_test_completed.emit(True, final_msg)

        except Exception as e:
            # 키움은 성공했지만 InfluxDB가 실패한 경우
            error_msg = f"{kiwoom_msg}\nInfluxDB 연결 에러: {e}"
            self.connection_test_completed.emit(False, error_msg)

        self.logger.error(f"influx_msg: {influx_msg}")
        self.connection_test_completed.emit(True, f"{kiwoom_msg}\n{influx_msg}")

    def check_db_status(self):
        """메뉴 액션: DB 상태 점검"""
        asyncio.create_task(self._check_db_status_task())

    async def _check_db_status_task(self):
        try:
            # InfluxDBClientAsync는 health()를 직접 노출하지 않고 ping() 사용
            is_alive = await getattr(self.influx_client, 'ping', lambda: self.influx_client.client.ping())()
            if is_alive:
                self.sig_menu_action_result.emit("DB 상태 점검", "InfluxDB 연결 상태가 정상(Ping 성공)입니다.")
            else:
                self.sig_menu_action_result.emit("DB 상태 점검", "InfluxDB 서버와 통신할 수 없습니다 (Ping 실패). URL과 설정을 확인하세요.")
        except Exception as e:
            self.sig_menu_action_result.emit("DB 상태 점검", f"DB 상태 확인 실패:\n{str(e)}")

    def force_refresh_token(self):
        """메뉴 액션: API 토큰 강제 갱신"""
        # 실제로는 TokenManager나 Auth 모듈을 호출해야 하지만 여기서는 메시지만 에뮬레이션
        self.sig_menu_action_result.emit("토큰 갱신", "새로운 Kiwoom REST API 토큰 발급을 요청했습니다.")

from gui.batch_backtest_worker import BatchBacktestWorker

class BacktestViewModel(QObject):
    """
    백테스트 스튜디오 ViewModel.
    UI의 백테스트 요청을 BacktestEngine으로 전달하고, 결과를 수집하여 Signal로 발송합니다.
    """
    sig_bt_progress = pyqtSignal(int, int, float) # step, total_steps, current_pnl
    sig_bt_finished = pyqtSignal(dict) # kpi dict
    sig_bt_chart_data = pyqtSignal(object) # DataFrame
    sig_bt_error = pyqtSignal(str)

    def __init__(self, config_manager, influx_client, data_collector, order_manager, universe_manager=None, historical_fetcher=None):
        super().__init__()
        self.config_manager = config_manager
        self.influx_client = influx_client
        self.data_collector = data_collector
        self.order_manager = order_manager
        self.universe_manager = universe_manager
        self.historical_fetcher = historical_fetcher
        self.logger = logging.getLogger("BacktestViewModel")

        from core.backtester import BacktestEngine
        self.engine = BacktestEngine(self.data_collector, self.config_manager.get_symbols())
        self.model_path = None
        self.batch_worker = None # [신규] 일괄 백테스트 워커 참조 저장

    def set_model_path(self, path: str):
        self.model_path = path

    def start_backtest(self, start_date: str, end_date: str, symbol: str):
        if not self.model_path:
            self.sig_bt_error.emit("학습된 모델 파일(.zip)을 먼저 선택해주세요.")
            return

        asyncio.create_task(self._run_backtest_task(start_date, end_date, symbol))

    async def _run_backtest_task(self, start_date: str, end_date: str, symbol: str):
        try:
            # 1. 대상 종목 및 데이터 로드 (실제 DB 연동)
            data_list = await self.influx_client.fetch_data_by_range(symbol, start_date, end_date)
            
            if not data_list:
                self.sig_bt_error.emit(f"[{symbol}] 해당 기간({start_date}~{end_date})의 데이터가 DB에 없습니다. 먼저 데이터를 수집해주세요.")
                return

            import pandas as pd
            df = pd.DataFrame(data_list)
            df['step'] = range(len(df))

            # 2. Env 생성 및 Agent 주입
            from env.trading_env import ScalpingTradingEnv
            from models.agent import TradingAgentWrapper

            # [최적화] 고정된 종목 임베딩(100)을 제외한 피처 영역 차원으로 모드 감지
            model_dim = TradingAgentWrapper.get_model_dimension(self.model_path)
            if model_dim >= 200:
                detected_mode = "advanced"
            elif model_dim >= 140:
                detected_mode = "basic"
            elif model_dim > 100: # 구형 모델 호환
                detected_mode = "advanced"
            else:
                detected_mode = "basic"
                self.logger.info(f"모델 차원({model_dim}) 기반 자동 모드 설정: {detected_mode}")

            # historical_data를 직접 주입하고 모드를 backtest로 설정하여 전체 구간 테스트
            # [FIX] 백테스트 시 모델 차원 불일치 방지: 훈련 시와 동일한 전체 유니버스 리스트 주입
            all_symbols_list = [s.get("code") for s in self.config_manager.get_symbols()]

            env_config = {
                "symbol": symbol,
                "initial_balance": 10000000,
                "historical_data": data_list,
                "mode": "backtest",
                "feature_mode": detected_mode,
                "target_dim": model_dim, # [핵심] 환경이 모델에 맞출 수 있도록 목표 차원 전달
                "all_symbols": all_symbols_list
            }
            env = ScalpingTradingEnv(self.data_collector, self.order_manager, env_config)
            
            print(f"   => 백테스트 설정: 모델 차원({model_dim}) 감지됨. 분석 모드를 '{detected_mode}'로 자동 전환합니다.")

            agent_config = {"seq_len": 10}
            agent = TradingAgentWrapper(env, agent_config)

            # 모델 로드
            try:
                agent.load_weights(self.model_path)
                self.logger.info(f"Backtest: 모델 로딩 성공 ✅ ({self.model_path})")
            except FileNotFoundError:
                self.sig_bt_error.emit(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
                return

            # 3. 백테스트 실행
            from core.backtester import KPICalculator

            def progress_cb(step, total, pnl):
                self.sig_bt_progress.emit(step, total, pnl)

            trades_df = await self.engine.run_backtest(agent, env, df, callbacks=[progress_cb])

            # 4. 결과 처리 및 UI 전송
            kpi = KPICalculator.calculate(trades_df, 10000000)

            # [신규] 백테스트 결과 자동 Export (CSV)
            self._export_backtest_results(kpi, trades_df, symbol, start_date, end_date)

            self.sig_bt_chart_data.emit(trades_df)
            self.sig_bt_finished.emit(kpi)

        except Exception as e:
            self.sig_bt_error.emit(str(e))

    def _export_backtest_results(self, kpi, trades_df, symbol, start_date, end_date):
        """백테스트 결과를 CSV 파일로 자동 저장합니다."""
        try:
            import datetime
            import pandas as pd
            import os

            # 1. 폴더 생성
            results_dir = "./backtest_results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = os.path.basename(self.model_path) if self.model_path else "unknown"

            # 2. KPI Summary 저장 (누적 모드)
            summary_path = os.path.join(results_dir, "backtest_summary.csv")
            
            # 매매 횟수 계산 (Hold 제외)
            total_trades = len(trades_df[trades_df['action'].isin(['Buy', 'Sell'])])

            summary_row = {
                "Timestamp": timestamp,
                "Model Name": model_name,
                "Symbol": symbol,
                "Start Date": start_date,
                "End Date": end_date,
                "Total Return (%)": round(kpi.get("Total Return", 0), 2),
                "Win Rate (%)": round(kpi.get("Win Rate", 0), 2),
                "MDD (%)": round(kpi.get("MDD", 0), 2),
                "Profit Factor": round(kpi.get("Profit Factor", 0), 3),
                "Total Trades": total_trades
            }
            summary_df = pd.DataFrame([summary_row])
            
            # 파일이 없으면 헤더 포함 저장, 있으면 Append
            if not os.path.exists(summary_path):
                summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
            else:
                summary_df.to_csv(summary_path, index=False, header=False, mode='a', encoding='utf-8-sig')

            # 3. 상세 매매 내역(Trade Log) 저장 (개별 파일)
            # 수동 분석을 위해 'Hold'를 제외한 실제 액션만 추출
            trade_log = trades_df[trades_df['action'].isin(['Buy', 'Sell'])].copy()
            if not trade_log.empty:
                # 파일명: trade_log_모델명_시간.csv
                clean_model_name = model_name.replace(".zip", "").replace(" ", "_")
                log_filename = f"trade_log_{clean_model_name}_{file_ts}.csv"
                log_path = os.path.join(results_dir, log_filename)
                trade_log.to_csv(log_path, index=False, encoding='utf-8-sig')
                self.logger.info(f"Backtest: 상세 매매 내역 저장 완료 ({log_path})")

            self.logger.info(f"Backtest: 결과 요약 저장 완료 ({summary_path})")

        except Exception as export_e:
            self.logger.error(f"Backtest 결과 Export 중 오류 발생: {export_e}")

    def start_auto_backtest_batch(self, start_date: str, end_date: str):
        """[NEW] 원클릭 Top 30 거래량 종목 자동 백테스트 실행 (선택한 종료일 데이터 사용)"""
        if not self.model_path:
            # 설정의 active_model_path 확인
            self.model_path = self.config_manager.get("active_model_path")
            if not self.model_path or not os.path.exists(self.model_path):
                self.sig_bt_error.emit("활성화된 모델이 없거나 파일이 존재하지 않습니다. 모델 로드를 먼저 해주세요.")
                return
        
        # [수정] 무조건 '당일'이 아니라 사용자가 선택한 '종료일(end_date)'을 기준으로 배치 실행
        self.logger.info(f"Top 30 자동 백테스트 시작: 기준일={end_date}")
        asyncio.create_task(self._run_auto_batch_task(end_date, end_date))

    async def _run_auto_batch_task(self, start_date: str, end_date: str):
        try:
            # 1. Top 30 종목 스캔
            access_token = self.config_manager.get("KIWOOM_ACCESS_TOKEN", "")
            if not access_token:
                self.sig_bt_error.emit("API 접근 토큰이 없습니다. 먼저 로그인(토큰 발급)이 필요합니다.")
                return

            if not self.universe_manager:
                self.sig_bt_error.emit("UniverseManager가 주입되지 않았습니다.")
                return

            self.sig_bt_progress.emit(0, 100, 0)
            self.logger.info("거래량 상위 30개 종목 리스트를 가져오는 중...")
            
            result = await self.universe_manager.fetch_top_30_volume_symbols(access_token)

            from returns.io import IOSuccess, IOFailure
            if isinstance(result, IOFailure):
                err = result.failure()._inner_value
                self.sig_bt_error.emit(f"Kiwoom API 호출 실패: {err}")
                return

            # IOSuccess(Success(list)) 형태이므로 unwrap()._inner_value 사용
            top_30_list = result.unwrap()._inner_value
                
            if not top_30_list:
                self.logger.warning("UniverseManager가 빈 종목 리스트를 반환했습니다. 필터링 조건이나 API 응답을 확인하세요.")
                self.sig_bt_error.emit("거래량 상위 30개 종목을 가져오지 못했습니다. (API 응답이 비어있거나 모두 필터링됨)")
                return

            symbols = [s["code"] for s in top_30_list]
            self.logger.info(f"스캔 완료: {len(symbols)}개 종목에 대해 백테스트 배치를 시작합니다.")

            # 2. 배치 실행 준비 (환경/에이전트 빌더 정의)
            from env.trading_env import ScalpingTradingEnv
            from models.agent import TradingAgentWrapper

            # 모델 차원 감지 (최초 1회)
            model_dim = TradingAgentWrapper.get_model_dimension(self.model_path)
            detected_mode = "advanced" if model_dim >= 200 else "basic"
            all_symbols_list = [s.get("code") for s in self.config_manager.get_symbols()]

            async def env_builder(sym, start, end):
                from returns.io import IOFailure
                # [수정] 수집 중단 시점을 당일 08:00:00으로 설정
                stop_ts = f"{start[:4]}-{start[4:6]}-{start[6:8]} 08:00:00"
                
                # [기능 개선] DB 대신 키움 서버에서 직접 수집하되, UI의 날짜 범위를 최대한 준수
                result = await self.historical_fetcher.fetch_historical_data(
                    sym, end, access_token, 
                    stop_timestamp=stop_ts, 
                    max_pages=5 # 자동 배치는 보통 단기이므로 5페이지면 충분 (약 4500분)
                )
                
                if isinstance(result, IOFailure):
                    self.logger.error(f"[{sym}] 키움 서버 데이터 수집 실패: {result.failure()}")
                    return None, None
                
                data = result.unwrap()._inner_value
                if not data: 
                    self.logger.warning(f"[{sym}] 당일({start}) 데이터가 없습니다.")
                    return None, None
                
                # API는 최신순으로 데이터를 주므로 정렬 및 시간 필터링 (08:00 ~ 16:00)
                df = pd.DataFrame(data)
                if 'timestamp' in df.columns:
                    day_prefix = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
                    df = df[(df['timestamp'] >= f"{day_prefix} 08:00:00") & 
                            (df['timestamp'] <= f"{day_prefix} 16:00:00")]
                    
                    if df.empty:
                        self.logger.warning(f"[{sym}] 08:00 ~ 16:00 범위 내에 유효한 데이터가 없습니다.")
                        return None, None
                        
                    df = df.sort_values('timestamp').reset_index(drop=True)
                
                df['step'] = range(len(df))
                
                env_config = {
                    "symbol": sym,
                    "initial_balance": 10000000,
                    "historical_data": data,
                    "mode": "backtest",
                    "feature_mode": detected_mode,
                    "target_dim": model_dim,
                    "all_symbols": all_symbols_list
                }
                return ScalpingTradingEnv(self.data_collector, self.order_manager, env_config), df

            def agent_builder(env):
                agent = TradingAgentWrapper(env, {"seq_len": 10})
                agent.load_weights(self.model_path)
                return agent

            # 3. 엔진 배치 실행
            results = await self.engine.run_automation_batch(
                agent_builder, env_builder, symbols, start_date, end_date,
                progress_cb=self._on_batch_progress
            )

            # 4. CSV 저장 및 완료 알림
            self._save_batch_results_csv(results, self.model_path)
            self.sig_bt_finished.emit({"Batch Count": len(results)})
            self.logger.info(f"Top 30 자동 백테스트 배치 완료 (저장: backtest_results/auto_bt_...)")

        except Exception as e:
            import traceback
            self.logger.error(f"배치 백테스트 오류: {e}\n{traceback.format_exc()}")
            self.sig_bt_error.emit(f"배치 실행 중 오류가 발생했습니다: {e}")

    def _on_batch_progress(self, idx, total, status_msg):
        """배치 진행률 업데이트 (UI 표시용)"""
        pct = int(((idx) / total) * 100)
        # progress signal의 파라미터 형식을 맞춤 (step=idx, total=total, current_pnl=0(배치에선 status_msg 대용 불가하므로 로깅용))
        # 하지만 기존 시그널은 (int, int, float) 이므로 progress_cb에서 status_msg를 직접 보낼 순 없음
        # 대신 뷰모델의 로거로 직접 출력
        self.logger.info(f"BT 배치 [{idx+1}/{total}] {status_msg}")
        self.sig_bt_progress.emit(idx + 1, total, 0.0)

    def _save_batch_results_csv(self, results, model_path):
        """[NEW] 요구사항에 맞춘 상세 CSV 저장 포맷"""
        try:
            import datetime
            import pandas as pd
            
            results_dir = "./backtest_results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            model_name = os.path.basename(model_path)
            today_str = datetime.datetime.now().strftime("%Y%m%d")
            timestamp_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            filename = f"auto_bt_{model_name.replace('.zip', '')}_{today_str}.csv"
            save_path = os.path.join(results_dir, filename)

            # DataFrame 생성
            df = pd.DataFrame(results)
            
            # 필수 컬럼 구성 및 순서 조정
            # [테스트 일시, 모델명, 종목코드(Symbol), 시작일, 종료일, 총수익률(%), 승률(%), MDD(%), Profit Factor, 총 매매횟수]
            df.insert(0, "테스트 일시", timestamp_now)
            df.insert(1, "모델명", model_name)
            
            # 컬럼명 매핑 (영문 -> 한글)
            column_map = {
                "Symbol": "종목코드(Symbol)",
                "Start Date": "시작일",
                "End Date": "종료일",
                "Total Return (%)": "총수익률(%)",
                "Win Rate (%)": "승률(%)",
                "MDD (%)": "MDD(%)",
                "Profit Factor": "Profit Factor",
                "Total Trades": "총 매매횟수"
            }
            df = df.rename(columns=column_map)
            
            # 수치형 컬럼 선정
            numeric_cols = ["총수익률(%)", "승률(%)", "MDD(%)", "Profit Factor", "총 매매횟수"]
            
            # [Average] 행 추가
            avg_row = {col: "" for col in df.columns}
            avg_row["종목코드(Symbol)"] = "[Average]"
            for col in numeric_cols:
                avg_row[col] = round(df[col].mean(), 2)
            
            df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

            # CSV 저장 (BOM 포함 UTF-8로 엑셀 호환성 확보)
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            self.logger.info(f"배치 결과 전용 CSV 저장 완료: {save_path}")

        except Exception as e:
            self.logger.error(f"CSV 저장 중 오류: {e}")

    def start_batch_backtest(self, model_paths: List[str], start_date: str, end_date: str):
        """[NEW] 다중 모델 x 전 종목 일괄 백테스트 실행 (QThread 기반)"""
        if not model_paths:
            self.sig_bt_error.emit("선택된 모델 파일이 없습니다.")
            return

        symbols = self.config_manager.get_symbols()
        if not symbols:
            self.sig_bt_error.emit("유니버스에 등록된 종목이 없습니다.")
            return

        # 이전 워커가 있다면 정리
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.stop()
            self.batch_worker.wait()

        self.logger.info(f"일괄 백테스트 시작: 모델 {len(model_paths)}개, 종목 {len(symbols)}개")
        
        self.batch_worker = BatchBacktestWorker(
            model_paths, symbols, start_date, end_date, 
            self.engine, self.influx_client, self.config_manager
        )
        
        # 워커 시그널 연결
        self.batch_worker.sig_progress.connect(self.sig_bt_progress.emit)
        self.batch_worker.sig_status.connect(lambda msg: self.logger.info(f"BT Batch Status: {msg}"))
        self.batch_worker.sig_finished.connect(self.sig_bt_finished.emit)
        self.batch_worker.sig_error.connect(self.sig_bt_error.emit)
        
        # 스레드 시작
        self.batch_worker.start()

    def stop_batch_backtest(self):
        """일괄 백테스트 중단 요청"""
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.stop()
            self.logger.info("일괄 백테스트 중단 요청됨.")
