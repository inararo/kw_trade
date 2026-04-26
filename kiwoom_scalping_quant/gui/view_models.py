import asyncio
import logging
import os
import glob
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, Qt
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

    # 스레드 브릿지: 백그라운드 -> 메인 스레드 (내부용)
    _sig_raw_data = pyqtSignal(object)

    # Risk Limits and Alerts
    sig_risk_metrics_updated = pyqtSignal(float, float) # current PnL, available invest limit
    sig_status_alert = pyqtSignal(str)
    
    # [제어 상태 시그널]
    sig_trading_paused = pyqtSignal(bool)    # True: 일시정지, False: 재개
    sig_monitoring_stopped = pyqtSignal(bool) # True: 중지, False: 감시중

    def __init__(self, data_collector, order_manager, config_manager):
        super().__init__()
        self.data_collector = data_collector
        self.order_manager = order_manager
        self.config_manager = config_manager
        self._is_running = False
        self._mock_task = None

        # 현재 화면에 상세를 띄울 대상 종목
        self.selected_symbol = None
        self.symbols_summary = {}

        # 종목명 캐시 (Code -> Name): 접미사(_AL) 제거 후 순수 코드와 매핑
        self._symbol_names = {
            s.get("code", "").split('_')[0]: s.get("name") 
            for s in self.config_manager.get_symbols() if s.get("code")
        }

        # UI logging hook for Signal Only mode bypass messages
        if hasattr(self.order_manager, 'signals'):
            self.order_manager.signals.signal_only_log.connect(self.append_log)

        # DataCollector 측에서 데이터가 들어올 때 콜백받을 수 있도록 설정
        self.data_collector.set_ui_callback(self._on_data_received)

    def append_log(self, msg: str):
        self.sig_log_appended.emit(msg)

    def set_selected_symbol(self, symbol: str):
        self.selected_symbol = symbol

    @pyqtSlot(object)
    def _on_data_received(self, data: dict):
        """DataCollector에서 호출되는 UI 업데이트 콜백 (qasync: 동일 스레드)"""
        try:
            raw_symbol = data.get("symbol", "")
            if not raw_symbol:
                return
            symbol = raw_symbol.split('_')[0].strip()

            if symbol not in self.symbols_summary:
                name = self._symbol_names.get(symbol)
                if not name:
                    # 캐시에 없으면 config에서 새로 조회 (중간에 추가된 종목 대응)
                    for s in self.config_manager.get_symbols():
                        cfg_code = s.get("code", "").split('_')[0]
                        if cfg_code == symbol:
                            name = s.get("name")
                            self._symbol_names[symbol] = name
                            break
                
                name = name or "-"
                self.symbols_summary[symbol] = {"name": name, "price": 0, "ai_signal": "-", "holdings": 0}

            if "price" in data:
                self.symbols_summary[symbol]["price"] = data["price"]

            self.symbols_summary[symbol]["holdings"] = self.order_manager.holdings.get(symbol, 0)

            self.sig_symbols_summary_updated.emit(self.symbols_summary)

            if symbol == self.selected_symbol or not self.selected_symbol:
                if "price" in data:
                    self.sig_price_updated.emit(float(data["price"]))
                if "orderbook" in data:
                    self.sig_orderbook_updated.emit(dict(data["orderbook"]))

        except Exception as e:
            import traceback
            self.sig_log_appended.emit(f"[UI 오류] {e} | {traceback.format_exc()[-300:]}")


    async def start_polling(self):
        """실전 매매/백테스트 모드에서의 일반 폴링 (mock 사용 시 제외)"""
        self._is_running = True
        while self._is_running:
            # Emit Risk Manager details periodically
            if hasattr(self.order_manager, 'risk_manager') and self.order_manager.risk_manager:
                rm = self.order_manager.risk_manager
                pnl = rm.daily_realized_pnl
                max_invest = rm.get_max_invest_per_symbol()

                # Calculate basic rough available limit (e.g. max_invest - current holding of selected symbol)
                # Using 0 if none selected for simple UI purpose
                curr_invested = 0
                if self.selected_symbol:
                    curr_invested = self.order_manager.holdings.get(self.selected_symbol, 0) * self.order_manager.avg_entry_prices.get(self.selected_symbol, 0)

                avail_limit = max(0, max_invest - curr_invested)
                self.sig_risk_metrics_updated.emit(pnl, avail_limit)

            await asyncio.sleep(1.0)

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
            sm.set_ai_paused(paused)
            self.sig_trading_paused.emit(paused)
            status = "일시정지" if paused else "재개"
            msg = f"[시스템] 🤖 AI 매매 의사결정이 {status}되었습니다."
            self.sig_log_appended.emit(msg)

    def toggle_monitoring(self, stopped: bool):
        """실시간 데이터 수집(웹소켓) 자체를 중단하거나 재개"""
        if stopped:
            self._is_monitoring_stopped = True
            asyncio.create_task(self.data_collector.stop())
            self.sig_monitoring_stopped.emit(True)
            self.sig_log_appended.emit("[시스템] 📡 실시간 종목 감시가 중단되었습니다. (웹소켓 연결 해제)")
        else:
            self._is_monitoring_stopped = False
            # 재시작 전 안전하게 플래그 리셋
            self.data_collector.is_running = True 
            asyncio.create_task(self.data_collector.start())
            self.sig_monitoring_stopped.emit(False)
            self.sig_log_appended.emit("[시스템] 📡 실시간 종목 감시를 재개합니다. (재연결 시도 중...)")

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

    def __init__(self, config_manager, historical_fetcher, influx_client, universe_manager, token_manager):
        super().__init__()
        self.config_manager = config_manager
        self.historical_fetcher = historical_fetcher
        self.influx_client = influx_client
        self.universe_manager = universe_manager
        self.token_manager = token_manager
        self.logger = logging.getLogger("AssetDataViewModel")

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
        
        for raw_code in symbols:
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

    async def _build_universe_task(self, top_n: int = 20, is_auto=False):
        self.sig_progress_updated.emit(0)
        if not is_auto:
            self.sig_status_updated.emit(f"시장 전체 종목 조회 및 주도주 필터링 중 (Top {top_n})...")

        access_token = self.config_manager.get("KIWOOM_ACCESS_TOKEN", "")
        if not access_token:
            self.fetch_failed.emit("API 접근 토큰이 없습니다. 설정에서 발급해 주세요.")
            return False

        # @future_safe에 의해 감싸진 async 함수는 await하면 반환값이 Result 타입 객체입니다.
        result = await self.universe_manager.build_top_n_universe(access_token, top_n=top_n)

        # @future_safe returns IOFailure on exception and IOSuccess on success
        if isinstance(result, IOFailure):
            # IOFailure.failure() returns the unwrapped exception inside an IO, so we use _inner_value or str()
            err_msg = str(result.failure()._inner_value if hasattr(result.failure(), '_inner_value') else result.failure())
            self.fetch_failed.emit(f"유니버스 생성 실패: {err_msg}")
            return False

        # unwrap() on IOSuccess returns an IO object. We extract the raw list with _inner_value
        try:
            top_stocks = result.unwrap()._inner_value
        except Exception as e:
            top_stocks = []

        if not isinstance(top_stocks, list):
            top_stocks = []

        # Bulk save newly selected top_stocks to config
        new_symbols = []
        for stock in top_stocks:
            if isinstance(stock, dict) and "code" in stock and "name" in stock:
                new_symbols.append({
                    "code": stock["code"], 
                    "name": stock["name"],
                    "price": stock.get("price", 0.0),
                    "flu_rt": stock.get("flu_rt", 0.0),
                    "volume": stock.get("volume", 0.0)
                })

        # [버그 수정] 장외 시간이거나 API 응답이 없어 리스트가 비어있을 경우 덮어쓰지 않음
        if not new_symbols:
            existing_symbols = self.config_manager.get_symbols()
            if existing_symbols:
                msg = "현재 장외 시간이거나 API 수신 데이터가 없습니다. 기존 유니버스 리스트를 유지합니다."
                self.sig_status_updated.emit(msg)
                if not is_auto:
                    self.fetch_completed.emit(msg)
                self.logger.info(msg)
                self.load_symbols() # 부팅 시퀀스 Event Set을 위해 호출 필수
                return True

        self.config_manager.set_symbols(new_symbols)

        self.sig_progress_updated.emit(100)

        msg = f"상위 {len(top_stocks)}개 유니버스 생성 완료!"
        if is_auto:
            msg = "[POST-MARKET COLLECTION] " + msg

        self.sig_status_updated.emit(msg)
        if not is_auto:
            self.fetch_completed.emit(msg)

        self.load_symbols() # 갱신
        return True

    def load_symbols(self):
        # ConfigManager의 Result 처리
        result = self.config_manager.load_config()
        if isinstance(result, Success):
            symbols = self.config_manager.get_symbols()
            self.symbols_loaded.emit(symbols)
        else:
            self.symbol_update_failed.emit(f"설정 로드 실패: {result.failure()}")

    def add_symbol(self, code: str, name: str):
        result = self.config_manager.add_symbol(code, name)
        if isinstance(result, Success):
            self.symbol_update_success.emit(f"종목 추가 완료: {name}")
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
                # 데이터가 성공적으로 수집된 경우에만 해당 종목의 기존 데이터 삭제 (정밀 수집 반영)
                await self.influx_client.delete_data("historical_data", symbol)
                await self.influx_client.delete_data("tick_data", symbol)
                
                await self.influx_client.bulk_insert(data_list)
                self.logger.info(f"[{symbol}] InfluxDB 갱신 성공: {fetch_count}건 적재 완료.")
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
        self.logger.error(f"전체 수집 프로세스 종료: {msg}")

        if not is_auto:
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

    def start_training(self, total_timesteps: int, learning_rate: float, max_records: int, feature_mode: str = "basic", use_smart_sampling: bool = False):
        """UI에서 학습 시작 요청을 받아 파이프라인 조립 후 워커 실행"""
        if self.worker and self.worker.isRunning():
            self.sig_error.emit("이미 학습이 진행 중입니다.")
            return

        # 이전 태스크가 남아있다면 정리
        if self.prep_task and not self.prep_task.done():
            self.prep_task.cancel()

        self.prep_task = asyncio.create_task(self._prepare_and_start_training(total_timesteps, learning_rate, max_records, feature_mode, use_smart_sampling))

    async def _prepare_and_start_training(self, timesteps: int, lr: float, max_records: int, feature_mode: str, use_smart_sampling: bool = False):
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
            "ent_coef": ent_coef,
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
                    f"   => [{sampling_tag}] 최신 모델 계승: {os.path.basename(latest_model_zip)}"
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

    def __init__(self, config_manager, influx_client):
        super().__init__()
        self.config_manager = config_manager
        self.influx_client = influx_client
        self.logger = logging.getLogger("SettingsViewModel")

    def load_settings(self):
        """ConfigManager를 통해 통합 설정을 로드하고 UI로 Emit합니다."""
        result = self.config_manager.load_config()
        if isinstance(result, Success):
            self.settings_loaded.emit(result.unwrap())
        else:
            self.save_failed.emit(f"설정 로드 실패: {result.failure()}")

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
            self.save_completed.emit("설정이 성공적으로 저장되었습니다. (일부 설정은 재시작 시 적용됩니다.)")
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

class BacktestViewModel(QObject):
    """
    백테스트 스튜디오 ViewModel.
    UI의 백테스트 요청을 BacktestEngine으로 전달하고, 결과를 수집하여 Signal로 발송합니다.
    """
    sig_bt_progress = pyqtSignal(int, int, float) # step, total_steps, current_pnl
    sig_bt_finished = pyqtSignal(dict) # kpi dict
    sig_bt_chart_data = pyqtSignal(object) # DataFrame
    sig_bt_error = pyqtSignal(str)

    def __init__(self, config_manager, influx_client, data_collector, order_manager):
        super().__init__()
        self.config_manager = config_manager
        self.influx_client = influx_client
        self.data_collector = data_collector
        self.order_manager = order_manager
        self.logger = logging.getLogger("BacktestViewModel")

        from core.backtester import BacktestEngine
        self.engine = BacktestEngine(self.data_collector, self.config_manager.get_symbols())
        self.model_path = None

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
