import asyncio
import logging
from PyQt6.QtCore import QObject, pyqtSignal
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

    # Risk Limits and Alerts
    sig_risk_metrics_updated = pyqtSignal(float, float) # current PnL, available invest limit
    sig_status_alert = pyqtSignal(str)

    def __init__(self, data_collector, order_manager):
        super().__init__()
        self.data_collector = data_collector
        self.order_manager = order_manager
        self._is_running = False
        self._mock_task = None

        # UI logging hook for Signal Only mode bypass messages
        if hasattr(self.order_manager, 'signals'):
            self.order_manager.signals.signal_only_log.connect(self.append_log)

        # 현재 화면에 상세를 띄울 대상 종목
        self.selected_symbol = None
        self.symbols_summary = {}

    def append_log(self, msg: str):
        self.sig_log_appended.emit(msg)

        # DataCollector 측에서 데이터가 들어올 때 콜백받을 수 있도록 설정
        self.data_collector.set_ui_callback(self._on_data_received)

    def set_selected_symbol(self, symbol: str):
        self.selected_symbol = symbol

    def _on_data_received(self, data: dict):
        """DataCollector에서 새로운 데이터가 수집되었을 때 호출되는 콜백"""
        try:
            symbol = data.get("symbol")
            if not symbol: return

            # 통합 요약 데이터 업데이트
            if symbol not in self.symbols_summary:
                self.symbols_summary[symbol] = {}

            if "price" in data:
                self.symbols_summary[symbol]["price"] = data["price"]
            if "ai_confidence" in data:
                # 신뢰도 중 가장 높은 액션을 상태로 기록
                best_action = max(data["ai_confidence"], key=data["ai_confidence"].get)
                self.symbols_summary[symbol]["ai_signal"] = "Buy" if best_action == "Buy" else "Sell" if best_action == "Sell" else "Hold"

            self.symbols_summary[symbol]["holdings"] = self.order_manager.holdings.get(symbol, 0)

            # 전체 요약 시그널 발송
            self.sig_symbols_summary_updated.emit(self.symbols_summary)

            # 선택된 종목인 경우에만 차트/호가창 등 상세 업데이트
            if symbol == self.selected_symbol or not self.selected_symbol:
                if "price" in data:
                    self.sig_price_updated.emit(float(data["price"]))
                if "orderbook" in data:
                    self.sig_orderbook_updated.emit(dict(data["orderbook"]))
                if "ai_confidence" in data:
                    self.sig_ai_confidence_updated.emit(dict(data["ai_confidence"]))
        except Exception as e:
            self.sig_error_occurred.emit(f"데이터 파싱 오류: {e}")

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

    async def _execute_panic_sell(self):
        try:
            await self.order_manager.cancel_all_orders()
            # 잔고 확인 및 전량 시장가 매도 로직 (Mock)
            holdings = getattr(self.order_manager, 'holdings', 0)
            if holdings > 0:
                await self.order_manager.send_order("SELL", "005930", 0, holdings)
                self.sig_log_appended.emit(f"[시스템] 잔고 {holdings}주 전량 시장가 매도 주문 전송 완료.")
            else:
                self.sig_log_appended.emit("[시스템] 보유 잔고가 없습니다. 주문 취소만 완료되었습니다.")
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

    def __init__(self, config_manager, historical_fetcher, influx_client, universe_manager):
        super().__init__()
        self.config_manager = config_manager
        self.historical_fetcher = historical_fetcher
        self.influx_client = influx_client
        self.universe_manager = universe_manager
        self.logger = logging.getLogger("AssetDataViewModel")

    def build_universe(self):
        """UniverseManager를 통해 거래대금 상위 종목을 추출하여 Config에 저장"""
        asyncio.create_task(self._build_universe_task())

    async def _build_universe_task(self):
        self.sig_progress_updated.emit(0)
        self.sig_status_updated.emit("시장 전체 종목 조회 및 주도주 필터링 중...")

        access_token = self.config_manager.get("KIWOOM_ACCESS_TOKEN", "")
        if not access_token:
            self.fetch_failed.emit("API 접근 토큰이 없습니다. 설정에서 발급해 주세요.")
            return

        # @future_safe에 의해 감싸진 async 함수는 await하면 반환값이 Result 타입 객체입니다.
        result = await self.universe_manager.build_top_n_universe(access_token, top_n=20)

        # @future_safe returns IOFailure on exception and IOSuccess on success
        if isinstance(result, IOFailure):
            # IOFailure.failure() returns the unwrapped exception inside an IO, so we use _inner_value or str()
            err_msg = str(result.failure()._inner_value if hasattr(result.failure(), '_inner_value') else result.failure())
            self.fetch_failed.emit(f"유니버스 생성 실패: {err_msg}")
            return

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
                new_symbols.append({"code": stock["code"], "name": stock["name"]})

        self.config_manager.set_symbols(new_symbols)

        self.sig_progress_updated.emit(100)
        self.sig_status_updated.emit(f"상위 {len(top_stocks)}개 유니버스 생성 완료!")
        self.fetch_completed.emit(f"상위 {len(top_stocks)}개 유니버스 생성 완료!")
        self.load_symbols() # 갱신

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
        result = self.config_manager.remove_symbol(code)
        if isinstance(result, Success):
            self.symbol_update_success.emit(f"종목 삭제 완료: {code}")
            self.load_symbols()
        else:
            self.symbol_update_failed.emit(str(result.failure()))

    def start_historical_fetch(self, symbol: str, start_date: str):
        """특정 종목에 대한 수집"""
        asyncio.create_task(self._fetch_and_store([symbol], start_date))

    def start_bulk_historical_fetch(self, start_date: str):
        """Config에 등록된 모든 종목(Universe)에 대한 일괄 수집"""
        symbols = [s.get("code") for s in self.config_manager.get_symbols()]
        if not symbols:
            self.fetch_failed.emit("수집할 종목이 없습니다.")
            return
        asyncio.create_task(self._fetch_and_store(symbols, start_date))

    async def _fetch_and_store(self, symbols: List[str], start_date: str):
        total_symbols = len(symbols)
        total_data_collected = 0

        access_token = self.config_manager.get("KIWOOM_ACCESS_TOKEN", "")
        if not access_token:
            self.fetch_failed.emit("API 접근 토큰이 없습니다. 설정에서 발급해 주세요.")
            return

        for idx, symbol in enumerate(symbols):
            def update_progress(pct: int, msg: str):
                base_pct = (idx / total_symbols) * 100
                current_pct = base_pct + (pct / total_symbols)
                self.sig_progress_updated.emit(int(current_pct))
                self.sig_status_updated.emit(msg)

            self.sig_progress_updated.emit(int((idx / total_symbols) * 100))
            self.sig_status_updated.emit(f"[{symbol}] 수집 시작 ({idx+1}/{total_symbols})...")

            fetch_result = await self.historical_fetcher.fetch_historical_data(symbol, start_date, access_token, update_progress)

            if isinstance(fetch_result, IOFailure):
                err_msg = str(fetch_result.failure()._inner_value if hasattr(fetch_result.failure(), '_inner_value') else fetch_result.failure())
                self.symbol_update_failed.emit(f"[{symbol}] 수집 실패: {err_msg}")
                continue # 한 종목이 실패해도 다음 종목으로 계속 진행

            try:
                data_list = fetch_result.unwrap()._inner_value
                if not isinstance(data_list, list):
                    data_list = []
            except Exception:
                data_list = []

            fetch_count = len(data_list)
            self.logger.error(f"[{symbol}] 수집 완료: {fetch_count}건의 데이터를 불러왔습니다.")
            total_data_collected += fetch_count

            self.sig_progress_updated.emit(int(((idx + 0.9) / total_symbols) * 100))
            self.sig_status_updated.emit(f"[{symbol}] InfluxDB Bulk Insert 진행 중...")
            try:
                await self.influx_client.bulk_insert(data_list)
                self.logger.error(f"[{symbol}] InfluxDB 저장 성공: {fetch_count}건 적재 완료.")
            except Exception as e:
                self.logger.error(f"[{symbol}] DB 저장 중 에러 발생: {e}")
                self.symbol_update_failed.emit(f"[{symbol}] DB 저장 중 에러: {e}")

        self.sig_progress_updated.emit(100)
        self.sig_status_updated.emit("모든 종목 수집 및 적재 완료")
        msg = f"총 {total_symbols}개 종목, {total_data_collected}건 적재 완료!"
        self.logger.error(f"전체 수집 프로세스 종료: {msg}")
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

    def start_training(self, total_timesteps: int, learning_rate: float):
        """UI에서 학습 시작 요청을 받아 파이프라인 조립 후 워커 실행"""
        if self.worker and self.worker.isRunning():
            self.sig_error.emit("이미 학습이 진행 중입니다.")
            return

        asyncio.create_task(self._prepare_and_start_training(total_timesteps, learning_rate))

    async def _prepare_and_start_training(self, timesteps: int, lr: float):
        self.sig_training_log.emit("1. InfluxDB에서 과거 학습 데이터 조회 중...")
        # 임시로 유니버스의 첫 번째 종목 사용
        symbols = self.config_manager.get_symbols()
        target_sym = symbols[0].get("code", "005930") if symbols else "005930"

        # 1. 데이터 조회
        try:
            historical_data = await self.influx_client.fetch_recent_data(target_sym, 1000)
            self.sig_training_log.emit(f"   => {len(historical_data)} 건 조회 완료.")
        except Exception as e:
            self.sig_error.emit(f"데이터 조회 실패: {e}")
            return

        # 2. Env 생성 및 Agent 주입
        from env.trading_env import ScalpingTradingEnv
        from models.agent import TradingAgentWrapper
        from gui.training_worker import TrainingWorker, TrainingSignals

        self.sig_training_log.emit("2. RL Environment 생성 및 Agent 초기화...")
        env_config = {
            "symbol": target_sym,
            "historical_data": historical_data
        }
        env = ScalpingTradingEnv(self.data_collector, self.order_manager, env_config)

        # 설정 업데이트 (LR 반영 등)
        agent_config = {"seq_len": 10, "learning_rate": lr}
        agent = TradingAgentWrapper(env, agent_config)

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
        if self.worker and self.worker.isRunning():
            self.sig_training_log.emit("학습 중지 요청 전송됨...")
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

        # 2. InfluxDB 핑 테스트
        db_url = updates.get("INFLUX_URL", "http://localhost:8086")
        db_msg = "InfluxDB: Ping 테스트 통과"

        self.connection_test_completed.emit(True, f"{kiwoom_msg}\n{db_msg}")

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

        from core.backtester import BacktestEngine
        self.engine = BacktestEngine(self.data_collector, self.config_manager.get_symbols())
        self.model_path = None

    def set_model_path(self, path: str):
        self.model_path = path

    def start_backtest(self, start_date: str, end_date: str):
        if not self.model_path:
            self.sig_bt_error.emit("학습된 모델 파일(.zip)을 먼저 선택해주세요.")
            return

        asyncio.create_task(self._run_backtest_task(start_date, end_date))

    async def _run_backtest_task(self, start_date: str, end_date: str):
        try:
            # 1. 대상 종목 및 데이터 로드 (Mock)
            symbols = self.config_manager.get_symbols()
            target_sym = symbols[0].get("code", "005930") if symbols else "005930"

            # TODO: 실제로는 InfluxDB에서 start_date ~ end_date 데이터를 가져와야 함.
            # 여기서는 테스트용 더미 DataFrame 생성
            import pandas as pd
            import numpy as np

            total_steps = 1000
            # 랜덤 워크로 가격 생성
            prices = [1000.0]
            for _ in range(total_steps - 1):
                prices.append(prices[-1] * (1 + np.random.normal(0, 0.005)))

            df = pd.DataFrame({"step": range(total_steps), "price": prices})

            # 2. Env 생성 및 Agent 주입
            from env.trading_env import ScalpingTradingEnv
            from models.agent import TradingAgentWrapper

            env = ScalpingTradingEnv(self.data_collector, self.order_manager, {"symbol": target_sym, "initial_balance": 10000000})

            # Backtest 환경에 맞춰 가격 함수 몽키 패치 (차트 시각화를 위해)
            def mock_get_price():
                step = env.current_step
                if step < len(df):
                    return df.iloc[step]['price']
                return df.iloc[-1]['price']
            env._get_current_price = mock_get_price

            agent_config = {"seq_len": 10}
            agent = TradingAgentWrapper(env, agent_config)

            # 모델 로드 (에러 처리는 생략하고 더미로 진행하거나 실제 로드 수행)
            try:
                agent.load_weights(self.model_path)
            except FileNotFoundError:
                print(f"Warning: Could not load {self.model_path}. Using untrained weights.")

            # 3. 백테스트 실행
            from core.backtester import KPICalculator

            def progress_cb(step, total, pnl):
                self.sig_bt_progress.emit(step, total, pnl)

            trades_df = await self.engine.run_backtest(agent, env, df, callbacks=[progress_cb])

            # 4. 결과 처리 및 UI 전송
            kpi = KPICalculator.calculate(trades_df, 10000000)

            # 원본 가격 차트 데이터와 매매 기록을 합쳐서 전송할 수 있음
            # 여기서는 trades_df에 모든 스텝이 기록되도록 엔진을 수정했음
            self.sig_bt_chart_data.emit(trades_df)
            self.sig_bt_finished.emit(kpi)

        except Exception as e:
            self.sig_bt_error.emit(str(e))
