import asyncio
import glob
import logging
import os
import time
from typing import List, Dict, Any
import numpy as np

from env.trading_env import ScalpingTradingEnv
from models.agent import TradingAgentWrapper

class StrategyManager:
    """
    여러 종목(Multi-Symbol)의 트레이딩을 동시에 오케스트레이션하는 관리자 클래스.
    메모리 최적화를 위해 단일(Shared) Agent 모델을 유지하고,
    종목별로 별도의 TradingEnv 인스턴스를 생성하여 State를 추출/추론합니다.
    이벤트 드리븐 구조로 설계되어 새로운 틱 데이터 수신 시에만 추론을 수행합니다.
    """
    def __init__(self, config_manager, data_collector, order_manager, risk_manager):
        self.config_manager = config_manager
        self.data_collector = data_collector
        self.order_manager = order_manager
        self.risk_manager = risk_manager
        self.logger = logging.getLogger("StrategyManager")

        self.symbols: List[str] = [s.get('code') for s in self.config_manager.get_symbols()]
        if not self.symbols:
            self.symbols = ['005930'] # Fallback

        self.envs: Dict[str, ScalpingTradingEnv] = {}
        self.shared_agent: TradingAgentWrapper = None

        # [Multi-Model] Regime-Switching 대비 두 모델을 독립적으로 유지 — dual 모드사용
        self.model_random: TradingAgentWrapper = None  # random 샘플링 모델
        self.model_smart: TradingAgentWrapper  = None  # smart  샘플링 모델
        # active_model: shared_agent의 별칭으로, Regime-Switching 시 여기를 스와프하면 됨
        # (dual 모드에서는 shared_agent = model_random으로 초기화)

        self.is_running = False

        # 쿨다운 및 락 관리 (중복 주문 방지)
        self.last_action_times: Dict[str, float] = {}
        self.cooldown_seconds = 3.0

        # 동적 유니버스 스왑을 위한 락
        self._swap_lock = asyncio.Lock()
        
        # [제어] AI 매매 판단 일시정지 플래그 (Safety Switch)
        self.is_ai_paused = False

    def set_ai_paused(self, paused: bool):
        """AI의 매매 판단(추론)만 일시적으로 정지하거나 재개합니다."""
        self.is_ai_paused = paused
        self.logger.info(f"StrategyManager: AI Trading is {'PAUSED' if paused else 'RESUMED'}")

    # ------------------------------------------------------------------
    # [Private] 단일 모델 로드 헬퍼 — 특정 폴더에서 최신 .zip을 자동 탐색하여 로드
    # ------------------------------------------------------------------
    def _load_single_model(self, folder: str) -> TradingAgentWrapper:
        """
        folder: 'random' 또는 'smart'
        ./saved_models/{folder}/ 내에서 model_*_{folder}_*.zip 파일을 탐색하여
        가장 최신 파일을 로드하지 못하면 None 반환.
        """
        save_dir = f"./saved_models/{folder}/"
        # 파일명 패턴: model_{feature_mode}_{tag}_{timestamp}.zip
        pattern   = os.path.join(save_dir, f"model_*_{folder}_*.zip")
        files = sorted(glob.glob(pattern))

        # 취점 타입이 없으면 폴더 내 모든 zip 탐색 (하위 호환성)
        if not files:
            files = sorted(glob.glob(os.path.join(save_dir, "*.zip")))

        if not files:
            self.logger.warning(f"StrategyManager: [{folder}] 모델이 {save_dir}에 없습니다.")
            return None

        latest_zip = files[-1]
        model_path = latest_zip.replace(".zip", "")
        self.logger.info(f"StrategyManager: [{folder}] 모델 탐색 완료 → {latest_zip}")

        # 모델 차원 자동 감지 후 더미 환경 생성
        model_dim    = TradingAgentWrapper.get_model_dimension(model_path)
        detected_mode = "advanced" if model_dim >= 100 else "basic"
        dummy_config  = {"symbol": "DUMMY", "feature_mode": detected_mode, "target_dim": model_dim}
        dummy_env     = ScalpingTradingEnv(self.data_collector, self.order_manager, dummy_config)

        config_dict  = self.config_manager.get_dict() if hasattr(self.config_manager, "get_dict") else {}
        agent_config = {"seq_len": config_dict.get("seq_len", 10)}
        agent        = TradingAgentWrapper(dummy_env, agent_config)
        agent.load_weights(model_path)
        self.logger.error(
            f"StrategyManager: [{folder}] 모델 로딩 성공 ✅ — 차원={model_dim}, 모드={detected_mode} (Path: {model_path})"
        )
        print(f"시스템: [SUCCESS] '{folder}' 모델 가중치가 정상적으로 로드되었습니다. (차원: {model_dim})")
        return agent

    # ------------------------------------------------------------------
    # [Public] Config 라우팅 — live_trading_model_type에 따라 모델 선택
    # ------------------------------------------------------------------
    def load_model_from_config(self):
        """
        config.yaml의 live_trading_model_type 값을 읽어
        random / smart / dual 중 하나를 선택하여 모델을 로드합니다.
        GUI 입력이나 코드 수정 없이 실행 시 config 설정만으로 100% 자동화됩니다.
        """
        config_dict   = self.config_manager.get_dict() if hasattr(self.config_manager, "get_dict") else {}
        model_type    = config_dict.get("live_trading_model_type", "random").lower().strip()

        self.logger.info(f"StrategyManager: Config 모델 타입 = [{model_type}]")
        print(f"시스템: [Step 4] 모델 라우팅 모드 = '{model_type}'")

        if model_type == "smart":
            agent = self._load_single_model("smart")
            if agent:
                self.shared_agent = agent
                self.model_smart  = agent
            else:
                self._fallback_empty_model()

        elif model_type == "dual":
            # [Regime-Switching 뼈대] 두 모델을 모두 메모리에 로드
            self.model_random = self._load_single_model("random")
            self.model_smart  = self._load_single_model("smart")

            # 활성 모델: random 모델을 기본 활성으로 설정 (추후 스와프 가능)
            self.shared_agent = self.model_random or self.model_smart
            if not self.shared_agent:
                self._fallback_empty_model()
            else:
                print(
                    f"시스템: [Step 4] Dual 모드 — random={bool(self.model_random)}, "
                    f"smart={bool(self.model_smart)}, active=random"
                )

        else:  # 'random' 또는 미지정 기본값
            agent = self._load_single_model("random")
            if agent:
                self.shared_agent = agent
                self.model_random = agent
            else:
                self._fallback_empty_model()

        # 종목별 환경 동기화 (선택된 shared_agent의 모드/차원 기준)
        if self.shared_agent:
            self._sync_envs_with_agent(self.shared_agent)

    def _fallback_empty_model(self):
        """모델 파일이 없을 때 랜덤 가중치로 실행하는 폴백 처리."""
        print("시스템: [Step 4] 저장된 모델이 없습니다. 랜덤 초기 가중치로 실행합니다. (AI 학습 스튜디오에서 학습이 필요합니다)")
        config_dict  = self.config_manager.get_dict() if hasattr(self.config_manager, "get_dict") else {}
        dummy_config = {"symbol": "DUMMY", "feature_mode": "advanced"}
        dummy_env    = ScalpingTradingEnv(self.data_collector, self.order_manager, dummy_config)
        agent_config = {"seq_len": config_dict.get("seq_len", 10)}
        self.shared_agent = TradingAgentWrapper(dummy_env, agent_config)

    def _sync_envs_with_agent(self, agent: TradingAgentWrapper):
        """shared_agent의 모드/차원에 맞춰 종목별 환경을 재초기화."""
        config_dict  = self.config_manager.get_dict() if hasattr(self.config_manager, "get_dict") else {}
        try:
            detected_mode = agent.env.feature_mode
            obs_shape     = agent.env.observation_space.shape
            model_dim     = obs_shape[0] if obs_shape else 0
        except Exception:
            detected_mode = "advanced"
            model_dim     = 0

        for sym in self.symbols:
            clean_sym = sym.split('_')[0]
            from core.live_trading_engine import LiveTradingEngine
            engine = LiveTradingEngine(clean_sym, self.config_manager, self.order_manager, agent)
            self.envs[clean_sym] = engine
            self.last_action_times[clean_sym] = 0.0
            self.logger.info(f"StrategyManager: [{clean_sym}] 실시간 엔진(LiveTradingEngine) 초기화 완료.")

    # ------------------------------------------------------------------
    # [Legacy] 기존 load_model() 단일 경로 직접 지정 호환 유지
    # 주로 BacktestStudio나 수동 지정 시나리오에서 사용
    # ------------------------------------------------------------------
    def load_model(self, model_path: str):
        """[Legacy] 주어진 경로에서 단일 모델을 로드합니다."""
        try:
            model_dim     = TradingAgentWrapper.get_model_dimension(model_path) if model_path else 0
            detected_mode = "advanced" if model_dim >= 100 else "basic"
            self.logger.info(f"StrategyManager: [{model_dim}차원] → '{detected_mode}' 모드 자동 전환")

            dummy_config  = {"symbol": "DUMMY", "feature_mode": detected_mode, "target_dim": model_dim}
            dummy_env     = ScalpingTradingEnv(self.data_collector, self.order_manager, dummy_config)
            config_dict   = self.config_manager.get_dict() if hasattr(self.config_manager, "get_dict") else {}
            agent_config  = {"seq_len": config_dict.get("seq_len", 10)}
            self.shared_agent = TradingAgentWrapper(dummy_env, agent_config)

            if model_path:
                self.shared_agent.load_weights(model_path)
                self.logger.info(f"StrategyManager: Shared model weights loaded from {model_path}")
                for sym in self.symbols:
                    clean_sym = sym.split('_')[0]
                    from core.live_trading_engine import LiveTradingEngine
                    self.envs[clean_sym] = LiveTradingEngine(clean_sym, self.config_manager, self.order_manager, self.shared_agent)
                    self.last_action_times[clean_sym] = 0.0
                    self.logger.info(f"StrategyManager: [{clean_sym}] 실시간 엔진(LiveTradingEngine) 환경 초기화 완료.")

        except Exception as e:
            self.logger.error(f"StrategyManager 초기화 중 에러: {e}")

    async def start(self):
        """이벤트 드리븐 전략 루프 시작"""
        if not self.shared_agent:
            self.logger.warning("Agent가 로드되지 않았습니다. 매매 루프를 시작할 수 없습니다.")
            return

        self.is_running = True
        self.logger.info(f"StrategyManager: 멀티 종목({len(self.symbols)}개) 이벤트 드리븐 오케스트레이션 시작.")

        # [복구] 부동 속도 향상을 위해 최소 웜업(최근 1시간)만 백그라운드로 실행
        from core.scheduler import MarketState
        current_state = getattr(self.scheduler, "current_state", MarketState.OUT_OF_MARKET)
        
        if current_state == MarketState.TRADING:
            self.logger.info("StrategyManager: 최근 1시간 데이터를 백그라운드에서 로드하여 AI 판단 가동을 준비합니다.")
            
            token = None
            if hasattr(self.config_manager, "get"):
                token = self.config_manager.get("KIWOOM_ACCESS_TOKEN")
            if not token and hasattr(self.config_manager, "get_dict"):
                token = self.config_manager.get_dict().get("KIWOOM_ACCESS_TOKEN")

            # 초기 유니버스 종목들에 대해 웜업 개시 (0.2초 간격 분산)
            for sym, engine in self.envs.items():
                if token:
                    asyncio.create_task(engine.warmup(token))
                    await asyncio.sleep(0.2)
                else:
                    self.logger.warning(f"StrategyManager: [{sym}] 토큰 부재로 초기 웜업 생략")
        else:
            self.logger.info(f"StrategyManager: 현재 장 상태가 {current_state}이므로 웜업 및 AI 가동을 생략합니다.")

        # DataCollector의 상태 업데이트 콜백 리스트에 등록
        if hasattr(self.data_collector, 'on_state_updated_callbacks'):
            if self._on_tick_event not in self.data_collector.on_state_updated_callbacks:
                self.data_collector.on_state_updated_callbacks.append(self._on_tick_event)
                self.logger.info("StrategyManager: DataCollector 이벤트 구독 완료.")
        else:
            self.logger.error("StrategyManager: DataCollector에 이벤트 리스너 리스트가 없습니다.")
            
        # 빈 캔들 감시 데몬 구동
        asyncio.create_task(self._empty_candle_watchdog())

    async def update_universe(self, new_universe: List[Dict[str, Any]]):
        """동적 유니버스 스캐너가 호출하는 Safe Swap Logic"""
        async with self._swap_lock:
            # 모든 신규 심볼에서 접미사(_AL) 제거하여 순수 코드로 변환
            new_symbols = list(set([s.get("code").split('_')[0] for s in new_universe if s.get("code")]))
            
            # 기존 심볼들도 순수 코드로 변환하여 관리
            self.symbols = [s.split('_')[0] for s in list(self.symbols)]
            current_symbols = list(self.symbols)

            # 1. 퇴출(Out) 로직
            for sym in current_symbols:
                if sym not in new_symbols:
                    holdings = self.order_manager.holdings.get(sym, 0)
                    has_unexecuted = self.order_manager.has_unexecuted_orders(sym)

                    if holdings > 0 or has_unexecuted:
                        self.logger.warning(f"StrategyManager: [{sym}] 유니버스 탈락했으나 잔고/미체결이 있어 유지합니다.")
                        continue

                    self.logger.info(f"StrategyManager: [{sym}] 유니버스 퇴출.")
                    self.symbols.remove(sym)
                    if sym in self.envs: del self.envs[sym]
                    if sym in self.last_action_times: del self.last_action_times[sym]

                    if hasattr(self.data_collector, 'unsubscribe_symbol'):
                        await self.data_collector.unsubscribe_symbol(sym)

                    vm = getattr(self.config_manager, "_injected_live_vm", None)
                    if vm: vm.sig_log_appended.emit(f"[UNIVERSE UPDATE] OUT: {sym}")

            # 2. 진입(In) 로직
            config_dict = self.config_manager.get_dict() if hasattr(self.config_manager, "get_dict") else {}
            for sym in new_symbols:
                if sym not in self.symbols:
                    self.logger.info(f"StrategyManager: [{sym}] 신규 유니버스 편입.")
                    self.symbols.append(sym)
                    from core.live_trading_engine import LiveTradingEngine
                    engine = LiveTradingEngine(sym, self.config_manager, self.order_manager, self.shared_agent)
                    self.envs[sym] = engine
                    self.last_action_times[sym] = 0.0
                    
                    # [복구] 편입 즉시 최소 웜업 시작
                    token = None
                    if hasattr(self.config_manager, "get"):
                        token = self.config_manager.get("KIWOOM_ACCESS_TOKEN")
                    if not token and hasattr(self.config_manager, "get_dict"):
                        token = self.config_manager.get_dict().get("KIWOOM_ACCESS_TOKEN")
                    
                    if not token:
                        self.logger.warning(f"StrategyManager: [{sym}] 웜업을 위한 토큰이 없습니다. 추후 틱 데이터로만 축적합니다.")
                    else:
                        asyncio.create_task(engine.warmup(token))

                    if hasattr(self.data_collector, 'subscribe_symbol'):
                        await self.data_collector.subscribe_symbol(sym)

                    vm = getattr(self.config_manager, "_injected_live_vm", None)
                    if vm: vm.sig_log_appended.emit(f"[UNIVERSE UPDATE] IN: {sym}")

            # Config 반영 (실제 종목명 보존)
            name_map = {s.get("code").split('_')[0]: s.get("name") for s in new_universe if s.get("code")}
            updated_dicts = []
            for s in self.symbols:
                real_name = name_map.get(s, f"Stock_{s}")
                updated_dicts.append({"code": s, "name": real_name})
            
            self.config_manager.set_symbols(updated_dicts)
            
            # UI ViewModel에 실명 캐시 갱신 요청
            vm = getattr(self.config_manager, "_injected_live_vm", None)
            if vm and hasattr(vm, "update_symbol_names"):
                vm.update_symbol_names(updated_dicts)

    async def _on_tick_event(self, symbol: str, normalized_state=None, price=0.0, volume=0.0, timestamp=None):
        """데이터 수신 시 호출되는 핵심 리스너 (Event-Driven)"""
        if not self.is_running or self.is_ai_paused:
            return

        try:
            clean_symbol = symbol.split('_')[0]

            # --- GATE 1: MarketState 체크 ---
            from core.scheduler import MarketState
            scheduler = getattr(self.config_manager, "_injected_scheduler", None)
            if scheduler and scheduler.current_state != MarketState.TRADING:
                if time.time() % 30 < 1:  # 30초마다 1회 출력 (단순화)
                    self.logger.debug(f"[AI-GATE1] [{clean_symbol}] 장 외 시간 → 판단 차단")
                return

            # --- GATE 2: LiveTradingEngine Lazy Initialization ---
            engine = self.envs.get(clean_symbol)
            if engine is None:
                self.logger.info(f"[AI-GATE2] [{clean_symbol}] 실시간 엔진(LiveTradingEngine) 미등록 발견 → 즉시 생성 및 웜업")
                from core.live_trading_engine import LiveTradingEngine
                engine = LiveTradingEngine(clean_symbol, self.config_manager, self.order_manager, self.shared_agent)
                self.envs[clean_symbol] = engine
                token = None
                if hasattr(self.config_manager, "get"):
                    token = self.config_manager.get("KIWOOM_ACCESS_TOKEN")
                if not token and hasattr(self.config_manager, "get_dict"):
                    token = self.config_manager.get_dict().get("KIWOOM_ACCESS_TOKEN")
                
                if token:
                    asyncio.create_task(engine.warmup(token))
                else:
                    self.logger.warning(f"[AI-GATE2] [{clean_symbol}] 웜업 토큰 부재로 지연 실행 생략")

            # --- GATE 3: 틱 업데이트 (엔진으로 데이터 토스) ---
            if price > 0 and timestamp is not None:
                await engine.update_tick(price, int(volume), timestamp)

        except Exception as e:
            self.logger.error(f"[StrategyManager] _on_tick_event 처리 중 에러: {e}")

    async def _empty_candle_watchdog(self):
        """정기적으로 모든 엔진을 순회하며 거래량 0인 빈 캔들 케이스를 강제 확정시키는 데몬"""
        from datetime import datetime
        while self.is_running:
            await asyncio.sleep(1.0)
            now_dt = datetime.now()
            # 정각(0초) 무렵에 한 번씩 체크를 실행
            if now_dt.second == 0 or now_dt.second == 1:
                for sym, engine in list(self.envs.items()):
                    # 엔진에 empty_minute 체크 위임
                    if hasattr(engine, 'check_empty_minute'):
                        await engine.check_empty_minute(now_dt)
                # 동일 분 내 중복 실행 방지
                await asyncio.sleep(2.0)


    async def _on_tick_event_backup(self, symbol: str, normalized_state=None):
        """데이터 수신 시 호출되는 핵심 리스너 (Event-Driven)"""
        if not self.is_running or self.is_ai_paused:
            return

        try:
            # [진단] symbol 정규화
            clean_symbol = symbol.split('_')[0]

            # --- GATE 1: MarketState 체크 ---
            from core.scheduler import MarketState
            scheduler = getattr(self.config_manager, "_injected_scheduler", None)
            if scheduler and scheduler.current_state != MarketState.TRADING:
                if int(time.time()) % 30 == 0:  # 30초마다 1회 출력
                    self.logger.error(f"[AI-GATE1] [{clean_symbol}] 장 외 시간 → 판단 차단 (MarketState={scheduler.current_state})")
                return

            # --- GATE 2: 쿨다운 체크 ---
            current_time = time.time()
            elapsed = current_time - self.last_action_times.get(clean_symbol, 0.0)
            if elapsed < self.cooldown_seconds:
                return  # 쿨다운은 정상 로직, 로그 불필요

            # --- GATE 3: env 존재 여부 (Lazy Initialization) ---
            env = self.envs.get(clean_symbol)
            if env is None:
                self.logger.info(f"[AI-GATE3] [{clean_symbol}] 실시간 환경(env) 미등록 발견 → 즉시 생성 및 등록 시도")
                from env.trading_env import ScalpingTradingEnv

                # 전역 설정에서 현재 feature_mode 및 초기 자산 획득
                config_dict = self.config_manager.get_dict() if hasattr(self.config_manager, "get_dict") else {}
                feature_mode = config_dict.get("feature_mode", "basic")
                initial_balance = config_dict.get("initial_balance", 10000000)

                env_config = {
                    "symbol": clean_symbol,
                    "initial_balance": initial_balance,
                    "feature_mode": feature_mode
                }

                # 환경 생성 및 등록
                self.envs[clean_symbol] = ScalpingTradingEnv(self.data_collector, self.order_manager, env_config)
                self.last_action_times[clean_symbol] = 0.0
                env = self.envs[clean_symbol]
                self.logger.info(f"   => [{clean_symbol}] 환경 등록 완료 (Mode: {feature_mode})")

            # --- GATE 4: 데이터 버퍼 충분 여부 ---
            seq_len = self.shared_agent.seq_len
            actual_buffer = self.data_collector.state_buffers.get(symbol, [])
            buf_len = len(actual_buffer)
            if buf_len < seq_len:
                if int(time.time()) % 10 == 0:  # 10초마다 1회 출력
                    self.logger.error(f"[AI-GATE4] [{clean_symbol}] 버퍼 부족 ({buf_len}/{seq_len}) → 대기 중")
                return

            # --- GATE 5: 관측값 유효성 ---
            obs = self.data_collector.get_latest_state(symbol, seq_len=seq_len)
            if np.all(obs == 0):
                self.logger.info(f"[AI-GATE5] [{clean_symbol}] 관측값 전체 0 → 추론 불가")
                return

            # --- 추론 실행 ---
            # Gymnasium v1.0 호환성: 래퍼 체인 내 속성 탐색을 위해 get_wrapper_attr 사용
            action_masks = env.get_wrapper_attr('action_masks')()
            obs_batch = np.expand_dims(obs, axis=0)

            self.logger.info(f"[AI-INFER] [{clean_symbol}] 추론 시작 | masks={action_masks} | buf={buf_len}")

            result = self.shared_agent.predict(obs_batch, action_masks=action_masks, return_probs=True)
            action, probs = result
            if isinstance(action, np.ndarray): action = int(action[0])

            # 확률 분포 및 최신 임계값 확인 (전역 설정 연동)
            ai_threshold = self.config_manager.get("ai_confidence_threshold", 0.5)
            max_prob = float(max(probs))
            raw_action = action

            if max_prob < ai_threshold:
                action = 0  # 신뢰도 부족 → Hold 강제

            action_names = {0: "Hold", 1: "Buy", 2: "Sell"}

            self.logger.error(
                f"[AI-RESULT] [{clean_symbol}] 원본={action_names.get(raw_action, '?')} "
                f"| 최종={action_names.get(action, '?')} "
                f"| Hold={probs[0]:.2f} Buy={probs[1]:.2f} Sell={probs[2]:.2f} "
                f"| max_conf={max_prob:.2f}"
            )

            # AI 신뢰도 UI 업데이트 (0: Hold, 1: Buy, 2: Sell)
            confidence_dict = {
                "Hold": int(probs[0] * 100),
                "Buy": int(probs[1] * 100),
                "Sell": int(probs[2] * 100)
            }

            signal_text = "Hold"
            if action == 1:
                signal_text = "Buy"
            elif action == 2:
                signal_text = "Sell"

            vm = getattr(self.config_manager, "_injected_live_vm", None)
            if vm:
                if clean_symbol not in vm.symbols_summary:
                    vm.symbols_summary[clean_symbol] = {"price": 0, "ai_signal": "-", "holdings": 0}
                vm.symbols_summary[clean_symbol]["ai_signal"] = signal_text
                vm.sig_symbols_summary_updated.emit(vm.symbols_summary)

                if vm.selected_symbol == clean_symbol or not vm.selected_symbol:
                    vm.sig_ai_confidence_updated.emit(confidence_dict)

            # 3. Action 수행 (1: BUY, 2: SELL)
            if action in [1, 2]:
                str_action = "BUY" if action == 1 else "SELL"
                current_price = self.data_collector.get_latest_price(symbol)
                if current_price <= 0:
                    self.logger.error(f"[AI-ORDER] [{clean_symbol}] {str_action} 신호이나 현재가 0 → 주문 생략")
                    return

                if action == 1:  # BUY - 동적 수량 계산
                    max_invest = self.risk_manager.get_max_invest_per_symbol()
                    qty = int(max_invest // current_price)

                    if qty <= 0:
                        msg = f"[SYSTEM] {clean_symbol} 매수 신호 발생했으나, 1주 가격({current_price:,.0f}원)이 설정된 최대 한도({max_invest:,.0f}원)를 초과하여 매수를 생략합니다."
                        self.logger.error(f"[AI-ORDER] [{clean_symbol}] {msg}")
                        vm = getattr(self.config_manager, "_injected_live_vm", None)
                        if vm: vm.sig_log_appended.emit(msg)
                        return
                    target_qty = qty
                else:  # SELL
                    target_qty = self.order_manager.holdings.get(clean_symbol, 0)

                if target_qty > 0:
                    self.logger.error(f"StrategyManager: [{clean_symbol}] 에이전트 결단 - {str_action} {target_qty}주")
                    self.last_action_times[clean_symbol] = current_time
                    asyncio.create_task(
                        self.order_manager.execute_smart_order(str_action, clean_symbol, target_qty, self.data_collector)
                    )

        except Exception as e:
            self.logger.error(f"StrategyManager: [{symbol}] 이벤트 처리 에러: {e}")

    async def stop(self):
        """오케스트레이션 정지 및 구독 해제"""
        self.is_running = False
        if hasattr(self.data_collector, 'on_state_updated_callbacks'):
            if self._on_tick_event in self.data_collector.on_state_updated_callbacks:
                self.data_collector.on_state_updated_callbacks.remove(self._on_tick_event)
        self.logger.info("StrategyManager: 모든 매매 로직 및 이벤트 구독이 정지되었습니다.")
