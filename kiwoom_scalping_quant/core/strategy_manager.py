import asyncio
import glob
import logging
import os
import time
import threading
from datetime import datetime
from typing import List, Dict, Any, Callable
import numpy as np

from env.trading_env import ScalpingTradingEnv
from models.agent import TradingAgentWrapper
from utils.daily_logger import log_universe_snapshot
from core.scheduler import MarketState

class StrategyManager:
    """
    여러 종목(Multi-Symbol)의 트레이딩을 동시에 오케스트레이션하는 관리자 클래스.
    이벤트 드리븐 구조로 설계되어 새로운 틱 데이터 수신 시에만 추론을 수행합니다.
    """
    def __init__(self, config_manager, data_collector, order_manager, risk_manager, system_config=None):
        self.config_manager = config_manager
        self.system_config = system_config
        self.data_collector = data_collector
        self.order_manager = order_manager
        self.risk_manager = risk_manager
        self.logger = logging.getLogger("StrategyManager")
        self.broker_api = None # [신규] 토큰 재발급용 API 핸들

        # [수정] 부팅 시 전체 유니버스를 미리 로드하지 않고 빈 상태로 시작합니다.
        # 실제 감시 종목은 init_engines() 또는 handle_condition_insert()를 통해 동적으로 추가됩니다.
        self.symbols: List[str] = []

        self.envs: Dict[str, Any] = {}
        self.shared_agent: TradingAgentWrapper = None

        self.model_random: TradingAgentWrapper = None
        self.model_smart: TradingAgentWrapper  = None

        self.is_running = False
        self.last_action_times: Dict[str, float] = {}
        self.cooldown_seconds = 3.0
        self._swap_lock = asyncio.Lock()
        self.is_ai_paused = False
        self.last_global_buy_time = 0.0 # [신규] 전역 매수 쿨타임 관리용
        self.global_buy_cooldown = 2.5  # [신규] 글로벌 매수 쿨타임 (초)
        self._order_lock = asyncio.Lock() # [신규] 비동기 레이스 컨디션 방지용 락
        self._pending_buy_symbols: set = set() # [신규] 동기적 중복 진입 차단용 집합
        
        # [동적 유니버스 필터링] 8슬롯 로직 명시적 설정
        self.MAX_CONCURRENT_STOCKS = 8
        self.pending_universe_queue: List[str] = [] # 조건검색 대기열 (초과분 저장)
        
        # [신규] 순차 웜업 큐 및 워커
        self._warmup_queue = asyncio.Queue()
        self._warmup_worker_task = None
        self._warmup_worker_task = None
        self._dashboard_task = None
        
        # [신규] 조건식 스위칭 이벤트 콜백
        self.on_condition_switched_callbacks: List[Callable] = []

        # ── [멀티 모델 스위칭] ──────────────────────────────────────────
        # 두 모델을 동시에 메모리에 로드해 두고, 포인터만 교체하는 방식으로
        # 매매 루프 중단 없는 실시간 스위칭(Hot-Swap)을 지원합니다.
        self._model_offensive: TradingAgentWrapper = None   # 공격형 모델
        self._model_defensive: TradingAgentWrapper = None   # 방어형 모델
        self._current_model_mode: str = "offensive"          # 현재 활성 모드
        self._model_swap_lock = threading.Lock()             # Thread-Safe 교체용 Lock
        # ────────────────────────────────────────────────────────────────

        # [신규] 최소 감시 보장(Minimum Lock Time) 관련 상태
        self.inserted_at: Dict[str, float] = {} # {symbol: timestamp}
        self.MIN_LOCK_TIME = 60 # 초 단위 (깜빡임 방지를 위해 60초로 연장)

    def set_ai_paused(self, paused: bool):
        if self.is_ai_paused != paused:
            self.is_ai_paused = paused
            self.logger.info(f"StrategyManager: AI Trading is {'PAUSED' if paused else 'RESUMED'}")

    def _load_model_by_path(self, model_path: str) -> TradingAgentWrapper:
        """지정된 경로의 모델 파이을 다이렉트로 로드합니다."""
        if not model_path or not os.path.exists(model_path):
            self.logger.warning(f"StrategyManager: 지정된 모델 경로가 유효하지 않습니다: {model_path}")
            return None

        # .zip 확장자 제거 (Stable Baselines3 규격 대응)
        model_load_path = model_path.replace(".zip", "")
        self.logger.info(f"StrategyManager: 명시적 모델 로드 시작 → {model_path}")

        try:
            # 모델 차원 자동 감지 및 더미 환경 생성
            model_dim = TradingAgentWrapper.get_model_dimension(model_load_path)
            detected_mode = "advanced" if model_dim >= 100 else "basic"
            dummy_config = {"symbol": "DUMMY", "feature_mode": detected_mode, "target_dim": model_dim}
            dummy_env = ScalpingTradingEnv(self.data_collector, self.order_manager, dummy_config)

            config_dict = self.config_manager.get_dict() if hasattr(self.config_manager, "get_dict") else {}
            agent_config = {"seq_len": config_dict.get("seq_len", 10)}
            agent = TradingAgentWrapper(dummy_env, agent_config)
            
            agent.load_weights(model_load_path)
            print(f"StrategyManager: 모델 로딩 성공 ✅ (차원={model_dim}, 경로={model_path})")
            return agent
        except Exception as e:
            self.logger.warning(f"StrategyManager: 모델 가중치 로드 중 치명적 오류: {e}")
            return None

    def load_model_from_config(self):
        """
        [하위 호환] config.yaml의 설정을 기반으로 두 모델을 모두 프리로드합니다.
        preload_all_models()의 앨리어스입니다.
        """
        self.preload_all_models()

    def preload_all_models(self):
        """
        부팅 시 공격형·방어형 두 모델을 모두 메모리에 로드합니다.
        스위칭 시 파일 I/O 없이 포인터만 교체하는 Zero-Latency Hot-Swap을 위한 사전 작업입니다.
        """
        config_dict = self.config_manager.get_dict() if hasattr(self.config_manager, "get_dict") else {}

        offensive_path = config_dict.get("model_offensive_path", "").strip()
        defensive_path = config_dict.get("model_defensive_path", "").strip()
        # bool(True=공격형) 또는 구버전 str("offensive") 모두 처리
        _raw_mode = config_dict.get("active_model_mode", True)
        if isinstance(_raw_mode, bool):
            initial_mode = "offensive" if _raw_mode else "defensive"
        else:
            initial_mode = str(_raw_mode).strip().lower()

        self.logger.info(f"StrategyManager: 멀티 모델 프리로드 시작 | 공격형={offensive_path} | 방어형={defensive_path}")

        # 1. 공격형 모델 로드
        if offensive_path:
            agent = self._load_model_by_path(offensive_path)
            if agent:
                self._model_offensive = agent
                self.logger.info("StrategyManager: ✅ 공격형 모델(Offensive) 프리로드 완료")
            else:
                self.logger.warning("StrategyManager: ❌ 공격형 모델 로드 실패")
        else:
            self.logger.warning("StrategyManager: model_offensive_path가 설정되지 않았습니다.")

        # 2. 방어형 모델 로드
        if defensive_path:
            agent = self._load_model_by_path(defensive_path)
            if agent:
                self._model_defensive = agent
                self.logger.info("StrategyManager: ✅ 방어형 모델(Defensive) 프리로드 완료")
            else:
                self.logger.warning("StrategyManager: ❌ 방어형 모델 로드 실패")
        else:
            self.logger.warning("StrategyManager: model_defensive_path가 설정되지 않았습니다.")

        # 3. 초기 모드에 따라 shared_agent 설정
        self._current_model_mode = initial_mode
        if initial_mode == "defensive" and self._model_defensive:
            self.shared_agent = self._model_defensive
        elif self._model_offensive:
            self.shared_agent = self._model_offensive
            self._current_model_mode = "offensive"
        else:
            self.logger.warning("StrategyManager: 두 모델 모두 로드 실패. 폴백 모드로 전환합니다.")
            self._fallback_empty_model()
            return

        self.logger.info(
            f"StrategyManager: 🧠 초기 AI 모드 설정 완료 → [{self._current_model_mode.upper()}] "
            f"(공격형={'OK' if self._model_offensive else 'FAIL'}, "
            f"방어형={'OK' if self._model_defensive else 'FAIL'})"
        )

    def switch_model_by_mode(self, mode: str, trigger_source: str = "UNKNOWN") -> bool:
        """
        [Hot-Swap] 매매 루프 중단 없이 AI 모델을 실시간으로 교체합니다.
        threading.Lock()으로 보호되어 멀티스레드 환경에서도 안전합니다.

        Args:
            mode:           "offensive" 또는 "defensive"
            trigger_source: 로그 기록용 트리거 소스 ("LOCAL_UI", "FIREBASE", 등)

        Returns:
            True: 교체 성공, False: 이미 같은 모드이거나 모델 없음
        """
        mode = mode.strip().lower()
        if mode not in ("offensive", "defensive"):
            self.logger.warning(f"StrategyManager: switch_model_by_mode — 알 수 없는 모드 '{mode}'")
            return False

        with self._model_swap_lock:
            # 이미 같은 모드라면 스킵
            if mode == self._current_model_mode:
                self.logger.info(f"StrategyManager: 이미 [{mode.upper()}] 모드입니다. 스위칭을 건너뜁니다.")
                return False

            new_agent = self._model_offensive if mode == "offensive" else self._model_defensive
            if new_agent is None:
                self.logger.warning(
                    f"StrategyManager: [{mode.upper()}] 모델이 메모리에 없습니다. "
                    "preload_all_models()가 먼저 호출되었는지 확인하세요."
                )
                return False

            prev_mode = self._current_model_mode
            self._current_model_mode = mode

            # shared_agent 포인터 교체
            self.shared_agent = new_agent

            # 모든 활성 엔진에 새 에이전트 전파 (방법 A)
            success_count = 0
            for sym, engine in self.envs.items():
                if hasattr(engine, 'update_agent'):
                    if engine.update_agent(new_agent):
                        success_count += 1

        # Lock 해제 후 로그 기록 (I/O는 Lock 밖에서 수행)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = (
            f"[{now_str}] [MODEL_SWITCH] "
            f"Trigger={trigger_source} | "
            f"Before={prev_mode.upper()} | "
            f"After={mode.upper()} | "
            f"Engines Updated={success_count}/{len(self.envs)}"
        )
        self.logger.warning(f"🔄 {log_msg}")
        print(log_msg)  # 콘솔 직접 출력 (가시성 보장)

        # UI 로그 창에도 출력
        live_vm = getattr(self.config_manager, "_injected_live_vm", None)
        if live_vm and hasattr(live_vm, 'append_log'):
            mode_label = "🔴 공격형 (Offensive)" if mode == "offensive" else "🔵 방어형 (Defensive)"
            live_vm.append_log(
                f"🔄 [모델 전환] {mode_label} | 트리거: {trigger_source} "
                f"| 엔진 {success_count}개 갱신 완료"
            )

        return True

    def can_execute_buy(self) -> bool:
        """글로벌 매수 쿨타임 상태를 확인합니다."""
        now = time.time()
        return (now - self.last_global_buy_time) >= self.global_buy_cooldown

    def record_buy(self):
        """글로벌 매수 발생 시점을 기록합니다."""
        self.last_global_buy_time = time.time()

    async def init_engines(self, universe_list: List[Dict[str, Any]]):
        """
        초기 유니버스 확정 시 호출됩니다. (스냅샷 대응)
        이제 기존 엔진을 clear() 하지 않고, 새 리스트와의 차이점만 찾아 업데이트합니다.
        """
        new_codes = set()
        protected_list = self.config_manager.get("protected_symbols", [])
        protected_symbols = set(str(s).split('_')[0] for s in protected_list)

        for s in universe_list:
            orig = s.get("code")
            if not orig: continue
            clean = orig.split('_')[0]
            if clean not in protected_symbols:
                new_codes.add(clean)

        # 보유 종목 추가 (선택 사항: 사용자가 조건검색만 원하더라도 보유 종목 관리는 필요할 수 있음)
        # 하지만 사용자 요청이 '조건검색에 포착된 종목만'이므로 보유 종목은 제외하거나 
        # 필요 시 LiveDashboardViewModel에서 처리하도록 위임합니다.
        
        current_codes = set(self.symbols)
        to_add = list(new_codes - current_codes)
        to_remove = list(current_codes - new_codes)

        # [🚨 중요] 이제 직접 update_engines를 호출하여 슬롯 관리를 수행합니다.
        await self.update_engines(to_add, to_remove)

    async def update_engines(self, to_add: List[str], to_remove: List[str]):
        """매매 엔진 동기화 (추가/삭제)"""
        if not self.shared_agent:
            self.logger.warning("StrategyManager: 모델이 로드되지 않아 엔진을 업데이트할 수 없습니다.")
            return

        # 1. 제거 처리
        for sym in to_remove:
            # [Edge Case] 대기열에 있는 경우 리스트에서만 제거
            if sym in self.pending_universe_queue:
                self.pending_universe_queue.remove(sym)
                self.logger.info(f"StrategyManager: [{sym}] 대기열에서 조용히 삭제되었습니다.")
                continue

            if sym in self.envs:
                engine = self.envs.pop(sym)
                if hasattr(engine, 'destroy'):
                    await engine.destroy()
                if sym in self.symbols: self.symbols.remove(sym)
                if sym in self.last_action_times: del self.last_action_times[sym]
                
                # 실시간 구독 해제
                if hasattr(self.data_collector, 'unsubscribe_symbol'):
                    asyncio.create_task(self.data_collector.unsubscribe_symbol(sym))
                
                self.logger.info(f"StrategyManager: [{sym}] 활성 엔진 파괴 및 구독 해제 완료.")

        # 2. 추가 처리 (슬롯 제한 확인)
        from core.live_trading_engine import LiveTradingEngine
        for sym in to_add:
            if sym in self.symbols or sym in self.pending_universe_queue:
                continue

            if len(self.symbols) < self.MAX_CONCURRENT_STOCKS:
                self.logger.info(f"StrategyManager: [{sym}] 활성 슬롯 진입 ({len(self.symbols)}/{self.MAX_CONCURRENT_STOCKS})")
                self.symbols.append(sym)
                engine = LiveTradingEngine(sym, self.config_manager, self.order_manager, self.shared_agent, 
                                           strategy_manager=self, broker_api=self.broker_api)
                self.envs[sym] = engine
                self.last_action_times[sym] = 0.0
                self.inserted_at[sym] = time.time() # [추가] 진입 시간 기록
                
                # [🚨 중요] 활성 슬롯일 때만 시세 구독 요청
                if hasattr(self.data_collector, 'subscribe_symbol'):
                    asyncio.create_task(self.data_collector.subscribe_symbol(sym))
                
                # 웜업 큐 투입
                self._warmup_queue.put_nowait(engine)
            else:
                self.logger.info(f"StrategyManager: [{sym}] 슬롯 초과로 대기열(Queue)에 추가됩니다.")
                self.pending_universe_queue.append(sym)

        # 3. 빈 슬롯이 생겼다면 대기열에서 보충
        if len(self.symbols) < self.MAX_CONCURRENT_STOCKS and self.pending_universe_queue:
            await self._process_pending_queue()

    def _fallback_empty_model(self):
        config_dict = self.config_manager.get_dict() if hasattr(self.config_manager, "get_dict") else {}
        dummy_config = {"symbol": "DUMMY", "feature_mode": "advanced"}
        dummy_env = ScalpingTradingEnv(self.data_collector, self.order_manager, dummy_config)
        agent_config = {"seq_len": config_dict.get("seq_len", 10)}
        self.shared_agent = TradingAgentWrapper(dummy_env, agent_config)
        self.logger.warning("StrategyManager: 기본 폴백(Random) 모델로 기동합니다.")

    async def start(self):
        if not self.shared_agent:
            self.logger.warning("Agent가 로드되지 않았습니다. 매매 루프를 시작할 수 없습니다.")
            return

        self.is_running = True
        self.logger.info(f"StrategyManager: 멀티 종목({len(self.symbols)}개) 전략 루프 시작.")

        # [수정] 주입된 스케줄러를 통해 현재 장 상태 확인
        scheduler = getattr(self.config_manager, "_injected_scheduler", None)
        current_state = getattr(scheduler, "current_state", MarketState.IDLE)

        bypass = getattr(self.system_config, "BYPASS_MARKET_HOURS", False)

        if current_state == MarketState.TRADING or bypass:
            self.logger.info(f"StrategyManager: 장중 부팅(또는 테스트 모드={bypass}) - 종목별 순차 웜업(백그라운드)을 시작합니다.")
            # 기존에 큐에 들어간 엔진들이 있다면 워커가 시작되면서 처리함
        else:
            self.logger.info(f"StrategyManager: 장외 시간({current_state}) - 웜업을 생략합니다.")

        # 웜업 워커 루프 시작
        self._warmup_worker_task = asyncio.create_task(self._warmup_worker_loop())
        
        # [신규] 실시간 감시 현황 대시보드 로그 루프 시작
        self._dashboard_task = asyncio.create_task(self._status_dashboard_loop())
        
        asyncio.create_task(self._empty_candle_watchdog())
        asyncio.create_task(self._balance_sync_loop()) 

    async def update_universe(self, new_universe: List[Dict[str, Any]]):
        """장중 유니버스 동적 교체"""
        try:
            from utils.daily_logger import log_universe_snapshot
            log_universe_snapshot(new_universe, reason="장중 유니버스 갱신")
        except Exception as e:
            self.logger.error(f"주도주 스냅샷 로깅 에러 (update): {e}")

        async with self._swap_lock:
            # [완전 해결] 장중 교체 시에도 순수 숫자로 통일
            unique_symbols = set()
            protected_list = self.config_manager.get("protected_symbols", [])
            protected_symbols = set(str(s).split('_')[0] for s in protected_list)

            for s in new_universe:
                orig = s.get("code")
                if not orig: continue
                clean = orig.split('_')[0]
                if clean not in protected_symbols:
                    unique_symbols.add(clean)
            
            for orig in self.order_manager.bot_holdings.keys():
                clean = orig.split('_')[0]
                if clean not in protected_symbols:
                    unique_symbols.add(clean)
            
            # [추가] 미체결 주문이 있는 종목도 유니버스에 강제 포함
            for sym in self.symbols:
                if self.order_manager.has_unexecuted_orders(sym):
                    unique_symbols.add(sym.split('_')[0])

            new_symbols_list = list(unique_symbols)
            current_symbols = list(self.symbols)

            # 1. 기존 유니버스에서 빠진 종목 및 추가될 종목 분류
            to_remove = []
            to_add = []

            for sym in current_symbols:
                if sym not in new_symbols_list:
                    if sym in protected_symbols:
                        self.logger.warning(f"🚫 [보호 종목] {sym} 즉시 제거")
                    elif sym in self.order_manager.bot_holdings or \
                         self.order_manager.holdings.get(sym, 0) > 0 or \
                         self.order_manager.has_unexecuted_orders(sym):
                        continue
                    to_remove.append(sym)

            for sym in new_symbols_list:
                if sym not in current_symbols and sym not in self.pending_universe_queue:
                    to_add.append(sym)

            # [🚨 최적화 통합] update_engines를 통해 모든 로직(슬롯, 구독, 엔진관리)을 일원화합니다.
            await self.update_engines(to_add, to_remove)

            # DataCollector 구독 업데이트 및 웜업 큐 투입은 이제 update_engines 내부에서 처리됨

            # 웜업은 update_engines 내부에서 수행됨

            # 4. UI 쪽에 종목 리스트가 교체되었음을 알림 (보유 종목 합산 및 보호 종목 필터링)
            live_vm = getattr(self.config_manager, "_injected_live_vm", None)
            if live_vm:
                # 1. 새로운 주도주 리스트 필터링
                display_universe = [s for s in new_universe if str(s.get("code")).split('_')[0] not in protected_symbols]
                display_codes = set(str(s.get("code")).split('_')[0] for s in display_universe)
                
                # 2. 주도주에 없지만 보유 중인 종목 강제 추가
                for sym, qty in self.order_manager.bot_holdings.items():
                    clean = sym.split('_')[0]
                    # 잔고가 있거나 미체결이 있는 경우 리스트에 강제 추가
                    if clean not in display_codes and clean not in protected_symbols:
                        has_unex = self.order_manager.has_unexecuted_orders(clean)
                        if qty > 0 or has_unex:
                            label = f"{clean} (보유)" if qty > 0 else f"{clean} (미체결)"
                            display_universe.append({"code": clean, "name": label, "is_holding": True})
                
                live_vm.update_universe_list(display_universe)

    async def _on_tick_event(self, symbol: str, normalized_state=None, price=0.0, volume=0.0, timestamp=None, **kwargs):
        if not self.is_running or self.is_ai_paused: return
        
        # [개선] _order_lock은 update_tick 전체를 감싸면 모든 종목 틱이 직렬화되어 AI 추론이 가로막힙니다.
        # 대신 _pending_buy_symbols(CPython GIL로 보호되는 동기적 set 연산)만으로 레이스 컨디션을 방어합니다.
        try:
            clean_symbol = symbol.split('_')[0]
            engine = self.envs.get(clean_symbol)

            if engine and price > 0 and timestamp:
                # [중복 진입 차단] 이미 매수 요청이 진행 중인 종목은 즐시 바이패스
                if clean_symbol in self._pending_buy_symbols:
                    self.logger.warning(f"🛡️ [Lock 방어] {clean_symbol}은 이미 주문 진행 중이므로 중복 진입을 차단합니다.")
                    return

                # 추가 인자(change_rate 등)를 포함하여 엔진에 전달
                await engine.update_tick(symbol, normalized_state, price=price, volume=int(volume), 
                                         timestamp=timestamp, **kwargs)
        except Exception as e:
            self.logger.error(f"StrategyManager: 틱 이벤트 처리 중 에러 ({symbol}): {e}")

    async def _empty_candle_watchdog(self):
        from datetime import datetime
        while self.is_running:
            await asyncio.sleep(1.0)
            now_dt = datetime.now()
            if now_dt.second == 1:
                for engine in list(self.envs.values()):
                    if hasattr(engine, 'check_empty_minute'):
                        await engine.check_empty_minute(now_dt)
                await asyncio.sleep(2.0)

    async def _balance_sync_loop(self):
        """실전 매매 모드일 때 주기적으로 계좌 잔고를 동기화합니다."""
        self.logger.info("StrategyManager: 잔고 동기화 루프 시작 (주기: 60초)")
        while self.is_running:
            try:
                # 60초마다 동기화 시도 (OrderManager 내부의 30초 쿨다운과 별개)
                await self.order_manager.sync_balance()
            except Exception as e:
                self.logger.error(f"잔고 동기화 루프 오류: {e}")
            
            await asyncio.sleep(60.0)

    async def stop(self):
        """전략 매니저 및 하위 엔진 정지"""
        self.is_running = False
        self.logger.info("StrategyManager: 정지 시퀀스 시작...")
        
        if self._warmup_worker_task:
            self._warmup_worker_task.cancel()
        if self._dashboard_task:
            self._dashboard_task.cancel()
            
        # 하위 엔진들 정지
        for sym, engine in self.envs.items():
            if hasattr(engine, 'destroy'):
                await engine.destroy()
        
        # 콜백 제거
        if hasattr(self.data_collector, 'on_state_updated_callbacks'):
            if self._on_tick_event in self.data_collector.on_state_updated_callbacks:
                self.data_collector.on_state_updated_callbacks.remove(self._on_tick_event)
                
        self.logger.info("StrategyManager: 모든 서비스가 정지되었습니다.")

    async def _warmup_worker_loop(self):
        """웜업 큐에서 엔진을 하나씩 꺼내어 순차적으로 웜업을 수행하는 워커 루프"""
        self.logger.info("🚀 StrategyManager: 순차 웜업 워커 루프 가동 시작")
        
        while self.is_running:
            try:
                # 큐에서 엔진 대기
                engine = await self._warmup_queue.get()
                
                token = self.config_manager.get("KIWOOM_ACCESS_TOKEN")
                if not token:
                    self.logger.error("StrategyManager: 웜업 실패 - KIWOOM_ACCESS_TOKEN이 없습니다.")
                    self._warmup_queue.task_done()
                    continue
                
                sym = getattr(engine, 'symbol', 'UNKNOWN')
                self.logger.info(f"⏳ [{sym}] 순차 웜업 시작 (남은 대기: {self._warmup_queue.qsize()})")
                
                try:
                    # 개별 종목 웜업에 타임아웃 적용
                    await asyncio.wait_for(engine.warmup(token), timeout=25.0) # 타임아웃 25초로 상향 (토큰 재발급 고려)
                    
                    if getattr(engine, 'is_warmed_up', False):
                        # [검증] 최소 데이터(예: 30개) 확보 여부 확인
                        if len(getattr(engine, 'minute_buffer', [])) < 30:
                            if not self.config_manager.get("OFFLINE_MODE", False):
                                self.logger.error(f"❌ [{sym}] 웜업 데이터 부족 ({len(engine.minute_buffer)}/30). 감시 대상에서 제외합니다.")
                            await self._remove_dynamic_symbol(sym)
                            await self._process_pending_queue()
                        else:
                            self.logger.info(f"✅ [{sym}] 웜업 완료 및 AI 감시 준비됨.")
                    else:
                        self.logger.error(f"⚠️ [{sym}] 웜업 실패. 감시 대상에서 제외합니다.")
                        await self._remove_dynamic_symbol(sym)
                        await self._process_pending_queue()
                except asyncio.TimeoutError:
                    self.logger.error(f"⏰ [{sym}] 웜업 타임아웃 발생. 감시 대상에서 제외합니다.")
                    await self._remove_dynamic_symbol(sym)
                    await self._process_pending_queue()
                except Exception as e:
                    self.logger.error(f"❌ [{sym}] 웜업 중 치명적 오류 발생: {e}")
                    await self._remove_dynamic_symbol(sym)
                    await self._process_pending_queue()
                
                # 큐 작업 완료 보고
                self._warmup_queue.task_done()
                
                # API Rate Limit 방지를 위한 필수 지연 시간 (0.5초)
                await asyncio.sleep(0.5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"StrategyManager: 웜업 워커 루프 에러: {e}")
                await asyncio.sleep(1.0)

    async def _status_dashboard_loop(self):
        """주기적으로 현재 감시 중인 종목 리스트와 상태를 통합 리포팅합니다."""
        # [오프라인 모드] 실시간 API 요청 차단
        if self.config.get("OFFLINE_MODE", False):
            self.logger.info(f"🚫 오프라인 모드: 실시간 감시가 차단합니다.")
            return {"return_code": "OFFLINE", "return_msg": "System is running in OFFLINE mode."}

        self.logger.info("📡 StrategyManager: 실시간 감시 현황 대시보드 루프 가동")
        while self.is_running:
            try:
                await asyncio.sleep(30.0) # 30초 간격
                
                active_list = list(self.symbols)
                curr_count = len(active_list)
                max_count = self.MAX_CONCURRENT_STOCKS
                queue_count = len(self.pending_universe_queue)
                warmup_count = self._warmup_queue.qsize()
                
                # 가독성을 위해 리스트 출력
                print(
                    f"[📡 시스템 현황] 현재 집중 감시 종목: {curr_count}/{max_count}개 ({active_list}) "
                    f"| 대기열(Queue): {queue_count}개 | 웜업대기: {warmup_count}개"
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"StrategyManager: 대시보드 루프 에러: {e}")
                await asyncio.sleep(5.0)

    async def _safe_sequential_warmup(self, engines_to_warmup: list, token: str):
        """API 조회 제한(Rate Limit)을 피하며 안정적으로 웜업 수행"""
        if not engines_to_warmup: return
        
        self.logger.info(f"🚀 StrategyManager: {len(engines_to_warmup)}개 종목 웜업 큐 가동 시작 (토큰 확인됨)")
        
        count = 0
        success_count = 0
        
        for engine in engines_to_warmup:
            count += 1
            sym = getattr(engine, 'symbol', 'UNKNOWN')
            
            try:
                self.logger.info(f"⏳ [{sym}] 웜업 시작 ({count}/{len(engines_to_warmup)})")
                
                # 개별 종목 웜업에 최대 10초 타임아웃 적용 (한 종목에 묶이지 않도록)
                await asyncio.wait_for(engine.warmup(token), timeout=10.0)
                
                # 웜업 후 상태 확인
                is_ready = getattr(engine, 'is_warmed_up', False)
                if is_ready:
                    success_count += 1
                    self.logger.info(f"✅ [{sym}] 웜업 성공 (현재 {success_count}개 완료)")
                else:
                    self.logger.warning(f"⚠️ [{sym}] 웜업 완료되었으나 상태가 False입니다. (데이터 부족 가능성)")
                    
            except asyncio.TimeoutError:
                self.logger.error(f"⏰ [{sym}] 웜업 타임아웃! (10초 경과 - 다음 종목으로 넘어감)")
            except Exception as e:
                self.logger.error(f"❌ [{sym}] 웜업 중 치명적 오류: {e}")

            # 증권사 API TR 조회 제한 회피를 위한 필수 딜레이 (0.4~0.6초)
            await asyncio.sleep(0.5)

        self.logger.info(f"🏁 StrategyManager: 유니버스 웜업 종료. (성공: {success_count}/{len(engines_to_warmup)})")

    # ==========================================
    # [동적 유니버스 필터링] 조건검색 이벤트 핸들러 (8슬롯 & Queue)
    # ==========================================
    async def handle_condition_snapshot(self, symbols: List[str]):
        """초기 조건검색 스냅샷(예: 31개) 수신 처리"""
        self.logger.info(f"📋 [조건검색 스냅샷] 전체 {len(symbols)} 종목 수신 (8슬롯 & Queue 로직 적용)")
        
        async with self._swap_lock:
            # 1. 현재 관리 중인 종목(보유 종목 등)은 유지하고, 
            #    새로 들어온 스냅샷 중 빈 자리에 들어갈 수 있는 것과 대기열로 갈 것을 구분합니다.
            
            # 현재 활성 슬롯에 있는 종목들
            current_active = set(self.symbols)
            
            to_add_active = []
            to_add_queue = []
            
            for sym in symbols:
                clean_sym = sym.split('_')[0]
                
                # [🚨 중요] 이미 활성 슬롯에 있는 종목이라도, 데이터 수집기 재시작 시점일 수 있으므로 구독을 재확인합니다.
                if clean_sym in current_active:
                    if hasattr(self.data_collector, 'subscribe_symbol'):
                        asyncio.create_task(self.data_collector.subscribe_symbol(clean_sym))
                    continue
                
                if clean_sym in self.pending_universe_queue:
                    continue
                
                if len(current_active) + len(to_add_active) < self.MAX_CONCURRENT_STOCKS:
                    to_add_active.append(clean_sym)
                else:
                    to_add_queue.append(clean_sym)
            
            # 대기열 갱신 (기존 대기열은 스냅샷으로 교체하되, 현재 활성 중인 것은 제외)
            self.pending_universe_queue = to_add_queue
            
            if to_add_active:
                self.logger.info(f"🚀 활성 슬롯 추가 할당: {to_add_active}")
                await self.update_engines(to_add_active, [])
            
            if self.pending_universe_queue:
                self.logger.info(f"⏳ 대기열(Queue) 구성 완료: {len(self.pending_universe_queue)} 종목")

    async def handle_condition_insert(self, symbol: str, event_data: dict = None):
        """조건검색 편입 이벤트 수신"""
        clean_symbol = symbol.split('_')[0]
        self.logger.info(f"📥 [StrategyManager] handle_condition_insert 호출됨: {clean_symbol}")
        
        async with self._swap_lock:
            # [보완] 이미 관리 중인 종목이라도 이탈 예약 상태라면 해제
            if clean_symbol in self.symbols:
                engine = self.envs.get(clean_symbol)
                if engine and getattr(engine, 'is_condition_deleted', False):
                    engine.is_condition_deleted = False
                    self.logger.info(f"♻️ {clean_symbol} 재편입 확인: 이탈 예약을 취소하고 감시를 유지합니다.")
                else:
                    self.logger.info(f"ℹ️ {clean_symbol}은 이미 활성 감시 중입니다. (Skip) 현재 목록: {self.symbols}")
                return

            if clean_symbol in self.pending_universe_queue:
                self.logger.info(f"ℹ️ {clean_symbol}은 이미 대기열에 있습니다. (Skip)")
                return

            # 최대 감시 종목 수 여유가 있는지 확인
            if len(self.symbols) < self.MAX_CONCURRENT_STOCKS:
                self.logger.info(f"🌟 [조건검색 편입] {clean_symbol} 활성 슬롯 즉시 배정 (현재 {len(self.symbols)}/{self.MAX_CONCURRENT_STOCKS})")
                await self.update_engines([clean_symbol], [])
            else:
                if clean_symbol not in self.pending_universe_queue:
                    self.pending_universe_queue.append(clean_symbol)
                    self.logger.info(f"⏳ [조건검색 대기] {clean_symbol} 슬롯 포화. 대기열 추가 (Queue: {len(self.pending_universe_queue)}개)")

    async def handle_condition_delete(self, symbol: str, event_data: dict = None, is_retry: bool = False):
        """조건검색 이탈 이벤트 수신"""
        clean_symbol = symbol.split('_')[0]
        self.logger.info(f"📤 [StrategyManager] handle_condition_delete 호출됨: {clean_symbol}")
        
        async with self._swap_lock:
            if clean_symbol in self.pending_universe_queue:
                self.pending_universe_queue.remove(clean_symbol)
                self.logger.info(f"🗑️ [조건검색 이탈] {clean_symbol} 대기열에서 제거")
                return
                
            if clean_symbol in self.symbols:
                # 잔고와 미체결 내역이 있는지 확인
                holdings_qty = self.order_manager.holdings.get(clean_symbol, 0)
                has_unex = self.order_manager.has_unexecuted_orders(clean_symbol)
                
                engine = self.envs.get(clean_symbol)
                
                # [상태 업데이트] 이탈 예정임을 표시 (최초 이탈 이벤트 발생 시에만)
                if not is_retry and engine:
                    engine.is_condition_deleted = True

                if holdings_qty > 0 or has_unex:
                    self.logger.info(f"⚠️ [조건검색 이탈 보류] {clean_symbol} 잔고({holdings_qty}주)/미체결({has_unex}) 존재. 슬롯 유지")
                else:
                    # [신규] 최소 감시 시간(MIN_LOCK_TIME) 보호 로직
                    import time
                    entry_time = self.inserted_at.get(clean_symbol, 0)
                    elapsed = time.time() - entry_time
                    # float precision 오차 방지를 위해 0.1초 마진
                    if elapsed < self.MIN_LOCK_TIME - 0.1:
                        if not is_retry:
                            wait_time = self.MIN_LOCK_TIME - elapsed
                            self.logger.info(f"⏳ [{clean_symbol}] 조건 이탈 지연 (최소 감시 시간 보호: 경과 {elapsed:.1f}초, 남은 시간 {wait_time:.1f}초)")
                            asyncio.create_task(self._delayed_condition_delete(clean_symbol, wait_time))
                        return

                    # [최종 검증] 지연 대기 중에 다시 편입되지 않았는지 확인
                    if engine and not engine.is_condition_deleted:
                        if is_retry:
                            self.logger.info(f"🛡️ {clean_symbol} 지연 이탈 취소: 대기 중 재편입되었습니다.")
                        return

                    self.logger.info(f"🗑️ [조건검색 이탈] {clean_symbol} 활성 슬롯 비움 및 엔진 제거")
                    await self.update_engines([], [clean_symbol])
                    if clean_symbol in self.inserted_at: del self.inserted_at[clean_symbol]
                    
                    # [핵심] 빈 자리가 생겼으므로 대기열에서 보충
                    await self._process_pending_queue()
            else:
                if not is_retry:
                    self.logger.info(f"ℹ️ {clean_symbol}은 감시 중인 종목이 아닙니다. (Skip)")

    async def _delayed_condition_delete(self, symbol: str, delay: float):
        """지정된 시간 대기 후 이탈 처리를 다시 시도합니다."""
        await asyncio.sleep(delay)
        self.logger.info(f"⏰ [{symbol}] 최소 감시 시간 경과. 이탈 처리 재시도...")
        await self.handle_condition_delete(symbol, is_retry=True)

    async def _process_pending_queue(self):
        """
        대기열(Queue)에서 다음 종목을 꺼내어 활성 슬롯으로 배치합니다.
        주의: 이 메서드는 호출자가 반드시 self._swap_lock을 보유한 상태에서 호출해야 합니다. (Deadlock 방지)
        """
        next_sym = None
        if self.pending_universe_queue and len(self.symbols) < self.MAX_CONCURRENT_STOCKS:
            next_sym = self.pending_universe_queue.pop(0)
            self.logger.info(f"🔄 [슬롯 교체] 대기열에서 '{next_sym}'를 꺼내어 활성 슬롯으로 이동합니다.")
        
        if next_sym:
            # update_engines는 동기화 상태에서 안전하게 내부 속성을 수정합니다.
            await self.update_engines([next_sym], [])

    async def _add_dynamic_symbol(self, symbol: str):
        """단일 종목 동적 추가 및 엔진 구동"""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            from core.live_trading_engine import LiveTradingEngine
            new_engine = LiveTradingEngine(symbol, self.config_manager, self.order_manager, self.shared_agent, 
                                           strategy_manager=self, broker_api=self.broker_api)
            self.envs[symbol] = new_engine
            self.last_action_times[symbol] = 0.0
            self.inserted_at[symbol] = time.time() # [추가] 진입 시간 기록
            
            # 실시간 구독 요청
            if hasattr(self.data_collector, 'subscribe_symbol'):
                await self.data_collector.subscribe_symbol(symbol)
                
            # 즉시 웜업 큐에 등록
            token = self.config_manager.get("KIWOOM_ACCESS_TOKEN")
            if token:
                self._warmup_queue.put_nowait(new_engine)

    async def _remove_dynamic_symbol(self, symbol: str):
        """단일 종목 동적 제거 및 엔진 파괴"""
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            if symbol in self.envs:
                engine = self.envs.pop(symbol)
                if hasattr(engine, 'destroy'):
                    await engine.destroy()
            
            if hasattr(self.data_collector, 'unsubscribe_symbol'):
                await self.data_collector.unsubscribe_symbol(symbol)

    async def switch_condition(self, new_name: str):
        """
        [Shared Core] 지정된 시간에 조건식을 스위칭합니다.
        1. 기존 대기열 비우기 (활성 8슬롯은 유지)
        2. 데이터 수집기에 새 조건명 주입
        3. 새 조건식 서버 요청 (CNSRLST -> CNSRREQ)
        """
        print(f"🚀 [시스템 스위칭] 조건식 변경 시작: {self.data_collector.target_condition_name} ➡️ {new_name}")
        
        async with self._swap_lock:
            # 1. 대기열 비우기 (기존 조건식의 대기 종목들은 더 이상 유효하지 않음)
            old_queue_count = len(self.pending_universe_queue)
            self.pending_universe_queue.clear()
            self.logger.info(f"StrategyManager: 기존 대기열({old_queue_count}개)을 초기화했습니다.")

            # 2. 데이터 수집기에 새 조건명 설정
            if hasattr(self.data_collector, 'target_condition_name'):
                self.data_collector.target_condition_name = new_name
            
            # 3. 서버에 새 조건식 목록 및 실시간 감시 요청
            if hasattr(self.data_collector, 'request_condition_list'):
                await self.data_collector.request_condition_list()
            
            # 4. 외부 콜백 호출 (UI 갱신 등)
            for cb in self.on_condition_switched_callbacks:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb(new_name))
                else:
                    try: cb(new_name)
                    except: pass
        
        self.logger.info(f"✅ [시스템 스위칭] {new_name} 모드로 전환 완료.")
