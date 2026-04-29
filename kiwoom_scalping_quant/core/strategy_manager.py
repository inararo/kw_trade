import asyncio
import glob
import logging
import os
import time
from typing import List, Dict, Any
import numpy as np

from env.trading_env import ScalpingTradingEnv
from models.agent import TradingAgentWrapper
from utils.daily_logger import log_universe_snapshot

class StrategyManager:
    """
    여러 종목(Multi-Symbol)의 트레이딩을 동시에 오케스트레이션하는 관리자 클래스.
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
            self.symbols = ['005930']

        self.envs: Dict[str, Any] = {}
        self.shared_agent: TradingAgentWrapper = None

        self.model_random: TradingAgentWrapper = None
        self.model_smart: TradingAgentWrapper  = None

        self.is_running = False
        self.last_action_times: Dict[str, float] = {}
        self.cooldown_seconds = 3.0
        self._swap_lock = asyncio.Lock()
        self.is_ai_paused = False

    def set_ai_paused(self, paused: bool):
        self.is_ai_paused = paused
        self.logger.info(f"StrategyManager: AI Trading is {'PAUSED' if paused else 'RESUMED'}")

    def _load_model_by_path(self, model_path: str) -> TradingAgentWrapper:
        """지정된 경로의 모델 파이을 다이렉트로 로드합니다."""
        if not model_path or not os.path.exists(model_path):
            self.logger.error(f"StrategyManager: 지정된 모델 경로가 유효하지 않습니다: {model_path}")
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
            self.logger.error(f"StrategyManager: 모델 가중치 로드 중 치명적 오류: {e}")
            return None

    def load_model_from_config(self):
        """
        config.yaml의 active_model_path를 직접 참조하여 모델을 로드합니다.
        더 이상 디렉토리를 스캔하며 모델을 자동 탐색하지 않습니다.
        """
        try:
            config_dict = self.config_manager.get_dict() if hasattr(self.config_manager, "get_dict") else {}
            active_path = config_dict.get("active_model_path", "").strip()
            
            self.logger.info(f"StrategyManager: 설정된 활성 모델 경로 = [{active_path}]")
            
            if active_path:
                agent = self._load_model_by_path(active_path)
                if agent:
                    self.shared_agent = agent
                    # 하위 호환성을 위해 모델 타입 분기 생략하고 shared_agent로 단일화
                else:
                    self._fallback_empty_model()
            else:
                self.logger.warning("StrategyManager: active_model_path가 설정되지 않았습니다. 폴백 모드로 진입합니다.")
                self._fallback_empty_model()
                
        except Exception as e:
            self.logger.error(f"StrategyManager: 부팅 중 모델 로드 프로세스 실패 ({e}). 폴백 모드로 전환합니다.")
            self._fallback_empty_model()

    def init_engines(self, universe_list: List[Dict[str, Any]]):
        """유니버스 확정 후 실시간 매매 엔진 초기화 (Lazy Initialization)"""
        # [NEW] 유니버스 전체 스냅샷 로깅
        try:
            log_universe_snapshot(universe_list, reason="초기 유니버스 설정")
        except Exception as e:
            self.logger.error(f"주도주 스냅샷 로깅 에러 (init): {e}")

        if not self.shared_agent:
            self.logger.error("StrategyManager: 엔진 초기화 실패 - 로드된 에이전트가 없습니다.")
            return

        # [완전 해결] 모든 식별자를 순수 숫자 코드로 통일하여 중복 방지
        unique_symbols = set()
        protected_list = self.config_manager.get("protected_symbols", [])
        protected_symbols = set(str(s).split('_')[0] for s in protected_list)

        # 1. 주도주 리스트 정제 및 추가
        for s in universe_list:
            orig = s.get("code")
            if not orig: continue
            clean = orig.split('_')[0]
            if clean not in protected_symbols:
                unique_symbols.add(clean)

        # 2. 보유 종목 정제 및 추가
        for orig in self.order_manager.bot_holdings.keys():
            clean = orig.split('_')[0]
            if clean not in protected_symbols:
                unique_symbols.add(clean)

        # 최종 심볼 리스트 (순수 숫자로 통일)
        self.symbols = list(unique_symbols)
        self.envs.clear()
        
        self.logger.info(f"StrategyManager: 확정된 유니버스 {len(self.symbols)}개에 대해 엔진 초기화 시작.")
        new_engines = []
        for sym in self.symbols:
            from core.live_trading_engine import LiveTradingEngine
            engine = LiveTradingEngine(sym, self.config_manager, self.order_manager, self.shared_agent)
            self.envs[sym] = engine
            new_engines.append(engine)
            self.last_action_times[sym] = 0.0
            self.logger.info(f"StrategyManager: [{sym}] 실시간 엔진 결합 완료.")

        # [보강] 초기화 시 토큰이 있다면 즉시 웜업 시도 (start() 호출을 기다리지 않음)
        token = self.config_manager.get("KIWOOM_ACCESS_TOKEN")
        if token and new_engines:
            asyncio.create_task(self._safe_sequential_warmup(new_engines, token))
        
        print(f"시스템: [SUCCESS] 총 {len(self.symbols)}개의 매매 엔진 배치 완료.")

        # [추가] UI 쪽에 초기 유니버스 리스트 전달 (보유 종목 합산 및 보호 종목 필터링 적용)
        live_vm = getattr(self.config_manager, "_injected_live_vm", None)
        if live_vm:
            # 1. 주도주 리스트 필터링
            display_universe = [s for s in universe_list if str(s.get("code")).split('_')[0] not in protected_symbols]
            display_codes = set(str(s.get("code")).split('_')[0] for s in display_universe)
            
            # 2. 주도주에 없는 보유 종목 추가
            for sym, qty in self.order_manager.bot_holdings.items():
                if sym not in display_codes and sym not in protected_symbols:
                    display_universe.append({"code": sym, "name": f"{sym} (보유)", "is_holding": True})
            
            live_vm.update_universe_list(display_universe)

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

        current_state = getattr(scheduler, "current_state", MarketState.OUT_OF_MARKET)

        if current_state == MarketState.TRADING:
            self.logger.info("StrategyManager: 장중 부팅 - 종목별 순차 웜업(백그라운드)을 시작합니다.")
            token = self.config_manager.get("KIWOOM_ACCESS_TOKEN")
            if token:
                engines_list = list(self.envs.values())
                asyncio.create_task(self._safe_sequential_warmup(engines_list, token))
        else:
            self.logger.info(f"StrategyManager: 장외 시간({current_state}) - 웜업을 생략합니다.")

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

            new_symbols_list = list(unique_symbols)
            current_symbols = list(self.symbols)

            # 1. 기존 유니버스에서 빠진 종목 정리
            for sym in current_symbols:
                if sym not in new_symbols_list:
                    # [중요] 보호 종목은 잔고 상관없이 무조건 제거!
                    if sym in protected_symbols:
                        self.logger.warning(f"🚫 [보호 종목] {sym} 종목은 잔고 여부와 상관없이 감시 리스트에서 즉시 제거됩니다.")
                    else:
                        # 실제 잔고가 있거나 봇 관리 종목이면 유지 (clean 코드로 비교)
                        if sym in self.order_manager.bot_holdings or self.order_manager.holdings.get(sym, 0) > 0:
                            self.logger.warning(f"⚠️ [Zombie Position 방어] {sym} 종목이 탈락했으나 잔고/미체결로 인해 강제 유지됩니다.")
                            continue
                    
                    self.symbols.remove(sym)
                    if sym in self.envs: del self.envs[sym]
                    if hasattr(self.data_collector, 'unsubscribe_symbol'):
                        await self.data_collector.unsubscribe_symbol(sym)

            # 2. 새로운 종목 추가 및 엔진 생성
            new_engines = []
            for sym in new_symbols_list:
                if sym not in self.symbols:
                    # [이중 안전장치] 여기서도 보호 종목 체크
                    if sym in protected_symbols: continue
                    
                    self.symbols.append(sym)
                    from core.live_trading_engine import LiveTradingEngine
                    new_engine = LiveTradingEngine(sym, self.config_manager, self.order_manager, self.shared_agent)
                    self.envs[sym] = new_engine
                    new_engines.append(new_engine)
                    if hasattr(self.data_collector, 'subscribe_symbol'):
                        await self.data_collector.subscribe_symbol(sym)

            # 3. 새로 추가된 엔진들만 모아서 순차 웜업 큐로 넘김
            if new_engines:
                token = self.config_manager.get("KIWOOM_ACCESS_TOKEN")
                if token:
                    asyncio.create_task(self._safe_sequential_warmup(new_engines, token))

            # 4. UI 쪽에 종목 리스트가 교체되었음을 알림 (보유 종목 합산 및 보호 종목 필터링)
            live_vm = getattr(self.config_manager, "_injected_live_vm", None)
            if live_vm:
                # 1. 새로운 주도주 리스트 필터링
                display_universe = [s for s in new_universe if str(s.get("code")).split('_')[0] not in protected_symbols]
                display_codes = set(str(s.get("code")).split('_')[0] for s in display_universe)
                
                # 2. 주도주에 없지만 보유 중인 종목 강제 추가
                for sym, qty in self.order_manager.bot_holdings.items():
                    clean = sym.split('_')[0]
                    if clean not in display_codes and clean not in protected_symbols:
                        display_universe.append({"code": clean, "name": f"{clean} (보유)", "is_holding": True})
                
                live_vm.update_universe_list(display_universe)

    async def _on_tick_event(self, symbol: str, normalized_state=None, price=0.0, volume=0.0, timestamp=None):
        if not self.is_running or self.is_ai_paused: return
        try:
            # [단일화] 무조건 순수 코드로 엔진 조회
            clean_symbol = symbol.split('_')[0]
            engine = self.envs.get(clean_symbol)

            if engine and price > 0 and timestamp:
                # ==============================================================
                # [긴급 패치] 엔진이 오해하지 않도록 인자 개수와 키워드를 완벽하게 맞춰서 던짐!
                # ==============================================================
                await engine.update_tick(symbol, normalized_state, price=price, volume=int(volume), timestamp=timestamp)
        except Exception as e:
            self.logger.error(f"[StrategyManager] _on_tick_event 오류: {e}")

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
        self.is_running = False
        if hasattr(self.data_collector, 'on_state_updated_callbacks'):
            if self._on_tick_event in self.data_collector.on_state_updated_callbacks:
                self.data_collector.on_state_updated_callbacks.remove(self._on_tick_event)
        self.logger.info("StrategyManager Stopped.")

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
