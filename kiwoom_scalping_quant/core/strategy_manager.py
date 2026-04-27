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

    def _load_single_model(self, folder: str) -> TradingAgentWrapper:
        save_dir = f"./saved_models/{folder}/"
        pattern = os.path.join(save_dir, f"model_*_{folder}_*.zip")
        files = sorted(glob.glob(pattern))

        if not files:
            files = sorted(glob.glob(os.path.join(save_dir, "*.zip")))

        if not files:
            self.logger.warning(f"StrategyManager: [{folder}] 모델이 {save_dir}에 없습니다.")
            return None

        latest_zip = files[-1]
        model_path = latest_zip.replace(".zip", "")
        self.logger.info(f"StrategyManager: [{folder}] 모델 탐색 완료 → {latest_zip}")

        model_dim = TradingAgentWrapper.get_model_dimension(model_path)
        detected_mode = "advanced" if model_dim >= 100 else "basic"
        dummy_config = {"symbol": "DUMMY", "feature_mode": detected_mode, "target_dim": model_dim}
        dummy_env = ScalpingTradingEnv(self.data_collector, self.order_manager, dummy_config)

        config_dict = self.config_manager.get_dict() if hasattr(self.config_manager, "get_dict") else {}
        agent_config = {"seq_len": config_dict.get("seq_len", 10)}
        agent = TradingAgentWrapper(dummy_env, agent_config)
        agent.load_weights(model_path)
        self.logger.info(f"StrategyManager: [{folder}] 모델 로딩 성공 ✅ (차원={model_dim})")
        return agent

    def load_model_from_config(self):
        try:
            config_dict = self.config_manager.get_dict() if hasattr(self.config_manager, "get_dict") else {}
            model_type = config_dict.get("live_trading_model_type", "random").lower().strip()
            self.logger.info(f"StrategyManager: Config 모델 타입 = [{model_type}]")

            if model_type == "smart":
                agent = self._load_single_model("smart")
                self.shared_agent = agent or self.shared_agent
                if agent: self.model_smart = agent
                else: self._fallback_empty_model()
            elif model_type == "dual":
                self.model_random = self._load_single_model("random")
                self.model_smart = self._load_single_model("smart")
                self.shared_agent = self.model_random or self.model_smart
                if not self.shared_agent: self._fallback_empty_model()
            else:
                agent = self._load_single_model("random")
                self.shared_agent = agent or self.shared_agent
                if agent: self.model_random = agent
                else: self._fallback_empty_model()
        except Exception as e:
            self.logger.error(f"StrategyManager: 모델 로딩 중 오류 발생 ({e}). 폴백 모드로 전환합니다.")
            self._fallback_empty_model()

    def init_engines(self, universe_list: List[Dict[str, Any]]):
        """유니버스 확정 후 실시간 매매 엔진 초기화 (Lazy Initialization)"""
        if not self.shared_agent:
            self.logger.error("StrategyManager: 엔진 초기화 실패 - 로드된 에이전트가 없습니다.")
            return

        self.symbols = [s.get("code").split('_')[0] for s in universe_list if s.get("code")]
        self.envs.clear()
        
        self.logger.info(f"StrategyManager: 확정된 유니버스 {len(self.symbols)}개에 대해 엔진 초기화 시작.")
        for sym in self.symbols:
            from core.live_trading_engine import LiveTradingEngine
            self.envs[sym] = LiveTradingEngine(sym, self.config_manager, self.order_manager, self.shared_agent)
            self.last_action_times[sym] = 0.0
            self.logger.info(f"StrategyManager: [{sym}] 실시간 엔진 결합 완료.")
        
        print(f"시스템: [SUCCESS] 총 {len(self.symbols)}개의 매매 엔진 배치 완료.")

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

        from core.scheduler import MarketState
        scheduler = getattr(self.config_manager, "_injected_scheduler", None)
        current_state = getattr(scheduler, "current_state", MarketState.OUT_OF_MARKET)
        
        if current_state == MarketState.TRADING:
            self.logger.info("StrategyManager: 장중 부팅 - 종목별 웜업(백그라운드)을 시작합니다.")
            token = self.config_manager.get("KIWOOM_ACCESS_TOKEN")
            for sym, engine in self.envs.items():
                if token:
                    asyncio.create_task(engine.warmup(token))
                    await asyncio.sleep(0.1)
        else:
            self.logger.info(f"StrategyManager: 장외 시간({current_state}) - 웜업을 생략합니다.")

        if hasattr(self.data_collector, 'on_state_updated_callbacks'):
            if self._on_tick_event not in self.data_collector.on_state_updated_callbacks:
                self.data_collector.on_state_updated_callbacks.append(self._on_tick_event)

        asyncio.create_task(self._empty_candle_watchdog())

    async def update_universe(self, new_universe: List[Dict[str, Any]]):
        async with self._swap_lock:
            new_symbols = list(set([s.get("code").split('_')[0] for s in new_universe if s.get("code")]))
            current_symbols = list(self.symbols)

            for sym in current_symbols:
                if sym not in new_symbols:
                    holdings = self.order_manager.holdings.get(sym, 0)
                    if holdings > 0: continue
                    self.symbols.remove(sym)
                    if sym in self.envs: del self.envs[sym]
                    if hasattr(self.data_collector, 'unsubscribe_symbol'):
                        await self.data_collector.unsubscribe_symbol(sym)

            for sym in new_symbols:
                if sym not in self.symbols:
                    self.symbols.append(sym)
                    from core.live_trading_engine import LiveTradingEngine
                    self.envs[sym] = LiveTradingEngine(sym, self.config_manager, self.order_manager, self.shared_agent)
                    token = self.config_manager.get("KIWOOM_ACCESS_TOKEN")
                    if token: asyncio.create_task(self.envs[sym].warmup(token))
                    if hasattr(self.data_collector, 'subscribe_symbol'):
                        await self.data_collector.subscribe_symbol(sym)

    async def _on_tick_event(self, symbol: str, normalized_state=None, price=0.0, volume=0.0, timestamp=None):
        if not self.is_running or self.is_ai_paused: return
        try:
            clean_symbol = symbol.split('_')[0]
            engine = self.envs.get(clean_symbol)
            if engine and price > 0 and timestamp:
                await engine.update_tick(price, int(volume), timestamp)
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

    async def stop(self):
        self.is_running = False
        if hasattr(self.data_collector, 'on_state_updated_callbacks'):
            if self._on_tick_event in self.data_collector.on_state_updated_callbacks:
                self.data_collector.on_state_updated_callbacks.remove(self._on_tick_event)
        self.logger.info("StrategyManager Stopped.")
