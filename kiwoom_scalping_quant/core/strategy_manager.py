import asyncio
import logging
from typing import List, Dict, Any
import numpy as np

from env.trading_env import ScalpingTradingEnv
from models.agent import TradingAgentWrapper

class StrategyManager:
    """
    여러 종목(Multi-Symbol)의 트레이딩을 동시에 오케스트레이션하는 관리자 클래스.
    메모리 최적화를 위해 단일(Shared) Agent 모델을 유지하고,
    종목별로 별도의 TradingEnv 인스턴스를 생성하여 State를 추출/추론합니다.
    각 종목의 에러가 다른 종목에 영향을 주지 않도록 격리(Isolation) 처리합니다.
    """
    def __init__(self, config_manager, data_collector, order_manager):
        self.config_manager = config_manager
        self.data_collector = data_collector
        self.order_manager = order_manager
        self.logger = logging.getLogger("StrategyManager")

        self.symbols: List[str] = [s.get('code') for s in self.config_manager.get_symbols()]
        if not self.symbols:
            self.symbols = ['005930'] # Fallback

        self.envs: Dict[str, ScalpingTradingEnv] = {}
        self.shared_agent: TradingAgentWrapper = None
        self.is_running = False
        self._tasks: List[asyncio.Task] = []

    def load_model(self, model_path: str):
        """
        초기 통합 모델 생성 및 가중치 로드
        (Env는 더미 하나를 넣어서 AgentWrapper를 인스턴스화하고 weights를 로드합니다)
        """
        try:
            dummy_env = ScalpingTradingEnv(self.data_collector, self.order_manager, {"symbol": "DUMMY"})
            # Config에서 seq_len 등 추출
            config_dict = self.config_manager.get_dict() if hasattr(self.config_manager, "get_dict") else {}
            agent_config = {"seq_len": config_dict.get("seq_len", 10)}

            self.shared_agent = TradingAgentWrapper(dummy_env, agent_config)

            # 실제 모델 가중치가 있다면 로드
            if model_path:
                self.shared_agent.load_weights(model_path)
                self.logger.info(f"StrategyManager: Shared model weights loaded from {model_path}")
            else:
                self.logger.info("StrategyManager: Running with initialized untrained weights (No model path provided).")

            # 종목별 독립 환경 구성
            for sym in self.symbols:
                env_config = {"symbol": sym, "initial_balance": config_dict.get("initial_balance", 10000000)}
                self.envs[sym] = ScalpingTradingEnv(self.data_collector, self.order_manager, env_config)

        except Exception as e:
            self.logger.error(f"StrategyManager 초기화 중 에러: {e}")

    async def start(self):
        """오케스트레이션 루프 시작"""
        if not self.shared_agent:
            self.logger.warning("Agent가 로드되지 않았습니다. 매매 루프를 시작할 수 없습니다.")
            return

        self.is_running = True
        self.logger.info(f"StrategyManager: 멀티 종목({len(self.symbols)}개) 오케스트레이션 시작.")

        # 각 종목별로 비동기 무한 루프 태스크(마이크로스레드) 생성
        for sym in self.symbols:
            task = asyncio.create_task(self._run_symbol_loop(sym))
            self._tasks.append(task)

        # 모든 태스크 대기 (오류 발생 시에도 개별적으로 무시/재시작되도록 묶어둠)
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _run_symbol_loop(self, symbol: str):
        """특정 종목에 대한 독립적인 추론 및 주문 집행 루프"""
        env = self.envs.get(symbol)
        if not env:
            return

        # 루프 주기 제어용
        poll_interval = 1.0 # 1초마다 상태 확인 및 액션 추론

        while self.is_running:
            try:
                # 1. State 조회 (DataCollector에서 해당 종목의 정규화된 롤링 버퍼 획득)
                seq_len = self.shared_agent.seq_len
                obs = self.data_collector.get_latest_state(symbol, seq_len=seq_len)

                # 2. Action Mask 계산 (미체결 주문 여부, 잔고 등)
                action_masks = env.action_masks()

                # 3. Model Inference (추론)
                # numpy 배열 차원 맞춤 (1, seq_len * feature_dim)
                obs_batch = np.expand_dims(obs, axis=0)
                action = self.shared_agent.predict(obs_batch, action_masks=action_masks)

                # SB3가 1D 배열을 반환할 수 있으므로 언패킹
                if isinstance(action, np.ndarray):
                    action = int(action[0])

                # 4. Action 집행
                if action in [1, 2]: # 1: Buy, 2: Sell
                    str_action = "BUY" if action == 1 else "SELL"

                    # 수량 로직 (일단 임시로 10주 또는 보유량 전량)
                    target_qty = 10 if action == 1 else self.order_manager.holdings.get(symbol, 0)

                    if target_qty > 0:
                        self.logger.info(f"StrategyManager: [{symbol}] 에이전트 결단 - {str_action} {target_qty}주")
                        # execute_smart_order는 Cancel & Replace 기능이 있으므로 asyncio.create_task로 분리 실행 (비동기 병렬)
                        asyncio.create_task(
                            self.order_manager.execute_smart_order(str_action, symbol, target_qty, self.data_collector)
                        )

            except asyncio.CancelledError:
                self.logger.info(f"StrategyManager: [{symbol}] 루프 중지됨.")
                break
            except Exception as e:
                # 오류 격리(Isolation): 한 종목의 오류가 다른 종목에 영향을 미치지 않도록 함
                self.logger.error(f"StrategyManager: [{symbol}] 매매 루프 중 에러 발생: {e}")

            await asyncio.sleep(poll_interval)

    async def stop(self):
        """모든 종목의 매매 루프 정지"""
        self.is_running = False
        for task in self._tasks:
            if not task.done():
                task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self.logger.info("StrategyManager: 모든 매매 루프가 안전하게 종료되었습니다.")
