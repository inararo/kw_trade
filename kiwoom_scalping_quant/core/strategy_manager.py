import asyncio
import logging
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
        self.is_running = False

        # 쿨다운 및 락 관리 (중복 주문 방지)
        self.last_action_times: Dict[str, float] = {}
        self.cooldown_seconds = 3.0

        # 동적 유니버스 스왑을 위한 락
        self._swap_lock = asyncio.Lock()

    def load_model(self, model_path: str):
        """초기 통합 모델 생성 및 가격 로드"""
        try:
            dummy_env = ScalpingTradingEnv(self.data_collector, self.order_manager, {"symbol": "DUMMY"})
            config_dict = self.config_manager.get_dict() if hasattr(self.config_manager, "get_dict") else {}
            agent_config = {"seq_len": config_dict.get("seq_len", 10)}

            self.shared_agent = TradingAgentWrapper(dummy_env, agent_config)

            if model_path:
                self.shared_agent.load_weights(model_path)
                self.logger.info(f"StrategyManager: Shared model weights loaded from {model_path}")
            else:
                self.logger.info("StrategyManager: Running with initialized untrained weights.")

            # 종목별 독립 환경 구성
            for sym in self.symbols:
                env_config = {"symbol": sym, "initial_balance": config_dict.get("initial_balance", 10000000)}
                self.envs[sym] = ScalpingTradingEnv(self.data_collector, self.order_manager, env_config)
                self.last_action_times[sym] = 0.0

        except Exception as e:
            self.logger.error(f"StrategyManager 초기화 중 에러: {e}")

    async def start(self):
        """이벤트 드리븐 전략 루프 시작"""
        if not self.shared_agent:
            self.logger.warning("Agent가 로드되지 않았습니다. 매매 루프를 시작할 수 없습니다.")
            return

        self.is_running = True
        self.logger.info(f"StrategyManager: 멀티 종목({len(self.symbols)}개) 이벤트 드리븐 오케스트레이션 시작.")

        # DataCollector의 상태 업데이트 콜백 리스트에 등록
        if hasattr(self.data_collector, 'on_state_updated_callbacks'):
            if self._on_tick_event not in self.data_collector.on_state_updated_callbacks:
                self.data_collector.on_state_updated_callbacks.append(self._on_tick_event)
                self.logger.info("StrategyManager: DataCollector 이벤트 구독 완료.")
        else:
            self.logger.error("StrategyManager: DataCollector에 이벤트 리스너 리스트가 없습니다.")

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
                    env_config = {"symbol": sym, "initial_balance": config_dict.get("initial_balance", 10000000)}
                    self.envs[sym] = ScalpingTradingEnv(self.data_collector, self.order_manager, env_config)
                    self.last_action_times[sym] = 0.0

                    if hasattr(self.data_collector, 'subscribe_symbol'):
                        await self.data_collector.subscribe_symbol(sym)

                    vm = getattr(self.config_manager, "_injected_live_vm", None)
                    if vm: vm.sig_log_appended.emit(f"[UNIVERSE UPDATE] IN: {sym}")

            # Config 반영
            updated_dicts = [{"code": s, "name": f"Dynamic_{s}"} for s in self.symbols]
            self.config_manager.set_symbols(updated_dicts)

    async def _on_tick_event(self, symbol: str, normalized_state=None):
        """데이터 수신 시 호출되는 핵심 리스너 (Event-Driven)"""
        if not self.is_running:
            return

        try:
            # 1. 상태 및 쿨다운 체크
            from core.scheduler import MarketState
            scheduler = getattr(self.config_manager, "_injected_scheduler", None)
            if scheduler and scheduler.current_state != MarketState.TRADING:
                return

            current_time = time.time()
            if current_time - self.last_action_times.get(symbol, 0.0) < self.cooldown_seconds:
                return

            env = self.envs.get(symbol)
            if not env: return

            # 2. 추론 수행
            seq_len = self.shared_agent.seq_len
            
            # [버그 수정] 실제 쌓인 데이터가 seq_len에 도달했는지 먼저 확인
            # 기존의 np.all(obs == 0) 체크는 패딩된 non-zero 배열을 통과시키는 허점이 있었음
            actual_buffer = self.data_collector.state_buffers.get(symbol, [])
            if len(actual_buffer) < seq_len:
                return  # 데이터 충분히 쌓이지 않으면 추론하지 않음
            
            obs = self.data_collector.get_latest_state(symbol, seq_len=seq_len)
            if np.all(obs == 0): return  # 이중 방어

            action_masks = env.action_masks()
            obs_batch = np.expand_dims(obs, axis=0)
            
            # 신뢰도(probs)와 함께 추론
            result = self.shared_agent.predict(obs_batch, action_masks=action_masks, return_probs=True)
            action, probs = result
            if isinstance(action, np.ndarray): action = int(action[0])

            # [버그 수정] 확률 분포가 거의 동일한 경우(학습 초기/랜덤 상태)는 Hold로 강제
            # 가장 높은 확률이 임계값(예: 50%) 이상일 때만 액션을 신뢰함
            MIN_ACTION_CONFIDENCE = 0.50
            max_prob = max(probs)
            if max_prob < MIN_ACTION_CONFIDENCE:
                action = 0  # 신뢰도 부족 → Hold 강제

            # AI 신뢰도 UI 업데이트 (0: Hold, 1: Buy, 2: Sell)
            confidence_dict = {
                "Hold": int(probs[0] * 100),
                "Buy": int(probs[1] * 100),
                "Sell": int(probs[2] * 100)
            }
            
            # 결정된 신호 텍스트
            signal_text = "Hold"
            if action == 1: signal_text = "Buy"
            elif action == 2: signal_text = "Sell"

            vm = getattr(self.config_manager, "_injected_live_vm", None)
            if vm:
                # 1. 요약 정보 업데이트 (대시보드 테이블용)
                if symbol not in vm.symbols_summary:
                    vm.symbols_summary[symbol] = {"price": 0, "ai_signal": "-", "holdings": 0}
                
                vm.symbols_summary[symbol]["ai_signal"] = signal_text
                vm.sig_symbols_summary_updated.emit(vm.symbols_summary)

                # 2. 상세 시각화 업데이트 (선택된 종목이거나 선택이 없을 때)
                if vm.selected_symbol == symbol or not vm.selected_symbol:
                    vm.sig_ai_confidence_updated.emit(confidence_dict)

            # 3. Action 수행 (1: BUY, 2: SELL)
            if action in [1, 2]:
                str_action = "BUY" if action == 1 else "SELL"
                current_price = self.data_collector.get_latest_price(symbol)
                if current_price <= 0: return

                if action == 1: # BUY - 동적 수량 계산
                    max_invest = self.risk_manager.get_max_invest_per_symbol()
                    qty = int(max_invest // current_price)
                    
                    if qty <= 0:
                        msg = f"[SYSTEM] {symbol} 매수 신호 발생했으나, 1주 가격({current_price:,.0f}원)이 설정된 최대 한도({max_invest:,.0f}원)를 초과하여 매수를 생략합니다."
                        self.logger.warning(msg)
                        vm = getattr(self.config_manager, "_injected_live_vm", None)
                        if vm: vm.sig_log_appended.emit(msg)
                        return
                    target_qty = qty
                else: # SELL
                    target_qty = self.order_manager.holdings.get(symbol, 0)

                if target_qty > 0:
                    self.logger.error(f"StrategyManager: [{symbol}] 에이전트 결단 - {str_action} {target_qty}주")
                    self.last_action_times[symbol] = current_time
                    asyncio.create_task(
                        self.order_manager.execute_smart_order(str_action, symbol, target_qty, self.data_collector)
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
