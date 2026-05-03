import asyncio
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List

class BacktestEngine:
    """
    비동기 백테스팅 엔진.
    에이전트와 환경을 주입받아 메인 UI 스레드 블로킹 없이 과거 데이터에 대해 시뮬레이션을 수행합니다.
    """
    def __init__(self, data_collector, config: Dict[str, Any]):
        self.data_collector = data_collector
        self.config = config
        self.is_running = False
        self.trades = [] # 매매 기록: [{"time": ts, "price": p, "action": a, "pnl": pnl, "cum_pnl": cum_pnl}, ...]
        self.daily_pnl = [] # 일별 손익 기록

    async def run_backtest(self, agent, env, df: pd.DataFrame, callbacks: List[Any] = None) -> pd.DataFrame:
        """
        주어진 데이터프레임과 에이전트를 기반으로 환경(Env) 위에서 백테스트 시뮬레이션을 돌립니다.
        결과로 모든 스텝의 기록이 담긴 DataFrame을 반환합니다.
        """
        self.is_running = True
        self.history = [] # 모든 스텝의 기록: [{"step": s, "price": p, "action": a, "reward": r, "balance": b}, ...]

        obs, info = env.reset()
        done = False
        truncated = False
        step = 0
        total_steps = len(df)

        # [FIX] 초기 자본금 기준을 현금이 아닌 '총자산 가치'로 선언
        initial_balance = info.get('net_worth', info.get('balance', 10000000))

        # UI 업데이트용 콜백
        def _notify_progress(s: int, tot: int, pnl: float):
            if callbacks:
                for cb in callbacks:
                    cb(s, tot, pnl)

        while not done and self.is_running and step < total_steps:
            # 1. Action Masking 적용
            action_masks = env.get_wrapper_attr('action_masks')()

            # 2. Agent 예측 (Confidence 필터링 적용)
            action, confidence = self._predict_with_confidence(agent, obs, action_masks)
            
            # [규칙] 매수(1) 예측 시 확률이 60% 미만이면 강제 Hold(0)
            if action == 1 and confidence < 0.6:
                action = 0

            # 3. 환경 Step 실행
            next_obs, reward, done, truncated, info = env.step(action)

            # 4. 정보 기록
            current_price = getattr(env, '_get_current_price', lambda: 1000.0)()
            action_executed = info.get("action_executed", action)
            action_map = {0: "Hold", 1: "Buy", 2: "Sell"}
            
            self.history.append({
                "step": step,
                "price": current_price,
                "action": action_map.get(action_executed, "Hold"),
                "reward": reward,
                "balance": info.get('net_worth', initial_balance)
            })

            # [FIX] 백테스트 시 시계열 연속성을 위해 중간 리셋 로직 제거
            # 날짜가 바뀌더라도(truncated) 환경을 초기화하지 않고 그대로 시점(obs)을 이어가서 LSTM 버퍼를 보존함
            obs = next_obs
            
            step += 1

            if step % 100 == 0:
                _notify_progress(step, total_steps, info.get('net_worth', initial_balance) - initial_balance)
                await asyncio.sleep(0) # 이벤트 루프 양보

        self.is_running = False
        return pd.DataFrame(self.history)

    def _predict_with_confidence(self, agent, obs, action_masks=None):
        """에이전트로부터 액션과 해당 액션의 확률(Confidence)을 추출"""
        try:
            # 1. TradingAgentWrapper인 경우 (우리가 만든 래퍼 클래스)
            if hasattr(agent, 'model') and hasattr(agent, 'predict'):
                result = agent.predict(obs, action_masks=action_masks, return_probs=True)
                if isinstance(result, tuple) and len(result) == 2:
                    action, probs = result
                    confidence = float(probs[action])
                    return int(action), confidence
                else:
                    return int(result), 1.0

            # 2. SB3 모델인 경우 직접 policy 활용
            if hasattr(agent, 'policy'):
                import torch
                obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(agent.device)
                
                with torch.no_grad():
                    # MaskablePPO 여부 확인
                    if action_masks is not None and hasattr(agent.policy, "get_distribution"):
                        masks_tensor = torch.as_tensor(action_masks).unsqueeze(0).to(agent.device)
                        # MaskablePPO의 경우 masking이 적용된 분포를 가져옴
                        latent_pi, _, latent_sde = agent.policy._get_latent(obs_tensor)
                        distribution = agent.policy._get_action_dist_from_latent(latent_pi, latent_sde)
                        distribution.apply_masking(masks_tensor)
                        probs = distribution.distribution.probs.cpu().numpy()[0]
                    else:
                        # 일반 PPO
                        dist = agent.policy.get_distribution(obs_tensor)
                        probs = dist.distribution.probs.cpu().numpy()[0]
                
                action = int(probs.argmax())
                confidence = float(probs[action])
                return action, confidence
            else:
                # 일반 객체인 경우 (0-d array 언패킹 방지)
                result = agent.predict(obs, action_masks=action_masks)
                if isinstance(result, tuple):
                    return int(result[0]), 1.0
                return int(result), 1.0
        except Exception as e:
            # 에러 시 기본 추론으로 폴백
            try:
                result = agent.predict(obs, action_masks=action_masks)
                action = result[0] if isinstance(result, tuple) else result
                return int(action), 0.5
            except:
                return 0, 0.0

    def stop(self):
        self.is_running = False

    async def run_automation_batch(self, agent_builder_cb, env_builder_cb, symbol_list: List[str], 
                                  start_date: str, end_date: str, progress_cb=None) -> List[Dict[str, Any]]:
        """
        [NEW] 여러 종목에 대해 독립적으로 백테스트를 실행하는 배치 프로세스.
        특정 종목 에러 시에도 중단되지 않고 다음 종목으로 넘어갑니다.
        """
        batch_results = []
        total_symbols = len(symbol_list)

        for i, symbol in enumerate(symbol_list):
            try:
                if progress_cb:
                    progress_cb(i, total_symbols, f"[{symbol}] 데이터 로딩 중...")

                # 1. 환경 및 에이전트 생성 (콜백 활용)
                env, df = await env_builder_cb(symbol, start_date, end_date)
                if env is None or df is None or df.empty:
                    raise ValueError(f"데이터가 없거나 환경 생성 실패: {symbol}")

                agent = agent_builder_cb(env)

                # 2. 백테스트 실행
                # 내부 run_backtest 활용 (콜백은 배치 진행 상황 위주로 업데이트)
                def inner_cb(step, total, pnl):
                    if progress_cb:
                        progress_cb(i, total_symbols, f"[{symbol}] 진행 중... {step}/{total}")

                history_df = await self.run_backtest(agent, env, df, callbacks=[inner_cb])

                # 3. KPI 계산
                kpi = KPICalculator.calculate(history_df)
                total_trades = len(history_df[history_df['action'].isin(['Buy', 'Sell'])])
                
                # 결과 수집
                result_row = {
                    "Symbol": symbol,
                    "Start Date": start_date,
                    "End Date": end_date,
                    "Total Return (%)": round(kpi.get("Total Return", 0), 2),
                    "Win Rate (%)": round(kpi.get("Win Rate", 0), 2),
                    "MDD (%)": round(kpi.get("MDD", 0), 2),
                    "Profit Factor": round(kpi.get("Profit Factor", 0), 3),
                    "Total Trades": total_trades
                }
                batch_results.append(result_row)

            except Exception as e:
                import logging
                logging.getLogger("BacktestEngine").error(f"[{symbol}] 배치 테스트 중 에러 발생: {e}")
                # 에러 발생 시에도 결과 리스트에 실패 기록을 남겨 행 개수를 맞춤
                batch_results.append({
                    "Symbol": symbol,
                    "Start Date": start_date,
                    "End Date": end_date,
                    "Total Return (%)": 0,
                    "Win Rate (%)": 0,
                    "MDD (%)": 0,
                    "Profit Factor": 0,
                    "Total Trades": 0,
                    "Error": str(e)
                })

        return batch_results

class KPICalculator:
    @staticmethod
    def calculate(history_df: pd.DataFrame, initial_balance: float = 10000000) -> Dict[str, float]:
        if history_df is None or history_df.empty:
            return {"Total Return": 0.0, "Win Rate": 0.0, "MDD": 0.0, "Profit Factor": 0.0}

        # 1. 총 수익률 (마지막 잔고 기준)
        final_balance = history_df.iloc[-1]['balance']
        total_return = ((final_balance - initial_balance) / initial_balance) * 100

        # 2. 매매 기록 필터링 (완료된 매매인 Sell 기반으로 성과 측정)
        sell_trades = history_df[history_df['action'] == 'Sell']
        
        # 승률: Sell 시점의 reward가 양수인 경우를 승리로 판단
        win_trades = len(sell_trades[sell_trades['reward'] > 0])
        total_sells = len(sell_trades)
        win_rate = (win_trades / total_sells * 100) if total_sells > 0 else 0.0

        # Profit Factor (매도 시 발생한 총수익 / 총손실)
        gross_profit = sell_trades[sell_trades['reward'] > 0]['reward'].sum()
        gross_loss = abs(sell_trades[sell_trades['reward'] < 0]['reward'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        # 3. MDD (Max Drawdown) - 전체 이력의 balance 기준
        balances = history_df['balance'].values
        running_max = np.maximum.accumulate(balances)
        drawdowns = (running_max - balances) / running_max
        mdd = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0.0

        return {
            "Total Return": total_return,
            "Win Rate": win_rate,
            "MDD": mdd,
            "Profit Factor": profit_factor
        }
