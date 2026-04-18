import asyncio
import time
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
        결과로 매매 기록이 담긴 DataFrame을 반환합니다.
        """
        self.is_running = True
        self.trades = []
        self.daily_pnl = []

        obs, info = env.reset()
        done = False
        truncated = False
        step = 0
        total_steps = len(df)

        initial_balance = info.get('balance', 10000000)
        current_balance = initial_balance

        # UI 업데이트용 콜백
        def _notify_progress(s: int, tot: int, pnl: float):
            if callbacks:
                for cb in callbacks:
                    cb(s, tot, pnl)

        while not done and not truncated and self.is_running and step < total_steps:
            # 1. Action Masking 적용
            action_masks = env.action_masks()

            # 2. Agent 예측
            action = agent.predict(obs, action_masks=action_masks)

            # 3. 환경 Step 실행
            next_obs, reward, done, truncated, info = env.step(action)

            # 4. 거래 발생 시 기록
            # df에서 시간, 가격 정보 추출 (여기서는 인덱스나 특정 컬럼 활용 가정)
            # 목업 코드에서는 환경에서 주는 가격이나 step 번호로 대체
            current_price = getattr(env, '_get_current_price', lambda: 1000.0)()

            if action in [1, 2]: # Buy or Sell
                trade = {
                    "step": step,
                    "price": current_price,
                    "action": "Buy" if action == 1 else "Sell",
                    "reward": reward,
                    "balance": info.get('balance', 0)
                }
                self.trades.append(trade)

            obs = next_obs
            step += 1

            if step % 100 == 0:
                _notify_progress(step, total_steps, info.get('balance', initial_balance) - initial_balance)
                await asyncio.sleep(0) # 이벤트 루프 양보 (UI 반응성 유지)

        self.is_running = False

        if self.trades:
            return pd.DataFrame(self.trades)
        return pd.DataFrame()

    def stop(self):
        self.is_running = False

class KPICalculator:
    @staticmethod
    def calculate(trades_df: pd.DataFrame, initial_balance: float = 10000000) -> Dict[str, float]:
        if trades_df is None or trades_df.empty:
            return {"Total Return": 0.0, "Win Rate": 0.0, "MDD": 0.0, "Profit Factor": 0.0}

        final_balance = trades_df.iloc[-1]['balance']
        total_return = ((final_balance - initial_balance) / initial_balance) * 100

        # 승률: 수익이 0 초과인 거래 / 전체 거래 (단순화)
        win_trades = len(trades_df[trades_df['reward'] > 0])
        total_trades = len(trades_df)
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0.0

        # Profit Factor (총수익 / 총손실)
        gross_profit = trades_df[trades_df['reward'] > 0]['reward'].sum()
        gross_loss = abs(trades_df[trades_df['reward'] < 0]['reward'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        # MDD (Max Drawdown)
        balances = trades_df['balance'].values
        running_max = np.maximum.accumulate(balances)
        drawdowns = (running_max - balances) / running_max
        mdd = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0.0

        return {
            "Total Return": total_return,
            "Win Rate": win_rate,
            "MDD": mdd,
            "Profit Factor": profit_factor
        }
