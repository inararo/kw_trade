import asyncio
import logging
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# ─────────────────────────────────────────────────────
# 5-액션 정의 (trading_env.py와 동기화)
# 0:Hold  1:Buy40%  2:Buy60%  3:Sell60%  4:Sell40%
# ─────────────────────────────────────────────────────
ACTION_MAP = {
    0: "Hold",
    1: "Buy40%",
    2: "Buy60%",
    3: "Sell60%",
    4: "Sell40%",
}
BUY_ACTIONS  = (1, 2)
SELL_ACTIONS = (3, 4)
BUY_THRESHOLD  = 0.6
SELL_THRESHOLD = 0.6


def _compute_indicators(data: list) -> pd.DataFrame:
    """
    [보조지표 자동 계산] trading_env._compute_indicators()와 완전히 동일한 로직.
    SMA_20 / SMA_60 / RSI_14 계산 후 ffill → 0 채움.
    반환: 원본 데이터와 인덱스가 1:1 대응되는 DataFrame
    """
    prices = pd.Series([float(d.get('price', 0)) for d in data], dtype=np.float64)

    sma20 = prices.rolling(window=20, min_periods=1).mean()
    sma60 = prices.rolling(window=60, min_periods=1).mean()

    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
    rs    = gain / (loss + 1e-9)
    rsi14 = 100.0 - (100.0 / (1.0 + rs))

    df = pd.DataFrame({'SMA_20': sma20, 'SMA_60': sma60, 'RSI_14': rsi14})
    df.ffill(inplace=True)
    df.fillna(0.0, inplace=True)
    return df


class BacktestEngine:
    """
    비동기 백테스팅 엔진.
    에이전트와 환경을 주입받아 메인 UI 스레드 블로킹 없이
    과거 데이터에 대해 시뮬레이션을 수행합니다.
    5-액션(Hold / Buy40% / Buy60% / Sell60% / Sell40%) 완전 대응.
    """
    def __init__(self, data_collector, config: Dict[str, Any]):
        self.data_collector = data_collector
        self.config = config
        self.is_running = False
        self.trades  = []
        self.history = []

    async def run_backtest(
        self,
        agent,
        env,
        df: pd.DataFrame,
        callbacks: List[Any] = None
    ) -> pd.DataFrame:
        """
        주어진 DataFrame과 에이전트를 기반으로 Env 위에서 백테스트를 수행합니다.
        결과로 모든 스텝의 기록이 담긴 DataFrame을 반환합니다.
        """
        self.is_running = True
        self.history = []

        # ── 보조지표 계산 (Env 내부와 동일한 로직) ──────────
        logger = logging.getLogger("BacktestEngine")
        raw_data = getattr(env, 'historical_data', None)
        indicator_df = None
        if raw_data:
            indicator_df = _compute_indicators(raw_data)
            logger.debug(f"보조지표 계산 완료 ({len(indicator_df)}행)")

        # ── 환경 초기화 ──────────────────────────────────
        obs, info = env.reset()
        done      = False
        step      = 0
        total_steps = len(df)
        initial_balance = info.get('net_worth', info.get('balance', 10000000))

        def _notify(s, tot, pnl):
            if callbacks:
                for cb in callbacks:
                    cb(s, tot, pnl)

        while not done and self.is_running and step < total_steps:

            # ── 1. Action Masking ────────────────────────
            action_masks = env.get_wrapper_attr('action_masks')()

            # ── 2. 예측 (Confidence 필터 포함) ───────────
            action, probs = self._predict_with_confidence(agent, obs, action_masks)

            # Confidence 필터 (매수/매도 계열 분리 적용)
            if action in BUY_ACTIONS and float(probs[action]) < BUY_THRESHOLD:
                action = 0
            elif action in SELL_ACTIONS and float(probs[action]) < SELL_THRESHOLD:
                action = 0

            # ── 3. 환경 Step ─────────────────────────────
            next_obs, reward, done, truncated, info = env.step(action)

            # ── 4. 기록 ──────────────────────────────────
            current_price  = getattr(env, '_get_current_price', lambda: 1000.0)()
            action_executed = info.get("action_executed", action)

            # 보조지표 값 추가 (디버깅·분석용)
            indic_row = {}
            if indicator_df is not None and step < len(indicator_df):
                row = indicator_df.iloc[step]
                indic_row = {
                    "SMA_20": round(float(row['SMA_20']), 2),
                    "SMA_60": round(float(row['SMA_60']), 2),
                    "RSI_14": round(float(row['RSI_14']), 2),
                }

            self.history.append({
                "step":    step,
                "price":   current_price,
                "action":  ACTION_MAP.get(action_executed, "Hold"),
                "reward":  reward,
                "balance": info.get('net_worth', initial_balance),
                # 분할 매매 분석을 위한 추가 필드
                "holdings":          info.get('holdings', 0),
                "unrealized_pnl_pct": info.get('unrealized_pnl_pct', 0.0),
                **indic_row,
            })

            # 백테스트는 날짜가 바뀌어도 Env 초기화하지 않음
            # (LSTM 버퍼의 시계열 연속성 보존)
            obs = next_obs
            step += 1

            if step % 100 == 0:
                _notify(step, total_steps, info.get('net_worth', initial_balance) - initial_balance)
                await asyncio.sleep(0)

        self.is_running = False
        return pd.DataFrame(self.history)

    # ──────────────────────────────────────────────────────────
    # 예측 헬퍼 (5-액션 대응 / probs 배열 반환)
    # ──────────────────────────────────────────────────────────
    def _predict_with_confidence(self, agent, obs, action_masks=None):
        """
        에이전트로부터 액션과 전체 확률 배열(probs)을 반환.
        반환: (action: int, probs: np.ndarray[5])
        """
        n_actions = 5
        fallback_probs = np.ones(n_actions) / n_actions

        try:
            if hasattr(agent, 'model') and hasattr(agent, 'predict'):
                result = agent.predict(obs, action_masks=action_masks, return_probs=True)
                if isinstance(result, tuple) and len(result) == 2:
                    action, probs = result
                    # probs가 스칼라거나 길이가 부족하면 패딩
                    probs = np.atleast_1d(np.array(probs, dtype=np.float32))
                    if len(probs) < n_actions:
                        probs = np.pad(probs, (0, n_actions - len(probs)))
                    return int(action), probs
                return int(result), fallback_probs

            if hasattr(agent, 'policy'):
                obs_t = torch.as_tensor(obs).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    if action_masks is not None and hasattr(agent.policy, "get_distribution"):
                        masks_t = torch.as_tensor(action_masks).unsqueeze(0).to(agent.device)
                        latent_pi, _, latent_sde = agent.policy._get_latent(obs_t)
                        dist = agent.policy._get_action_dist_from_latent(latent_pi, latent_sde)
                        dist.apply_masking(masks_t)
                        probs = dist.distribution.probs.cpu().numpy()[0]
                    else:
                        dist  = agent.policy.get_distribution(obs_t)
                        probs = dist.distribution.probs.cpu().numpy()[0]
                action = int(probs.argmax())
                if len(probs) < n_actions:
                    probs = np.pad(probs, (0, n_actions - len(probs)))
                return action, probs

            result = agent.predict(obs, action_masks=action_masks)
            action = int(result[0]) if isinstance(result, tuple) else int(result)
            return action, fallback_probs

        except Exception:
            try:
                result = agent.predict(obs, action_masks=action_masks)
                action = result[0] if isinstance(result, tuple) else result
                return int(action), fallback_probs
            except Exception:
                return 0, fallback_probs

    def stop(self):
        self.is_running = False

    # ──────────────────────────────────────────────────────────
    # 배치 실행
    # ──────────────────────────────────────────────────────────
    async def run_automation_batch(
        self,
        agent_builder_cb,
        env_builder_cb,
        symbol_list: List[str],
        start_date: str,
        end_date: str,
        progress_cb=None
    ) -> List[Dict[str, Any]]:
        """여러 종목에 대해 독립적으로 백테스트를 실행하는 배치 프로세스."""
        batch_results = []
        total = len(symbol_list)

        for i, symbol in enumerate(symbol_list):
            try:
                if progress_cb:
                    progress_cb(i, total, f"[{symbol}] 데이터 로딩 중...")

                env, df = await env_builder_cb(symbol, start_date, end_date)
                if env is None or df is None or df.empty:
                    raise ValueError(f"데이터가 없거나 환경 생성 실패: {symbol}")

                agent = agent_builder_cb(env)

                def inner_cb(step, tot, pnl):
                    if progress_cb:
                        progress_cb(i, total, f"[{symbol}] 진행 중... {step}/{tot}")

                history_df = await self.run_backtest(agent, env, df, callbacks=[inner_cb])
                kpi = KPICalculator.calculate(history_df)

                buy_actions  = ['Buy40%', 'Buy60%']
                sell_actions = ['Sell60%', 'Sell40%']
                total_trades = len(history_df[history_df['action'].isin(buy_actions + sell_actions)])

                batch_results.append({
                    "Symbol":           symbol,
                    "Start Date":       start_date,
                    "End Date":         end_date,
                    "Total Return (%)": round(kpi.get("Total Return", 0), 2),
                    "Win Rate (%)":     round(kpi.get("Win Rate", 0), 2),
                    "MDD (%)":          round(kpi.get("MDD", 0), 2),
                    "Profit Factor":    round(kpi.get("Profit Factor", 0), 3),
                    "Total Trades":     total_trades,
                })

            except Exception as e:
                logging.getLogger("BacktestEngine").error(f"[{symbol}] 배치 테스트 에러: {e}")
                batch_results.append({
                    "Symbol": symbol, "Start Date": start_date, "End Date": end_date,
                    "Total Return (%)": 0, "Win Rate (%)": 0, "MDD (%)": 0,
                    "Profit Factor": 0, "Total Trades": 0, "Error": str(e)
                })

        return batch_results


class KPICalculator:
    """
    백테스트 결과 DataFrame으로부터 KPI를 계산합니다.
    5-액션 분할 매매(Buy40%/Buy60%/Sell60%/Sell40%) 완전 대응.
    """

    @staticmethod
    def calculate(history_df: pd.DataFrame, initial_balance: float = 10000000) -> Dict[str, float]:
        if history_df is None or history_df.empty:
            return {
                "Total Return": 0.0, "Win Rate": 0.0,
                "MDD": 0.0, "Profit Factor": 0.0,
            }

        # ── 1. 총 수익률 ──────────────────────────────
        final_balance = history_df.iloc[-1]['balance']
        total_return  = (final_balance - initial_balance) / initial_balance * 100

        # ── 2. 승률 & Profit Factor ───────────────────
        # 분할 매도(Sell60%, Sell40%) 시점마다 reward가 기록됨
        # reward가 거래 수익에 직결되는 값은 아니므로,
        # '매도 체결 시 balance 증분'을 실현 손익으로 사용
        sell_mask = history_df['action'].isin(['Sell60%', 'Sell40%'])
        sell_rows = history_df[sell_mask].copy()

        if len(sell_rows) > 0:
            # 각 매도 직전과 직후의 balance 차이 = 실현 손익 근사
            # (env.step()이 net_worth = balance + holdings*price를 반환하므로
            #  매도 순간의 reward 부호를 승패 판단에 사용)
            sell_rows = sell_rows.copy()
            win_trades  = len(sell_rows[sell_rows['reward'] > 0])
            total_sells = len(sell_rows)
            win_rate    = win_trades / total_sells * 100

            gross_profit = sell_rows[sell_rows['reward'] > 0]['reward'].sum()
            gross_loss   = abs(sell_rows[sell_rows['reward'] < 0]['reward'].sum())
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        else:
            win_rate      = 0.0
            profit_factor = 0.0

        # ── 3. MDD (Max Drawdown) ─────────────────────
        balances    = history_df['balance'].values
        running_max = np.maximum.accumulate(balances)
        drawdowns   = (running_max - balances) / (running_max + 1e-9)
        mdd         = float(np.max(drawdowns) * 100) if len(drawdowns) > 0 else 0.0

        # ── 4. 분할 매매 횟수 분석 ────────────────────
        action_counts = history_df['action'].value_counts()

        return {
            "Total Return":  total_return,
            "Win Rate":      win_rate,
            "MDD":           mdd,
            "Profit Factor": profit_factor,
            # 상세 액션 카운트 (배치 결과 집계용)
            "Buy40_Count":   int(action_counts.get('Buy40%',  0)),
            "Buy60_Count":   int(action_counts.get('Buy60%',  0)),
            "Sell60_Count":  int(action_counts.get('Sell60%', 0)),
            "Sell40_Count":  int(action_counts.get('Sell40%', 0)),
            "Hold_Count":    int(action_counts.get('Hold',    0)),
        }
