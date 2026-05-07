import os
import json
import numpy as np
import pandas as pd
import torch
from collections import deque
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QPushButton, QLabel, QLineEdit, QSpinBox,
                             QTextEdit, QFileDialog, QMessageBox)
from PyQt6.QtCore import QThread, pyqtSignal
from sb3_contrib import MaskablePPO


def _load_training_universe(config_manager, model_path: str) -> list:
    """
    훈련 시의 Universe(all_symbols)를 복원합니다.
    우선순위: 모델파일 옆 symbols.json 캐시 → InfluxDB schema.tagValues → config_manager.get_symbols()
    training_env.__init__와 동일하게 sorted() 훈 반환.
    """
    # 1. 모델 옆 캐시 파일
    cache = model_path.replace(".zip", "") + ".symbols.json"
    if os.path.exists(cache):
        try:
            with open(cache, 'r', encoding='utf-8') as f:
                syms = json.load(f)
            if isinstance(syms, list) and syms:
                return sorted(list(set(syms)))
        except Exception:
            pass

    # 2. InfluxDB 전체 심볼 조회 (Sync)
    try:
        from influxdb_client import InfluxDBClient
        url    = config_manager.get("INFLUX_URL",    "http://localhost:8086")
        token  = config_manager.get("INFLUX_TOKEN",  "")
        org    = config_manager.get("INFLUX_ORG",    "my-trade")
        bucket = config_manager.get("influx_bucket", "stock_data")
        sc     = InfluxDBClient(url=url, token=token, org=org, timeout=30000)
        qa     = sc.query_api()
        q      = f'import "influxdata/influxdb/schema" schema.tagValues(bucket: "{bucket}", tag: "symbol")'
        tables = qa.query(q, org=org)
        raw    = []
        for table in tables:
            for rec in table.records:
                v = rec.get_value()
                if v and v != "UNKNOWN":
                    raw.append(v.split('_')[0].strip())
        sc.close()
        if raw:
            syms = sorted(list(set(raw)))
            try:
                with open(cache, 'w', encoding='utf-8') as f:
                    json.dump(syms, f, ensure_ascii=False)
            except Exception:
                pass
            return syms
    except Exception:
        pass

    # 3. config_manager fallback
    try:
        raw_list = config_manager.get_symbols()
        syms = sorted(list(set(s.get("code", "") for s in raw_list if s.get("code"))))
        if syms:
            return syms
    except Exception:
        pass
    return []

# ─── 5-액션 정의 (trading_env.py 동기화) ─────────────────────
ACTION_LABELS = {
    0: ("🛑", "Hold    "),
    1: ("🟢", "Buy 40% "),
    2: ("💚", "Buy 60% "),
    3: ("🔴", "Sell 60%"),
    4: ("🟠", "Sell 40%"),
}
NUM_ACTIONS    = 5
BUY_THRESHOLD  = 0.3
SELL_THRESHOLD = 0.3

# ─── Obs 구성 상수 (trading_env.__init__ 역산과 동일) ─────────
WINDOW_SIZE     = 10
SINGLE_FEAT_DIM = 5
PORTFOLIO_DIM   = 2
INDICATOR_DIM   = 7
INITIAL_BALANCE = 10_000_000
SLIPPAGE        = 0.0005


# ════════════════════════════════════════════════════════════════
# trading_env.py 메서드 100% 복제 함수군
# ════════════════════════════════════════════════════════════════

def _compute_indicators(data):
    df = pd.DataFrame(data)
    
    close_series = pd.to_numeric(df.get('price', df.get('close', df.get('cur_prc', 0))), errors='coerce').fillna(0)
    high_series = pd.to_numeric(df.get('high', close_series), errors='coerce').fillna(0)
    low_series = pd.to_numeric(df.get('low', close_series), errors='coerce').fillna(0)
    vol_series = pd.to_numeric(df.get('volume', df.get('trde_qty', 0)), errors='coerce').fillna(0)

    sma20 = close_series.rolling(20, min_periods=1).mean()
    sma60 = close_series.rolling(60, min_periods=1).mean()

    delta = close_series.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rsi14 = 100.0 - (100.0 / (1.0 + gain / (loss + 1e-9)))

    if 'timestamp' in df.columns:
        date_str = df['timestamp'].astype(str).str[:8]
        vp = close_series * vol_series
        cum_vp = vp.groupby(date_str).cumsum()
        cum_v = vol_series.groupby(date_str).cumsum()
        vwap = cum_vp / (cum_v + 1e-9)
    else:
        vwap = close_series.copy()

    std20 = close_series.rolling(20, min_periods=1).std()
    bb_upper = sma20 + (std20 * 2)
    bb_lower = sma20 - (std20 * 2)

    tr1 = high_series - low_series
    tr2 = (high_series - close_series.shift(1)).abs()
    tr3 = (low_series - close_series.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=1).mean()

    res_df = pd.DataFrame({
        'SMA_20': sma20, 
        'SMA_60': sma60, 
        'RSI_14': rsi14,
        'VWAP': vwap,
        'BB_UPPER': bb_upper,
        'BB_LOWER': bb_lower,
        'ATR_14': atr14
    })
    res_df.ffill(inplace=True); res_df.fillna(0.0, inplace=True)
    return res_df


def _extract_single_feature(data, idx, initial_price):
    idx = min(idx, len(data) - 1)
    row, prev = data[idx], data[max(0, idx-1)]
    cp, pp    = float(row.get('price', 1000)), float(prev.get('price', 1000))
    vc, vp    = float(row.get('volume', 0)),   float(prev.get('volume', 0))
    ret       = (cp - pp) / (pp + 1e-9) * 100.0
    rel_p     = (cp - initial_price) / (initial_price + 1e-9) * 10.0
    vol_ret   = (vc - vp) / (vp + 1e-9)
    return np.array([
        np.clip(ret, -5, 5), np.clip(rel_p, -10, 10), np.clip(vol_ret, -10, 10),
        float(row.get('OIR', 0.0)), float(row.get('Volatility', 0.0))
    ], dtype=np.float32)


def _get_portfolio_state(balance, holdings, avg_entry_price, curr_p):
    net  = balance + holdings * curr_p
    pos  = float(np.clip(holdings * curr_p / (net + 1e-9), 0.0, 1.0))
    pnl  = float(np.clip((curr_p - avg_entry_price) / (avg_entry_price + 1e-9), -1.0, 1.0)) \
           if holdings > 0 and avg_entry_price > 0 else 0.0
    return np.array([pos, pnl], dtype=np.float32)


def _get_indicator_obs(idf, idx, curr_p):
    row = idf.iloc[min(idx, len(idf)-1)]
    p   = curr_p + 1e-9
    
    sma20 = float(row['SMA_20'])
    sma60 = float(row['SMA_60'])
    rsi14 = float(row['RSI_14'])
    vwap = float(row.get('VWAP', 0.0))
    bb_upper = float(row.get('BB_UPPER', 0.0))
    bb_lower = float(row.get('BB_LOWER', 0.0))
    atr14 = float(row.get('ATR_14', 0.0))
    
    return np.array([
        float(np.clip((curr_p - sma20) / p, -1.0, 1.0)),
        float(np.clip((curr_p - sma60) / p, -1.0, 1.0)),
        float(np.clip(rsi14 / 100.0, 0.0, 1.0)),
        float(np.clip((curr_p - vwap) / p, -1.0, 1.0)),
        float(np.clip((bb_upper - curr_p) / p, -1.0, 1.0)),
        float(np.clip((curr_p - bb_lower) / p, -1.0, 1.0)),
        float(np.clip(atr14 / p, 0.0, 1.0))
    ], dtype=np.float32)


def _build_obs(buf, stock_id_dim, symbol_idx, portfolio_state, indicator_obs):
    """trading_env._get_observation() 완전 동일. symbol_idx는 훈련 시 정렬된 Universe 기준."""
    features     = np.concatenate(list(buf)).astype(np.float32)
    stock_onehot = np.zeros(stock_id_dim, dtype=np.float32)
    if 0 <= symbol_idx < stock_id_dim:
        stock_onehot[symbol_idx] = 1.0
    return np.concatenate([features, stock_onehot, portfolio_state, indicator_obs])


def _simulate_action(action, curr_p, balance, holdings, avg_entry_price, net_worth):
    bp, sp = curr_p * (1 + SLIPPAGE), curr_p * (1 - SLIPPAGE)
    if action == 1:
        sh = int(min(net_worth * 0.40, balance * 0.99) / bp)
        if sh > 0:
            cost = sh * bp; prev = holdings * avg_entry_price
            holdings += sh; balance -= cost
            avg_entry_price = (prev + cost) / (holdings + 1e-9)
    elif action == 2:
        sh = int(min(net_worth * 0.60, balance * 0.99) / bp)
        if sh > 0:
            cost = sh * bp; prev = holdings * avg_entry_price
            holdings += sh; balance -= cost
            avg_entry_price = (prev + cost) / (holdings + 1e-9)
    elif action == 3:
        sq = max(1, int(holdings * 0.6))
        if holdings > 0:
            balance += sq * sp; holdings -= sq
            if holdings == 0: avg_entry_price = 0.0
    elif action == 4:
        sq = max(1, int(holdings * 0.4))
        if holdings > 0:
            balance += sq * sp; holdings -= sq
            if holdings == 0: avg_entry_price = 0.0
    return balance, holdings, avg_entry_price


def _is_pullback(idf, idx):
    row = idf.iloc[min(idx, len(idf)-1)]
    return (row['SMA_20'] > row['SMA_60']) and (row['RSI_14'] < 40.0)


# ════════════════════════════════════════════════════════════════
# Worker
# ════════════════════════════════════════════════════════════════

class PolicyCheckWorker(QThread):
    sig_finished = pyqtSignal(str)
    sig_error    = pyqtSignal(str)
    sig_log      = pyqtSignal(str)

    def __init__(self, config_manager, model_path, symbol, max_steps):
        super().__init__()
        self.config_manager = config_manager
        self.model_path     = model_path
        self.symbol         = symbol
        self.max_steps      = max_steps
        self.is_running     = True

    def stop(self): self.is_running = False

    def run(self):
        try:
            import traceback

            # ── 1. 모델 로드 ──────────────────────────────────
            self.sig_log.emit(f"📡 모델 로드 중: {os.path.basename(self.model_path)}")
            try:
                model = MaskablePPO.load(self.model_path)
            except Exception as e:
                self.sig_error.emit(f"모델 로드 실패: {e}"); return

            model_action_n = model.action_space.n
            target_dim     = model.observation_space.shape[0]
            self.sig_log.emit(
                f"✅ 모델 로드 완료 | Action:Discrete({model_action_n}) | Obs:{target_dim}"
            )

            # ── 2. Universe 복원 ───────────────────────────────
            all_syms = _load_training_universe(self.config_manager, self.model_path)
            sym_to_idx = {s: i for i, s in enumerate(all_syms)}
            symbol_idx = sym_to_idx.get(self.symbol, -1)
            FEAT_DIM     = 5 * 10
            stock_id_dim = max(1, target_dim - FEAT_DIM - 2 - 3)
            self.sig_log.emit(
                f"   Universe: {len(all_syms)}종목 | stock_id_dim={stock_id_dim} | "
                + (f"{self.symbol} → idx={symbol_idx} ✅"
                   if symbol_idx >= 0 and symbol_idx < stock_id_dim
                   else f"⚠️ '{self.symbol}' Universe 미포함 또는 idx 초과")
            )

            # ── 3. 데이터 로드 (InfluxDB Sync) ─────────────────
            self.sig_log.emit(f"📂 [{self.symbol}] 데이터 조회 중...")
            from influxdb_client import InfluxDBClient
            url    = self.config_manager.get("INFLUX_URL")
            token  = self.config_manager.get("INFLUX_TOKEN")
            org    = self.config_manager.get("INFLUX_ORG")
            bucket = self.config_manager.get("influx_bucket", "stock_data")
            clean  = self.symbol.split('_')[0].strip()

            client    = InfluxDBClient(url=url, token=token, org=org, timeout=300000)
            query_api = client.query_api()
            query = f'''
                from(bucket: "{bucket}")
                |> range(start: -1y)
                |> filter(fn: (r) => r["_measurement"] == "historical_data" or r["_measurement"] == "tick_data")
                |> filter(fn: (r) => r["symbol"] == "{self.symbol}" or r["symbol"] == "{clean}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"], desc: true)
                |> limit(n: {self.max_steps + 200})
            '''
            tables    = query_api.query(query, org=org)
            data_list = []
            for table in tables:
                for record in table.records:
                    data_list.append({
                        "timestamp": record.get_time(),
                        "price":  float(record.values.get("price",  0.0)),
                        "volume": float(record.values.get("volume", 0.0)),
                        "OIR":    float(record.values.get("OIR",    0.0)),
                        "Volatility": float(record.values.get("Volatility", 0.0)),
                    })
            client.close()

            if len(data_list) < 60:
                self.sig_error.emit(f"데이터 부족 ({len(data_list)}개)"); return

            sorted_data = sorted(data_list, key=lambda x: x["timestamp"])
            self.sig_log.emit(f"✅ 데이터 준비 완료: {len(sorted_data)}개 캔들")

            # ── 4. ScalpingTradingEnv 인스턴스화 ───────────────
            # 수동 feature 복제 없이 env 자체가 obs를 생성
            self.sig_log.emit("🏗️ ScalpingTradingEnv 생성 중...")
            from env.trading_env import ScalpingTradingEnv
            env_config = {
                "symbol":          self.symbol,
                "initial_balance": 10_000_000,
                "historical_data": sorted_data,
                "mode":            "backtest",
                "feature_mode":    "basic",
                "target_dim":      target_dim,
                "all_symbols":     all_syms,
            }
            env = ScalpingTradingEnv(None, None, env_config)
            obs, _ = env.reset()
            self.sig_log.emit(
                f"✅ Env 생성 완료 | obs.shape={obs.shape}\n"
                f"   [DEBUG] obs[:5]={np.round(obs[:5], 4)}  obs[-5:]={np.round(obs[-5:], 4)}\n"
                f"   obs범위=[{obs.min():.4f}, {obs.max():.4f}]  "
                f"(|val|>50: {(np.abs(obs)>50).sum()}개"
                + (" ⚠️raw가격섞임!" if (np.abs(obs)>50).sum() > 0 else " ✅") + ")"
            )
            self.sig_log.emit(
                f"🧠 시뮬레이션 시작 (Buy≥{BUY_THRESHOLD}, Sell≥{SELL_THRESHOLD})..."
            )

            # ── 5. 추론 루프 ───────────────────────────────────
            total_steps      = 0
            raw_counts       = {i: 0 for i in range(NUM_ACTIONS)}
            filtered_counts  = {i: 0 for i in range(NUM_ACTIONS)}
            max_probs        = {i: 0.0 for i in range(NUM_ACTIONS)}
            probs_accumulator = np.zeros(NUM_ACTIONS, dtype=np.float64)
            filter_overrides = 0
            buy_conf_list    = []
            sell_conf_list   = []
            obs_warn         = 0
            detail_logs      = []
            done             = False

            while not done and self.is_running and total_steps < self.max_steps:
                if (np.abs(obs) > 50).sum() > 0:
                    obs_warn += 1

                # 마스킹
                action_masks = env.action_masks()
                masks_arr    = np.array(action_masks, dtype=bool)
                if len(masks_arr) < model_action_n:
                    masks_arr = np.pad(masks_arr, (0, model_action_n - len(masks_arr)),
                                       constant_values=True)
                else:
                    masks_arr = masks_arr[:model_action_n]

                state_input  = np.expand_dims(obs.astype(np.float32), 0)
                raw_action, _ = model.predict(state_input, action_masks=masks_arr, deterministic=True)
                raw_action    = int(raw_action[0] if isinstance(raw_action, np.ndarray) else raw_action)

                # 확률 추출 (사용자 요청: model.policy.get_distribution)
                try:
                    obs_t, _ = model.policy.obs_to_tensor(state_input)
                    with torch.no_grad():
                        distribution = model.policy.get_distribution(obs_t)
                        if hasattr(distribution.distribution, 'probs'):
                            probs = distribution.distribution.probs.detach().cpu().numpy()[0]
                        else:
                            probs = torch.softmax(distribution.distribution.logits, -1).detach().cpu().numpy()[0]
                    if len(probs) < NUM_ACTIONS:
                        probs = np.pad(probs, (0, NUM_ACTIONS - len(probs)))
                except Exception:
                    probs = np.ones(NUM_ACTIONS) / NUM_ACTIONS

                total_steps += 1
                raw_counts[raw_action] += 1
                conf = float(probs[raw_action])
                probs_accumulator += probs[:NUM_ACTIONS]
                
                # 최대 확률 갱신
                for k in range(NUM_ACTIONS):
                    max_probs[k] = max(max_probs[k], float(probs[k]))

                # Confidence 필터
                filtered_action = raw_action
                if raw_action in (1, 2):
                    buy_conf_list.append(conf)
                    if conf < BUY_THRESHOLD: filtered_action = 0; filter_overrides += 1
                elif raw_action in (3, 4):
                    sell_conf_list.append(conf)
                    if conf < SELL_THRESHOLD: filtered_action = 0; filter_overrides += 1
                filtered_counts[filtered_action] += 1

                # 상세 로그 (신호 발생 시 또는 잠재적 신호 발견 시)
                potential_action = np.argmax(probs[1:]) + 1
                potential_prob   = probs[potential_action]
                
                if (raw_action in (1, 2, 3, 4) or potential_prob > 0.05) and len(detail_logs) < 100:
                    icon, lbl = ACTION_LABELS[raw_action]
                    omk = " ⚡→Hold" if filtered_action == 0 else ""
                    inds = env._current_indicators
                    pmk  = ""
                    if raw_action in (1, 2) and inds:
                        if inds.get('SMA_20', 0) > inds.get('SMA_60', 0) and inds.get('RSI_14', 100) < 40:
                            pmk = " 🎯눌림목!"
                    
                    ps = "  ".join(
                        f"{ACTION_LABELS.get(j,('?','?'))[1].strip()}:{probs[j]:.3f}"
                        for j in range(NUM_ACTIONS)
                    )
                    
                    curr_price = env._get_current_price()
                    log_entry = f"  {icon} step={env.current_step:4d} | {lbl.strip()} | 신뢰도:{conf:.4f} | 현재가:{curr_price:,.0f}원{omk}{pmk}\n     [{ps}]"
                    if potential_prob > 0.1 and raw_action == 0:
                        log_entry = " ⭐ [잠재] " + log_entry
                    
                    detail_logs.append(log_entry)

                # env.step (필터 적용 액션)
                obs, reward, done, truncated, info = env.step(filtered_action)
                done = done or truncated

            # ── 6. 결과 리포트 ────────────────────────────────
            if total_steps == 0:
                self.sig_error.emit("유효한 추론 스텝이 없습니다."); return

            sep = "─" * 50
            raw_lines = "\n".join(
                f"    {ACTION_LABELS[k][0]} {ACTION_LABELS[k][1]}({k}): "
                f"{raw_counts[k]:4d}회  ({raw_counts[k]/total_steps*100:5.1f}%)"
                for k in range(NUM_ACTIONS)
            )
            flt_lines = "\n".join(
                f"    {ACTION_LABELS[k][0]} {ACTION_LABELS[k][1]}({k}): "
                f"{filtered_counts[k]:4d}회  ({filtered_counts[k]/total_steps*100:5.1f}%)"
                for k in range(NUM_ACTIONS)
            )
            avg_probs = probs_accumulator / max(total_steps, 1)
            prob_lines = []
            for k in range(NUM_ACTIONS):
                icon, lbl = ACTION_LABELS[k]
                prob_lines.append(f"    {icon} {lbl}({k}): Avg:{avg_probs[k]*100:6.3f}% | Max:{max_probs[k]*100:6.3f}%")
            prob_summary = "\n".join(prob_lines)

            # 통계 변수 복구
            avg_bc = np.mean(buy_conf_list)  if buy_conf_list  else 0.0
            max_bc = np.max(buy_conf_list)   if buy_conf_list  else 0.0
            avg_sc = np.mean(sell_conf_list) if sell_conf_list else 0.0
            rbp    = (raw_counts[1] + raw_counts[2]) / total_steps * 100
            rsp    = (raw_counts[3] + raw_counts[4]) / total_steps * 100

            if raw_counts[0] == total_steps:
                concl = "🚨 100% Hold — 모델이 모든 상황을 관망으로 판단합니다.\n   (Hold 평균 확률이 99%를 넘는지 확인하세요.)"
            elif rbp > 80:
                concl = "⚠️ 정책 붕괴 위험! 매수 편향이 심각합니다."
            elif rbp + rsp < 5:
                concl = "⚠️ 과소 추론 위험! 모델이 지나치게 소극적입니다."
            elif avg_bc > 0.9:
                concl = "⚠️ 신뢰도 포화(Saturation) 의심!"
            else:
                concl = "✅ 모델 정책이 비교적 균형 잡혀 있습니다."

            result_text = (
                f"{sep}\n"
                f"  테스트 종목    : {self.symbol}\n"
                f"  총 추론 스텝   : {total_steps}\n"
                f"  Obs Dim        : {target_dim}\n"
                f"  Action Space   : Discrete({model_action_n})\n"
                f"  Obs 스케일 경고: {obs_warn}건\n"
                f"{sep}\n\n"
                f"  ▶ 전체 스텝 평균 및 최대 확률 (Softmax)\n{prob_summary}\n\n"
                f"  ▶ 순수 모델 판단 (필터 미적용)\n{raw_lines}\n\n"
                f"  ▶ Confidence 필터 적용 후 (Buy≥{BUY_THRESHOLD}, Sell≥{SELL_THRESHOLD})\n{flt_lines}\n\n"
                f"  ▶ 필터→Hold 전환: {filter_overrides}회 ({filter_overrides/total_steps*100:.1f}%)\n\n"
                f"  ▶ 매수 신뢰도 — 평균:{avg_bc:.4f}  최대:{max_bc:.4f} (n={len(buy_conf_list)})\n"
                f"  ▶ 매도 신뢰도 — 평균:{avg_sc:.4f} (n={len(sell_conf_list)})\n\n"
                f"  ▶ 매수 편향:{rbp:.1f}%  매도 편향:{rsp:.1f}%\n\n"
                f"  [진단 결론]\n  {concl}\n\n"
                f"{sep}\n"
                f"  ▶ 매수/매도 신호 상세 (최대 50건)\n"
                + ("\n".join(detail_logs) if detail_logs else "  (신호 없음)") + f"\n{sep}\n"
            )
            self.sig_finished.emit(result_text)

        except Exception as e:
            import traceback
            self.sig_error.emit(f"진단 중 오류: {e}\n{traceback.format_exc()}")



# ════════════════════════════════════════════════════════════════
# Tab UI
# ════════════════════════════════════════════════════════════════

class PolicyInspectorTab(QWidget):
    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.model_path     = ""
        self.worker         = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        ig     = QGroupBox("진단 설정")
        il     = QVBoxLayout()

        ml = QHBoxLayout()
        ml.addWidget(QLabel("진단 모델:"))
        self.lbl_model = QLabel("선택된 모델 없음")
        self.lbl_model.setStyleSheet("color:#888;font-style:italic;")
        btn = QPushButton("모델 파일 선택 (.zip)")
        btn.clicked.connect(self._on_select_model)
        ml.addWidget(self.lbl_model, 1); ml.addWidget(btn)
        il.addLayout(ml)

        pl = QHBoxLayout()
        pl.addWidget(QLabel("종목 코드:"))
        self.edit_symbol = QLineEdit("001440")
        self.edit_symbol.setPlaceholderText("예: 005930")
        pl.addWidget(self.edit_symbol)
        pl.addSpacing(20)
        pl.addWidget(QLabel("진단 스텝 수:"))
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(100, 5000); self.spin_steps.setValue(1000); self.spin_steps.setSingleStep(100)
        pl.addWidget(self.spin_steps); pl.addStretch(1)
        self.btn_run = QPushButton("🚀 진단 시작")
        self.btn_run.setFixedHeight(35)
        self.btn_run.setStyleSheet("background-color:#2b5b84;color:white;font-weight:bold;")
        self.btn_run.clicked.connect(self._on_run_diagnosis)
        pl.addWidget(self.btn_run)
        il.addLayout(pl)
        ig.setLayout(il); layout.addWidget(ig)

        og = QGroupBox("진단 결과 (Diagnosis Output)")
        ol = QVBoxLayout()
        self.txt_output = QTextEdit(); self.txt_output.setReadOnly(True)
        self.txt_output.setStyleSheet(
            "background-color:#0c0c0c;color:#dcdcdc;"
            "font-family:'Consolas','Courier New',monospace;font-size:10pt;border:1px solid #333;"
        )
        ol.addWidget(self.txt_output); og.setLayout(ol); layout.addWidget(og, 1)

    def _on_select_model(self):
        base = os.path.join(os.getcwd(), "saved_models")
        os.makedirs(base, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(self, "진단할 모델 선택", base, "Zip Files (*.zip)")
        if path:
            self.model_path = path
            self.lbl_model.setText(os.path.basename(path))
            self.lbl_model.setStyleSheet("color:#007acc;font-weight:bold;")

    def _on_run_diagnosis(self):
        if not self.model_path:
            QMessageBox.warning(self, "경고", "모델 파일을 먼저 선택하세요."); return
        sym = self.edit_symbol.text().strip()
        if not sym:
            QMessageBox.warning(self, "경고", "종목 코드를 입력하세요."); return

        self.btn_run.setEnabled(False)
        self.txt_output.clear()
        self.txt_output.append(">>> 진단 프로세스 시작...")

        self.worker = PolicyCheckWorker(
            self.config_manager, self.model_path, sym, self.spin_steps.value()
        )
        self.worker.sig_log.connect(lambda m: self.txt_output.append(f"[*] {m}"))
        self.worker.sig_finished.connect(self._on_finished)
        self.worker.sig_error.connect(self._on_error)
        self.worker.start()

    def _on_finished(self, result):
        self.txt_output.append("\n" + result)
        self.btn_run.setEnabled(True)
        QMessageBox.information(self, "완료", "모델 정책 진단이 완료되었습니다.")

    def _on_error(self, err):
        self.txt_output.append(f"\n❌ [ERROR] {err}")
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "에러", f"진단 중 오류:\n{err}")
