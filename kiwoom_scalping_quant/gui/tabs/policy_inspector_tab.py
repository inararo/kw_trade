import os
import numpy as np
import pandas as pd
import torch
from collections import deque
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QPushButton, QLabel, QLineEdit, QSpinBox,
                             QTextEdit, QFileDialog, QMessageBox)
from PyQt6.QtCore import QThread, pyqtSignal
from sb3_contrib import MaskablePPO

from core.feature_engineer import AdvancedFeatureEngineer
from models.lstm_extractor import OnlineRollingNormalizer

# ─────────────────────────────────────────────────────────────
# 5-액션 정의 (trading_env.py / check_model_policy.py 와 동기화)
# 0:Hold  1:Buy40%  2:Buy60%  3:Sell60%  4:Sell40%
# ─────────────────────────────────────────────────────────────
ACTION_LABELS = {
    0: ("🛑", "Hold    "),
    1: ("🟢", "Buy 40% "),
    2: ("💚", "Buy 60% "),
    3: ("🔴", "Sell 60%"),
    4: ("🟠", "Sell 40%"),
}
NUM_ACTIONS    = 5
BUY_THRESHOLD  = 0.6
SELL_THRESHOLD = 0.6


def _compute_indicators(data: list) -> pd.DataFrame:
    """
    [보조지표 자동 계산] trading_env._compute_indicators()와 완전히 동일한 로직.
    SMA_20 / SMA_60 / RSI_14 계산 후 ffill → 0 채움.
    """
    prices = pd.Series([float(d.get('price', 0)) for d in data], dtype=np.float64)
    sma20  = prices.rolling(window=20, min_periods=1).mean()
    sma60  = prices.rolling(window=60, min_periods=1).mean()

    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
    rs    = gain / (loss + 1e-9)
    rsi14 = 100.0 - (100.0 / (1.0 + rs))

    df = pd.DataFrame({'SMA_20': sma20, 'SMA_60': sma60, 'RSI_14': rsi14})
    df.ffill(inplace=True)
    df.fillna(0.0, inplace=True)
    return df


def _get_indicator_obs(indicator_df: pd.DataFrame, idx: int, current_price: float) -> np.ndarray:
    """trading_env._get_indicator_obs()와 동일한 스케일링."""
    idx = min(idx, len(indicator_df) - 1)
    row = indicator_df.iloc[idx]
    p   = current_price + 1e-9
    sma20_s = float(np.clip((current_price - row['SMA_20']) / p, -1.0, 1.0))
    sma60_s = float(np.clip((current_price - row['SMA_60']) / p, -1.0, 1.0))
    rsi14_s = float(row['RSI_14']) / 100.0
    return np.array([sma20_s, sma60_s, rsi14_s], dtype=np.float32)


def _is_pullback(indicator_df: pd.DataFrame, idx: int) -> bool:
    """눌림목 조건: SMA_20 > SMA_60 AND RSI_14 < 40"""
    idx = min(idx, len(indicator_df) - 1)
    row = indicator_df.iloc[idx]
    return (row['SMA_20'] > row['SMA_60']) and (row['RSI_14'] < 40.0)


class PolicyCheckWorker(QThread):
    """모델 정책 진단을 수행하는 백그라운드 워커 (5-액션 / 보조지표 대응)"""
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

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            # ── 1. 모델 로드 ───────────────────────────
            self.sig_log.emit(f"📡 모델 로드 중: {os.path.basename(self.model_path)}")
            try:
                model = MaskablePPO.load(self.model_path)
            except Exception as e:
                self.sig_error.emit(f"모델 로드 실패: {e}")
                return

            model_action_n = model.action_space.n
            target_dim     = model.observation_space.shape[0]
            self.sig_log.emit(
                f"✅ 모델 로드 완료 | Action Space: Discrete({model_action_n}) | Obs Dim: {target_dim}"
            )
            if model_action_n != NUM_ACTIONS:
                self.sig_log.emit(
                    f"⚠️ 모델 액션 수({model_action_n})가 기준({NUM_ACTIONS})과 다릅니다. "
                    f"마스크를 {model_action_n}개로 조정합니다."
                )

            # ── 2. 데이터 로드 (InfluxDB 동기 클라이언트) ──
            self.sig_log.emit(f"📂 [{self.symbol}] 과거 데이터 조회 중...")
            from influxdb_client import InfluxDBClient
            url    = self.config_manager.get("INFLUX_URL")
            token  = self.config_manager.get("INFLUX_TOKEN")
            org    = self.config_manager.get("INFLUX_ORG")
            bucket = self.config_manager.get("influx_bucket", "stock_data")

            sync_client = InfluxDBClient(url=url, token=token, org=org, timeout=300000)
            query_api   = sync_client.query_api()

            clean_symbol = self.symbol.split('_')[0].strip()
            query = f'''
                from(bucket: "{bucket}")
                |> range(start: -1y)
                |> filter(fn: (r) => r["_measurement"] == "historical_data" or r["_measurement"] == "tick_data")
                |> filter(fn: (r) => r["symbol"] == "{self.symbol}" or r["symbol"] == "{clean_symbol}")
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
                        "price":     float(record.values.get("price", 0.0)),
                        "volume":    float(record.values.get("volume", 0.0)),
                        "open":      float(record.values.get("open", 0.0)),
                        "high":      float(record.values.get("high", 0.0)),
                        "low":       float(record.values.get("low", 0.0)),
                    })
            sync_client.close()

            if len(data_list) < 60:
                self.sig_error.emit(
                    f"데이터 부족 ({len(data_list)}개, 최소 60개 필요)\n"
                    "InfluxDB에 해당 종목 데이터가 있는지 확인하세요."
                )
                return

            sorted_data = sorted(data_list, key=lambda x: x["timestamp"])
            self.sig_log.emit(f"✅ 데이터 준비 완료: {len(sorted_data)}개 캔들")

            # ── 3. 보조지표 전처리 ─────────────────────
            # trading_env._compute_indicators()와 완전히 동일한 로직
            self.sig_log.emit("📐 보조지표(SMA_20 / SMA_60 / RSI_14) 계산 중...")
            indicator_df = _compute_indicators(sorted_data)
            self.sig_log.emit(
                f"✅ 보조지표 계산 완료 (NaN: {indicator_df.isna().sum().sum()}개)"
            )

            # ── 4. 추론 루프 ────────────────────────────
            normalizer = OnlineRollingNormalizer(window_size=200)
            seq_len    = self.config_manager.get("seq_len", 10)

            # 통계 카운터
            total_steps     = 0
            raw_counts      = {i: 0 for i in range(NUM_ACTIONS)}   # 필터 미적용
            filtered_counts = {i: 0 for i in range(NUM_ACTIONS)}   # 필터 적용 후
            filter_overrides = 0
            buy_confidences  = []   # action in (1, 2)
            sell_confidences = []   # action in (3, 4)
            detail_logs      = []   # 매수/매도 상세 로그

            minute_buffer = deque(maxlen=100)
            self.sig_log.emit(
                f"🧠 정책 분석 시뮬레이션 시작... "
                f"(Confidence 필터: Buy≥{BUY_THRESHOLD}, Sell≥{SELL_THRESHOLD})"
            )

            for i, candle in enumerate(sorted_data):
                if not self.is_running:
                    break

                minute_buffer.append(candle)
                if len(minute_buffer) < 30:
                    continue

                # Feature Engineering
                try:
                    features = AdvancedFeatureEngineer.process_historical_data(list(minute_buffer))
                except Exception:
                    continue
                if len(features) < seq_len:
                    continue

                obs_1d         = features[-seq_len:].flatten()
                obs_normalized = normalizer.normalize(obs_1d)

                # 보조지표 추가 (trading_env._get_observation()과 동일한 순서)
                current_price   = float(candle.get('price', 0))
                indicator_obs   = _get_indicator_obs(indicator_df, i, current_price)
                obs_full        = np.concatenate([obs_normalized, indicator_obs])

                # 패딩 (Target Dim 동기화)
                if len(obs_full) < target_dim:
                    obs_full = np.pad(obs_full, (0, target_dim - len(obs_full)), 'constant')
                elif len(obs_full) > target_dim:
                    obs_full = obs_full[:target_dim]

                state_input  = np.expand_dims(obs_full, axis=0)
                action_masks = np.array([True] * model_action_n)

                # 예측
                raw_action, _ = model.predict(state_input, action_masks=action_masks, deterministic=True)
                if isinstance(raw_action, np.ndarray):
                    raw_action = int(raw_action[0])

                # 확률 추출
                with torch.no_grad():
                    obs_tensor, _ = model.policy.obs_to_tensor(state_input)
                    dist = model.policy.get_distribution(obs_tensor)
                    if hasattr(dist.distribution, 'probs'):
                        probs = dist.distribution.probs.cpu().numpy()[0]
                    else:
                        probs = torch.softmax(dist.distribution.logits, dim=-1).cpu().numpy()[0]

                if len(probs) < NUM_ACTIONS:
                    probs = np.pad(probs, (0, NUM_ACTIONS - len(probs)))

                total_steps += 1
                raw_counts[raw_action] += 1

                # Confidence 필터
                filtered_action = raw_action
                conf            = float(probs[raw_action])

                if raw_action in (1, 2):    # 매수 계열
                    buy_confidences.append(conf)
                    if conf < BUY_THRESHOLD:
                        filtered_action  = 0
                        filter_overrides += 1
                elif raw_action in (3, 4):  # 매도 계열
                    sell_confidences.append(conf)
                    if conf < SELL_THRESHOLD:
                        filtered_action  = 0
                        filter_overrides += 1

                filtered_counts[filtered_action] += 1

                # 매수/매도 신호 상세 로그 (최대 50건만 보관)
                if raw_action in (1, 2, 3, 4) and len(detail_logs) < 50:
                    icon, lbl    = ACTION_LABELS[raw_action]
                    override_mrk = " ⚡→Hold" if filtered_action == 0 else ""
                    pullback_mrk = " 🎯눌림목!" if raw_action in (1, 2) and _is_pullback(indicator_df, i) else ""
                    ts_str       = str(candle.get('timestamp', ''))[:16]
                    p_str        = "  ".join(
                        [f"{ACTION_LABELS.get(j, ('?','?'))[1].strip()}:{probs[j]:.3f}"
                         for j in range(min(NUM_ACTIONS, len(probs)))]
                    )
                    detail_logs.append(
                        f"  {icon} {ts_str} | {lbl.strip()} | 신뢰도:{conf:.4f}"
                        f"{override_mrk}{pullback_mrk}\n"
                        f"     [{p_str}]"
                    )

                if total_steps >= self.max_steps:
                    break

            # ── 5. 결과 텍스트 생성 ─────────────────────
            if total_steps > 0:
                sep = "─" * 48

                # 순수 판단 비율
                raw_buy_pct  = (raw_counts[1] + raw_counts[2]) / total_steps * 100
                raw_sell_pct = (raw_counts[3] + raw_counts[4]) / total_steps * 100

                # 신뢰도
                avg_buy_conf  = np.mean(buy_confidences)  if buy_confidences  else 0.0
                max_buy_conf  = np.max(buy_confidences)   if buy_confidences  else 0.0
                avg_sell_conf = np.mean(sell_confidences) if sell_confidences else 0.0

                # 진단 결론
                if raw_buy_pct > 80:
                    conclusion = "⚠️ 정책 붕괴 위험! 매수 편향이 심각합니다.\n   → 학습률 조정 또는 Hold 페널티 완화를 검토하세요."
                elif raw_buy_pct + raw_sell_pct < 5:
                    conclusion = "⚠️ 과소 추론 위험! 모델이 지나치게 소극적입니다.\n   → 거래 보상 배율(reward_multiplier)을 높이세요."
                elif avg_buy_conf > 0.9:
                    conclusion = "⚠️ 신뢰도 포화(Saturation) 의심! 매수 확률이 0.9999에 수렴합니다.\n   → 정규화(Normalizer) 또는 클리핑 로직을 점검하세요."
                else:
                    conclusion = "✅ 모델 정책이 비교적 균형 잡혀 있습니다."

                # 순수 판단 줄
                raw_lines = "\n".join(
                    f"    {ACTION_LABELS[k][0]} {ACTION_LABELS[k][1]}({k}): "
                    f"{raw_counts[k]:4d}회  ({raw_counts[k]/total_steps*100:5.1f}%)"
                    for k in range(NUM_ACTIONS)
                )
                # 필터 적용 후 줄
                flt_lines = "\n".join(
                    f"    {ACTION_LABELS[k][0]} {ACTION_LABELS[k][1]}({k}): "
                    f"{filtered_counts[k]:4d}회  ({filtered_counts[k]/total_steps*100:5.1f}%)"
                    for k in range(NUM_ACTIONS)
                )
                # 상세 로그
                detail_block = "\n".join(detail_logs) if detail_logs else "  (매수/매도 신호 없음)"

                result_text = (
                    f"{sep}\n"
                    f"  테스트 종목   : {self.symbol}\n"
                    f"  총 추론 스텝  : {total_steps}\n"
                    f"  Obs Dim       : {target_dim}\n"
                    f"  Action Space  : Discrete({model_action_n})\n"
                    f"{sep}\n\n"
                    f"  ▶ 순수 모델 판단 (Confidence 필터 미적용)\n"
                    f"{raw_lines}\n\n"
                    f"  ▶ Confidence 필터 적용 후 (Buy≥{BUY_THRESHOLD}, Sell≥{SELL_THRESHOLD})\n"
                    f"{flt_lines}\n\n"
                    f"  ▶ 필터로 Hold 전환: {filter_overrides}회 "
                    f"({filter_overrides/total_steps*100:.1f}%)\n\n"
                    f"  ▶ 매수 신뢰도 — 평균: {avg_buy_conf:.4f}  최대: {max_buy_conf:.4f} "
                    f"(n={len(buy_confidences)})\n"
                    f"  ▶ 매도 신뢰도 — 평균: {avg_sell_conf:.4f} "
                    f"(n={len(sell_confidences)})\n\n"
                    f"  ▶ 매수 편향도: {raw_buy_pct:.1f}%  |  매도 편향도: {raw_sell_pct:.1f}%\n\n"
                    f"  [진단 결론]\n"
                    f"  {conclusion}\n\n"
                    f"{sep}\n"
                    f"  ▶ 매수/매도 신호 상세 (최대 50건)\n"
                    f"{detail_block}\n"
                    f"{sep}\n"
                )
                self.sig_finished.emit(result_text)
            else:
                self.sig_error.emit("유효한 추론 스텝이 없습니다. 데이터 기간을 늘리거나 다른 종목을 선택해 보세요.")

        except Exception as e:
            import traceback
            self.sig_error.emit(f"진단 중 오류 발생: {str(e)}\n{traceback.format_exc()}")


class PolicyInspectorTab(QWidget):
    """모델 정책 진단 (Policy Inspector) 탭 UI"""

    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.model_path     = ""
        self.worker         = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # ── 입력 영역 ────────────────────────────────
        input_group  = QGroupBox("진단 설정")
        input_layout = QVBoxLayout()

        # 모델 선택
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("진단 모델:"))
        self.lbl_model = QLabel("선택된 모델 없음")
        self.lbl_model.setStyleSheet("color: #888; font-style: italic;")
        btn_select = QPushButton("모델 파일 선택 (.zip)")
        btn_select.clicked.connect(self._on_select_model)
        model_layout.addWidget(self.lbl_model, 1)
        model_layout.addWidget(btn_select)
        input_layout.addLayout(model_layout)

        # 종목 및 스텝
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("종목 코드:"))
        self.edit_symbol = QLineEdit("001440")
        self.edit_symbol.setPlaceholderText("예: 005930")
        params_layout.addWidget(self.edit_symbol)
        params_layout.addSpacing(20)
        params_layout.addWidget(QLabel("진단 스텝 수:"))
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(100, 5000)
        self.spin_steps.setValue(1000)
        self.spin_steps.setSingleStep(100)
        params_layout.addWidget(self.spin_steps)
        params_layout.addStretch(1)

        self.btn_run = QPushButton("🚀 진단 시작")
        self.btn_run.setFixedHeight(35)
        self.btn_run.setStyleSheet(
            "background-color: #2b5b84; color: white; font-weight: bold;"
        )
        self.btn_run.clicked.connect(self._on_run_diagnosis)
        params_layout.addWidget(self.btn_run)

        input_layout.addLayout(params_layout)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # ── 결과 출력 영역 ────────────────────────────
        output_group  = QGroupBox("진단 결과 (Diagnosis Output)")
        output_layout = QVBoxLayout()
        self.txt_output = QTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setStyleSheet("""
            background-color: #0c0c0c;
            color: #dcdcdc;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 10pt;
            border: 1px solid #333;
        """)
        output_layout.addWidget(self.txt_output)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group, 1)

    def _on_select_model(self):
        base_dir = os.path.join(os.getcwd(), "saved_models")
        os.makedirs(base_dir, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self, "진단할 모델 선택", base_dir, "Zip Files (*.zip)"
        )
        if path:
            self.model_path = path
            self.lbl_model.setText(os.path.basename(path))
            self.lbl_model.setStyleSheet("color: #007acc; font-weight: bold;")

    def _on_run_diagnosis(self):
        if not self.model_path:
            QMessageBox.warning(self, "경고", "진단할 모델 파일을 먼저 선택해 주세요.")
            return
        symbol = self.edit_symbol.text().strip()
        if not symbol:
            QMessageBox.warning(self, "경고", "종목 코드를 입력해 주세요.")
            return

        self.btn_run.setEnabled(False)
        self.txt_output.clear()
        self.txt_output.append(">>> 진단 프로세스 시작...")

        self.worker = PolicyCheckWorker(
            self.config_manager, self.model_path,
            symbol, self.spin_steps.value()
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
        QMessageBox.critical(self, "에러", f"진단 중 오류가 발생했습니다:\n{err}")
