import os
import asyncio
import numpy as np
import pandas as pd
import torch
from collections import deque
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QPushButton, QLabel, QLineEdit, QSpinBox, QTextEdit, QFileDialog, QMessageBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from sb3_contrib import MaskablePPO

from core.feature_engineer import AdvancedFeatureEngineer
from models.agent import TradingAgentWrapper
from models.lstm_extractor import OnlineRollingNormalizer

class PolicyCheckWorker(QThread):
    """
    모델 정책 진단을 수행하는 백그라운드 워커
    """
    # (결과 텍스트)
    sig_finished = pyqtSignal(str)
    sig_error = pyqtSignal(str)
    sig_log = pyqtSignal(str)

    def __init__(self, config_manager, model_path, symbol, max_steps):
        super().__init__()
        self.config_manager = config_manager
        self.model_path = model_path
        self.symbol = symbol
        self.max_steps = max_steps
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            self.sig_log.emit(f"📡 모델 로드 중: {os.path.basename(self.model_path)}")
            
            # 1. 모델 로드
            try:
                model = MaskablePPO.load(self.model_path)
            except Exception as e:
                self.sig_error.emit(f"모델 로드 실패: {e}")
                return

            # 2. 데이터 로드 (InfluxDB 사용 - 스레드 안정성을 위해 동기식 조회 추천이나 여기서는 워커 전용 루프 사용)
            self.sig_log.emit(f"📂 [{self.symbol}] 과거 데이터 조회 중...")
            
            # 워커 전용 DB 클라이언트 생성
            from influxdb_client import InfluxDBClient
            url = self.config_manager.get("INFLUX_URL")
            token = self.config_manager.get("INFLUX_TOKEN")
            org = self.config_manager.get("INFLUX_ORG")
            bucket = self.config_manager.get("influx_bucket", "stock_data")
            
            sync_client = InfluxDBClient(url=url, token=token, org=org, timeout=300000)
            query_api = sync_client.query_api()
            
            clean_symbol = self.symbol.split('_')[0].strip()
            self.sig_log.emit(f"📂 [{self.symbol}/{clean_symbol}] 데이터 조회 중 (범위: 최근 1년)...")
            
            # 충분한 데이터를 위해 max_steps 보다 200개 더 가져옴 (Feature Engineering용)
            query = f'''
                from(bucket: "{bucket}")
                |> range(start: -1y)
                |> filter(fn: (r) => r["_measurement"] == "historical_data" or r["_measurement"] == "tick_data")
                |> filter(fn: (r) => r["symbol"] == "{self.symbol}" or r["symbol"] == "{clean_symbol}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"], desc: true)
                |> limit(n: {self.max_steps + 100})
            '''
            
            tables = query_api.query(query, org=org)
            data_list = []
            for table in tables:
                for record in table.records:
                    data_list.append({
                        "timestamp": record.get_time(),
                        "open": float(record.values.get("open", 0.0)),
                        "high": float(record.values.get("high", 0.0)),
                        "low": float(record.values.get("low", 0.0)),
                        "price": float(record.values.get("price", 0.0)),
                        "volume": float(record.values.get("volume", 0.0))
                    })
            sync_client.close()

            if len(data_list) < 50:
                self.sig_error.emit(f"데이터가 부족합니다. (조회된 건수: {len(data_list)})\nInfluxDB에 해당 종목의 과거 데이터가 있는지 확인해 주세요.")
                return

            # 시간순 정렬
            sorted_data = sorted(data_list, key=lambda x: x["timestamp"])
            self.sig_log.emit(f"✅ 데이터 준비 완료: {len(sorted_data)}개 캔들")

            # 3. 진단 시작
            normalizer = OnlineRollingNormalizer(window_size=200)
            seq_len = self.config_manager.get("seq_len", 10)
            
            total_steps = 0
            counts = {0: 0, 1: 0, 2: 0}
            buy_confidences = []
            
            minute_buffer = deque(maxlen=100)
            
            self.sig_log.emit("🧠 정책 분석 시뮬레이션 시작...")
            
            for i in range(len(sorted_data)):
                if not self.is_running: break
                
                minute_buffer.append(sorted_data[i])
                if len(minute_buffer) < 30: continue
                    
                # 피처 추출
                features = AdvancedFeatureEngineer.process_historical_data(list(minute_buffer))
                if len(features) < seq_len: continue
                    
                obs_1d = features[-seq_len:].flatten()
                obs_normalized = normalizer.normalize(obs_1d)
                
                # 차원 패딩
                target_dim = model.observation_space.shape[0]
                if len(obs_normalized) < target_dim:
                    obs_normalized = np.pad(obs_normalized, (0, target_dim - len(obs_normalized)), 'constant')
                    
                state_input = np.expand_dims(obs_normalized, axis=0)
                
                # 예측
                with torch.no_grad():
                    action, _ = model.predict(state_input, deterministic=True)
                    if isinstance(action, np.ndarray): action = int(action[0])
                    
                    # 확률(Confidence) 추출
                    obs_tensor, _ = model.policy.obs_to_tensor(state_input)
                    distribution = model.policy.get_distribution(obs_tensor)
                    if hasattr(distribution.distribution, 'probs'):
                        probs = distribution.distribution.probs.cpu().numpy()[0]
                    else:
                        logits = distribution.distribution.logits
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                
                total_steps += 1
                counts[action] += 1
                if action == 1: # Buy
                    buy_confidences.append(probs[1])

                if total_steps >= self.max_steps: break

            # 4. 결과 요약 생성
            if total_steps > 0:
                p_hold = (counts[0] / total_steps) * 100
                p_buy  = (counts[1] / total_steps) * 100
                p_sell = (counts[2] / total_steps) * 100
                avg_buy_conf = np.mean(buy_confidences) if buy_confidences else 0.0
                
                diagnosis_msg = ""
                if p_buy > 80: diagnosis_msg = "⚠️ [진단] 정책 붕괴 위험! 매수 편향이 심각합니다."
                elif p_buy < 3: diagnosis_msg = "⚠️ [진단] 과소 추론 위험! 모델이 지나치게 소극적입니다."
                else: diagnosis_msg = "✅ [진단] 모델 정책이 비교적 균형 잡혀 있습니다."

                result_text = (
                    f"----------------------------------------\n"
                    f"- 테스트 종목: {self.symbol}\n"
                    f"- 총 테스트 스텝 수: {total_steps}\n"
                    f"- 🛑 Hold (0) 선택 횟수: {counts[0]} ({p_hold:.1f}%)\n"
                    f"- 🟢 Buy  (1) 선택 횟수: {counts[1]} ({p_buy:.1f}%)\n"
                    f"- 🔴 Sell (2) 선택 횟수: {counts[2]} ({p_sell:.1f}%)\n"
                    f"- 🧠 Buy 평균 신뢰도: {avg_buy_conf:.4f}\n"
                    f"----------------------------------------\n"
                    f"{diagnosis_msg}\n"
                )
                self.sig_finished.emit(result_text)
            else:
                self.sig_error.emit("유효한 추론 스텝이 없습니다. 데이터 기간을 늘리거나 다른 종목을 선택해 보세요.")

        except Exception as e:
            import traceback
            self.sig_error.emit(f"진단 중 오류 발생: {str(e)}\n{traceback.format_exc()}")

class PolicyInspectorTab(QWidget):
    """
    모델 정책 진단 (Policy Inspector) 탭 UI
    """
    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.model_path = ""
        self.worker = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 1. 입력 영역
        input_group = QGroupBox("진단 설정")
        input_layout = QVBoxLayout()
        
        # 모델 선택
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("진단 모델:"))
        self.lbl_model = QLabel("선택된 모델 없음")
        self.lbl_model.setStyleSheet("color: #888; font-style: italic;")
        btn_select_model = QPushButton("모델 파일 선택 (.zip)")
        btn_select_model.clicked.connect(self._on_select_model)
        model_layout.addWidget(self.lbl_model, 1)
        model_layout.addWidget(btn_select_model)
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
        self.btn_run.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold;")
        self.btn_run.clicked.connect(self._on_run_diagnosis)
        params_layout.addWidget(self.btn_run)
        
        input_layout.addLayout(params_layout)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # 2. 결과 출력 영역
        output_group = QGroupBox("진단 결과 (Diagnosis Output)")
        output_layout = QVBoxLayout()
        self.txt_output = QTextEdit()
        self.txt_output.setReadOnly(True)
        # Monospace 폰트 설정
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
        if not os.path.exists(base_dir): os.makedirs(base_dir, exist_ok=True)
        
        path, _ = QFileDialog.getOpenFileName(self, "진단할 모델 선택", base_dir, "Zip Files (*.zip)")
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
            self.config_manager, 
            self.model_path, 
            symbol, 
            self.spin_steps.value()
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
