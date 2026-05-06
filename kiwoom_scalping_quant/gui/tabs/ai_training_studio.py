from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QSpinBox, QDoubleSpinBox, QFormLayout, QListWidget, QComboBox, QAbstractSpinBox, QCheckBox
from PyQt6.QtCore import pyqtSlot, Qt
import pyqtgraph as pg

class AITrainingStudioTab(QWidget):
    """
    탭 C: AI 학습 및 모니터링
    학습 파라미터 제어 및 시각화 패널.
    """
    def __init__(self, view_model):
        super().__init__()
        self.view_model = view_model
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        # 1. 좌측: 학습 파라미터 폼
        params_group = QGroupBox("학습 파라미터 설정")
        form_layout = QFormLayout()

        # HTS 스타일 가로형 컨트롤 공통 스타일 (슬림화 및 정밀 정렬)
        base_style = """
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                border-radius: 2px;
                padding: 0 5px;
                font-size: 12px;
                font-family: 'Consolas', monospace;
            }
            QPushButton#spin_btn {
                min-width: 40px;
                max-width: 40px;
                background-color: #383838;
                color: #ffffff;
                border: 1px solid #4d4d4d;
                border-radius: 2px;
                padding: 0;
                margin: 0;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton#spin_btn:hover {
                background-color: #007acc;
                border: 1px solid #0098ff;
            }
            QPushButton#spin_btn:pressed {
                background-color: #005a9e;
            }
        """

        def create_h_spin(widget, label, form):
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2) 
            
            widget.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            widget.setStyleSheet(base_style)
            widget.setFixedHeight(24) # 높이 24px 강제 고정
            
            btn_up = QPushButton("▲")
            btn_up.setObjectName("spin_btn")
            btn_up.setStyleSheet(base_style)
            btn_up.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_up.setFixedHeight(24) # 버튼 높이도 24px로 완전 일치
            btn_up.clicked.connect(widget.stepUp)
            
            btn_down = QPushButton("▼")
            btn_down.setObjectName("spin_btn")
            btn_down.setStyleSheet(base_style)
            btn_down.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_down.setFixedHeight(24) # 버튼 높이도 24px로 완전 일치
            btn_down.clicked.connect(widget.stepDown)
            
            layout.addWidget(widget, stretch=1)
            layout.addWidget(btn_up)
            layout.addWidget(btn_down)
            form.addRow(label, container)

        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1000, 10000000)
        self.spin_steps.setValue(1000000)
        self.spin_steps.setSingleStep(100000)
        self.spin_steps.setGroupSeparatorShown(True)
        create_h_spin(self.spin_steps, "총 스텝 수:", form_layout)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(5)
        self.spin_lr.setRange(0.00001, 0.1)
        self.spin_lr.setValue(0.00030)
        self.spin_lr.setSingleStep(0.00001)
        create_h_spin(self.spin_lr, "학습률:", form_layout)

        self.spin_max_records = QSpinBox()
        self.spin_max_records.setRange(1000, 10000000)
        self.spin_max_records.setValue(100000)
        self.spin_max_records.setSingleStep(10000)
        self.spin_max_records.setGroupSeparatorShown(True)
        create_h_spin(self.spin_max_records, "데이터 로드 건수 (종목당):", form_layout)

        self.combo_feature_mode = QComboBox()
        self.combo_feature_mode.addItems(["Basic (단순 가격/거래량)", "Advanced (보조지표 추가)"])
        self.combo_feature_mode.setCurrentIndex(1)
        form_layout.addRow("데이터 분석 모드:", self.combo_feature_mode)

        # --- Advanced PPO Settings ---
        self.spin_ent = QDoubleSpinBox()
        self.spin_ent.setDecimals(3)
        self.spin_ent.setRange(0.0, 0.1)
        self.spin_ent.setValue(0.010)
        self.spin_ent.setSingleStep(0.001)
        self.spin_ent.setToolTip("Entropy Coef: 낮을수록 기존 타점 유지, 높을수록 새로운 탐험 시도")
        create_h_spin(self.spin_ent, "탐험 계수 (Entropy):", form_layout)

        self.spin_clip = QDoubleSpinBox()
        self.spin_clip.setRange(0.0, 0.4)
        self.spin_clip.setValue(0.15)
        self.spin_clip.setSingleStep(0.05)
        self.spin_clip.setToolTip("Clip Range: 정책 업데이트 시 변화폭 제한 (안정성)")
        create_h_spin(self.spin_clip, "클립 범위 (Clip):", form_layout)

        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(0.8, 0.999)
        self.spin_gamma.setValue(0.995)
        self.spin_gamma.setSingleStep(0.001)
        self.spin_gamma.setDecimals(3)
        self.spin_gamma.setToolTip("Gamma: 할인율. 낮을수록 단기 수익, 높을수록 장기 수익 중시")
        create_h_spin(self.spin_gamma, "할인율 (Gamma):", form_layout)

        self.spin_gae = QDoubleSpinBox()
        self.spin_gae.setRange(0.8, 1.0)
        self.spin_gae.setValue(0.95)
        self.spin_gae.setSingleStep(0.01)
        self.spin_gae.setToolTip("GAE Lambda: 어드밴티지 추정 시 편향-분산 트레이드오프 조절")
        create_h_spin(self.spin_gae, "GAE 람다:", form_layout)

        # [스마트 샘플링 토글] 데이터 분석 모드 바로 아래
        self.chk_smart_sampling = QCheckBox("활황장(Volume Spike) 구간 집중 학습")
        self.chk_smart_sampling.setChecked(False)  # 기본값: OFF (Pure Random)
        self.chk_smart_sampling.setToolTip(
            "ON: 거래량이 폭발하는 활황장 구간(20봉 평균의 1.5배 이상)에서 80%의 확률로\n"
            "에피소드 시작점을 선택합니다. 스캘핑 타점 학습 가속화에 효과적입니다.\n"
            "OFF: 전체 구간에서 순수 무작위(Pure Random)로 시작점을 선택합니다. (기본값)"
        )
        self.chk_smart_sampling.setStyleSheet("""
            QCheckBox { color: #dcdcdc; font-size: 12px; }
            QCheckBox::indicator { width: 14px; height: 14px; }
            QCheckBox::indicator:checked { background-color: #007acc; border: 1px solid #0098ff; border-radius: 2px; }
            QCheckBox::indicator:unchecked { background-color: #383838; border: 1px solid #4d4d4d; border-radius: 2px; }
        """)
        form_layout.addRow("스마트 샘플링:", self.chk_smart_sampling)

        # [추가] 에피소드 길이 개선 옵션
        self.chk_day_begin = QCheckBox("항상 장 시작(09:00)부터 학습")
        self.chk_day_begin.setChecked(True)
        self.chk_day_begin.setToolTip("ON: 샘플링된 지점이 속한 날짜의 09:00부터 학습을 시작하여 에피소드 길이를 확보합니다.")
        self.chk_day_begin.setStyleSheet(self.chk_smart_sampling.styleSheet())
        form_layout.addRow("시작점 보정:", self.chk_day_begin)

        self.chk_overnight = QCheckBox("날짜 변경 시에도 학습 유지 (Overnight)")
        self.chk_overnight.setChecked(False)
        self.chk_overnight.setToolTip("ON: 날짜가 바뀌어도 에피소드를 종료하지 않고 계속 진행합니다. (장기 학습용)")
        self.chk_overnight.setStyleSheet(self.chk_smart_sampling.styleSheet())
        form_layout.addRow("연속 학습:", self.chk_overnight)

        self.btn_start = QPushButton("학습 시작")
        self.btn_start.setStyleSheet("background-color: green; color: white;")
        self.btn_start.clicked.connect(self._on_start_clicked)
        form_layout.addRow("", self.btn_start)

        self.btn_stop = QPushButton("학습 중지")
        self.btn_stop.setStyleSheet("background-color: orange; color: white;")
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.btn_stop.setEnabled(False)
        form_layout.addRow("", self.btn_stop)

        params_group.setLayout(form_layout)
        
        # 1-2. 좌측 하단: AI 뇌 구조 요약 패널 (MLOps 정보)
        self.info_group = QGroupBox("AI 모델 및 훈련 환경 정보")
        info_layout = QVBoxLayout()
        
        from PyQt6.QtWidgets import QTextEdit
        self.txt_info_summary = QTextEdit()
        self.txt_info_summary.setReadOnly(True)
        self.txt_info_summary.setStyleSheet("""
            background-color: #1e1e1e; 
            color: #dcdcdc; 
            border: 1px solid #3d3d3d;
            font-size: 11px;
        """)
        info_layout.addWidget(self.txt_info_summary)
        self.info_group.setLayout(info_layout)
        
        # 좌측 레이아웃 구성
        left_panel = QVBoxLayout()
        left_panel.addWidget(params_group, stretch=1)
        left_panel.addWidget(self.info_group, stretch=1)
        
        main_layout.addLayout(left_panel, stretch=1)

        # 초기 요약 정보 수립
        self._update_info_summary()

        # 2. 우측: 실시간 학습 곡선 (PyQtGraph)
        chart_group = QGroupBox("학습 진행 상황 및 보상 곡선")
        chart_layout = QVBoxLayout()

        pg.setConfigOption('background', '#2b2b2b')
        pg.setConfigOption('foreground', 'w')

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', "진행 스텝")
        self.plot_widget.setLabel('left', "평균 보상")
        self.plot_widget.showGrid(x=True, y=True)

        self.reward_curve = self.plot_widget.plot(pen=pg.mkPen('g', width=2))
        chart_layout.addWidget(self.plot_widget, stretch=2)

        # 훈련 로그 출력 리스트
        self.log_list = QListWidget()
        self.log_list.setStyleSheet("background-color: black; color: lime; font-family: monospace;")
        chart_layout.addWidget(self.log_list, stretch=1)

        chart_group.setLayout(chart_layout)
        main_layout.addWidget(chart_group, stretch=2)

        self._step_data = []
        self._reward_data = []

    def _connect_signals(self):
        self.view_model.sig_training_started.connect(self.on_training_started)
        self.view_model.sig_training_progress.connect(self.on_training_progress)
        self.view_model.sig_training_log.connect(self.on_log_msg)
        self.view_model.sig_training_finished.connect(self.on_training_finished)
        self.view_model.sig_error.connect(self.on_error)
        
        # [신규] 분석 모드 및 하이퍼파라미터 변경 시 정보 패널 즉시 갱신
        self.combo_feature_mode.currentIndexChanged.connect(self._update_info_summary)
        self.chk_smart_sampling.stateChanged.connect(self._update_info_summary)
        self.spin_lr.valueChanged.connect(self._update_info_summary)
        self.spin_ent.valueChanged.connect(self._update_info_summary)
        self.spin_clip.valueChanged.connect(self._update_info_summary)
        self.spin_gamma.valueChanged.connect(self._update_info_summary)
        self.spin_gae.valueChanged.connect(self._update_info_summary)
        self.chk_day_begin.stateChanged.connect(self._update_info_summary)
        self.chk_overnight.stateChanged.connect(self._update_info_summary)

    def _on_start_clicked(self):
        timesteps = self.spin_steps.value()
        lr = self.spin_lr.value()
        max_records = self.spin_max_records.value()
        feature_mode = "advanced" if self.combo_feature_mode.currentIndex() == 1 else "basic"
        use_smart_sampling = self.chk_smart_sampling.isChecked()
        
        # 신규 하이퍼파라미터 추출
        ppo_params = {
            "ent_coef": self.spin_ent.value(),
            "clip_range": self.spin_clip.value(),
            "gamma": self.spin_gamma.value(),
            "gae_lambda": self.spin_gae.value()
        }

        # [즉각 반응] 시작 버튼을 먼저 비활성화하여 중복 클릭 방지
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self._step_data.clear()
        self._reward_data.clear()
        self.reward_curve.setData([], [])
        self.log_list.clear()

        self.view_model.start_training(timesteps, lr, max_records, 
                                     feature_mode=feature_mode, 
                                     use_smart_sampling=use_smart_sampling,
                                     ppo_params=ppo_params,
                                     always_start_day_begin=self.chk_day_begin.isChecked(),
                                     allow_overnight_episodes=self.chk_overnight.isChecked())

    def _on_stop_clicked(self):
        self.btn_stop.setEnabled(False) # 중복 중단 요청 방지
        self.view_model.stop_training()

    @pyqtSlot()
    def on_training_started(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    @pyqtSlot(int, float, float)
    def on_training_progress(self, step: int, reward: float, loss: float):
        self._step_data.append(step)
        self._reward_data.append(reward)
        self.reward_curve.setData(self._step_data, self._reward_data)

    @pyqtSlot(str)
    def on_log_msg(self, msg: str):
        self.log_list.addItem(msg)
        self.log_list.scrollToBottom()

    @pyqtSlot()
    def on_training_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    @pyqtSlot(str)
    def on_error(self, err: str):
        self.log_list.addItem(f"[오류] {err}")
        self.log_list.scrollToBottom()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _update_info_summary(self):
        """현재 UI 설정 및 환경 상수를 기반으로 AI 모델 명세서를 생성합니다."""
        is_advanced = self.combo_feature_mode.currentIndex() == 1
        is_smart = self.chk_smart_sampling.isChecked()
        
        # 하드코딩된 환경 상수 (TradingEnv와 동기화된 정보)
        window_size = 10
        cooldown = 10
        grace_period = 10
        
        if is_advanced:
            mode_title = "Advanced (고도화 분석)"
            features = "가격변동률, 거래량스파이크, RSI, MA이격도, BB위치, OIR, 틱변동성, VWAP이격도, 추세변동성, 장중시간, 정규시장여부"
            dim = 110
        else:
            mode_title = "Basic (단순 지표)"
            features = "가격변동률, 수익률, 거래량, OIR, 틱변동성"
            dim = 50

        summary = f"""
<b>🧠 알고리즘:</b> Maskable PPO (연속 의사결정 최적화)<br><br>
<b>👀 입력 데이터 (State): {mode_title}</b>
<ul>
    <li><b>지표:</b> {features}</li>
    <li><b>기억력:</b> 최근 {window_size}스텝 Lookback Window</li>
    <li><b>입력 차원:</b> {dim}차원 (1개 지표 x {window_size}개 시점)</li>
</ul>
<b>✋ 액션 및 제어 (Action):</b>
<ul>
    <li><b>공간:</b> 매수, 매도, 관망 (3개 이산 액션)</li>
    <li><b>제약:</b> Action Masking (잔고/보유량 유효성 검사)</li>
    <li><b>장벽:</b> 매매 쿨다운 {cooldown}스텝 (뇌동매매 방지)</li>
</ul>
<b>🎯 보상 체계 (Reward):</b>
<ul>
    <li><b>승수:</b> 실현 수익률 10x 가중치 적용 (도파민 강화)</li>
    <li><b>인내심:</b> 초기 {grace_period}스텝 패널티 유예 (패닉셀 방지)</li>
    <li><b>감가:</b> 보유 시간당 -0.005 패널티 (장기보유 방지)</li>
</ul>
<b>⚙️ 하이퍼파라미터 (Hyperparams):</b>
<ul>
    <li><b>Learning Rate:</b> {self.spin_lr.value():.6f}</li>
    <li><b>Entropy:</b> {self.spin_ent.value():.3f}</li>
    <li><b>Clip Range:</b> {self.spin_clip.value():.2f}</li>
    <li><b>Gamma:</b> {self.spin_gamma.value()}</li>
    <li><b>GAE Lambda:</b> {self.spin_gae.value()}</li>
</ul>
        """
        self.txt_info_summary.setHtml(summary)
