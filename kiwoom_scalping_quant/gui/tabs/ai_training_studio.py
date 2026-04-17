from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QSpinBox, QDoubleSpinBox, QFormLayout

class AITrainingStudioTab(QWidget):
    """
    탭 C: AI 학습 및 모니터링 (AI Training Studio)
    학습 파라미터 제어 및 시각화 패널.
    """
    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        # 1. 좌측: 학습 파라미터 폼
        params_group = QGroupBox("Training Parameters")
        form_layout = QFormLayout()

        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1000, 1000000)
        self.spin_steps.setValue(10000)
        form_layout.addRow("Total Timesteps:", self.spin_steps)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(5)
        self.spin_lr.setRange(0.00001, 0.1)
        self.spin_lr.setValue(0.00030)
        form_layout.addRow("Learning Rate:", self.spin_lr)

        self.btn_start = QPushButton("Start Training")
        self.btn_start.setStyleSheet("background-color: green; color: white;")
        form_layout.addRow("", self.btn_start)

        self.btn_stop = QPushButton("Stop Training")
        self.btn_stop.setStyleSheet("background-color: orange; color: white;")
        form_layout.addRow("", self.btn_stop)

        params_group.setLayout(form_layout)
        main_layout.addWidget(params_group, stretch=1)

        # 2. 우측: 학습 곡선 (임시 텍스트)
        chart_group = QGroupBox("Training Progress / Reward Curve")
        chart_layout = QVBoxLayout()

        self.lbl_chart = QLabel("[TensorBoard / Matplotlib Chart Placeholder]\n\nWaiting for training to start...")
        self.lbl_chart.setStyleSheet("background-color: #2b2b2b; color: white; text-align: center;")
        chart_layout.addWidget(self.lbl_chart)

        chart_group.setLayout(chart_layout)
        main_layout.addWidget(chart_group, stretch=2)
