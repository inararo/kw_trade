from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QSpinBox, QDoubleSpinBox, QFormLayout, QListWidget
from PyQt6.QtCore import pyqtSlot
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

        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1000, 1000000)
        self.spin_steps.setValue(10000)
        form_layout.addRow("총 스텝 수:", self.spin_steps)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(5)
        self.spin_lr.setRange(0.00001, 0.1)
        self.spin_lr.setValue(0.00030)
        form_layout.addRow("학습률:", self.spin_lr)

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
        main_layout.addWidget(params_group, stretch=1)

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

    def _on_start_clicked(self):
        timesteps = self.spin_steps.value()
        lr = self.spin_lr.value()

        self._step_data.clear()
        self._reward_data.clear()
        self.reward_curve.setData([], [])
        self.log_list.clear()

        self.view_model.start_training(timesteps, lr)

    def _on_stop_clicked(self):
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
