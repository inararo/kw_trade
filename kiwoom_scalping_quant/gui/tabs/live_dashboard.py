from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QProgressBar, QListWidget
)
from PyQt6.QtCore import pyqtSlot
from gui.components.orderbook_ladder import OrderbookLadderWidget
import time

class LiveDashboardTab(QWidget):
    """
    탭 A: 실시간 매매
    호가창 래더, AI 신뢰도 모니터, 패닉 버튼 등을 포함.
    """
    def __init__(self, view_model):
        super().__init__()
        self.view_model = view_model
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        # 1. 좌측 패널: 호가창 래더
        ladder_group = QGroupBox("호가창 래더")
        ladder_layout = QVBoxLayout()
        self.orderbook_widget = OrderbookLadderWidget(self.view_model)
        ladder_layout.addWidget(self.orderbook_widget)
        ladder_group.setLayout(ladder_layout)
        main_layout.addWidget(ladder_group, stretch=1)

        # 2. 우측 패널: AI 모니터 및 컨트롤
        control_group = QGroupBox("AI 모니터링 및 시스템 제어")
        control_layout = QVBoxLayout()

        # AI 신뢰도 모니터
        ai_layout = QVBoxLayout()
        ai_layout.addWidget(QLabel("AI 에이전트 행동 신뢰도:"))

        self.prog_hold = QProgressBar()
        self.prog_hold.setStyleSheet("QProgressBar::chunk { background-color: gray; }")
        self.prog_hold.setFormat("관망: %p%")

        self.prog_buy = QProgressBar()
        self.prog_buy.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        self.prog_buy.setFormat("매수: %p%")

        self.prog_sell = QProgressBar()
        self.prog_sell.setStyleSheet("QProgressBar::chunk { background-color: blue; }")
        self.prog_sell.setFormat("매도: %p%")

        ai_layout.addWidget(self.prog_hold)
        ai_layout.addWidget(self.prog_buy)
        ai_layout.addWidget(self.prog_sell)
        control_layout.addLayout(ai_layout)

        # 장외 시간 테스트용 Mock 데이터 실행 버튼
        self.btn_mock = QPushButton("가상 데이터 스트림 실행")
        self.btn_mock.clicked.connect(self.view_model.start_mock_stream)
        control_layout.addWidget(self.btn_mock)

        # 체결 및 시스템 로그 리스트
        control_layout.addWidget(QLabel("시스템 및 체결 로그:"))
        self.log_list = QListWidget()
        self.log_list.setStyleSheet("background-color: #2b2b2b; color: #a9b7c6; font-family: monospace;")
        control_layout.addWidget(self.log_list, stretch=1)

        # 패닉 버튼
        self.panic_btn = QPushButton("🚨 전량 시장가 매도 및 전체 주문 취소 🚨")
        self.panic_btn.setStyleSheet("background-color: darkred; color: white; font-size: 16px; font-weight: bold; height: 50px;")
        self.panic_btn.clicked.connect(self.view_model.trigger_panic_sell)
        control_layout.addWidget(self.panic_btn)

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group, stretch=1)

    def _connect_signals(self):
        self.view_model.sig_ai_confidence_updated.connect(self.on_ai_confidence_updated)
        self.view_model.sig_log_appended.connect(self.on_log_appended)
        self.view_model.sig_error_occurred.connect(self.on_error)

    @pyqtSlot(dict)
    def on_ai_confidence_updated(self, conf: dict):
        self.prog_hold.setValue(conf.get("Hold", 0))
        self.prog_buy.setValue(conf.get("Buy", 0))
        self.prog_sell.setValue(conf.get("Sell", 0))

    @pyqtSlot(str)
    def on_log_appended(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_list.addItem(f"[{ts}] {msg}")
        self.log_list.scrollToBottom()

    @pyqtSlot(str)
    def on_error(self, msg: str):
        self.on_log_appended(f"[오류] {msg}")
