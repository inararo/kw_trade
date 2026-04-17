from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox
from gui.components.orderbook_ladder import OrderbookLadderWidget

class LiveDashboardTab(QWidget):
    """
    탭 A: 실시간 매매 (Live Dashboard)
    호가창 래더, AI 신뢰도 모니터, 패닉 버튼 등을 포함.
    """
    def __init__(self, view_model):
        super().__init__()
        self.view_model = view_model
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        # 1. 좌측 패널: 호가창 래더
        ladder_group = QGroupBox("Orderbook Ladder")
        ladder_layout = QVBoxLayout()
        self.orderbook_widget = OrderbookLadderWidget(self.view_model)
        ladder_layout.addWidget(self.orderbook_widget)
        ladder_group.setLayout(ladder_layout)
        main_layout.addWidget(ladder_group, stretch=1)

        # 2. 우측 패널: AI 모니터 및 컨트롤
        control_group = QGroupBox("AI Monitor & Controls")
        control_layout = QVBoxLayout()

        # AI 신뢰도 모니터 (임시 라벨)
        self.ai_chart_label = QLabel("[Chart Placeholder] AI Confidence: Hold 60%, Buy 30%, Sell 10%")
        self.ai_chart_label.setMinimumHeight(150)
        self.ai_chart_label.setStyleSheet("background-color: black; color: lime; font-weight: bold; padding: 10px;")
        control_layout.addWidget(self.ai_chart_label)

        # 체결 로그 (임시 라벨)
        self.execution_log = QLabel("Execution Log:\n- Waiting for events...")
        self.execution_log.setStyleSheet("border: 1px solid gray; padding: 5px;")
        control_layout.addWidget(self.execution_log)

        control_layout.addStretch()

        # 패닉 버튼
        self.panic_btn = QPushButton("🚨 PANIC SELL & CANCEL ALL 🚨")
        self.panic_btn.setStyleSheet("background-color: red; color: white; font-size: 16px; font-weight: bold; height: 50px;")
        # 클릭 시 ViewModel을 거치거나 OrderManager로 신호 전달 로직 필요
        control_layout.addWidget(self.panic_btn)

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group, stretch=1)
