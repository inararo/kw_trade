from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QProgressBar, QListWidget, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import pyqtSlot, Qt
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
        main_vertical_layout = QVBoxLayout(self)

        # Risk / Status Bar (Top)
        self.status_bar = QLabel("시스템 정상 대기 중")
        self.status_bar.setStyleSheet("background-color: #2b5b84; color: white; padding: 10px; font-weight: bold;")
        self.status_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_vertical_layout.addWidget(self.status_bar)

        self.risk_bar = QLabel("당일 누적 손익: 0원 | 잔여 매수 가능 한도: 계산 중...")
        self.risk_bar.setStyleSheet("background-color: #3b3b3b; color: #a9b7c6; padding: 5px; font-weight: bold;")
        self.risk_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_vertical_layout.addWidget(self.risk_bar)

        main_layout = QHBoxLayout()
        main_vertical_layout.addLayout(main_layout)

        # 1. 좌측 패널: 통합 다중 종목 마스터 테이블
        master_group = QGroupBox("전체 감시 종목 (Universe)")
        master_layout = QVBoxLayout()

        self.summary_table = QTableWidget(0, 4)
        self.summary_table.setHorizontalHeaderLabels(["종목코드", "현재가", "AI 신호", "보유량"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.summary_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.summary_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        master_layout.addWidget(self.summary_table)

        master_group.setLayout(master_layout)
        main_layout.addWidget(master_group, stretch=1)

        # 2. 중앙 패널: 상세 호가창 래더 (선택된 종목)
        ladder_group = QGroupBox("상세 호가창 래더")
        ladder_layout = QVBoxLayout()
        self.orderbook_widget = OrderbookLadderWidget(self.view_model)
        ladder_layout.addWidget(self.orderbook_widget)
        ladder_group.setLayout(ladder_layout)
        main_layout.addWidget(ladder_group, stretch=1)

        # 3. 우측 패널: AI 모니터 및 컨트롤
        control_group = QGroupBox("상세 AI 모니터링 및 시스템 제어")
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
        self.view_model.sig_symbols_summary_updated.connect(self.on_symbols_summary_updated)
        self.view_model.sig_ai_confidence_updated.connect(self.on_ai_confidence_updated)
        self.view_model.sig_log_appended.connect(self.on_log_appended)
        self.view_model.sig_risk_metrics_updated.connect(self.on_risk_metrics_updated)
        self.view_model.sig_status_alert.connect(self.on_status_alert)
        self.view_model.sig_error_occurred.connect(self.on_error)

    def _on_table_selection_changed(self):
        selected_items = self.summary_table.selectedItems()
        if selected_items:
            # 첫 번째 컬럼(종목코드) 가져오기
            row = selected_items[0].row()
            symbol = self.summary_table.item(row, 0).text()
            if hasattr(self.view_model, 'set_selected_symbol'):
                self.view_model.set_selected_symbol(symbol)
                self.on_log_appended(f"[UI] 상세 뷰 종목 변경: {symbol}")

    @pyqtSlot(dict)
    def on_symbols_summary_updated(self, summary_dict: dict):
        """테이블 갱신. UI 병목을 피하기 위해 최적화가 필요할 수 있으나 현재는 전체를 다시 그림"""
        self.summary_table.setRowCount(len(summary_dict))

        for row, (symbol, data) in enumerate(summary_dict.items()):
            self.summary_table.setItem(row, 0, QTableWidgetItem(symbol))

            price_str = f"{data.get('price', 0):,.0f}"
            self.summary_table.setItem(row, 1, QTableWidgetItem(price_str))

            ai_sig = data.get('ai_signal', '-')
            item_sig = QTableWidgetItem(ai_sig)
            if ai_sig == "Buy":
                item_sig.setForeground(Qt.GlobalColor.red)
            elif ai_sig == "Sell":
                item_sig.setForeground(Qt.GlobalColor.blue)
            self.summary_table.setItem(row, 2, item_sig)

            self.summary_table.setItem(row, 3, QTableWidgetItem(str(data.get('holdings', 0))))

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

    @pyqtSlot(float, float)
    def on_risk_metrics_updated(self, pnl: float, available_limit: float):
        self.risk_bar.setText(f"당일 누적 손익: {pnl:,.0f} 원 | 잔여 매수 가능 한도: {available_limit:,.0f} 원")

    @pyqtSlot(str)
    def on_status_alert(self, alert_msg: str):
        self.status_bar.setText(alert_msg)
        self.status_bar.setStyleSheet("background-color: darkred; color: yellow; padding: 10px; font-weight: bold;")

    @pyqtSlot(str)
    def on_error(self, msg: str):
        self.on_log_appended(f"[오류] {msg}")
