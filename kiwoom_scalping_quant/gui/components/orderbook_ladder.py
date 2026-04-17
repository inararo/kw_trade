from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import pyqtSlot
from typing import Dict

class OrderbookLadderWidget(QWidget):
    """
    ViewModel의 시그널을 받아 호가창을 렌더링하는 순수 뷰 컴포넌트.
    비즈니스 로직(DataCollector 등)을 직접 참조하지 않음.
    """
    def __init__(self, view_model):
        super().__init__()
        self.view_model = view_model
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        self.layout = QVBoxLayout(self)
        self.title_label = QLabel("Orderbook Ladder (DOM)", self)
        self.layout.addWidget(self.title_label)

        # 임시 가격/호가 표시 라벨들
        self.price_label = QLabel("Current Price: -", self)
        self.layout.addWidget(self.price_label)

        self.orderbook_label = QLabel("Waiting for data...", self)
        self.layout.addWidget(self.orderbook_label)

    def _connect_signals(self):
        # ViewModel의 시그널을 UI 슬롯에 연결
        self.view_model.price_updated.connect(self.on_price_updated)
        self.view_model.orderbook_updated.connect(self.on_orderbook_updated)
        self.view_model.error_occurred.connect(self.on_error_occurred)

    @pyqtSlot(float)
    def on_price_updated(self, price: float):
        self.price_label.setText(f"Current Price: {price:,.0f} KRW")

    @pyqtSlot(dict)
    def on_orderbook_updated(self, orderbook: Dict):
        # 호가창 데이터 렌더링 로직 (예시로 텍스트화)
        ask1 = orderbook.get("ask1", 0)
        bid1 = orderbook.get("bid1", 0)
        self.orderbook_label.setText(f"Ask 1: {ask1} \nBid 1: {bid1}")

    @pyqtSlot(str)
    def on_error_occurred(self, error_msg: str):
        # 예외 상황을 UI 붉은색 텍스트 등으로 렌더링
        self.orderbook_label.setText(f"<html><font color='red'>Error: {error_msg}</font></html>")
