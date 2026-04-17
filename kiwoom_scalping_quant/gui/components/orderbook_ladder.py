from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import pyqtSlot
from typing import Dict
import pyqtgraph as pg
import numpy as np

class OrderbookLadderWidget(QWidget):
    """
    ViewModel의 시그널을 받아 호가창을 렌더링하는 순수 뷰 컴포넌트.
    pyqtgraph의 BarGraphItem을 사용하여 양방향 호가 잔량을 히트맵처럼 표시합니다.
    """
    def __init__(self, view_model):
        super().__init__()
        self.view_model = view_model
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        self.layout = QVBoxLayout(self)
        self.title_label = QLabel("호가창 래더")
        self.layout.addWidget(self.title_label)

        self.price_label = QLabel("현재가: -")
        self.price_label.setStyleSheet("font-size: 18px; font-weight: bold; color: orange;")
        self.layout.addWidget(self.price_label)

        # PyQtGraph 설정
        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'w')

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=False, y=True)
        self.plot_widget.setLabel('bottom', "잔량")
        self.plot_widget.setLabel('left', "가격")

        # 양방향 바 그래프 생성
        # Asks (매도) -> Red, Negative X (왼쪽)
        self.bar_asks = pg.BarGraphItem(x0=[0]*10, y=np.arange(10), height=0.6, width=0, brush='r')
        # Bids (매수) -> Blue, Positive X (오른쪽)
        self.bar_bids = pg.BarGraphItem(x0=[0]*10, y=np.arange(10), height=0.6, width=0, brush='b')

        self.plot_widget.addItem(self.bar_asks)
        self.plot_widget.addItem(self.bar_bids)

        self.layout.addWidget(self.plot_widget, stretch=1)

    def _connect_signals(self):
        # 새로운 시그널 이름에 맞추어 연결
        if hasattr(self.view_model, 'sig_price_updated'):
            self.view_model.sig_price_updated.connect(self.on_price_updated)
        if hasattr(self.view_model, 'sig_orderbook_updated'):
            self.view_model.sig_orderbook_updated.connect(self.on_orderbook_updated)

    @pyqtSlot(float)
    def on_price_updated(self, price: float):
        self.price_label.setText(f"현재가: {price:,.0f} 원")

    @pyqtSlot(dict)
    def on_orderbook_updated(self, orderbook: Dict):
        # Mock 데이터 구조: {"asks": [{"price": p, "qty": q}, ...], "bids": [...]}
        asks = orderbook.get("asks", [])
        bids = orderbook.get("bids", [])

        if not asks or not bids: return

        ask_prices = [item["price"] for item in asks]
        ask_qtys = [-item["qty"] for item in asks] # 왼쪽 방향으로 그리기 위해 음수 처리

        bid_prices = [item["price"] for item in bids]
        bid_qtys = [item["qty"] for item in bids]

        # Y축을 가격으로 변경하고 렌더링
        self.bar_asks.setOpts(y=ask_prices, width=ask_qtys, height=(ask_prices[1]-ask_prices[0])*0.8 if len(ask_prices)>1 else 50)
        self.bar_bids.setOpts(y=bid_prices, width=bid_qtys, height=(bid_prices[0]-bid_prices[1])*0.8 if len(bid_prices)>1 else 50)
