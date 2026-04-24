import os
import pandas as pd
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QPushButton, QLabel, QDateEdit, QFileDialog, QMessageBox, QSplitter)
from PyQt6.QtCore import QDate, Qt, pyqtSlot
import pyqtgraph as pg

class BacktestStudioTab(QWidget):
    def __init__(self, view_model):
        super().__init__()
        self.view_model = view_model
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # --- 1. Top Control Panel ---
        ctrl_group = QGroupBox("백테스트 설정")
        ctrl_layout = QHBoxLayout()

        ctrl_layout.addWidget(QLabel("종목 선택:"))
        self.combo_symbol = pg.ComboBox() # pyqtgraph's ComboBox is fine, or QComboBox
        from PyQt6.QtWidgets import QComboBox
        self.combo_symbol = QComboBox()
        self._populate_symbols()
        ctrl_layout.addWidget(self.combo_symbol)

        ctrl_layout.addWidget(QLabel("시작일:"))
        self.date_start = QDateEdit(QDate.currentDate().addMonths(-1))
        self.date_start.setCalendarPopup(True)
        ctrl_layout.addWidget(self.date_start)

        ctrl_layout.addWidget(QLabel("종료일:"))
        self.date_end = QDateEdit(QDate.currentDate())
        self.date_end.setCalendarPopup(True)
        ctrl_layout.addWidget(self.date_end)

        self.btn_load_model = QPushButton("모델 로드 (.zip)")
        self.btn_load_model.clicked.connect(self._on_load_model)
        ctrl_layout.addWidget(self.btn_load_model)
        self.lbl_model_path = QLabel("선택된 모델: 없음")
        ctrl_layout.addWidget(self.lbl_model_path)

        self.btn_start = QPushButton("백테스트 시작")
        self.btn_start.setStyleSheet("background-color: #2b5b84; color: white;")
        self.btn_start.clicked.connect(self._on_start_backtest)
        ctrl_layout.addWidget(self.btn_start)

        ctrl_group.setLayout(ctrl_layout)
        main_layout.addWidget(ctrl_group)

        # Splitter for Chart and Results
        splitter = QSplitter(Qt.Orientation.Vertical)

        # --- 2. Center Chart Panel (pyqtgraph) ---
        chart_group = QGroupBox("시가총액 및 매매 타점 시각화 (Trade Visualizer)")
        chart_layout = QVBoxLayout()

        pg.setConfigOption('background', '#1e1e1e')
        pg.setConfigOption('foreground', 'w')
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', "스텝 (Step)")
        self.plot_widget.setLabel('left', "가격 (Price)")
        self.plot_widget.showGrid(x=True, y=True)

        # Price Line
        self.price_curve = self.plot_widget.plot(pen=pg.mkPen('w', width=1.5), name="Price")

        # Scatter plots for Buy/Sell markers
        self.buy_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 200), symbol='t1') # Up pointing triangle
        self.sell_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(0, 0, 255, 200), symbol='t')  # Down pointing triangle
        self.plot_widget.addItem(self.buy_scatter)
        self.plot_widget.addItem(self.sell_scatter)

        chart_layout.addWidget(self.plot_widget)
        chart_group.setLayout(chart_layout)
        splitter.addWidget(chart_group)

        # --- 3. Bottom KPI Panel ---
        kpi_group = QGroupBox("성과 분석 (KPI Dashboard)")
        kpi_layout = QHBoxLayout()

        self.lbl_return = QLabel("총 수익률: - %")
        self.lbl_return.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.lbl_winrate = QLabel("승률: - %")
        self.lbl_mdd = QLabel("MDD: - %")
        self.lbl_profit_factor = QLabel("Profit Factor: -")
        self.lbl_progress = QLabel("진행률: 대기 중")

        kpi_layout.addWidget(self.lbl_return)
        kpi_layout.addWidget(self.lbl_winrate)
        kpi_layout.addWidget(self.lbl_mdd)
        kpi_layout.addWidget(self.lbl_profit_factor)
        kpi_layout.addStretch()
        kpi_layout.addWidget(self.lbl_progress)

        kpi_group.setLayout(kpi_layout)
        splitter.addWidget(kpi_group)

        main_layout.addWidget(splitter, stretch=1)

    def _connect_signals(self):
        from PyQt6.QtCore import Qt
        self.view_model.sig_bt_progress.connect(self.on_bt_progress)
        self.view_model.sig_bt_finished.connect(self.on_bt_finished, Qt.ConnectionType.QueuedConnection)
        self.view_model.sig_bt_error.connect(self.on_bt_error, Qt.ConnectionType.QueuedConnection)
        self.view_model.sig_bt_chart_data.connect(self.on_bt_chart_data)

    def _on_load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "학습 모델 선택", "", "Zip Files (*.zip)")
        if file_path:
            self.lbl_model_path.setText(f"선택된 모델: {os.path.basename(file_path)}")
            self.view_model.set_model_path(file_path)

    def _on_start_backtest(self):
        start_dt = self.date_start.date().toString("yyyyMMdd")
        end_dt = self.date_end.date().toString("yyyyMMdd")
        symbol = self.combo_symbol.currentData() # code
        
        if not symbol:
            QMessageBox.warning(self, "경고", "테스트할 종목을 선택해주세요.")
            return

        self.btn_start.setEnabled(False)
        self.price_curve.setData([], [])
        self.buy_scatter.setData([])
        self.sell_scatter.setData([])
        self.lbl_progress.setText("진행률: 시작...")

        self.view_model.start_backtest(start_dt, end_dt, symbol)

    @pyqtSlot()
    @pyqtSlot(list)
    def _populate_symbols(self, symbols=None):
        """종목 선택 콤보박스 아이템 갱신"""
        self.combo_symbol.clear()
        if symbols is None:
            symbols = self.view_model.config_manager.get_symbols()
            
        for s in symbols:
            name = s.get("name", "Unknown")
            code = s.get("code", "")
            self.combo_symbol.addItem(f"{name} ({code})", code)

    @pyqtSlot(int, int, float)
    def on_bt_progress(self, step: int, total: int, pnl: float):
        pct = (step / total) * 100 if total > 0 else 0
        self.lbl_progress.setText(f"진행률: {step}/{total} ({pct:.1f}%) | 누적 PnL: {pnl:,.0f}")

    @pyqtSlot(dict)
    def on_bt_finished(self, kpi: dict):
        self.btn_start.setEnabled(True)
        self.lbl_progress.setText("진행률: 완료")

        self.lbl_return.setText(f"총 수익률: {kpi.get('Total Return', 0):.2f} %")
        if kpi.get('Total Return', 0) > 0:
            self.lbl_return.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
        else:
            self.lbl_return.setStyleSheet("color: green; font-size: 16px; font-weight: bold;")

        self.lbl_winrate.setText(f"승률: {kpi.get('Win Rate', 0):.2f} %")
        self.lbl_mdd.setText(f"MDD: {kpi.get('MDD', 0):.2f} %")
        pf = kpi.get('Profit Factor', 0)
        self.lbl_profit_factor.setText(f"Profit Factor: {'inf' if pf == float('inf') else f'{pf:.2f}'}")

        QMessageBox.information(self, "백테스트 완료", "백테스트 시뮬레이션 및 분석이 완료되었습니다.")

    @pyqtSlot(object)
    def on_bt_chart_data(self, df):
        if df is None or df.empty:
            return

        # 차트 데이터 시각화
        steps = df['step'].values
        prices = df['price'].values

        self.price_curve.setData(steps, prices)

        # [혁신] Y축 자동 스케일링: 데이터 범위에 맞춰 축 최적화
        if len(prices) > 0:
            y_min, y_max = prices.min(), prices.max()
            # 가격 변동폭의 10%를 마진으로 적용 (변동이 없는 경우 대비 0.01% 최소 마진)
            margin = (y_max - y_min) * 0.1 if y_max > y_min else prices[0] * 0.01
            # 만약 가격이 0이라면 마진을 기본값으로 설정
            if margin == 0: margin = 100 
            
            self.plot_widget.setYRange(y_min - margin, y_max + margin, padding=0)
            self.plot_widget.setXRange(steps.min(), steps.max(), padding=0.02)
            self.plot_widget.enableAutoRange(axis='y', enable=False) # 수동 설정 후 자동추적 중지 (고정)

        buys = df[df['action'] == 'Buy']
        sells = df[df['action'] == 'Sell']

        buy_x = buys['step'].values
        buy_y = buys['price'].values
        self.buy_scatter.setData(x=buy_x, y=buy_y)

        sell_x = sells['step'].values
        sell_y = sells['price'].values
        self.sell_scatter.setData(x=sell_x, y=sell_y)

    @pyqtSlot(str)
    def on_bt_error(self, err_msg: str):
        self.btn_start.setEnabled(True)
        self.lbl_progress.setText("진행률: 에러 발생")
        QMessageBox.critical(self, "백테스트 에러", f"시뮬레이션 중 오류가 발생했습니다:\n{err_msg}")
