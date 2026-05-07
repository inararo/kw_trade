import os
import pandas as pd
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QPushButton, QLabel, QDateEdit, QFileDialog, QMessageBox, QSplitter, QComboBox, QProgressDialog)
from PyQt6.QtCore import QDate, Qt, pyqtSlot, QTimer
import pyqtgraph as pg

class BacktestStudioTab(QWidget):
    def __init__(self, view_model):
        super().__init__()
        self.view_model = view_model
        self.progress_dialog = None
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # --- 1. Top Control Panel ---
        ctrl_group = QGroupBox("백테스트 설정")
        group_layout = QVBoxLayout() # 세로 레이아웃으로 변경
        
        # 1-1. 설정 행 (종목, 날짜, 모델 로드)
        settings_layout = QHBoxLayout()
        settings_layout.setContentsMargins(0, 5, 0, 5)
        settings_layout.setSpacing(15)

        settings_layout.addWidget(QLabel("종목 선택:"))
        self.combo_symbol = QComboBox()
        self.combo_symbol.setMinimumWidth(150)
        self._populate_symbols()
        settings_layout.addWidget(self.combo_symbol)
        
        settings_layout.addWidget(QLabel("시작일:"))
        self.date_start = QDateEdit(QDate.currentDate().addMonths(-1))
        self.date_start.setCalendarPopup(True)
        self.date_start.setMinimumWidth(110)
        self.date_start.wheelEvent = lambda event: None # 휠 스크롤에 의한 날짜 변경 방지
        settings_layout.addWidget(self.date_start)
        
        settings_layout.addWidget(QLabel("종료일:"))
        self.date_end = QDateEdit(QDate.currentDate())
        self.date_end.setCalendarPopup(True)
        self.date_end.setMinimumWidth(110)
        self.date_end.wheelEvent = lambda event: None # 휠 스크롤에 의한 날짜 변경 방지
        settings_layout.addWidget(self.date_end)
        
        # 모델 로드 버튼 및 경로 레이블
        self.btn_load_model = QPushButton("모델 로드 (.zip)")
        self.btn_load_model.clicked.connect(self._on_load_model)
        settings_layout.addWidget(self.btn_load_model)
        
        self.lbl_model_path = QLabel("선택된 모델: 없음")
        self.lbl_model_path.setStyleSheet("color: #888888; font-size: 11px;")
        settings_layout.addWidget(self.lbl_model_path)
        
        settings_layout.addStretch(1)
        group_layout.addLayout(settings_layout)

        # 1-2. 버튼 행 (실행 버튼들)
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 5, 0, 5)
        btn_layout.setSpacing(10)

        # 실행 버튼 1: 단일 백테스트
        self.btn_start = QPushButton("백테스트 시작")
        self.btn_start.setMinimumWidth(150)
        self.btn_start.setFixedHeight(35)
        self.btn_start.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold;")
        self.btn_start.clicked.connect(self._on_start_backtest)
        btn_layout.addWidget(self.btn_start)

        # 실행 버튼 2: 자동 배치 (Top 30)
        self.btn_auto_batch = QPushButton("자동 백테스트 시작 (Top 30)")
        self.btn_auto_batch.setMinimumWidth(200)
        self.btn_auto_batch.setFixedHeight(35)
        self.btn_auto_batch.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_auto_batch.setToolTip("상위 30개 종목에 대해 일괄 백테스트를 수행합니다.")
        self.btn_auto_batch.clicked.connect(self._on_start_auto_batch)
        btn_layout.addWidget(self.btn_auto_batch)

        # 실행 버튼 3: 다중 모델 일괄 배치
        self.btn_multi_batch = QPushButton("일괄 백테스트 (다중 모델 x 전체 종목)")
        self.btn_multi_batch.setMinimumWidth(250)
        self.btn_multi_batch.setFixedHeight(35)
        self.btn_multi_batch.setStyleSheet("background-color: #673AB7; color: white; font-weight: bold;")
        self.btn_multi_batch.clicked.connect(self._on_start_multi_model_batch)
        btn_layout.addWidget(self.btn_multi_batch)

        btn_layout.addStretch(1) # 버튼들을 왼쪽으로 정렬 (필요시 양쪽에 Stretch를 주어 가운데 정렬 가능)
        group_layout.addLayout(btn_layout)

        ctrl_group.setLayout(group_layout)
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

        # [신규] 평단가 Line (보유 중일 때만 표시)
        self.avg_entry_curve = self.plot_widget.plot(pen=pg.mkPen('#FFD700', width=1, style=Qt.PenStyle.DashLine), name="AvgEntry")

        # Scatter plots for Buy/Sell markers
        # Buy 40% (Small Green), Buy 60% (Large Green)
        self.buy_40_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0, 200), symbol='t1')
        self.buy_60_scatter = pg.ScatterPlotItem(size=15, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0, 255), symbol='t1')
        
        # Sell 40% (Small Blue), Sell 60% (Large Blue)
        self.sell_40_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(0, 100, 255, 200), symbol='t')
        self.sell_60_scatter = pg.ScatterPlotItem(size=15, pen=pg.mkPen(None), brush=pg.mkBrush(0, 100, 255, 255), symbol='t')

        self.plot_widget.addItem(self.avg_entry_curve)
        self.plot_widget.addItem(self.buy_40_scatter)
        self.plot_widget.addItem(self.buy_60_scatter)
        self.plot_widget.addItem(self.sell_40_scatter)
        self.plot_widget.addItem(self.sell_60_scatter)

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
        # [FIX] 파일 다이얼로그 시작 경로를 saved_models 폴더로 고정
        base_dir = os.path.join(os.getcwd(), "saved_models")
        if not os.path.exists(base_dir): os.makedirs(base_dir, exist_ok=True)
        
        file_path, _ = QFileDialog.getOpenFileName(self, "학습 모델 선택", base_dir, "Zip Files (*.zip)")
        if file_path:
            self.lbl_model_path.setText(f"선택된 모델: {os.path.basename(file_path)}")
            self.view_model.set_model_path(file_path)

    def _validate_dates(self) -> bool:
        """시작일이 종료일보다 늦은지 검사"""
        if self.date_start.date() > self.date_end.date():
            QMessageBox.warning(self, "날짜 설정 오류", 
                                "시작일이 종료일보다 늦을 수 없습니다.\n날짜 범위를 다시 확인해 주세요.")
            return False
        return True

    def _on_start_backtest(self):
        if not self._validate_dates(): return
        start_dt = self.date_start.date().toString("yyyyMMdd")
        end_dt = self.date_end.date().toString("yyyyMMdd")
        symbol = self.combo_symbol.currentData() # code
        
        if not symbol:
            QMessageBox.warning(self, "경고", "테스트할 종목을 선택해주세요.")
            return

        self.btn_start.setEnabled(False)
        self.price_curve.setData([], [])
        self.avg_entry_curve.setData([], [])
        self.buy_40_scatter.setData([])
        self.buy_60_scatter.setData([])
        self.sell_40_scatter.setData([])
        self.sell_60_scatter.setData([])
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
        if self.progress_dialog and self.progress_dialog.isVisible():
            pct = int((step / total) * 100) if total > 0 else 0
            self.progress_dialog.setValue(pct)
            self.progress_dialog.setLabelText(f"배치 진행 중... ({step}/{total})")
            
        pct_text = (step / total) * 100 if total > 0 else 0
        self.lbl_progress.setText(f"진행률: {step}/{total} ({pct_text:.1f}%) | 누적 PnL: {pnl:,.0f}")

    def _on_start_auto_batch(self):
        """자동 백테스트 배치 시작 호출"""
        if not self._validate_dates(): return
        start_dt = self.date_start.date().toString("yyyyMMdd")
        end_dt = self.date_end.date().toString("yyyyMMdd")

        reply = QMessageBox.question(
            self, "자동 백테스트 확인",
            f"선택한 기간({start_dt} ~ {end_dt}) 동안 거래량 상위 30개 종목에 대해 일괄 백테스트를 시작하시겠습니까?\n"
            "(DB에 데이터가 없는 종목은 자동으로 제외됩니다.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.btn_auto_batch.setEnabled(False)
            self.lbl_progress.setText("진행률: 배치 시작 중...")
            
            # 프로그레스 다이얼로그 생성
            self.progress_dialog = QProgressDialog("자동 백테스트 배치 작업 중...", None, 0, 100, self)
            self.progress_dialog.setWindowTitle("배치 시뮬레이션")
            self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            self.progress_dialog.setAutoClose(True)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.show()

            self.view_model.start_auto_backtest_batch(start_dt, end_dt)

    def _on_start_multi_model_batch(self):
        """다중 모델 x 전 종목 일괄 백테스트 시작"""
        if not self._validate_dates(): return
        base_dir = os.path.join(os.getcwd(), "saved_models")
        files, _ = QFileDialog.getOpenFileNames(self, "백테스트 모델 다중 선택", base_dir, "Model Files (*.zip)")
        
        if not files:
            return

        start_dt = self.date_start.date().toString("yyyyMMdd")
        end_dt = self.date_end.date().toString("yyyyMMdd")

        reply = QMessageBox.question(
            self, "일괄 백테스트 확인",
            f"선택한 {len(files)}개의 모델과 모든 유니버스 종목에 대해 일괄 테스트를 시작하시겠습니까?\n"
            f"기간: {start_dt} ~ {end_dt}\n(결과는 CSV 파일로 저장됩니다.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.btn_multi_batch.setEnabled(False)
            self.lbl_progress.setText("일괄 백테스트 대기 중...")
            
            self.progress_dialog = QProgressDialog("다중 모델 일괄 백테스트 진행 중...", "중단", 0, 100, self)
            self.progress_dialog.setWindowTitle("일괄 시뮬레이션")
            self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            self.progress_dialog.canceled.connect(self.view_model.stop_batch_backtest)
            self.progress_dialog.show()

            self.view_model.start_batch_backtest(files, start_dt, end_dt)

    @pyqtSlot(dict)
    def on_bt_finished(self, kpi: dict):
        if self.progress_dialog:
            self.progress_dialog.close()
        self.btn_start.setEnabled(True)
        self.btn_auto_batch.setEnabled(True)
        self.lbl_progress.setText("진행률: 완료")

        if "Batch Count" in kpi:
            count = kpi["Batch Count"]
            path = kpi.get("CSV_Path", "알 수 없음")
            QMessageBox.information(self, "백테스트 완료", 
                                    f"일괄 백테스트가 완료되었습니다.\n\n"
                                    f"- 총 결과 수: {count}건\n"
                                    f"- 저장 경로: {path}")
            self.btn_multi_batch.setEnabled(True)
            return

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

        # 3. 평단가 데이터 (보유 중일 때만 유효값, 아니면 NaN)
        import numpy as np
        avg_entries = df['avg_entry_price'].values.copy()
        holdings = df['holdings'].values
        avg_entries[holdings == 0] = np.nan
        self.avg_entry_curve.setData(steps, avg_entries)

        # 4. 매매 마커 (Action별 분기)
        # Buy 40%
        b40 = df[df['action'] == 'Buy40%']
        self.buy_40_scatter.setData(x=b40['step'].values, y=b40['low'].values * 0.998)
        
        # Buy 60%
        b60 = df[df['action'] == 'Buy60%']
        self.buy_60_scatter.setData(x=b60['step'].values, y=b60['low'].values * 0.998)
        
        # Sell 40%
        s40 = df[df['action'] == 'Sell40%']
        self.sell_40_scatter.setData(x=s40['step'].values, y=s40['high'].values * 1.002)
        
        # Sell 60%
        s60 = df[df['action'] == 'Sell60%']
        self.sell_60_scatter.setData(x=s60['step'].values, y=s60['high'].values * 1.002)

    @pyqtSlot(str)
    def on_bt_error(self, err_msg: str):
        if self.progress_dialog:
            self.progress_dialog.close()
        self.btn_start.setEnabled(True)
        self.btn_auto_batch.setEnabled(True)
        self.lbl_progress.setText("진행률: 에러 발생")
        QMessageBox.critical(self, "백테스트 에러", f"시뮬레이션 중 오류가 발생했습니다:\n{err_msg}")

