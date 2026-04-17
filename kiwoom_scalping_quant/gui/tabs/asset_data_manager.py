from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QDateEdit, QProgressBar, QLabel, QGroupBox, QMessageBox, QInputDialog
from PyQt6.QtCore import QDate, pyqtSlot

class AssetDataManagerTab(QWidget):
    """
    탭 B: 종목 및 데이터 관리
    종목 리스트 편집(CRUD) 및 과거 데이터 수집 기능.
    """
    def __init__(self, view_model):
        super().__init__()
        self.view_model = view_model
        self._init_ui()
        self._connect_signals()

        # UI 로딩 완료 후 ViewModel에 데이터 요청
        self.view_model.load_symbols()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        # 1. 좌측: 종목 리스트 관리
        asset_group = QGroupBox("Asset Manager (Symbol List)")
        asset_layout = QVBoxLayout()

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Symbol", "Name"])
        asset_layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("Add Symbol")
        self.btn_add.clicked.connect(self._on_btn_add_clicked)

        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.clicked.connect(self._on_btn_remove_clicked)

        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        asset_layout.addLayout(btn_layout)

        asset_group.setLayout(asset_layout)
        main_layout.addWidget(asset_group, stretch=1)

        # 2. 우측: 과거 데이터 수집기
        data_group = QGroupBox("Historical Data Collector")
        data_layout = QVBoxLayout()

        data_layout.addWidget(QLabel("Target Symbol:"))
        # 실제로는 Table에서 선택된 값을 가져오도록 연동
        self.lbl_target = QLabel("005930 (Samsung)")
        data_layout.addWidget(self.lbl_target)

        data_layout.addWidget(QLabel("Start Date:"))
        self.date_start = QDateEdit(QDate.currentDate().addDays(-30))
        self.date_start.setCalendarPopup(True)
        data_layout.addWidget(self.date_start)

        self.btn_collect = QPushButton("Start Download to InfluxDB")
        self.btn_collect.clicked.connect(self._on_btn_collect_clicked)
        data_layout.addWidget(self.btn_collect)

        self.lbl_progress_msg = QLabel("Ready")
        data_layout.addWidget(self.lbl_progress_msg)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        data_layout.addWidget(self.progress_bar)

        data_layout.addStretch()
        data_group.setLayout(data_layout)
        main_layout.addWidget(data_group, stretch=1)

    def _connect_signals(self):
        self.view_model.symbols_loaded.connect(self.on_symbols_loaded)
        self.view_model.symbol_update_failed.connect(self.on_error)
        self.view_model.symbol_update_success.connect(self.on_success)

        self.view_model.fetch_progress_updated.connect(self.on_fetch_progress)
        self.view_model.fetch_completed.connect(self.on_fetch_completed)
        self.view_model.fetch_failed.connect(self.on_error)

    # --- UI Actions (View -> ViewModel) ---
    def _on_btn_add_clicked(self):
        code, ok1 = QInputDialog.getText(self, "Add Symbol", "Enter Symbol Code (e.g. 005930):")
        if ok1 and code:
            name, ok2 = QInputDialog.getText(self, "Add Symbol", "Enter Symbol Name:")
            if ok2 and name:
                # 비즈니스 로직(저장, 예외처리)은 ViewModel에 위임
                self.view_model.add_symbol(code, name)

    def _on_btn_remove_clicked(self):
        current_row = self.table.currentRow()
        if current_row >= 0:
            code_item = self.table.item(current_row, 0)
            if code_item:
                self.view_model.remove_symbol(code_item.text())
        else:
            QMessageBox.warning(self, "Warning", "Please select a symbol to remove.")

    def _on_btn_collect_clicked(self):
        current_row = self.table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Warning", "Please select a symbol from the table first.")
            return

        symbol = self.table.item(current_row, 0).text()
        self.btn_collect.setEnabled(False)
        start_date = self.date_start.date().toString("yyyyMMdd")
        self.view_model.start_historical_fetch(symbol, start_date)

    # --- Slots (ViewModel -> View) ---
    @pyqtSlot(list)
    def on_symbols_loaded(self, symbols: list):
        self.table.setRowCount(0)
        for row, sym in enumerate(symbols):
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(sym.get("code", "")))
            self.table.setItem(row, 1, QTableWidgetItem(sym.get("name", "")))

    @pyqtSlot(str)
    def on_success(self, msg: str):
        QMessageBox.information(self, "Success", msg)

    @pyqtSlot(str)
    def on_error(self, msg: str):
        self.btn_collect.setEnabled(True)
        QMessageBox.warning(self, "Error", msg)

    @pyqtSlot(int, str)
    def on_fetch_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.lbl_progress_msg.setText(msg)

    @pyqtSlot(str)
    def on_fetch_completed(self, msg: str):
        self.btn_collect.setEnabled(True)
        self.lbl_progress_msg.setText("Complete")
        QMessageBox.information(self, "Download Complete", msg)
