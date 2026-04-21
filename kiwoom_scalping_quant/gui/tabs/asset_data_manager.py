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
        asset_group = QGroupBox("종목 리스트 관리")
        asset_layout = QVBoxLayout()

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["종목코드", "종목명"])
        asset_layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("종목 추가")
        self.btn_add.clicked.connect(self._on_btn_add_clicked)

        self.btn_remove = QPushButton("선택 삭제")
        self.btn_remove.clicked.connect(self._on_btn_remove_clicked)

        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        asset_layout.addLayout(btn_layout)

        self.btn_auto_universe = QPushButton("주도주 유니버스 자동 생성 (Top 20)")
        self.btn_auto_universe.setStyleSheet("background-color: #2b5b84; color: white;")
        self.btn_auto_universe.clicked.connect(self._on_btn_auto_universe_clicked)
        asset_layout.addWidget(self.btn_auto_universe)

        asset_group.setLayout(asset_layout)
        main_layout.addWidget(asset_group, stretch=1)

        # 2. 우측: 과거 데이터 수집기
        data_group = QGroupBox("과거 데이터 수집기")
        data_layout = QVBoxLayout()

        data_layout.addWidget(QLabel("대상 종목: 표에서 선택"))

        data_layout.addWidget(QLabel("수집 시작일:"))
        self.date_start = QDateEdit(QDate.currentDate())
        self.date_start.setCalendarPopup(True)
        data_layout.addWidget(self.date_start)

        self.btn_collect = QPushButton("선택 종목 데이터 수집")
        self.btn_collect.clicked.connect(self._on_btn_collect_clicked)
        data_layout.addWidget(self.btn_collect)

        self.btn_collect_all = QPushButton("전체 종목 데이터 수집")
        self.btn_collect_all.setStyleSheet("background-color: #5b2b84; color: white;")
        self.btn_collect_all.clicked.connect(self._on_btn_collect_all_clicked)
        data_layout.addWidget(self.btn_collect_all)

        self.lbl_progress_msg = QLabel("대기 중")
        data_layout.addWidget(self.lbl_progress_msg)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        data_layout.addWidget(self.progress_bar)

        data_layout.addStretch()
        data_group.setLayout(data_layout)
        main_layout.addWidget(data_group, stretch=1)

    def _connect_signals(self):
        from PyQt6.QtCore import Qt
        self.view_model.symbols_loaded.connect(self.on_symbols_loaded)
        self.view_model.symbol_update_failed.connect(self.on_error, Qt.ConnectionType.QueuedConnection)
        self.view_model.symbol_update_success.connect(self.on_success, Qt.ConnectionType.QueuedConnection)

        self.view_model.sig_progress_updated.connect(self.on_progress_updated)
        self.view_model.sig_status_updated.connect(self.on_status_updated)
        self.view_model.fetch_completed.connect(self.on_fetch_completed, Qt.ConnectionType.QueuedConnection)
        self.view_model.fetch_failed.connect(self.on_error, Qt.ConnectionType.QueuedConnection)

    # --- UI Actions (View -> ViewModel) ---
    def _on_btn_add_clicked(self):
        code, ok1 = QInputDialog.getText(self, "종목 추가", "종목 코드를 입력하세요 (예: 005930):")
        if ok1 and code:
            name, ok2 = QInputDialog.getText(self, "종목 추가", "종목명을 입력하세요:")
            if ok2 and name:
                self.view_model.add_symbol(code, name)

    def _on_btn_remove_clicked(self):
        current_row = self.table.currentRow()
        if current_row >= 0:
            code_item = self.table.item(current_row, 0)
            if code_item:
                self.view_model.remove_symbol(code_item.text())
        else:
            QMessageBox.warning(self, "경고", "삭제할 종목을 표에서 선택해주세요.")

    def _on_btn_auto_universe_clicked(self):
        reply = QMessageBox.question(self, "확인", "기존 종목 리스트가 삭제되고 주도주 Top 20으로 교체됩니다. 진행하시겠습니까?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.view_model.build_universe()

    def _on_btn_collect_clicked(self):
        current_row = self.table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "경고", "먼저 표에서 수집할 종목을 선택해주세요.")
            return

        symbol = self.table.item(current_row, 0).text()
        self.btn_collect.setEnabled(False)
        self.btn_collect_all.setEnabled(False)
        start_date = self.date_start.date().toString("yyyyMMdd")
        self.view_model.start_historical_fetch(symbol, start_date)

    def _on_btn_collect_all_clicked(self):
        if self.table.rowCount() == 0:
            QMessageBox.warning(self, "경고", "수집할 종목이 없습니다. 먼저 종목을 추가해주세요.")
            return

        self.btn_collect.setEnabled(False)
        self.btn_collect_all.setEnabled(False)
        start_date = self.date_start.date().toString("yyyyMMdd")
        self.view_model.start_bulk_historical_fetch(start_date)

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
        QMessageBox.information(self, "성공", msg)

    @pyqtSlot(str)
    def on_error(self, msg: str):
        self.btn_collect.setEnabled(True)
        self.btn_collect_all.setEnabled(True)
        QMessageBox.warning(self, "오류", msg)

    @pyqtSlot(int)
    def on_progress_updated(self, pct: int):
        self.progress_bar.setValue(pct)

    @pyqtSlot(str)
    def on_status_updated(self, msg: str):
        self.lbl_progress_msg.setText(msg)

    @pyqtSlot(str)
    def on_fetch_completed(self, msg: str):
        self.btn_collect.setEnabled(True)
        self.btn_collect_all.setEnabled(True)
        self.lbl_progress_msg.setText("완료")
        QMessageBox.information(self, "다운로드 완료", msg)
