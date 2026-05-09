from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QDateEdit, QProgressBar, QLabel, QGroupBox, QMessageBox, QInputDialog, QSpinBox, QSplitter, QSizePolicy, QRadioButton, QButtonGroup
from PyQt6.QtCore import QDate, pyqtSlot, Qt

class AssetDataManagerTab(QWidget):
    """
    탭 B: 데이터 관리
    종목 리스트 편집(CRUD) 및 과거 데이터 수집 기능.
    """
    def __init__(self, view_model):
        super().__init__()
        self.view_model = view_model
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # 1. 좌측: 종목 리스트 관리
        asset_group = QGroupBox("종목 리스트 관리")
        asset_layout = QVBoxLayout()

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["선택", "종목코드", "종목명", "현재가", "등락률", "거래량"])
        self.table.horizontalHeader().setStretchLastSection(True)
        # 선택 열 너비 조정
        self.table.setColumnWidth(0, 40)
        self.table.setColumnWidth(2, 120)  # 종목명 컬럼 너비 충분히 확보
        
        # 선택된 행 하이라이트 강화 (밝은 파란색 계열)
        self.table.setStyleSheet("""
            QTableWidget::item:selected {
                background-color: #007acc;
                color: white;
                font-weight: bold;
            }
        """)
        
        # 테이블 사이즈 정책 조정 (하단 버튼이 밀리지 않도록)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        self.table.setMinimumHeight(200)
        
        asset_layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("전체 선택")
        self.btn_select_all.clicked.connect(self._on_btn_select_all_clicked)
        
        self.btn_deselect_all = QPushButton("전체 해제")
        self.btn_deselect_all.clicked.connect(self._on_btn_deselect_all_clicked)

        self.btn_add = QPushButton("종목 추가")
        self.btn_add.clicked.connect(self._on_btn_add_clicked)

        self.btn_remove = QPushButton("선택 삭제")
        self.btn_remove.clicked.connect(self._on_btn_remove_clicked)

        btn_layout.addWidget(self.btn_select_all)
        btn_layout.addWidget(self.btn_deselect_all)
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        asset_layout.addLayout(btn_layout)

        # 유니버스 생성 개수 입력 추가
        univ_ctrl_layout = QHBoxLayout()
        univ_ctrl_layout.addWidget(QLabel("수집 종목 수 (Top N):"))
        self.spin_top_n = QSpinBox()
        self.spin_top_n.setRange(1, 200)
        self.spin_top_n.setValue(20)
        self.spin_top_n.setMinimumSize(90, 30) # 너비와 높이를 충분히 확보
        self.spin_top_n.setStyleSheet("""
            QSpinBox {
                padding-right: 30px; 
                background-color: #2b2b2b; 
                color: white; 
                border: 1px solid #444;
            }
        """)
        univ_ctrl_layout.addWidget(self.spin_top_n)
        
        # [신규] 정렬 기준 라디오 버튼 추가
        self.sort_group = QButtonGroup(self)
        
        self.radio_vol = QRadioButton("거래량")
        self.radio_val = QRadioButton("거래대금")
        self.radio_flu = QRadioButton("등락률")
        
        # 거래량 기본 선택
        self.radio_vol.setChecked(True)
        
        self.sort_group.addButton(self.radio_vol)
        self.sort_group.addButton(self.radio_val)
        self.sort_group.addButton(self.radio_flu)
        
        univ_ctrl_layout.addSpacing(20)
        univ_ctrl_layout.addWidget(self.radio_vol)
        univ_ctrl_layout.addWidget(self.radio_val)
        univ_ctrl_layout.addWidget(self.radio_flu)
        
        univ_ctrl_layout.addStretch()
        asset_layout.addLayout(univ_ctrl_layout)

        self.btn_auto_universe = QPushButton("주도주 유니버스 생성 (Top N)")
        self.btn_auto_universe.setStyleSheet("background-color: #2b5b84; color: white;")
        self.btn_auto_universe.clicked.connect(self._on_btn_auto_universe_clicked)
        asset_layout.addWidget(self.btn_auto_universe)

        self.btn_fetch_db = QPushButton("DB 종목 가져오기 (기수집 데이터)")
        self.btn_fetch_db.setStyleSheet("background-color: #388e3c; color: white;") # Greenish to distinguish
        self.btn_fetch_db.clicked.connect(self._on_btn_fetch_db_clicked)
        asset_layout.addWidget(self.btn_fetch_db)

        asset_group.setLayout(asset_layout)
        main_splitter.addWidget(asset_group)

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

        # [신규] DB 데이터 삭제 버튼 (빨간색 계열)
        self.btn_delete_db = QPushButton("선택 종목 DB 데이터 삭제")
        self.btn_delete_db.setStyleSheet("""
            QPushButton {
                background-color: #c62828; 
                color: white; 
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e53935;
            }
        """)
        self.btn_delete_db.clicked.connect(self._on_btn_delete_db_clicked)
        data_layout.addWidget(self.btn_delete_db)

        self.lbl_progress_msg = QLabel("대기 중")
        data_layout.addWidget(self.lbl_progress_msg)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        data_layout.addWidget(self.progress_bar)

        data_layout.addStretch()

        # [신규] Firebase 테스트 로그 전송 섹션
        test_group = QGroupBox("Firebase 연동 테스트")
        test_layout = QHBoxLayout()
        
        self.btn_test_buy = QPushButton("테스트 BUY 로그 전송")
        self.btn_test_buy.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold;")
        self.btn_test_buy.clicked.connect(self._on_btn_test_buy_clicked)
        
        self.btn_test_sell = QPushButton("테스트 SELL 로그 전송")
        self.btn_test_sell.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold;")
        self.btn_test_sell.clicked.connect(self._on_btn_test_sell_clicked)
        
        test_layout.addWidget(self.btn_test_buy)
        test_layout.addWidget(self.btn_test_sell)
        test_group.setLayout(test_layout)
        data_layout.addWidget(test_group)

        data_group.setLayout(data_layout)
        main_splitter.addWidget(data_group)
        
        layout.addWidget(main_splitter)
        
        # 스플리터 초기 비율 설정 (6:4)
        main_splitter.setSizes([600, 400])

    def _connect_signals(self):
        # UI 업데이트 시그널들을 모두 QueuedConnection으로 설정하여 쓰레드/비동기 안전성 확보
        self.view_model.symbols_loaded.connect(self.on_symbols_loaded, Qt.ConnectionType.QueuedConnection)
        self.view_model.symbol_update_failed.connect(self.on_error, Qt.ConnectionType.QueuedConnection)
        self.view_model.symbol_update_success.connect(self.on_success, Qt.ConnectionType.QueuedConnection)

        self.view_model.sig_progress_updated.connect(self.on_progress_updated, Qt.ConnectionType.QueuedConnection)
        self.view_model.sig_status_updated.connect(self.on_status_updated, Qt.ConnectionType.QueuedConnection)
        self.view_model.fetch_completed.connect(self.on_fetch_completed, Qt.ConnectionType.QueuedConnection)
        self.view_model.fetch_failed.connect(self.on_error, Qt.ConnectionType.QueuedConnection)

    # --- UI Actions (View -> ViewModel) ---
    def _on_btn_add_clicked(self):
        code, ok = QInputDialog.getText(self, "종목 추가", "종목 코드를 입력하세요 (예: 005930):")
        if ok and code:
            self.view_model.add_symbol(code.strip())

    def _on_btn_remove_clicked(self):
        checked_codes = []
        for r in range(self.table.rowCount()):
            chk_item = self.table.item(r, 0)
            if chk_item and chk_item.checkState() == Qt.CheckState.Checked:
                checked_codes.append(self.table.item(r, 1).text())

        if not checked_codes:
            # 체크박스 선택이 없으면 현재 선택된 행이라도 삭제 시도 (하위 호환)
            current_row = self.table.currentRow()
            if current_row >= 0:
                checked_codes.append(self.table.item(current_row, 1).text())

        if checked_codes:
            reply = QMessageBox.question(self, "삭제 확인", f"선택한 {len(checked_codes)}개 종목을 정말 삭제하시겠습니까?", 
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.view_model.remove_symbols(checked_codes)
        else:
            QMessageBox.warning(self, "경고", "삭제할 종목의 체크박스를 선택해주세요.")

    def _on_btn_select_all_clicked(self):
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item:
                item.setCheckState(Qt.CheckState.Checked)

    def _on_btn_deselect_all_clicked(self):
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item:
                item.setCheckState(Qt.CheckState.Unchecked)

    def _on_btn_auto_universe_clicked(self):
        top_n = self.spin_top_n.value()
        
        # [신규] 선택된 정렬 기준 확인
        sort_by = "volume"
        if self.radio_val.isChecked():
            sort_by = "value"
        elif self.radio_flu.isChecked():
            sort_by = "flu_rt"
            
        sort_nm = {"volume": "거래량", "value": "거래대금", "flu_rt": "등락률"}.get(sort_by)
        
        reply = QMessageBox.question(self, "확인", f"기존 종목 리스트가 삭제되고 {sort_nm} Top {top_n}으로 교체됩니다. 진행하시겠습니까?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.view_model.build_universe(top_n=top_n, sort_by=sort_by)

    def _on_btn_fetch_db_clicked(self):
        reply = QMessageBox.question(self, "확인", "DB에 저장된 모든 종목을 가져와 현재 유니버스를 교체하시겠습니까?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.view_model.fetch_db_symbols()

    def _on_btn_collect_clicked(self):
        checked_symbols = []
        for r in range(self.table.rowCount()):
            chk_item = self.table.item(r, 0)
            if chk_item and chk_item.checkState() == Qt.CheckState.Checked:
                checked_symbols.append(self.table.item(r, 1).text())

        if not checked_symbols:
            current_row = self.table.currentRow()
            if current_row >= 0:
                checked_symbols.append(self.table.item(current_row, 1).text())

        if not checked_symbols:
            QMessageBox.warning(self, "경고", "먼저 수집할 종목의 체크박스를 선택해주세요.")
            return

        self.btn_collect.setEnabled(False)
        self.btn_collect_all.setEnabled(False)
        start_date = self.date_start.date().toString("yyyyMMdd")
        self.view_model.start_historical_fetch(checked_symbols, start_date)

    def _on_btn_collect_all_clicked(self):
        if self.table.rowCount() == 0:
            QMessageBox.warning(self, "경고", "수집할 종목이 없습니다. 먼저 종목을 추가해주세요.")
            return

        self.btn_collect.setEnabled(False)
        self.btn_collect_all.setEnabled(False)
        start_date = self.date_start.date().toString("yyyyMMdd")
        self.view_model.start_bulk_historical_fetch(start_date)

    def _on_btn_delete_db_clicked(self):
        """선택된 종목의 DB 데이터를 영구 삭제합니다."""
        checked_symbols = []
        for r in range(self.table.rowCount()):
            chk_item = self.table.item(r, 0)
            if chk_item and chk_item.checkState() == Qt.CheckState.Checked:
                checked_symbols.append(self.table.item(r, 1).text())

        if not checked_symbols:
            current_row = self.table.currentRow()
            if current_row >= 0:
                checked_symbols.append(self.table.item(current_row, 1).text())

        if not checked_symbols:
            QMessageBox.warning(self, "경고", "먼저 DB 데이터를 삭제할 종목의 체크박스를 선택해주세요.")
            return

        # 최종 확인 (매우 중요)
        reply = QMessageBox.critical(
            self, 
            "데이터 영구 삭제 경고", 
            f"선택한 {len(checked_symbols)}개 종목의 모든 과거 및 틱 데이터를 DB에서 영구히 삭제합니다.\n\n"
            "이 작업은 되돌릴 수 없습니다. 정말 진행하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.view_model.delete_db_data(checked_symbols)

    def _on_btn_test_buy_clicked(self):
        """테스트 BUY 로그를 전송합니다."""
        self.view_model.send_test_trade_log("BUY")

    def _on_btn_test_sell_clicked(self):
        """테스트 SELL 로그를 전송합니다."""
        self.view_model.send_test_trade_log("SELL")

    # --- Slots (ViewModel -> View) ---
    @pyqtSlot(list)
    def on_symbols_loaded(self, symbols: list):
        self.table.setRowCount(0)
        for row, sym in enumerate(symbols):
            self.table.insertRow(row)
            
            # 선택 체크박스
            chk_item = QTableWidgetItem()
            chk_item.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(row, 0, chk_item)

            self.table.setItem(row, 1, QTableWidgetItem(str(sym.get("code", ""))))
            
            # [UI 개선] 종목명 최소 10자리 보장 (공백 패딩)
            raw_name = str(sym.get("name", ""))
            padded_name = raw_name.ljust(10)
            self.table.setItem(row, 2, QTableWidgetItem(padded_name))
            
            # 현재가 표시 및 색상 적용
            price = sym.get("price", 0.0)
            flu_rt = sym.get("flu_rt", 0.0)
            price_item = QTableWidgetItem(f"{int(price):,}")
            
            # 등락률 아이템 생성
            rt_item = QTableWidgetItem(f"{flu_rt:+.2f}%")
            
            if flu_rt > 0:
                price_item.setForeground(Qt.GlobalColor.red)
                rt_item.setForeground(Qt.GlobalColor.red)
            elif flu_rt < 0:
                price_item.setForeground(Qt.GlobalColor.blue)
                rt_item.setForeground(Qt.GlobalColor.blue)
                
            self.table.setItem(row, 3, price_item)
            self.table.setItem(row, 4, rt_item)
            
            # 거래량 표시 (천 단위 콤마)
            volume = sym.get("volume", 0)
            vol_item = QTableWidgetItem(f"{int(volume):,}")
            self.table.setItem(row, 5, vol_item)

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
