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
        self._current_pnl = 0.0
        self._current_balance = 0.0
        self._available_limit = 0.0
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

        self.summary_table = QTableWidget(0, 8)
        self.summary_table.setHorizontalHeaderLabels(["종목코드", "종목명", "현재가", "등락률", "거래량", "AI 신호", "보유량", "수익률"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.summary_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        
        # 선택된 행 하이라이트 강화 (밝은 파란색 계열)
        self.summary_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #3b3b3b;
            }
            QTableWidget::item:selected {
                background-color: #007acc;
                color: white;
                font-weight: bold;
            }
        """)
        
        self.summary_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        master_layout.addWidget(self.summary_table)

        master_group.setLayout(master_layout)
        main_layout.addWidget(master_group, stretch=8) # [가로 사이즈 대폭 확대]

        # 2. 우측 통합 영역 (호가창 + AI제어 + 하단로그)
        dashboard_content_layout = QVBoxLayout()
        
        # 2-A. 상단 구역: 호가창(좌) + AI/제어(우)
        top_row_layout = QHBoxLayout()
        
        # [상세 호가창 래더]
        ladder_group = QGroupBox("상세 호가창 래더")
        ladder_layout = QVBoxLayout()
        self.orderbook_widget = OrderbookLadderWidget(self.view_model)
        ladder_layout.addWidget(self.orderbook_widget)
        ladder_group.setLayout(ladder_layout)
        top_row_layout.addWidget(ladder_group, stretch=5)

        # [상세 AI 모니터 및 컨트롤]
        control_group = QGroupBox("상세 AI 모니터링 및 제어")
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

        # 시스템 실시간 제어 패널 (위로 이동)
        sys_ctrl_group = QGroupBox("실시간 매매/감시 제어")
        sys_ctrl_layout = QVBoxLayout()
        self.btn_monitor_toggle = QPushButton("🛰️ 종목 감시 중지")
        self.btn_monitor_toggle.setCheckable(True)
        self.btn_monitor_toggle.setStyleSheet("background-color: #2b5b84; font-weight: bold; height: 35px;")
        self.btn_monitor_toggle.clicked.connect(self._on_monitor_toggle_clicked)
        sys_ctrl_layout.addWidget(self.btn_monitor_toggle)
        self.btn_ai_toggle = QPushButton("🤖 AI 매매 일시 정지")
        self.btn_ai_toggle.setCheckable(True)
        self.btn_ai_toggle.setStyleSheet("background-color: #5b2b84; font-weight: bold; height: 35px;")
        self.btn_ai_toggle.clicked.connect(self._on_ai_toggle_clicked)
        sys_ctrl_layout.addWidget(self.btn_ai_toggle)
        sys_ctrl_group.setLayout(sys_ctrl_layout)
        control_layout.addWidget(sys_ctrl_group)

        # 패닉 버튼
        self.panic_btn = QPushButton("🚨 전량 시장가 매도 🚨\n전체 주문 취소")
        self.panic_btn.setStyleSheet("background-color: darkred; color: white; font-size: 14px; font-weight: bold; height: 50px;")
        self.panic_btn.clicked.connect(self.view_model.trigger_panic_sell)
        control_layout.addWidget(self.panic_btn)
        
        control_layout.addStretch(1) # [여백] 나머지 요소를 위로 밀착
        control_group.setLayout(control_layout)
        top_row_layout.addWidget(control_group, stretch=2)

        dashboard_content_layout.addLayout(top_row_layout, stretch=5)

        # 2-B. 하단 구역: 시스템 및 체결 로그 (너비 확장)
        log_group = QGroupBox("시스템 및 체결 로그")
        log_layout = QVBoxLayout()
        self.log_list = QListWidget()
        self.log_list.setStyleSheet("background-color: #2b2b2b; color: #90EE90; font-family: monospace; font-size: 11px;")
        log_layout.addWidget(self.log_list)
        log_group.setLayout(log_layout)
        dashboard_content_layout.addWidget(log_group, stretch=2)
        
        main_layout.addLayout(dashboard_content_layout, stretch=7)

    # --- 실시간 제어 슬롯 ---
    def _on_monitor_toggle_clicked(self, checked):
        if checked:
            self.btn_monitor_toggle.setText("📡 종목 감시 재개")
            self.btn_monitor_toggle.setStyleSheet("background-color: #d32f2f; font-weight: bold; height: 35px;")
            self.btn_ai_toggle.setEnabled(False) # 감시 중단 시 AI 제어 불가
        else:
            self.btn_monitor_toggle.setText("🛰️ 종목 감시 중지")
            self.btn_monitor_toggle.setStyleSheet("background-color: #2b5b84; font-weight: bold; height: 35px;")
            self.btn_ai_toggle.setEnabled(True)
            
        self.view_model.toggle_monitoring(checked)

    def _on_ai_toggle_clicked(self, checked):
        if checked:
            # 상태: 일시정지됨 -> 버튼은 '재개'를 제안해야 함
            self.btn_ai_toggle.setText("🤖 AI 자동 매매 재개")
            self.btn_ai_toggle.setStyleSheet("background-color: #5b2b84; font-weight: bold; height: 35px;")
        else:
            # 상태: 가동 중 -> 버튼은 '일시정지'를 제안해야 함
            self.btn_ai_toggle.setText("⛔ AI 매매 일시 정지")
            self.btn_ai_toggle.setStyleSheet("background-color: #f57c00; font-weight: bold; height: 35px;")
            
        self.view_model.toggle_ai_trading(checked)

    def _connect_signals(self):
        self.view_model.sig_symbols_summary_updated.connect(self.on_symbols_summary_updated)
        self.view_model.sig_ai_confidence_updated.connect(self.on_ai_confidence_updated)
        self.view_model.sig_log_appended.connect(self.on_log_appended)
        self.view_model.sig_risk_metrics_updated.connect(self.on_risk_metrics_updated)
        self.view_model.sig_balance_updated.connect(self.on_balance_updated)
        self.view_model.sig_status_alert.connect(self.on_status_alert)
        self.view_model.sig_error_occurred.connect(self.on_error)
        self.view_model.sig_universe_changed.connect(self.on_universe_changed)
        # [원격 제어 연동] Firebase에서 제어 명령이 올 때 버튼 UI 상태를 즉시 갱신
        self.view_model.sig_trading_paused.connect(self.on_ai_trading_toggled)
        self.view_model.sig_monitoring_stopped.connect(self.on_monitoring_toggled)

    @pyqtSlot(bool)
    def on_monitoring_toggled(self, stopped: bool):
        """
        [원격/로컬 공통] 종목 감시 상태 변경 시 버튼 UI를 동기화합니다.
        stopped=True: 감시 중단(버튼 눌림 상태)
        stopped=False: 감시 중(버튼 해제 상태)
        """
        # blockSignals: setChecked 호출 시 clicked 시그널이 중복 발생하는 것을 방지
        self.btn_monitor_toggle.blockSignals(True)
        self.btn_monitor_toggle.setChecked(stopped)
        self.btn_monitor_toggle.blockSignals(False)

        if stopped:
            self.btn_monitor_toggle.setText("📡 종목 감시 재개")
            self.btn_monitor_toggle.setStyleSheet("background-color: #d32f2f; font-weight: bold; height: 35px;")
            self.btn_ai_toggle.setEnabled(False)
        else:
            self.btn_monitor_toggle.setText("🛰️ 종목 감시 중지")
            self.btn_monitor_toggle.setStyleSheet("background-color: #2b5b84; font-weight: bold; height: 35px;")
            self.btn_ai_toggle.setEnabled(True)

    @pyqtSlot(bool)
    def on_ai_trading_toggled(self, paused: bool):
        """
        [원격/로컬 공통] AI 매매 일시정지 상태 변경 시 버튼 UI를 동기화합니다.
        paused=True: 일시정지(버튼 눌림 상태)
        paused=False: 정상 작동(버튼 해제 상태)
        """
        self.btn_ai_toggle.blockSignals(True)
        self.btn_ai_toggle.setChecked(paused)
        self.btn_ai_toggle.blockSignals(False)

        if paused:
            self.btn_ai_toggle.setText("🤖 AI 자동 매매 재개")
            self.btn_ai_toggle.setStyleSheet("background-color: #5b2b84; font-weight: bold; height: 35px;")
        else:
            self.btn_ai_toggle.setText("⛔ AI 매매 일시 정지")
            self.btn_ai_toggle.setStyleSheet("background-color: #f57c00; font-weight: bold; height: 35px;")

    def _on_table_selection_changed(self):
        selected_items = self.summary_table.selectedItems()
        if selected_items:
            # 첫 번째 컬럼(종목코드) 가져오기
            row = selected_items[0].row()
            symbol = self.summary_table.item(row, 0).text()
            if hasattr(self.view_model, 'set_selected_symbol'):
                self.view_model.set_selected_symbol(symbol)
                self.on_log_appended(f"[UI] 상세 뷰 종목 변경: {symbol}")

    @pyqtSlot(list)
    def on_universe_changed(self, new_symbols: list):
        """
        [긴급 패치] 유니버스가 완전히 교체될 때 호출되어 테이블을 초기화합니다.
        다음 번 symbols_summary 업데이트 시 새로운 종목들로 테이블이 다시 그려집니다.
        """
        self.summary_table.setRowCount(0)
        self.on_log_appended(f"[UI] 장중 유니버스 교체 감지: 테이블을 초기화하고 다시 그립니다. ({len(new_symbols)} 종목)")

    @pyqtSlot(dict)
    def on_symbols_summary_updated(self, summary_dict: dict):
        """테이블 갱신. 최적화를 위해 기존 행을 찾아 업데이트함"""
        # 기존 맵 구성 (종목코드 -> row index)
        symbol_to_row = {}
        for r in range(self.summary_table.rowCount()):
            item = self.summary_table.item(r, 0)
            if item:
                symbol_to_row[item.text()] = r

        # 데이터 업데이트
        for symbol, data in summary_dict.items():
            if symbol not in symbol_to_row:
                # 새로운 종목인 경우 행 추가
                row = self.summary_table.rowCount()
                self.summary_table.insertRow(row)
                self.summary_table.setItem(row, 0, QTableWidgetItem(symbol))
                
                # 종목명 초기 설정 (index 1)
                name = data.get('name', '-')
                self.summary_table.setItem(row, 1, QTableWidgetItem(name))
                symbol_to_row[symbol] = row
            
            row = symbol_to_row[symbol]
            
            # 종목명 업데이트 (필요시)
            name = data.get('name', '-')
            if self.summary_table.item(row, 1) is None or self.summary_table.item(row, 1).text() != name:
                self.summary_table.setItem(row, 1, QTableWidgetItem(name))

            # 등락 및 색상 결정
            change_rate = data.get('change_rate', 0.0)
            text_color = Qt.GlobalColor.white # 기본값
            if change_rate > 0:
                text_color = Qt.GlobalColor.red
            elif change_rate < 0:
                text_color = Qt.GlobalColor.blue

            # 현재가 업데이트 (index 2) + 색상 적용
            price = data.get('price', 0)
            price_str = f"{price:,.0f}"
            price_item = self.summary_table.item(row, 2)
            if price_item is None or price_item.text() != price_str:
                price_item = QTableWidgetItem(price_str)
                self.summary_table.setItem(row, 2, price_item)
            price_item.setForeground(text_color)

            # [신규] 등락률 업데이트 (index 3) + 색상 적용
            chg_sign = "+" if change_rate > 0 else ""
            chg_str = f"{chg_sign}{change_rate:.2f}%"
            chg_item = self.summary_table.item(row, 3)
            if chg_item is None or chg_item.text() != chg_str:
                chg_item = QTableWidgetItem(chg_str)
                self.summary_table.setItem(row, 3, chg_item)
            chg_item.setForeground(text_color)

            # 거래량 업데이트 (index 4)
            volume = data.get('volume', 0)
            vol_str = f"{volume:,.0f}"
            if self.summary_table.item(row, 4) is None or self.summary_table.item(row, 4).text() != vol_str:
                self.summary_table.setItem(row, 4, QTableWidgetItem(vol_str))

            # AI 신호 업데이트 (index 5)
            ai_sig = data.get('ai_signal', '-')
            if self.summary_table.item(row, 5) is None or self.summary_table.item(row, 5).text() != ai_sig:
                item_sig = QTableWidgetItem(ai_sig)
                if ai_sig == "Buy":
                    item_sig.setForeground(Qt.GlobalColor.red)
                elif ai_sig == "Sell":
                    item_sig.setForeground(Qt.GlobalColor.blue)
                self.summary_table.setItem(row, 5, item_sig)

            # 보유량 업데이트 (index 6) + 빨간색 강조
            holdings = data.get('holdings', 0)
            holdings_str = str(holdings)
            hold_item = self.summary_table.item(row, 6)
            if hold_item is None or hold_item.text() != holdings_str:
                hold_item = QTableWidgetItem(holdings_str)
                self.summary_table.setItem(row, 6, hold_item)
            
            # 보유 중이면 빨간색, 아니면 기본색
            if holdings > 0:
                hold_item.setForeground(Qt.GlobalColor.red)
                hold_item.setSelected(True) # 시각적 강조 추가
            else:
                hold_item.setForeground(Qt.GlobalColor.white)

            # [신규] 수익률 업데이트 (index 7)
            avg_price = data.get('avg_price', 0.0)
            price = data.get('price', 0.0)
            pnl_rate = 0.0
            pnl_str = "-"
            pnl_color = Qt.GlobalColor.white
            
            if holdings > 0 and avg_price > 0:
                pnl_rate = ((price - avg_price) / avg_price) * 100
                pnl_sign = "+" if pnl_rate > 0 else ""
                pnl_str = f"{pnl_sign}{pnl_rate:.2f}%"
                if pnl_rate > 0:
                    pnl_color = Qt.GlobalColor.red
                elif pnl_rate < 0:
                    pnl_color = Qt.GlobalColor.blue
            
            pnl_item = self.summary_table.item(row, 7)
            if pnl_item is None or pnl_item.text() != pnl_str:
                pnl_item = QTableWidgetItem(pnl_str)
                self.summary_table.setItem(row, 7, pnl_item)
            pnl_item.setForeground(pnl_color)

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

    @pyqtSlot(float, float, float, float)
    def on_risk_metrics_updated(self, realized: float, evaluation: float, total_cash: float, per_symbol_limit: float):
        self._realized_pnl = realized
        self._evaluation_pnl = evaluation
        self._available_limit = total_cash # 전체 주문 가능 현금
        self._per_symbol_limit = per_symbol_limit
        self._update_risk_bar()

    @pyqtSlot(float)
    def on_balance_updated(self, balance: float):
        self._current_balance = balance
        self._update_risk_bar()

    def _update_risk_bar(self):
        """상단 리스크/자산 정보 레이블 갱신"""
        realized = getattr(self, '_realized_pnl', 0.0)
        evaluation = getattr(self, '_evaluation_pnl', 0.0)
        per_sym = getattr(self, '_per_symbol_limit', 0.0)
        
        text = (f"💵 실현 손익: {realized:,.0f} | "
                f"📈 평가 손익: {evaluation:,.0f} | "
                f"📊 총 자산: {self._current_balance:,.0f} | "
                f"💳 주문 가능: {self._available_limit:,.0f} | "
                f"🚫 종목 한도: {per_sym:,.0f}")
        self.risk_bar.setText(text)

    @pyqtSlot(str)
    def on_status_alert(self, alert_msg: str):
        self.status_bar.setText(alert_msg)
        self.status_bar.setStyleSheet("background-color: darkred; color: yellow; padding: 10px; font-weight: bold;")

    @pyqtSlot(str)
    def on_error(self, msg: str):
        self.on_log_appended(f"[오류] {msg}")
