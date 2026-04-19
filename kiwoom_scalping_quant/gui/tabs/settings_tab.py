from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QMessageBox
)
from PyQt6.QtCore import pyqtSlot

class SettingsTab(QWidget):
    """
    설정 탭: API 연동 정보(.env)와 매매 기본 설정(config.yaml)을 관리하는 UI
    """
    def __init__(self, view_model):
        super().__init__()
        self.view_model = view_model
        self._init_ui()
        self._connect_signals()

        # 탭 로드 시 초기 데이터 불러오기
        self.view_model.load_settings()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # 1. Broker API Group (.env)
        broker_group = QGroupBox("브로커 API 연결")
        broker_form = QFormLayout()

        self.input_app_key = QLineEdit()
        self.input_app_key.setEchoMode(QLineEdit.EchoMode.Password)
        broker_form.addRow("앱 키:", self.input_app_key)

        self.input_app_secret = QLineEdit()
        self.input_app_secret.setEchoMode(QLineEdit.EchoMode.Password)
        broker_form.addRow("앱 시크릿:", self.input_app_secret)

        self.input_account = QLineEdit()
        broker_form.addRow("계좌 번호:", self.input_account)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["모의투자", "실전투자"])
        broker_form.addRow("매매 모드:", self.combo_mode)

        broker_group.setLayout(broker_form)
        main_layout.addWidget(broker_group)

        # 2. Database Group (.env & config.yaml 혼합)
        db_group = QGroupBox("데이터베이스")
        db_form = QFormLayout()

        self.input_db_url = QLineEdit()
        db_form.addRow("주소:", self.input_db_url)

        self.input_db_token = QLineEdit()
        self.input_db_token.setEchoMode(QLineEdit.EchoMode.Password)
        db_form.addRow("토큰:", self.input_db_token)

        self.input_db_org = QLineEdit()
        db_form.addRow("조직:", self.input_db_org)

        self.input_db_bucket = QLineEdit()
        db_form.addRow("버킷:", self.input_db_bucket)

        db_group.setLayout(db_form)
        main_layout.addWidget(db_group)

        # 3. Risk Management Group (config.yaml)
        risk_group = QGroupBox("리스크 관리")
        risk_form = QFormLayout()

        self.spin_stop_loss = QDoubleSpinBox()
        self.spin_stop_loss.setSuffix(" %")
        self.spin_stop_loss.setDecimals(2)
        self.spin_stop_loss.setRange(-20.0, 0.0)
        risk_form.addRow("하드 손절 라인:", self.spin_stop_loss)

        self.spin_max_position = QDoubleSpinBox()
        self.spin_max_position.setSuffix(" %")
        self.spin_max_position.setRange(1.0, 100.0)
        risk_form.addRow("최대 진입 자금 비율:", self.spin_max_position)

        self.spin_cb_timeout = QSpinBox()
        self.spin_cb_timeout.setSuffix(" 초")
        self.spin_cb_timeout.setRange(1, 60)
        risk_form.addRow("서킷 브레이커 대기 시간:", self.spin_cb_timeout)

        risk_group.setLayout(risk_form)
        main_layout.addWidget(risk_group)

        # 4. System Group
        system_group = QGroupBox("시스템 알림 및 로깅")
        sys_form = QFormLayout()

        self.input_tg_token = QLineEdit()
        self.input_tg_token.setEchoMode(QLineEdit.EchoMode.Password)
        sys_form.addRow("텔레그램 봇 토큰:", self.input_tg_token)

        self.input_tg_chat = QLineEdit()
        sys_form.addRow("텔레그램 채팅 ID:", self.input_tg_chat)

        self.combo_log_level = QComboBox()
        self.combo_log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        sys_form.addRow("로그 레벨:", self.combo_log_level)

        system_group.setLayout(sys_form)
        main_layout.addWidget(system_group)

        # 5. 하단 제어 버튼
        btn_layout = QHBoxLayout()
        self.btn_test = QPushButton("연결 테스트")
        self.btn_test.clicked.connect(self._on_test_clicked)

        self.btn_save = QPushButton("설정 저장")
        self.btn_save.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold;")
        self.btn_save.clicked.connect(self._on_save_clicked)

        btn_layout.addWidget(self.btn_test)
        btn_layout.addWidget(self.btn_save)
        main_layout.addLayout(btn_layout)

    def _connect_signals(self):
        from PyQt6.QtCore import Qt
        self.view_model.settings_loaded.connect(self.on_settings_loaded)
        self.view_model.save_completed.connect(self.on_save_completed, Qt.ConnectionType.QueuedConnection)
        self.view_model.save_failed.connect(self.on_error, Qt.ConnectionType.QueuedConnection)
        self.view_model.connection_test_completed.connect(self.on_connection_test_completed, Qt.ConnectionType.QueuedConnection)

    def _get_current_data(self):
        """UI에 입력된 값을 통합된 딕셔너리로 반환"""
        return {
            "KIWOOM_APP_KEY": self.input_app_key.text(),
            "KIWOOM_APP_SECRET": self.input_app_secret.text(),
            "INFLUX_URL": self.input_db_url.text(),
            "INFLUX_TOKEN": self.input_db_token.text(),
            "INFLUX_ORG": self.input_db_org.text(),
            "TELEGRAM_BOT_TOKEN": self.input_tg_token.text(),
            "account_number": self.input_account.text(),
            "trading_mode": self.combo_mode.currentText(),
            "influx_bucket": self.input_db_bucket.text(),
            "stop_loss_pct": self.spin_stop_loss.value(),
            "max_position_pct": self.spin_max_position.value(),
            "cb_timeout_sec": self.spin_cb_timeout.value(),
            "telegram_chat_id": self.input_tg_chat.text(),
            "log_level": self.combo_log_level.currentText()
        }

    # --- UI Action Handlers ---
    def _on_test_clicked(self):
        self.btn_test.setEnabled(False)
        self.btn_test.setText("테스트 진행 중...")
        data = self._get_current_data()
        self.view_model.test_connection(data)

    def _on_save_clicked(self):
        data = self._get_current_data()
        self.view_model.save_settings(data)

    # --- ViewModel Signal Slots ---
    @pyqtSlot(dict)
    def on_settings_loaded(self, config: dict):
        # Env
        self.input_app_key.setText(config.get("KIWOOM_APP_KEY", ""))
        self.input_app_secret.setText(config.get("KIWOOM_APP_SECRET", ""))
        self.input_db_url.setText(config.get("INFLUX_URL", "http://localhost:8086"))
        self.input_db_token.setText(config.get("INFLUX_TOKEN", ""))
        self.input_db_org.setText(config.get("INFLUX_ORG", ""))
        self.input_tg_token.setText(config.get("TELEGRAM_BOT_TOKEN", ""))

        # Config
        self.input_account.setText(config.get("account_number", ""))
        self.combo_mode.setCurrentText(config.get("trading_mode", "모의투자"))
        self.input_db_bucket.setText(config.get("influx_bucket", "kiwoom_data"))
        self.spin_stop_loss.setValue(config.get("stop_loss_pct", -2.0))
        self.spin_max_position.setValue(config.get("max_position_pct", 50.0))
        self.spin_cb_timeout.setValue(config.get("cb_timeout_sec", 3))
        self.input_tg_chat.setText(config.get("telegram_chat_id", ""))
        self.combo_log_level.setCurrentText(config.get("log_level", "INFO"))

    @pyqtSlot(str)
    def on_save_completed(self, msg: str):
        QMessageBox.information(self, "저장 성공", msg)

    @pyqtSlot(str)
    def on_error(self, msg: str):
        QMessageBox.warning(self, "오류", msg)

    @pyqtSlot(bool, str)
    def on_connection_test_completed(self, success: bool, msg: str):
        self.btn_test.setEnabled(True)
        self.btn_test.setText("연결 테스트")

        if success:
            QMessageBox.information(self, "테스트 성공", msg)
        else:
            QMessageBox.critical(self, "테스트 실패", msg)
