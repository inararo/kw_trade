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
        broker_group = QGroupBox("Broker API (Kiwoom)")
        broker_form = QFormLayout()

        self.input_app_key = QLineEdit()
        self.input_app_key.setEchoMode(QLineEdit.EchoMode.Password)
        broker_form.addRow("App Key:", self.input_app_key)

        self.input_app_secret = QLineEdit()
        self.input_app_secret.setEchoMode(QLineEdit.EchoMode.Password)
        broker_form.addRow("App Secret:", self.input_app_secret)

        self.input_account = QLineEdit()
        broker_form.addRow("Account Number:", self.input_account)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["모의투자 (Virtual)", "실전투자 (Real)"])
        broker_form.addRow("Trading Mode:", self.combo_mode)

        broker_group.setLayout(broker_form)
        main_layout.addWidget(broker_group)

        # 2. Database Group (.env & config.yaml 혼합)
        db_group = QGroupBox("Database (InfluxDB)")
        db_form = QFormLayout()

        self.input_db_url = QLineEdit()
        db_form.addRow("URL:", self.input_db_url)

        self.input_db_token = QLineEdit()
        self.input_db_token.setEchoMode(QLineEdit.EchoMode.Password)
        db_form.addRow("Token:", self.input_db_token)

        self.input_db_org = QLineEdit()
        db_form.addRow("Organization:", self.input_db_org)

        self.input_db_bucket = QLineEdit()
        db_form.addRow("Bucket:", self.input_db_bucket)

        db_group.setLayout(db_form)
        main_layout.addWidget(db_group)

        # 3. Risk Management Group (config.yaml)
        risk_group = QGroupBox("Risk Management")
        risk_form = QFormLayout()

        self.spin_stop_loss = QDoubleSpinBox()
        self.spin_stop_loss.setSuffix(" %")
        self.spin_stop_loss.setDecimals(2)
        self.spin_stop_loss.setRange(-20.0, 0.0)
        risk_form.addRow("Hard Stop Loss:", self.spin_stop_loss)

        self.spin_max_position = QDoubleSpinBox()
        self.spin_max_position.setSuffix(" %")
        self.spin_max_position.setRange(1.0, 100.0)
        risk_form.addRow("Max Position Size:", self.spin_max_position)

        self.spin_cb_timeout = QSpinBox()
        self.spin_cb_timeout.setSuffix(" Sec")
        self.spin_cb_timeout.setRange(1, 60)
        risk_form.addRow("Circuit Breaker Timeout:", self.spin_cb_timeout)

        risk_group.setLayout(risk_form)
        main_layout.addWidget(risk_group)

        # 4. System Group
        system_group = QGroupBox("System & Alerts")
        sys_form = QFormLayout()

        self.input_tg_token = QLineEdit()
        self.input_tg_token.setEchoMode(QLineEdit.EchoMode.Password)
        sys_form.addRow("Telegram Bot Token:", self.input_tg_token)

        self.input_tg_chat = QLineEdit()
        sys_form.addRow("Telegram Chat ID:", self.input_tg_chat)

        self.combo_log_level = QComboBox()
        self.combo_log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        sys_form.addRow("Log Level:", self.combo_log_level)

        system_group.setLayout(sys_form)
        main_layout.addWidget(system_group)

        # 5. 하단 제어 버튼
        btn_layout = QHBoxLayout()
        self.btn_test = QPushButton("Test Connection")
        self.btn_test.clicked.connect(self._on_test_clicked)

        self.btn_save = QPushButton("Save Settings")
        self.btn_save.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold;")
        self.btn_save.clicked.connect(self._on_save_clicked)

        btn_layout.addWidget(self.btn_test)
        btn_layout.addWidget(self.btn_save)
        main_layout.addLayout(btn_layout)

    def _connect_signals(self):
        self.view_model.settings_loaded.connect(self.on_settings_loaded)
        self.view_model.save_completed.connect(self.on_save_completed)
        self.view_model.save_failed.connect(self.on_error)
        self.view_model.connection_test_completed.connect(self.on_connection_test_completed)

    def _get_current_data(self):
        """UI에 입력된 값을 env_dict와 config_dict로 분리하여 반환"""
        env_dict = {
            "KIWOOM_APP_KEY": self.input_app_key.text(),
            "KIWOOM_APP_SECRET": self.input_app_secret.text(),
            "INFLUX_URL": self.input_db_url.text(),
            "INFLUX_TOKEN": self.input_db_token.text(),
            "INFLUX_ORG": self.input_db_org.text(),
            "TELEGRAM_BOT_TOKEN": self.input_tg_token.text()
        }

        config_dict = {
            "account_number": self.input_account.text(),
            "trading_mode": self.combo_mode.currentText(),
            "influx_bucket": self.input_db_bucket.text(),
            "stop_loss_pct": self.spin_stop_loss.value(),
            "max_position_pct": self.spin_max_position.value(),
            "cb_timeout_sec": self.spin_cb_timeout.value(),
            "telegram_chat_id": self.input_tg_chat.text(),
            "log_level": self.combo_log_level.currentText()
        }
        return env_dict, config_dict

    # --- UI Action Handlers ---
    def _on_test_clicked(self):
        self.btn_test.setEnabled(False)
        self.btn_test.setText("Testing...")
        env_data, config_data = self._get_current_data()
        self.view_model.test_connection(env_data, config_data)

    def _on_save_clicked(self):
        env_data, config_data = self._get_current_data()
        self.view_model.save_settings(env_data, config_data)

    # --- ViewModel Signal Slots ---
    @pyqtSlot(dict, dict)
    def on_settings_loaded(self, env_dict: dict, config_dict: dict):
        # Env
        self.input_app_key.setText(env_dict.get("KIWOOM_APP_KEY", ""))
        self.input_app_secret.setText(env_dict.get("KIWOOM_APP_SECRET", ""))
        self.input_db_url.setText(env_dict.get("INFLUX_URL", "http://localhost:8086"))
        self.input_db_token.setText(env_dict.get("INFLUX_TOKEN", ""))
        self.input_db_org.setText(env_dict.get("INFLUX_ORG", ""))
        self.input_tg_token.setText(env_dict.get("TELEGRAM_BOT_TOKEN", ""))

        # Config
        self.input_account.setText(config_dict.get("account_number", ""))
        self.combo_mode.setCurrentText(config_dict.get("trading_mode", "모의투자 (Virtual)"))
        self.input_db_bucket.setText(config_dict.get("influx_bucket", "kiwoom_data"))
        self.spin_stop_loss.setValue(config_dict.get("stop_loss_pct", -2.0))
        self.spin_max_position.setValue(config_dict.get("max_position_pct", 50.0))
        self.spin_cb_timeout.setValue(config_dict.get("cb_timeout_sec", 3))
        self.input_tg_chat.setText(config_dict.get("telegram_chat_id", ""))
        self.combo_log_level.setCurrentText(config_dict.get("log_level", "INFO"))

    @pyqtSlot(str)
    def on_save_completed(self, msg: str):
        QMessageBox.information(self, "Success", msg)

    @pyqtSlot(str)
    def on_error(self, msg: str):
        QMessageBox.warning(self, "Error", msg)

    @pyqtSlot(bool, str)
    def on_connection_test_completed(self, success: bool, msg: str):
        self.btn_test.setEnabled(True)
        self.btn_test.setText("Test Connection")

        if success:
            QMessageBox.information(self, "Connection Test Passed", msg)
        else:
            QMessageBox.critical(self, "Connection Test Failed", msg)
