import yaml
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QDateEdit, QProgressBar, QLabel, QGroupBox, QMessageBox
from PyQt6.QtCore import QDate

class AssetDataManagerTab(QWidget):
    """
    탭 B: 종목 및 데이터 관리
    종목 리스트 편집(CRUD) 및 과거 데이터 수집 기능.
    """
    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path
        self._init_ui()
        self._load_config()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        # 1. 좌측: 종목 리스트 관리
        asset_group = QGroupBox("Asset Manager (Symbol List)")
        asset_layout = QVBoxLayout()

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Symbol", "Description"])
        asset_layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("Add")
        self.btn_save = QPushButton("Save to Config")
        self.btn_save.clicked.connect(self._save_config)

        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_save)
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
        data_layout.addWidget(self.btn_collect)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        data_layout.addWidget(self.progress_bar)

        data_layout.addStretch()
        data_group.setLayout(data_layout)
        main_layout.addWidget(data_group, stretch=1)

    def _load_config(self):
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            symbol = config.get("symbol", "")
            if symbol:
                self.table.insertRow(0)
                self.table.setItem(0, 0, QTableWidgetItem(symbol))
                self.table.setItem(0, 1, QTableWidgetItem("Primary Target"))
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to load config: {e}")

    def _save_config(self):
        # 1행의 종목을 config.yaml에 저장하는 예시
        if self.table.rowCount() > 0:
            symbol = self.table.item(0, 0).text()
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                config["symbol"] = symbol
                with open(self.config_path, "w", encoding="utf-8") as f:
                    yaml.dump(config, f)
                QMessageBox.information(self, "Saved", "Config saved successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Save Error", f"Failed to save config: {e}")
