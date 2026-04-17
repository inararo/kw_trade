import os
from PyQt6.QtWidgets import QMainWindow, QTabWidget, QWidget, QVBoxLayout, QMenuBar, QMenu, QStatusBar
from PyQt6.QtGui import QAction

from gui.tabs.live_dashboard import LiveDashboardTab
from gui.tabs.asset_data_manager import AssetDataManagerTab
from gui.tabs.ai_training_studio import AITrainingStudioTab

class MainWindow(QMainWindow):
    """
    HTS 스타일의 메인 윈도우 클래스.
    메뉴바, 상태 표시줄, 그리고 탭 위젯으로 구성됩니다.
    """
    def __init__(self, view_model, system):
        super().__init__()
        self.system = system
        self.view_model = view_model

        self.setWindowTitle("Kiwoom Scalping Quant - Professional HTS")
        self.setGeometry(100, 100, 1200, 800)

        self._init_menu_bar()
        self._init_tabs()
        self._init_status_bar()

    def _init_menu_bar(self):
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("파일(File)")
        exit_action = QAction("프로그램 종료", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        file_menu.addAction("로그 파일 열기")

        # Trading Menu
        trading_menu = menu_bar.addMenu("매매(Trading)")
        trading_menu.addAction("라이브 대시보드 열기")
        trading_menu.addAction("미체결 전체 취소")
        trading_menu.addAction("당일 손익 초기화")

        # Data Menu
        data_menu = menu_bar.addMenu("데이터(Data)")
        data_menu.addAction("종목 관리 열기")
        data_menu.addAction("DB 상태 점검")

        # AI Menu
        ai_menu = menu_bar.addMenu("AI 학습(AI Lab)")
        ai_menu.addAction("학습 스튜디오 열기")
        ai_menu.addAction("모델 검증 도구")

        # Settings Menu
        settings_menu = menu_bar.addMenu("설정(Settings)")
        settings_menu.addAction("환경 설정 창 열기")
        settings_menu.addAction("API 토큰 강제 갱신")

    def _init_tabs(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # 각 탭 초기화 및 의존성 주입
        self.tab_live = LiveDashboardTab(self.view_model)

        # `__file__` is kiwoom_scalping_quant/gui/main_window.py
        # 1 dirname -> gui/, 2 dirnames -> kiwoom_scalping_quant/
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        self.tab_asset = AssetDataManagerTab(config_path)

        self.tab_ai = AITrainingStudioTab()

        self.tabs.addTab(self.tab_live, "Live Dashboard")
        self.tabs.addTab(self.tab_asset, "Asset & Data")
        self.tabs.addTab(self.tab_ai, "AI Training Studio")

    def _init_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. API Disconnected. Latency: - ms")

        # 차후 ViewModel이나 DataCollector에서 latency 시그널을 연결하여 갱신 가능

    def closeEvent(self, event):
        """GUI 창 닫기 버튼 클릭 시 안전한 종료 트리거"""
        self.system.stop()
        event.accept()
