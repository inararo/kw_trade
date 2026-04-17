import os
from PyQt6.QtWidgets import QMainWindow, QTabWidget, QWidget, QVBoxLayout, QMenuBar, QMenu, QStatusBar
from PyQt6.QtGui import QAction

from gui.tabs.live_dashboard import LiveDashboardTab
from gui.tabs.asset_data_manager import AssetDataManagerTab
from gui.tabs.ai_training_studio import AITrainingStudioTab
from gui.tabs.settings_tab import SettingsTab

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

        # AssetDataViewModel은 main_window를 생성할 때 주입받은 시스템 객체나 별도 라우팅을 거쳐야 하지만
        # 단순화를 위해 시스템 뷰모델에서 가져오거나 DI 컨테이너에서 꺼내옵니다.
        # 여기서는 MainWindow 생성자가 asset_data_view_model도 받도록 가정하고
        # 임시로 getattr을 사용하여 확장 가능성을 열어둡니다. (또는 시스템 객체를 통해)
        asset_vm = getattr(self.system, "asset_data_view_model", None)
        if asset_vm is None and hasattr(self.system, "container"):
            asset_vm = self.system.container.asset_data_view_model()

        self.tab_asset = AssetDataManagerTab(asset_vm)

        self.tab_ai = AITrainingStudioTab()

        settings_vm = getattr(self.system, "settings_view_model", None)
        if settings_vm is None and hasattr(self.system, "container"):
            settings_vm = self.system.container.settings_view_model()

        self.tab_settings = SettingsTab(settings_vm)

        self.tabs.addTab(self.tab_live, "Live Dashboard")
        self.tabs.addTab(self.tab_asset, "Asset & Data")
        self.tabs.addTab(self.tab_ai, "AI Training Studio")
        self.tabs.addTab(self.tab_settings, "Settings")

    def _init_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. API Disconnected. Latency: - ms")

        # 차후 ViewModel이나 DataCollector에서 latency 시그널을 연결하여 갱신 가능

    def closeEvent(self, event):
        """GUI 창 닫기 버튼 클릭 시 비동기 종료 파이프라인 트리거"""
        event.ignore() # 즉시 닫히지 않도록 무시
        self.hide()    # 창부터 숨김 처리
        self.status_bar.showMessage("안전하게 종료 중입니다...")
        print("GUI: 종료 시그널 접수, 시스템 안전 종료 시작...")

        # 시스템의 비동기 종료 루틴을 백그라운드 태스크로 실행
        import asyncio
        asyncio.create_task(self.system.stop())
