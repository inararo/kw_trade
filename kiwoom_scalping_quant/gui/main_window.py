import os
from PyQt6.QtWidgets import QMainWindow, QTabWidget, QWidget, QVBoxLayout, QMenuBar, QMenu, QStatusBar, QMessageBox
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtCore import QUrl, pyqtSlot

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

        self.setWindowTitle("키움 스캘핑 퀀트 - 전문가용 HTS")
        self.setGeometry(100, 100, 1200, 800)

        self._init_menu_bar()
        self._init_tabs()
        self._init_status_bar()

    def _init_menu_bar(self):
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("파일")

        action_open_logs = QAction("로그 파일 열기", self)
        action_open_logs.triggered.connect(self._open_logs)
        file_menu.addAction(action_open_logs)

        exit_action = QAction("프로그램 종료", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Trading Menu
        trading_menu = menu_bar.addMenu("매매")

        action_open_live = QAction("라이브 대시보드 열기", self)
        action_open_live.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        trading_menu.addAction(action_open_live)

        action_cancel_all = QAction("미체결 전체 취소", self)
        action_cancel_all.triggered.connect(lambda: getattr(self.view_model, "cancel_orders_only", lambda: None)())
        trading_menu.addAction(action_cancel_all)

        action_reset_pnl = QAction("당일 손익 초기화", self)
        action_reset_pnl.triggered.connect(lambda: getattr(self.view_model, "reset_pnl", lambda: None)())
        trading_menu.addAction(action_reset_pnl)

        # Data Menu
        data_menu = menu_bar.addMenu("데이터")

        action_open_data = QAction("종목 관리 열기", self)
        action_open_data.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        data_menu.addAction(action_open_data)

        action_check_db = QAction("DB 상태 점검", self)
        data_menu.addAction(action_check_db)

        # AI Menu
        ai_menu = menu_bar.addMenu("AI 학습")

        action_open_ai = QAction("학습 스튜디오 열기", self)
        action_open_ai.triggered.connect(lambda: self.tabs.setCurrentIndex(2))
        ai_menu.addAction(action_open_ai)

        action_model_val = QAction("모델 검증 도구", self)
        action_model_val.triggered.connect(self._open_model_validation)
        ai_menu.addAction(action_model_val)

        # Settings Menu
        settings_menu = menu_bar.addMenu("설정")

        action_open_settings = QAction("환경 설정 창 열기", self)
        action_open_settings.triggered.connect(lambda: self.tabs.setCurrentIndex(3))
        settings_menu.addAction(action_open_settings)

        action_force_token = QAction("API 토큰 강제 갱신", self)
        settings_menu.addAction(action_force_token)

        # 나중에 SettingsViewModel이나 ViewModel 생성 시 connect 하기 위해 저장
        self._action_check_db = action_check_db
        self._action_force_token = action_force_token

    def _open_logs(self):
        """로컬 로그 폴더 열기"""
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(log_dir))

    def _open_model_validation(self):
        QMessageBox.information(self, "모델 검증 도구", "모델 검증 도구는 향후 업데이트에서 제공될 예정입니다.\n현재는 AI 학습 스튜디오 탭을 이용해주세요.")

    def _init_tabs(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # 각 탭 초기화 및 의존성 주입

        # LiveDashboardViewModel 추출 및 주입
        live_vm = getattr(self.system, "live_dashboard_view_model", None)
        if live_vm is None and hasattr(self.system, "container"):
            live_vm = self.system.container.live_dashboard_view_model()
        self.tab_live = LiveDashboardTab(live_vm)

        # AssetDataViewModel은 main_window를 생성할 때 주입받은 시스템 객체나 별도 라우팅을 거쳐야 하지만
        # 단순화를 위해 시스템 뷰모델에서 가져오거나 DI 컨테이너에서 꺼내옵니다.
        # 여기서는 MainWindow 생성자가 asset_data_view_model도 받도록 가정하고
        # 임시로 getattr을 사용하여 확장 가능성을 열어둡니다. (또는 시스템 객체를 통해)
        asset_vm = getattr(self.system, "asset_data_view_model", None)
        if asset_vm is None and hasattr(self.system, "container"):
            asset_vm = self.system.container.asset_data_view_model()

        self.tab_asset = AssetDataManagerTab(asset_vm)

        ai_vm = getattr(self.system, "ai_training_view_model", None)
        if ai_vm is None and hasattr(self.system, "container"):
            ai_vm = self.system.container.ai_training_view_model()
        self.tab_ai = AITrainingStudioTab(ai_vm)

        settings_vm = getattr(self.system, "settings_view_model", None)
        if settings_vm is None and hasattr(self.system, "container"):
            settings_vm = self.system.container.settings_view_model()

        self.tab_settings = SettingsTab(settings_vm)

        # Connect settings VM menu actions
        if hasattr(settings_vm, "check_db_status"):
            self._action_check_db.triggered.connect(settings_vm.check_db_status)
        if hasattr(settings_vm, "force_refresh_token"):
            self._action_force_token.triggered.connect(settings_vm.force_refresh_token)
        if hasattr(settings_vm, "sig_menu_action_result"):
            settings_vm.sig_menu_action_result.connect(self._on_menu_action_result)

        # Connect live VM menu actions
        if hasattr(live_vm, "sig_menu_action_result"):
            live_vm.sig_menu_action_result.connect(self._on_menu_action_result)

        self.tabs.addTab(self.tab_live, "라이브 대시보드")
        self.tabs.addTab(self.tab_asset, "종목 및 데이터 관리")
        self.tabs.addTab(self.tab_ai, "AI 학습 스튜디오")
        self.tabs.addTab(self.tab_settings, "환경 설정")

    @pyqtSlot(str, str)
    def _on_menu_action_result(self, title: str, message: str):
        QMessageBox.information(self, title, message)

    def _init_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("대기 중 | API 미연결 | 지연 시간: - ms")

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
