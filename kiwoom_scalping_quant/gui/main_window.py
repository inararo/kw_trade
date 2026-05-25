import os
from PyQt6.QtWidgets import QMainWindow, QTabWidget, QWidget, QVBoxLayout, QMenuBar, QMenu, QStatusBar, QMessageBox
from PyQt6.QtGui import QAction, QDesktopServices, QGuiApplication
from PyQt6.QtCore import QUrl, pyqtSlot

from gui.tabs.live_dashboard import LiveDashboardTab
from gui.tabs.asset_data_manager import AssetDataManagerTab
from gui.tabs.ai_training_studio import AITrainingStudioTab
from gui.tabs.settings_tab import SettingsTab
from gui.tabs.backtest_tab import BacktestStudioTab
from gui.tabs.policy_inspector_tab import PolicyInspectorTab

class MainWindow(QMainWindow):
    """
    HTS 스타일의 메인 윈도우 클래스.
    메뉴바, 상태 표시줄, 그리고 탭 위젯으로 구성됩니다.
    """
    def __init__(self, view_model, system):
        super().__init__()
        self.system = system
        self.view_model = view_model

        self.setWindowTitle("스캘핑 퀀트")
        self._apply_optimal_geometry()

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

        action_model_val = QAction("시각적 백테스트 열기", self)
        action_model_val.triggered.connect(lambda: self.tabs.setCurrentIndex(3))
        ai_menu.addAction(action_model_val)

        action_policy_check = QAction("모델 정책 진단 열기", self)
        action_policy_check.triggered.connect(lambda: self.tabs.setCurrentIndex(4))
        ai_menu.addAction(action_policy_check)

        # Settings Menu
        settings_menu = menu_bar.addMenu("설정")

        action_open_settings = QAction("환경 설정 창 열기", self)
        action_open_settings.triggered.connect(lambda: self.tabs.setCurrentIndex(5))
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


    def _init_tabs(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # 각 탭 초기화 및 의존성 주입

        # LiveDashboardViewModel 추출 및 주입
        live_vm = getattr(self.system, "live_vm", None)
        if live_vm is None and hasattr(self.system, "container"):
            live_vm = self.system.container.live_dashboard_view_model()
        self.tab_live = LiveDashboardTab(live_vm)

        # AssetDataViewModel은 main_window를 생성할 때 주입받은 시스템 객체에서 가져옵니다.
        asset_vm = getattr(self.system, "asset_vm", None)
        if asset_vm is None and hasattr(self.system, "container"):
            asset_vm = self.system.container.asset_data_view_model()

        self.tab_asset = AssetDataManagerTab(asset_vm)

        ai_vm = getattr(self.system, "ai_training_view_model", None)
        if ai_vm is None and hasattr(self.system, "container"):
            ai_vm = self.system.container.ai_training_view_model()
        self.tab_ai = AITrainingStudioTab(ai_vm)

        bt_vm = getattr(self.system, "backtest_view_model", None)
        if bt_vm is None and hasattr(self.system, "container"):
            bt_vm = self.system.container.backtest_view_model()
        self.tab_bt = BacktestStudioTab(bt_vm)

        settings_vm = getattr(self.system, "settings_view_model", None)
        if settings_vm is None and hasattr(self.system, "container"):
            settings_vm = self.system.container.settings_view_model()

        self.tab_settings = SettingsTab(settings_vm)

        # 모델 정책 진단 탭 초기화
        config_mgr = self.system.container.config_manager()
        self.tab_policy = PolicyInspectorTab(config_mgr)

        from PyQt6.QtCore import Qt

        # Connect settings VM menu actions
        if hasattr(settings_vm, "check_db_status"):
            self._action_check_db.triggered.connect(settings_vm.check_db_status)
        if hasattr(settings_vm, "force_refresh_token"):
            self._action_force_token.triggered.connect(settings_vm.force_refresh_token)
        if hasattr(settings_vm, "sig_menu_action_result"):
            settings_vm.sig_menu_action_result.connect(self._on_menu_action_result, Qt.ConnectionType.QueuedConnection)

        # Connect live VM menu actions
        if hasattr(live_vm, "sig_menu_action_result"):
            live_vm.sig_menu_action_result.connect(self._on_menu_action_result, Qt.ConnectionType.QueuedConnection)

        # 주도주 유니버스 갱신 시 백테스트 탭 및 정책 진단 탭의 종목 리스트 자동 업데이트 연결
        if asset_vm:
            if hasattr(self, 'tab_bt'):
                asset_vm.symbols_loaded.connect(self.tab_bt._populate_symbols)
            if hasattr(self, 'tab_policy'):
                asset_vm.symbols_loaded.connect(self.tab_policy._refresh_symbols)

        self.tabs.addTab(self.tab_live, "라이브 대시보드")
        self.tabs.addTab(self.tab_asset, "데이터 관리")
        self.tabs.addTab(self.tab_ai, "AI 학습 스튜디오")
        self.tabs.addTab(self.tab_bt, "Backtest Studio")
        self.tabs.addTab(self.tab_policy, "모델 정책 진단") # 인덱스 4
        self.tabs.addTab(self.tab_settings, "환경 설정")      # 인덱스 5

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

    def showEvent(self, event):
        super().showEvent(event)
        # 창이 화면에 올라간(Render) 직후, macOS 시스템의 창 복원 및 snapping 패스가 종료될 때까지 100ms 지연 후 크기 강제 패치
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, self._force_apply_geometry)

    def _force_apply_geometry(self):
        from PyQt6.QtCore import Qt
        # macOS의 창 상태 캐싱(Zoom/Maximized)을 원천 무력화하고 강제로 일반 창 모드 설정
        self.setWindowState(Qt.WindowState.WindowNoState)
        self.showNormal()
        self._apply_optimal_geometry()

    def _apply_optimal_geometry(self):
        from PyQt6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            ratio = screen.devicePixelRatio()
            
            # Retina 디스플레이 대응: 만약 물리적 해상도로 보고될 경우 디바이스 픽셀 비율로 나누어 완벽한 논리적 해상도 도출
            logical_w = geom.width()
            logical_h = geom.height()
            logical_x = geom.x()
            logical_y = geom.y()
            
            if logical_h > 1200 and ratio > 1.0:
                logical_w = int(logical_w / ratio)
                logical_h = int(logical_h / ratio)
                logical_x = int(logical_x / ratio)
                logical_y = int(logical_y / ratio)
                
            w = int(logical_w * 0.98)
            h = int(logical_h * 0.96)
            x = logical_x + (logical_w - w) // 2
            y = logical_y + (logical_h - h) // 2
            
            print(f"★ [DEBUG RETINA] ratio: {ratio}, logical: {logical_w}x{logical_h}, target: {w}x{h}, Y: {y}")
            self.resize(w, h)
            self.move(x, y)
            self.update()
        else:
            self.setGeometry(100, 80, 1500, 850)
