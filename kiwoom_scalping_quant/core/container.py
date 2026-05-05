from dependency_injector import containers, providers

import os
from core.data_collector import DataCollector
from core.order_manager import OrderManager
from core.config_manager import ConfigManager
from core.historical_fetcher import HistoricalFetcher
from core.universe_manager import UniverseManager
from core.strategy_manager import StrategyManager
from core.token_manager import TokenManager
from core.scheduler import MarketScheduler
from core.risk_manager import RiskManager
from core.telegram_notifier import TelegramNotifier
from db.influx_client import AsyncInfluxDBClient
from gui.view_models import AssetDataViewModel, SettingsViewModel, LiveDashboardViewModel, AITrainingViewModel, BacktestViewModel
from infrastructure.firebase_manager import FirebaseManager

class Container(containers.DeclarativeContainer):
    """
    중앙 집중식 의존성 주입(DI) 컨테이너
    모든 핵심 모듈의 생성과 생명주기를 여기서 관리하여 강한 결합을 피합니다.
    """

    # Configuration provider
    config = providers.Configuration()

    config_manager = providers.Singleton(
        ConfigManager,
        config_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    )

    historical_fetcher = providers.Singleton(
        HistoricalFetcher,
        config_manager=config_manager
    )

    universe_manager = providers.Singleton(
        UniverseManager,
        config_manager=config_manager
    )

    # DB Client (싱글톤)
    influx_client = providers.Singleton(
        AsyncInfluxDBClient,
        config=config_manager
    )

    # 텔레그램 알림 서비스 (싱글톤)
    telegram_notifier = providers.Singleton(
        TelegramNotifier,
        config_manager=config_manager
    )

    # Firebase Cloud Firestore 연동 매니저 (싱글톤)
    firebase_manager = providers.Singleton(
        FirebaseManager,
        config_manager=config_manager
    )

    # Core 비즈니스 로직 (싱글톤)
    token_manager = providers.Singleton(
        TokenManager,
        config_manager=config_manager
    )

    data_collector = providers.Singleton(
        DataCollector,
        config=config_manager
    )

    order_manager = providers.Singleton(
        OrderManager,
        config=config_manager,
        auth_manager=token_manager,
        telegram_notifier=telegram_notifier,
        firebase_manager=firebase_manager
    )

    risk_manager = providers.Singleton(
        RiskManager,
        config_manager=config_manager,
        order_manager=order_manager
    )

    strategy_manager = providers.Singleton(
        StrategyManager,
        config_manager=config_manager,
        data_collector=data_collector,
        order_manager=order_manager,
        risk_manager=risk_manager
    )


    market_scheduler = providers.Singleton(
        MarketScheduler,
        data_collector=data_collector,
        order_manager=order_manager,
        universe_manager=universe_manager,
        telegram_bot=telegram_notifier,
        firebase_manager=firebase_manager
    )

    # Presentation Layer - ViewModels (싱글톤으로 전환하여 상태 및 콜백 일관성 유지)
    live_dashboard_view_model = providers.Singleton(
        LiveDashboardViewModel,
        data_collector=data_collector,
        order_manager=order_manager,
        config_manager=config_manager
    )

    asset_data_view_model = providers.Singleton(
        AssetDataViewModel,
        config_manager=config_manager,
        historical_fetcher=historical_fetcher,
        influx_client=influx_client,
        universe_manager=universe_manager,
        token_manager=token_manager,
        firebase_manager=firebase_manager
    )

    settings_view_model = providers.Singleton(
        SettingsViewModel,
        config_manager=config_manager,
        influx_client=influx_client
    )

    ai_training_view_model = providers.Singleton(
        AITrainingViewModel,
        config_manager=config_manager,
        data_collector=data_collector,
        order_manager=order_manager,
        influx_client=influx_client
    )

    backtest_view_model = providers.Singleton(
        BacktestViewModel,
        config_manager=config_manager,
        influx_client=influx_client,
        data_collector=data_collector,
        order_manager=order_manager,
        universe_manager=universe_manager,
        historical_fetcher=historical_fetcher,
    )
