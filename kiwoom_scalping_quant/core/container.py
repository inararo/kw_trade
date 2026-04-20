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
from db.influx_client import AsyncInfluxDBClient
from gui.view_models import AssetDataViewModel, SettingsViewModel, LiveDashboardViewModel, AITrainingViewModel, BacktestViewModel

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
        config=config
    )

    # Core 비즈니스 로직 (싱글톤)
    data_collector = providers.Singleton(
        DataCollector,
        config=config_manager
    )

    order_manager = providers.Singleton(
        OrderManager,
        config=config_manager,
        auth_manager=None # 추후 AuthManager provider 주입 가능
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
        order_manager=order_manager
    )

    token_manager = providers.Singleton(
        TokenManager,
        config_manager=config_manager
    )

    market_scheduler = providers.Singleton(
        MarketScheduler,
        data_collector=data_collector,
        order_manager=order_manager,
        universe_manager=universe_manager,
        telegram_bot=None
    )

    token_manager = providers.Singleton(
        TokenManager,
        config_manager=config_manager
    )

    market_scheduler = providers.Singleton(
        MarketScheduler,
        data_collector=data_collector,
        order_manager=order_manager,
        universe_manager=universe_manager,
        telegram_bot=None
    )

    # Presentation Layer - ViewModels (팩토리 혹은 싱글톤으로 관리)
    live_dashboard_view_model = providers.Factory(
        LiveDashboardViewModel,
        data_collector=data_collector,
        order_manager=order_manager
    )

    asset_data_view_model = providers.Factory(
        AssetDataViewModel,
        config_manager=config_manager,
        historical_fetcher=historical_fetcher,
        influx_client=influx_client,
        universe_manager=universe_manager
    )

    settings_view_model = providers.Factory(
        SettingsViewModel,
        config_manager=config_manager,
        influx_client=influx_client
    )

    ai_training_view_model = providers.Factory(
        AITrainingViewModel,
        config_manager=config_manager,
        data_collector=data_collector,
        order_manager=order_manager,
        influx_client=influx_client
    )

    backtest_view_model = providers.Factory(
        BacktestViewModel,
        config_manager=config_manager,
        influx_client=influx_client,
        data_collector=data_collector,
        order_manager=order_manager
    )
