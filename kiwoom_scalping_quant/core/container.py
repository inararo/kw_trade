from dependency_injector import containers, providers

import os
from core.data_collector import DataCollector
from core.order_manager import OrderManager
from core.config_manager import ConfigManager
from core.historical_fetcher import HistoricalFetcher
from core.universe_manager import UniverseManager
from db.influx_client import AsyncInfluxDBClient
from gui.view_models import MarketDataViewModel, AssetDataViewModel, SettingsViewModel

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
        HistoricalFetcher
    )

    universe_manager = providers.Singleton(
        UniverseManager
    )

    # DB Client (싱글톤)
    influx_client = providers.Singleton(
        AsyncInfluxDBClient,
        config=config
    )

    # Core 비즈니스 로직 (싱글톤)
    data_collector = providers.Singleton(
        DataCollector,
        config=config
    )

    order_manager = providers.Singleton(
        OrderManager,
        config=config,
        auth_manager=None # 추후 AuthManager provider 주입 가능
    )

    # Presentation Layer - ViewModels (팩토리 혹은 싱글톤으로 관리)
    # ViewModel은 주입된 data_collector에 의존함
    market_data_view_model = providers.Factory(
        MarketDataViewModel,
        data_collector=data_collector
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
