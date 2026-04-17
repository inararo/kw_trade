from dependency_injector import containers, providers

from core.data_collector import DataCollector
from core.order_manager import OrderManager
from db.influx_client import AsyncInfluxDBClient
from gui.view_models import MarketDataViewModel

class Container(containers.DeclarativeContainer):
    """
    중앙 집중식 의존성 주입(DI) 컨테이너
    모든 핵심 모듈의 생성과 생명주기를 여기서 관리하여 강한 결합을 피합니다.
    """

    # Configuration provider
    config = providers.Configuration()

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
