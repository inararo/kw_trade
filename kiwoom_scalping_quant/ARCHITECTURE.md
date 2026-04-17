# Kiwoom Scalping Quant - Architecture

이 문서는 Kiwoom Scalping Quant 시스템의 전반적인 구조와 데이터 흐름, 그리고 소프트웨어 엔지니어링 설계 원칙을 설명하는 아키텍처 정의서입니다.

## 1. 시스템 구조도 (System Architecture)

다음은 시스템 내 핵심 컴포넌트 간의 상호작용과 데이터 흐름을 나타내는 시퀀스/컴포넌트 다이어그램입니다.

```mermaid
graph TD
    %% 외부 인터페이스
    Kiwoom[Kiwoom Securities API]
    WebSocket[Kiwoom WebSocket]

    %% 데이터/인프라 계층
    DB[(InfluxDB 2.x)]
    DataCollector[Data Collector\n(Rolling Buffer)]
    OrderManager[Order Manager\n(Throttling)]
    DIContainer[DI Container\n(dependency_injector)]

    %% 도메인/비즈니스 계층
    TradingEnv[Trading Environment\n(Gymnasium)]
    RLAgent[Maskable PPO Agent\n+ LSTM Extractor]
    Watchdog[Watchdog & Circuit Breaker]

    %% 프레젠테이션(GUI) 계층
    ViewModel[MarketData ViewModel\n(qasync & pyqtSignal)]
    GUI[PyQt6 Dashboard\n(Ladder, Chart)]

    %% 데이터 흐름
    WebSocket -->|Real-time Ticks/Orderbook| DataCollector
    Kiwoom <-->|REST API (Auth, Order, Execution)| OrderManager
    DataCollector -->|Tick Aggregation| TradingEnv
    TradingEnv <-->|State / Action| RLAgent
    RLAgent -->|Buy/Sell Signal| OrderManager

    DataCollector -->|Async Batch Write| DB
    DataCollector -->|State Events| ViewModel
    OrderManager -->|Execution Updates| ViewModel
    ViewModel -->|Emit Signals| GUI

    %% 제어 흐름
    Watchdog -.->|Monitor Latency & Heartbeat| DataCollector
    DIContainer -.->|Inject Dependencies| DataCollector
    DIContainer -.->|Inject Dependencies| OrderManager
    DIContainer -.->|Inject Dependencies| ViewModel
```

## 2. 계층형 설계 원칙 (Clean Architecture)

본 시스템은 높은 유지보수성과 유닛 테스트 용이성을 위해 철저한 계층 분리(Separation of Concerns) 원칙을 따릅니다.

1. **프레젠테이션 계층 (Presentation Layer - `gui/`)**
   - **역할:** 사용자에게 데이터를 시각화하고 입력을 받습니다.
   - **제약사항:** 비즈니스 로직(Core) 객체에 직접 접근하지 않습니다. 오직 `ViewModel`을 주입받아 `pyqtSignal` 이벤트만 구독(Subscribe)하며, 단방향 데이터 플로우를 유지합니다.

2. **도메인 계층 (Domain Layer - `env/`, `models/`)**
   - **역할:** 스캘핑 매매 로직, 강화학습 에피소드 관리, 피처 엔지니어링을 담당합니다.
   - **설계:** `TradingEnv`는 `DataCollector` 버퍼에서 데이터를 읽고, 액션 마스킹 규정(예수금 부족, 미체결 존재 등)을 적용합니다. `LSTMExtractor`는 Z-score 롤링 정규화를 통해 데이터 리키지(Data Leakage)를 원천 차단합니다.

3. **데이터 계층 (Data & Core Layer - `core/`, `db/`)**
   - **역할:** 외부 API 통신, 데이터 수집, 인메모리 버퍼링 및 DB 저장을 담당합니다.
   - **설계:** 모든 데이터 병목은 비동기 I/O와 메모리 내 `Rolling Buffer`(deque, Numpy array)로 해결하며, InfluxDB 저장은 `Batch` 처리로 최적화되었습니다. 연산이 많은 수식은 Numba JIT 컴파일로 C 언어 수준의 속도를 보장합니다.

## 3. 에러 처리 및 상태 관리 전략

HFT 환경에서는 단 한 번의 예외가 치명적인 손실로 이어질 수 있습니다.

* **함수형 에러 처리 (Result / Either Pattern)**
  전통적인 `try-except`로 예외를 던지는 대신 `returns` 라이브러리를 활용합니다. API 발주 등 네트워크 통신 함수는 성공 시 `Success(Value)`, 실패 시 `Failure(Exception)`로 래핑된 `FutureResult`를 반환합니다. 이를 통해 호출자는 강제적으로 에러 상황을 분기 처리해야 하며, 예기치 않은 크래시를 방지합니다.

* **Watchdog 및 Circuit Breaker**
  `DataCollector` 내부의 Watchdog 코루틴이 데이터 수신 주기를 지속적으로 모니터링합니다. 설정된 임계치(예: 3초) 이상 시세가 수신되지 않으면 즉각 **Circuit Breaker**가 발동하여, 보유 중인 미체결 주문을 일괄 취소하고 웹소켓의 재연결을 시도합니다.

* **불변성(Immutability)**
  멀티스레드/비동기 루프 간 상태 충돌을 막기 위해 뷰모델을 거쳐 UI로 전달되는 호가 데이터나 로직 내부의 주요 데이터는 깊은 복사(Deep Copy) 또는 Frozen Dataclass 형태로 다루어집니다.
