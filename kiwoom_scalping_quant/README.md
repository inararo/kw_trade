# Kiwoom Scalping Quant

파이썬, 강화학습(Maskable PPO), 비동기 통신, 그리고 PyQt6 GUI를 결합한 크로스 플랫폼(macOS/Windows) 초단타(Scalping) 자동 매매 시스템입니다.
키움증권의 신규 REST API 및 WebSocket을 활용하며 기존 32비트 OCX의 한계를 완벽히 극복했습니다.

## 주요 기능

1. **데이터 수집 모드:** InfluxDB를 연동하여 실시간 호가/틱 데이터를 로스 없이 비동기 배치(Batch) 방식으로 적재.
2. **모의투자(백테스트) 모드:** 수집된 데이터를 바탕으로 커스텀 Gymnasium 환경과 바닐라 LSTM + PPO 에이전트를 이용한 시뮬레이션.
3. **실전 매매 모드:** 지연 시간 50ms 미만을 목표로 하는 극초단타 매매. Throttling 회피, 액션 마스킹(예수금 부족, 미체결 방어), Watchdog 기반 Circuit Breaker 등 리스크 관리 기능 포함.

---

## 사전 요구사항 (Prerequisites)

* **OS:** Windows 10/11 또는 macOS (M1/M2 지원)
* **Python:** 3.10 이상
* **Database:** InfluxDB 2.x (Docker 설치 권장)
* **API:** 키움증권 Open API (신규 REST/WebSocket 기반) App Key 및 Secret 발급 완료

---

## 환경 변수 및 설정 세팅

### 1. `.env` 파일 (민감 정보 관리)
프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 아래 내용을 기입합니다.
```env
KIWOOM_APP_KEY=your_app_key_here
KIWOOM_APP_SECRET=your_app_secret_here
INFLUX_TOKEN=your_influx_db_token
TELEGRAM_WEBHOOK_URL=https://api.telegram.org/bot<TOKEN>/sendMessage
```

### 2. `config.yaml` 파일 (하이퍼파라미터 및 시스템 제어)
포함된 `config.yaml`을 수정하여 시스템 파라미터를 조절할 수 있습니다.
```yaml
symbol: '005930'              # 거래 대상 종목 코드 (예: 삼성전자)
initial_balance: 10000000     # 시작 예수금 (모의투자용)
slippage: 0.0005              # 슬리피지 페널티 (0.05%)
max_buffer_size: 10000        # 메모리 롤링 버퍼 크기
db_batch_size: 500            # InfluxDB 배치 단위
ws_url: 'ws://ops.kiwoom.com' # 키움 웹소켓 URL
```

---

## 설치 및 실행 가이드

1. **가상환경 생성 및 의존성 설치**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **프로그램 실행**
   비동기 루프와 GUI(PyQt6)가 qasync에 의해 통합 실행됩니다.
   ```bash
   python main.py
   ```

---

## GUI 매뉴얼 (Dashboard)

실행 시 표시되는 대시보드는 철저히 MVVM 패턴으로 구성되어 백그라운드 로직과 분리되어 있습니다.
* **호가창 래더 (DOM View):** 실시간 10호가 매수/매도 잔량을 히트맵 형태로 표시합니다.
* **AI 신뢰도 모니터:** 에이전트의 현재 Action Probability(Hold, Buy, Sell)를 실시간 차트로 보여줍니다.
* **시스템 헬스 LED:** WebSocket 연결 상태, API 지연 시간(50ms 이상 시 경고), Watchdog 생존 여부를 시각화합니다.
* **패닉 버튼:** 클릭 즉시 보유 중인 모든 잔고를 시장가 매도하고 미체결 주문을 취소하는 비상 기능입니다.

### 상단 메뉴바 기능 상세설명

* **파일 메뉴**
  * **로그 파일 열기**: 로컬 파일 탐색기를 통해 시스템 구동 및 로깅 정보가 기록된 `logs` 디렉토리를 즉시 엽니다.
  * **프로그램 종료**: 시스템 안전 종료 파이프라인(미체결 취소, 데이터 플러시)을 가동한 뒤 애플리케이션을 안전하게 닫습니다.

* **매매 메뉴**
  * **라이브 대시보드 열기**: 실시간 호가 및 매매가 진행되는 '라이브 대시보드' 탭으로 화면을 전환합니다.
  * **미체결 전체 취소**: 브로커 서버에 제출된 모든 미체결 주문을 비동기적으로 취소 요청합니다.
  * **당일 손익 초기화**: 당일 기록된 누적 손익 트래커를 초기화합니다.

* **데이터 메뉴**
  * **종목 관리 열기**: 새로운 종목을 추가하거나 과거 데이터를 수집할 수 있는 '데이터 관리' 탭으로 전환합니다.
  * **DB 상태 점검**: 연결된 InfluxDB의 헬스체크(Ping/Health API)를 수행하고 상태를 팝업 메시지로 보고합니다.

* **AI 학습 메뉴**
  * **학습 스튜디오 열기**: Stable-Baselines3 에이전트 파라미터를 조절하고 학습을 수행할 수 있는 'AI 학습 스튜디오' 탭으로 이동합니다.
  * **모델 검증 도구**: 향후 추가될 모델 검증 모듈을 위한 예약 메뉴로, 현재는 준비 중임을 알리는 팝업을 표시합니다.

* **설정 메뉴**
  * **환경 설정 창 열기**: API 연결, DB, 텔레그램 연동 등을 설정할 수 있는 '환경 설정' 탭으로 이동합니다.
  * **API 토큰 강제 갱신**: 만료되었거나 오류가 있는 Kiwoom REST API 토큰을 즉시 강제로 새로 발급받도록 요청합니다.

---

## 테스트 및 빌드 가이드

### 1. 단위 테스트 (Pytest)
시스템의 핵심 비즈니스 로직(에러 처리, 워치독 등)은 Mocking 기반 비동기 유닛 테스트로 검증됩니다. 주말 등 장이 열리지 않은 시간에도 테스트가 가능합니다.
```bash
# 전체 테스트 실행
pytest tests/ -v
```

### 2. 단독 실행 파일 빌드 (PyInstaller)
파이썬 환경이 없는 사용자도 더블클릭으로 실행할 수 있도록 패키징합니다.
Numba, PyTorch 등의 묵시적 의존성이 `.spec` 파일에 정의되어 있습니다.
```bash
# Windows (.exe) 또는 macOS (.app) 빌드
pyinstaller kiwoom_scalping_quant.spec
```
빌드 완료 후 `dist/` 폴더에서 실행 파일을 확인할 수 있습니다.
