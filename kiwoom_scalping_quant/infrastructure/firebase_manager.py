"""
FirebaseManager - 인프라 계층 (Infrastructure Layer)

Firebase Cloud Firestore를 통한 외부 서비스 연동 모듈.
- 체결 로그를 Firestore `trade_logs` 컬렉션에 기록
- 시스템 상태 및 엔진 하트비트를 Firestore `system_status/engine` 문서에 실시간 업데이트

[비동기 처리 전략]
firebase-admin의 Firestore SDK는 동기(Blocking) API입니다.
asyncio 이벤트 루프가 블로킹되지 않도록 asyncio.to_thread()를 사용하여
OS 스레드 풀에서 실행합니다.

[Fail-Safe 원칙]
Firebase 초기화/전송 실패 시 시스템 매매 흐름이 절대 중단되지 않도록
모든 public 메서드에서 예외를 내부적으로 처리합니다.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional, Any

logger = logging.getLogger("FirebaseManager")


class FirebaseManager:
    """
    Firebase Cloud Firestore 연동 매니저.

    DI 컨테이너에 Singleton으로 등록되며,
    OrderManager(체결 로그)와 MarketScheduler(시스템 상태)에서 호출됩니다.
    """

    def __init__(self, config_manager=None):
        """
        Args:
            config_manager: ConfigManager 인스턴스.
                            FIREBASE_KEY_PATH 설정값을 읽기 위해 사용.
                            None이면 프로젝트 루트의 firebase_key.json으로 폴백.
        """
        self.config_manager = config_manager
        self._db = None        # firestore.Client 인스턴스
        self._initialized = False
        self._init_firebase()

    # ─────────────────────────────────────────────────────────────
    # 내부 초기화
    # ─────────────────────────────────────────────────────────────

    def _init_firebase(self):
        """Firebase 앱 및 Firestore 클라이언트를 초기화합니다."""
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            # 1. 키 파일 경로 결정: 환경변수 > ConfigManager > 루트 폴백
            key_path = os.getenv("FIREBASE_KEY_PATH", "")
            if not key_path and self.config_manager:
                key_path = self.config_manager.get("FIREBASE_KEY_PATH", "")
            if not key_path:
                # 프로젝트 루트 기준으로 자동 탐색
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                key_path = os.path.join(root_dir, "firebase_key.json")

            if not os.path.exists(key_path):
                logger.warning(
                    f"FirebaseManager: 키 파일을 찾을 수 없습니다 ({key_path}). "
                    "Firestore 연동이 비활성화됩니다."
                )
                return

            # 2. 중복 초기화 방지 (앱이 이미 존재하면 재사용)
            try:
                app = firebase_admin.get_app()
            except ValueError:
                cred = credentials.Certificate(key_path)
                app = firebase_admin.initialize_app(cred)

            # 3. Firestore 클라이언트 생성
            self._db = firestore.client()
            self._initialized = True
            logger.info("FirebaseManager: Firestore 초기화 성공 ✅")

        except ImportError:
            logger.warning(
                "FirebaseManager: firebase-admin 패키지가 설치되지 않았습니다. "
                "`pip install firebase-admin`을 실행하세요."
            )
        except Exception as e:
            logger.error(f"FirebaseManager: 초기화 중 예외 발생 (연동 비활성화): {e}")

    # ─────────────────────────────────────────────────────────────
    # Public API - 체결 로그 전송 (Write)
    # ─────────────────────────────────────────────────────────────

    async def send_trade_log(
        self,
        log_type: str,
        symbol: str,
        symbol_name: str,
        price: float,
        qty: int,
        timestamp: str,
    ):
        """
        체결 데이터를 Firestore `trade_logs` 컬렉션에 비동기로 기록합니다.

        Args:
            log_type:    주문 유형 ('BUY' 또는 'SELL')
            symbol:      종목 코드 (예: '005930')
            symbol_name: 종목명 (예: '삼성전자')
            price:       체결 가격
            qty:         체결 수량
            timestamp:   체결 시각 문자열 (ISO 8601 권장)

        Firestore 저장 포맷:
        {
            "log_type":    "BUY",
            "symbol":      "005930",
            "symbol_name": "삼성전자",
            "price":       75000.0,
            "qty":         10,
            "timestamp":   "2026-04-29T09:05:00",
            "created_at":  <Firestore 서버 타임스탬프>
        }
        """
        trade_data = {
            "log_type":    log_type,
            "symbol":      symbol,
            "symbol_name": symbol_name,
            "price":       float(price),
            "qty":         int(qty),
            "timestamp_str": timestamp, # 기존 필드명과 충돌 피하기 위해 변경
        }
        await self.add_trade_log(trade_data)

    async def add_trade_log(self, trade_data: dict):
        """
        체결 내역을 Firestore trade_logs 컬렉션에 실시간으로 업로드합니다.
        
        Args:
            trade_data: 체결 정보 딕셔너리
                        (symbol, type, price, quantity, profit_loss 등 포함)
        """
        if not self._initialized:
            logger.error("FirebaseManager: 초기화되지 않아 로그를 전송할 수 없습니다.")
            return
        if not self._db:
            logger.error("FirebaseManager: DB 연결(Firestore)이 없어 로그를 전송할 수 없습니다.")
            return

        from firebase_admin import firestore as fs
        
        # 1. 서버 타임스탬프 강제 포함 (앱 정렬용)
        trade_data["timestamp"] = fs.SERVER_TIMESTAMP
        
        # 2. 전송 데이터 복사 (원본 딕셔너리 변조 방지)
        doc_data = trade_data.copy()
        logger.error(f"📤 [Firestore Payload] {doc_data}")

        try:
            # 3. asyncio.to_thread를 사용하여 블로킹 방지 (SDK가 동기 방식이므로 필수)
            await asyncio.to_thread(
                self._db.collection("trade_logs").add, doc_data
            )
            logger.error(
                f"✅ FirebaseManager: 체결 로그 실시간 업로드 완료 "
                f"({doc_data.get('symbol', 'UNKNOWN')})"
            )
        except Exception as e:
            # Fail-Safe: Firebase 전송 실패가 매매 흐름을 절대 중단시키지 않음
            logger.error(f"FirebaseManager: 체결 로그 업로드 중 에러 발생 (무시): {e}")

    # ─────────────────────────────────────────────────────────────
    # Public API - 시스템 상태 업데이트 (Upsert)
    # ─────────────────────────────────────────────────────────────

    async def update_engine_status(self, status: str):
        """
        엔진 가동 상태를 system_status/engine 문서에 기록합니다. (설정값과 분리)
        
        Args:
            status: "RUNNING" 또는 "OFFLINE"
        """
        if not self._initialized or not self._db:
            return

        from firebase_admin import firestore as fs
        try:
            # [리팩토링] settings/core에서 system_status/engine으로 경로 변경
            doc_ref = self._db.collection("system_status").document("engine")
            await asyncio.to_thread(
                doc_ref.set,
                {
                    "engine_status": status,
                    "last_heartbeat": fs.SERVER_TIMESTAMP
                },
                merge=True
            )
            logger.info(f"FirebaseManager: 엔진 상태 업데이트 (system_status/engine) → {status}")
        except Exception as e:
            logger.error(f"FirebaseManager: 엔진 상태 업데이트 실패: {e}")

    async def start_heartbeat(self):
        """
        1분마다 system_status/engine 문서의 last_heartbeat 필드를 갱신합니다.
        """
        if not self._initialized or not self._db:
            return

        from firebase_admin import firestore as fs
        logger.info("FirebaseManager: 실시간 하트비트(Heartbeat) 태스크를 시작합니다. (대상: system_status/engine)")
        
        while True:
            try:
                doc_ref = self._db.collection("system_status").document("engine")
                await asyncio.to_thread(
                    doc_ref.set,
                    {"last_heartbeat": fs.SERVER_TIMESTAMP},
                    merge=True
                )
                logger.debug("FirebaseManager: Heartbeat 갱신 완료")
            except Exception as e:
                logger.warning(f"FirebaseManager: Heartbeat 갱신 실패 (재시도 예정): {e}")
            
            await asyncio.sleep(60) # 1분 대기

    async def update_system_status(self, state: str):
        """
        현재 시스템(봇) 상태를 Firestore `system_status/engine` 문서에 업데이트합니다. (경로 통합)
        모바일 앱에서 봇의 현재 운영 상태를 실시간으로 확인할 수 있습니다.

        Args:
            state: MarketState 상태 문자열
                   예) 'BOOTING', 'IDLE', 'PREPARE', 'TRADING',
                       'CUTOFF', 'LIQUIDATING', 'STOPPED'

        Firestore 저장 포맷 (컬렉션: system_status / 도큐먼트: engine):
        {
            "current_state": "TRADING",
            "updated_at":    <Firestore 서버 타임스탬프>
        }
        """
        if not self._initialized or not self._db:
            return

        from firebase_admin import firestore as fs

        doc_data = {
            "current_state": state,
            "updated_at":    fs.SERVER_TIMESTAMP,
        }

        try:
            # merge=True: 도큐먼트가 없으면 생성, 있으면 해당 필드만 갱신
            await asyncio.to_thread(
                self._db.collection("system_status").document("engine").set,
                doc_data,
                merge=True,
            )
            logger.debug(f"FirebaseManager: system_status/engine 업데이트 → {state}")
        except Exception as e:
            logger.error(f"FirebaseManager: system_status/engine 업데이트 실패 (무시): {e}")

    # ─────────────────────────────────────────────────────────────
    # Public API - 실시간 리스너 (Listen)
    # ─────────────────────────────────────────────────────────────

    async def get_current_settings(self) -> dict:
        """
        부팅 시 settings/core 도큐먼트의 현재 상태를 1회 읽어옵니다.
        """
        if not self._initialized or not self._db:
            return {}

        try:
            doc_ref = self._db.collection("settings").document("core")
            doc = await asyncio.to_thread(doc_ref.get)
            if doc.exists:
                data = doc.to_dict()
                logger.info(f"FirebaseManager: settings/core 부팅 시 현재 상태 읽기 완료 ✅")
                return data
            return {}
        except Exception as e:
            logger.error(f"FirebaseManager: settings/core 초기 읽기 실패 (무시): {e}")
            return {}

    async def get_engine_status(self) -> dict:
        """
        부팅 시 system_status/engine 도큐먼트의 현재 상태를 1회 읽어옵니다.
        (is_monitoring_active, is_ai_trading_active 등 제어 플래그 확인용)
        """
        if not self._initialized or not self._db:
            return {}

        try:
            doc_ref = self._db.collection("system_status").document("engine")
            doc = await asyncio.to_thread(doc_ref.get)
            if doc.exists:
                data = doc.to_dict()
                logger.info(f"FirebaseManager: system_status/engine 부팅 시 현재 상태 읽기 완료 ✅")
                return data
            return {}
        except Exception as e:
            logger.error(f"FirebaseManager: system_status/engine 초기 읽기 실패 (무시): {e}")
            return {}

    async def update_setting_to_remote(self, key: str, value: Any):
        """
        로컬에서 변경된 특정 설정값을 Firestore의 settings/core 문서에 즉시 동기화합니다.

        Args:
            key: Firestore 필드 이름
            value: 업데이트할 값
        """
        if not self._initialized or not self._db:
            return

        try:
            doc_ref = self._db.collection("settings").document("core")
            # 딕셔너리 형태로 감싸서 단일 필드 업데이트
            await asyncio.to_thread(doc_ref.update, {key: value})
            logger.info(f"[INFO] 로컬 설정 변경사항을 파이어베이스에 성공적으로 동기화했습니다: {key} -> {value}")
        except Exception as e:
            logger.error(f"FirebaseManager: 로컬 설정 동기화 실패 ({key}): {e}")

    async def initialize_default_settings(self, default_config: dict):
        """
        부팅 시 settings/core 도큐먼트에 기본 설정값을 안전하게 업로드합니다.

        [핵심 동작]
        merge=True를 사용하므로:
        - 도큐먼트가 없을 경우: default_config 그대로 생성
        - 도큐먼트가 이미 있을 경우: 모바일 앱에서 변경한 값은 유지하고,
          default_config에만 존재하는 새 키(신규 설정 항목)만 추가합니다.

        Args:
            default_config: 업로드할 기본 설정 딕셔너리
                예) {"max_position_pct": 10.0, "stop_loss_pct": -3.0, ...}
        """
        if not self._initialized or not self._db:
            logger.warning("FirebaseManager: settings 초기화 건너뜀 (Firestore 비활성)")
            return

        try:
            doc_ref = self._db.collection("settings").document("core")
            
            # [리팩토링] 기존 settings/core에 남아있을 수 있는 엔진 상태 키들을 제거
            from firebase_admin import firestore as fs
            cleanup_data = {
                "engine_status": fs.DELETE_FIELD,
                "last_heartbeat": fs.DELETE_FIELD,
                "last_updated_by_engine": fs.DELETE_FIELD
            }
            # DELETE_FIELD는 update()에서만 동작합니다.
            try:
                await asyncio.to_thread(doc_ref.update, cleanup_data)
                logger.info("FirebaseManager: settings/core 내 엔진 상태 필드 정리 완료 (Path 분리 대응)")
            except:
                pass # 필드가 이미 없으면 에러날 수 있음 (무시)

            await asyncio.to_thread(doc_ref.set, default_config, merge=True)
            logger.info(
                f"FirebaseManager: settings/core 기본값 업로드 완료 ✅ "
                f"(키 {len(default_config)}개, merge=True)"
            )
        except Exception as e:
            logger.error(f"FirebaseManager: settings 초기화 실패 (무시): {e}")

    async def report_settings_applied(self):
        """
        엔진이 원격 설정을 성공적으로 반영했음을 system_status/engine에 기록합니다.
        (settings/core의 에코 방지를 위해 경로를 분리했습니다.)
        """
        if not self._initialized or not self._db:
            return

        from firebase_admin import firestore as fs
        try:
            doc_ref = self._db.collection("system_status").document("engine")
            await asyncio.to_thread(
                doc_ref.set, 
                {"last_updated_by_engine": fs.SERVER_TIMESTAMP},
                merge=True
            )
            logger.debug("FirebaseManager: settings 반영 시각 보고 완료 (system_status/engine)")
        except Exception as e:
            logger.error(f"FirebaseManager: settings 반영 보고 실패: {e}")

    def listen_to_settings(self, callback_func):
        """
        settings/core 도큐먼트의 변경사항을 실시간으로 감시합니다.
        on_snapshot은 백그라운드 스레드에서 실행되므로,
        callback_func 내부에서 call_soon_threadsafe 등으로 루프와 연결해야 합니다.

        Args:
            callback_func: 데이터 변경 시 호출될 함수. 매개변수: dict
        """
        if not self._initialized or not self._db:
            logger.warning("FirebaseManager: settings 리스너 비활성 (초기화 실패)")
            return

        def on_snapshot(doc_snapshot, changes, read_time):
            # [신규] 무시할 시스템 필드 목록 (엔진 스스로 업데이트하는 값들 및 다른 경로로 이동된 제어 필드)
            IGNORE_KEYS = [
                'last_heartbeat', 'engine_status', 'last_updated_by_engine', 
                'current_state', 'updated_at',
                'is_monitoring_active', 'is_ai_trading_active' # [이동] system_status/engine에서 별도 관리
            ]

            for doc in doc_snapshot:
                if doc.exists:
                    data = doc.to_dict()
                    
                    # 1. 실제 설정값만 추출 (시스템 필드 제외)
                    filtered_data = {k: v for k, v in data.items() if k not in IGNORE_KEYS}
                    
                    # 2. 에코 방지 로직: 로컬 메모리의 설정값과 실제로 다른 항목이 있는지 검사
                    has_real_change = False
                    if self.config_manager:
                        for k, v in filtered_data.items():
                            if self.config_manager.get(k) != v:
                                has_real_change = True
                                break
                    else:
                        has_real_change = True # 비교 대상이 없으면 일단 통과

                    # 3. 하트비트만 변경된 경우 로그 없이 즉시 리턴하여 루프 차단
                    if not has_real_change:
                        continue

                    # 실제 사용자가 값을 바꿨을 때만 로그 출력 및 적용
                    logger.info(f"FirebaseManager: 원격 설정 변경 감지 (실제 변경 있음) → {filtered_data}")
                    try:
                        callback_func(data)
                    except Exception as e:
                        logger.error(f"FirebaseManager: settings 콜백 오류: {e}")

        try:
            doc_ref = self._db.collection("settings").document("core")
            self._settings_watcher = doc_ref.on_snapshot(on_snapshot)
            logger.info("FirebaseManager: settings/core 리스너 활성화 ✅")
        except Exception as e:
            logger.error(f"FirebaseManager: settings 리스너 설정 중 오류: {e}")

    def listen_to_engine_status(self, callback_func):
        """
        system_status/engine 도큐먼트의 변경사항을 실시간으로 감시합니다.
        (is_monitoring_active, is_ai_trading_active 등 제어 플래그 전용)
        """
        if not self._initialized or not self._db:
            logger.warning("FirebaseManager: engine status 리스너 비활성 (초기화 실패)")
            return

        def on_snapshot(doc_snapshot, changes, read_time):
            for doc in doc_snapshot:
                if doc.exists:
                    data = doc.to_dict()
                    try:
                        callback_func(data)
                    except Exception as e:
                        logger.error(f"FirebaseManager: engine status 콜백 오류: {e}")

        try:
            doc_ref = self._db.collection("system_status").document("engine")
            self._status_watcher = doc_ref.on_snapshot(on_snapshot)
            logger.info("FirebaseManager: system_status/engine 리스너 활성화 ✅")
        except Exception as e:
            logger.error(f"FirebaseManager: engine status 리스너 설정 중 오류: {e}")

    def listen_to_commands(self, callback_func):
        """
        commands 컬렉션의 PENDING 상태 명령을 실시간으로 감시합니다.
        ADDED 이벤트(새 문서 추가)만 처리하여 중복 실행을 방지합니다.

        Args:
            callback_func: 명령 감지 시 호출될 함수. 매개변수: (doc_id: str, data: dict)
        """
        if not self._initialized or not self._db:
            logger.warning("FirebaseManager: commands 리스너 비활성 (초기화 실패)")
            return

        def on_snapshot(col_snapshot, changes, read_time):
            for change in changes:
                # ADDED만 처리: 최초 연결 시 기존 PENDING 문서 재처리 방지
                if change.type.name == 'ADDED':
                    doc = change.document
                    data = doc.to_dict()
                    logger.warning(
                        f"FirebaseManager: 원격 명령 수신 "
                        f"→ action={data.get('action')} (ID: {doc.id})"
                    )
                    try:
                        callback_func(doc.id, data)
                    except Exception as e:
                        logger.error(f"FirebaseManager: commands 콜백 오류: {e}")

        try:
            query = self._db.collection("commands").where("status", "==", "PENDING")
            self._commands_watcher = query.on_snapshot(on_snapshot)
            logger.info("FirebaseManager: commands 리스너 활성화 ✅")
        except Exception as e:
            logger.error(f"FirebaseManager: commands 리스너 설정 중 오류: {e}")

    async def update_command_status(self, doc_id: str, status: str):
        """
        명령 처리 결과를 Firestore에 기록합니다 (예: PENDING → COMPLETED).

        Args:
            doc_id: commands 컬렉션의 문서 ID
            status: 변경할 상태 문자열 (예: 'COMPLETED', 'FAILED')
        """
        if not self._initialized or not self._db:
            return

        from firebase_admin import firestore as fs

        try:
            doc_ref = self._db.collection("commands").document(doc_id)
            await asyncio.to_thread(
                doc_ref.update,
                {"status": status, "completed_at": fs.SERVER_TIMESTAMP}
            )
            logger.info(f"FirebaseManager: 명령 상태 업데이트 완료 ({doc_id} → {status})")
        except Exception as e:
            logger.error(f"FirebaseManager: 명령 상태 업데이트 실패: {e}")

    async def update_control_status(self, is_monitoring_active: bool, is_ai_trading_active: bool):
        """
        종목 감시 및 AI 매매 활성화 상태를 system_status/engine 문서에 기록합니다.
        
        Args:
            is_monitoring_active: 실시간 데이터 수집 활성화 여부
            is_ai_trading_active: AI 매매 결정 활성화 여부
        """
        if not self._initialized or not self._db:
            return

        from firebase_admin import firestore as fs
        try:
            doc_ref = self._db.collection("system_status").document("engine")
            await asyncio.to_thread(
                doc_ref.set,
                {
                    "is_monitoring_active": is_monitoring_active,
                    "is_ai_trading_active": is_ai_trading_active,
                    "updated_at": fs.SERVER_TIMESTAMP
                },
                merge=True
            )
            logger.info(f"FirebaseManager: 제어 상태 업데이트 (Monitoring: {is_monitoring_active}, AI: {is_ai_trading_active})")
        except Exception as e:
            logger.error(f"FirebaseManager: 제어 상태 업데이트 실패: {e}")

