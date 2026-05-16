import os
import yaml
import dotenv
from typing import Dict, Any, List
from returns.result import Result, Success, Failure
from returns.future import FutureResult, future_safe

class ConfigManager:
    """
    통합 설정 관리자.
    - 민감한 정보(.env): App Key, Secret, Influx Token, Telegram API Key (.gitignore 대상)
    - 일반 매매 설정(config.yaml): 모의/실전 모드, 하드 손절 라인, 진입 자금 비율 등
    두 파일의 출처를 신경 쓰지 않고 `.get('key')` 형태로 쉽게 값을 가져올 수 있는 래퍼(Wrapper)를 제공합니다.
    불변성을 지향하며, Result 패턴으로 I/O 에러를 안전하게 핸들링합니다.
    """
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        self._config_cache: Dict[str, Any] = {}
        self._env_keys = {"KIWOOM_APP_KEY", "KIWOOM_APP_SECRET", "KIWOOM_ACCESS_TOKEN", "INFLUX_URL", "INFLUX_TOKEN", "INFLUX_ORG", "TELEGRAM_BOT_TOKEN", "FIREBASE_KEY_PATH"}
        self._runtime_keys = {"OFFLINE_MODE"} # [신규] 메모리(런타임)에만 유지하고 파일에 저장하지 않을 키 목록
        self.firebase_manager = None # [NEW] 역방향 동기화를 위한 매니저 주입용

        self.load_config(skip_symbols=True) # 초기 생성 시에는 종목 리스트를 비워둠 (이중 로드 방지)

    def load_config(self, skip_symbols: bool = False) -> Result[Dict[str, Any], Exception]:
        """config.yaml과 .env 파일을 모두 로드하여 캐시합니다."""
        try:
            # 1. Load config.yaml
            if not os.path.exists(self.config_path):
                self._config_cache = {
                    "symbols": [], 
                    "ws_url": "ws://localhost:8080",
                    "ai_confidence_threshold": 0.5
                }
                self.save_config()
            else:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if not data:
                        data = {"symbols": []}
                    
                    # 'universe' 키에 데이터가 있고 'symbols'가 비어있을 경우 마이그레이션 지원
                    if not data.get("symbols") and data.get("universe"):
                        data["symbols"] = data.get("universe")
                    
                    # symbols를 건너뛰어야 하는 경우 빈 리스트로 설정
                    if skip_symbols:
                        data["symbols"] = []
                        if "universe" in data: data["universe"] = []

                    self._config_cache = data

            # 2. Load .env
            dotenv.load_dotenv(self.env_path)
            for key in self._env_keys:
                self._config_cache[key] = os.getenv(key, "")

            return Success(self._config_cache)
        except Exception as e:
            return Failure(e)

    def get(self, key: str, default: Any = None) -> Any:
        """두 파일의 출처를 신경 쓰지 않고 쉽게 값을 가져갈 수 있는 래퍼 메서드."""
        return self._config_cache.get(key, default)

    def get_dict(self) -> Dict[str, Any]:
        """현재 캐시된 모든 설정을 딕셔너리 형태로 반환합니다."""
        return self._config_cache

    def hot_reload_settings(self, new_values: Dict[str, Any]) -> bool:
        """
        외부(Firebase 리스너 등)에서 전달받은 설정값으로 메모리 캐시를 즉시 업데이트합니다.
        실제로 값이 변경된 항목이 있을 경우에만 config.yaml을 저장하고 True를 반환합니다.

        Args:
            new_values: Firestore에서 수신한 변경 딕셔너리 {key: new_value}
        Returns:
            bool: 실제 변경 사항이 있어 저장까지 완료했는지 여부
        """
        import logging
        logger = logging.getLogger("ConfigManager")
        
        changed_keys = []
        for key, value in new_values.items():
            old_value = self._config_cache.get(key)
            
            # 값이 실제로 다른 경우에만 처리 (무한 루프 방지 핵심)
            if old_value != value:
                self._config_cache[key] = value
                changed_keys.append(key)
                logger.info(f"[설정값 변경 감지] {key}: {old_value} -> {value}")

                # [로깅 레벨 즉시 업데이트]
                if key == "log_level":
                    try:
                        new_level = str(value).upper()
                        numeric_level = getattr(logging, new_level, None)
                        if isinstance(numeric_level, int):
                            logging.getLogger().setLevel(numeric_level)
                            logger.critical(f"🚀 [시스템 로깅 레벨 변경] {new_level}로 즉시 적용되었습니다.")
                    except Exception as e:
                        logger.error(f"로깅 레벨 변경 중 오류: {e}")

        if not changed_keys:
            return False

        # 변경된 항목이 있을 때만 config.yaml에 영구 저장
        result = self.save_config()
        if isinstance(result, Failure):
            logger.error(f"[설정값 파일 저장 실패] {result.failure()}")
            return False
        
        logger.info(f"[설정값 파일 저장 완료] config.yaml 업데이트 (대상: {', '.join(changed_keys)})")
        return True

    def on_settings_changed(self, data: dict, loop: Any):
        """백그라운드 스레드에서 호출됨 → call_soon_threadsafe로 메인 루프에서 안전하게 실행"""
        def _apply():
            self.hot_reload_settings(data)
        loop.call_soon_threadsafe(_apply)

    def get_rest_url(self) -> str:
        """현재 설정된 trading_mode에 따른 REST API Base URL을 반환합니다."""
        kiwoom_config = self.get("kiwoom", {})
        mode = kiwoom_config.get("trading_mode", "virtual")
        urls = kiwoom_config.get("rest_base_url", {})
        return urls.get(mode, "https://mockapi.kiwoom.com")

    def get_ws_url(self) -> str:
        """현재 설정된 trading_mode에 따른 WebSocket URL을 반환합니다."""
        kiwoom_config = self.get("kiwoom", {})
        mode = kiwoom_config.get("trading_mode", "virtual")
        urls = kiwoom_config.get("ws_url", {})
        return urls.get(mode, "wss://mockapi.kiwoom.com:10000/api/dostk/websocket")

    def save_config(self) -> Result[bool, Exception]:
        """현재 캐시된 설정들을 .env와 config.yaml에 분리하여 저장합니다."""
        try:
            # 1. Save .env (python-dotenv set_key 사용)
            if not os.path.exists(self.env_path):
                open(self.env_path, 'a').close() # 빈 파일 생성

            for key in self._env_keys:
                if key in self._config_cache:
                    dotenv.set_key(self.env_path, key, str(self._config_cache[key]))

            # 2. Save config.yaml (env 키를 제외한 나머지)
            # [안전 장치] YAML 저장 시 복잡한 Python 객체(Firestore Timestamp 등)가 포함되지 않도록 기본 타입만 필터링
            yaml_data = {
                k: v for k, v in self._config_cache.items() 
                if k not in self._env_keys and k not in self._runtime_keys and isinstance(v, (str, int, float, bool, list, dict))
            }
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

            return Success(True)
        except Exception as e:
            return Failure(e)

    def update_settings(self, updates: Dict[str, Any]) -> Result[bool, Exception]:
        """UI에서 전달받은 수정값들을 캐시에 갱신 후 파일에 저장합니다."""
        import asyncio
        
        # 1. 캐시 업데이트 전, 실제로 값이 바뀐 항목들만 추출 (무한 루프 방지 및 효율성)
        actual_updates = {}
        for k, v in updates.items():
            if self._config_cache.get(k) != v:
                actual_updates[k] = v
        
        if not actual_updates:
            return Success(False) # 변경 사항 없음

        self._config_cache.update(actual_updates)
        
        # 2. 로컬 파일 저장
        save_result = self.save_config()
        if isinstance(save_result, Failure):
            return save_result

        # 3. [역방향 동기화] Firebase에 즉시 반영
        if self.firebase_manager:
            _EXCLUDED = {
                "account_number", "KIWOOM_APP_KEY", "KIWOOM_APP_SECRET", "KIWOOM_ACCESS_TOKEN",
                "INFLUX_URL", "INFLUX_TOKEN", "INFLUX_ORG", "influx_bucket", "INFLUX_BUCKET",
                "TELEGRAM_BOT_TOKEN", "telegram_chat_id", "FIREBASE_KEY_PATH",
                "kiwoom", "ws_url", "symbols", "universe"
            }
            for key, value in actual_updates.items():
                if key not in _EXCLUDED and isinstance(value, (int, float, str, bool)):
                    asyncio.create_task(self.firebase_manager.update_setting_to_remote(key, value))
        
        return Success(True)

    def set_runtime(self, key: str, value: Any):
        """[신규] 파일에 저장하지 않고 메모리(캐시)에서만 유효한 설정을 추가합니다."""
        self._config_cache[key] = value
        self._runtime_keys.add(key)

    def get_symbols(self) -> List[Dict[str, str]]:
        """저장된 종목 리스트 반환 (symbols를 우선하며 universe를 폴백으로 사용)"""
        symbols = self._config_cache.get("symbols", [])
        if not symbols:
            symbols = self._config_cache.get("universe", [])
        return symbols

    def add_symbol(self, code: str, name: str) -> Result[bool, Exception]:
        symbols = self.get_symbols()
        if any(s.get('code') == code for s in symbols):
            return Failure(ValueError(f"Symbol {code} already exists."))

        symbols.append({"code": code, "name": name})
        self._config_cache["symbols"] = symbols
        return self.save_config()

    def remove_symbol(self, code: str) -> Result[bool, Exception]:
        """단일 종목 삭제"""
        return self.remove_symbols([code])

    def remove_symbols(self, codes: List[str]) -> Result[bool, Exception]:
        """다중 종목 일괄 삭제"""
        try:
            symbols = self.get_symbols()
            code_set = set(codes)
            filtered = [s for s in symbols if s.get('code') not in code_set]

            if len(symbols) == len(filtered):
                return Failure(ValueError(f"지정한 종목들을 찾을 수 없습니다: {codes}"))

            self._config_cache["symbols"] = filtered
            return self.save_config()
        except Exception as e:
            return Failure(e)

    def set_symbols(self, new_symbols: List[Dict[str, str]]) -> Result[bool, Exception]:
        """새로운 종목 리스트로 전체를 덮어씁니다 (Bulk update)."""
        self._config_cache["symbols"] = new_symbols
        return self.save_config()
