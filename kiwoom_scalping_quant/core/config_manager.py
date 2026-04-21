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
        self._env_keys = {"KIWOOM_APP_KEY", "KIWOOM_APP_SECRET", "KIWOOM_ACCESS_TOKEN", "INFLUX_URL", "INFLUX_TOKEN", "INFLUX_ORG", "TELEGRAM_BOT_TOKEN"}

        self.load_config() # 동기 로드

    def load_config(self) -> Result[Dict[str, Any], Exception]:
        """config.yaml과 .env 파일을 모두 로드하여 캐시합니다."""
        try:
            # 1. Load config.yaml
            if not os.path.exists(self.config_path):
                self._config_cache = {"symbols": [], "ws_url": "ws://localhost:8080"}
                self.save_config()
            else:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if not data:
                        data = {"symbols": []}
                    
                    # 'universe' 키에 데이터가 있고 'symbols'가 비어있을 경우 마이그레이션 지원
                    if not data.get("symbols") and data.get("universe"):
                        data["symbols"] = data.get("universe")
                    
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
            yaml_data = {k: v for k, v in self._config_cache.items() if k not in self._env_keys}
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

            return Success(True)
        except Exception as e:
            return Failure(e)

    def update_settings(self, updates: Dict[str, Any]) -> Result[bool, Exception]:
        """UI에서 전달받은 수정값들을 캐시에 갱신 후 파일에 저장합니다."""
        self._config_cache.update(updates)
        return self.save_config()

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
        symbols = self.get_symbols()
        filtered = [s for s in symbols if s.get('code') != code]

        if len(symbols) == len(filtered):
            return Failure(ValueError(f"Symbol {code} not found."))

        self._config_cache["symbols"] = filtered
        return self.save_config()

    def set_symbols(self, new_symbols: List[Dict[str, str]]) -> Result[bool, Exception]:
        """새로운 종목 리스트로 전체를 덮어씁니다 (Bulk update)."""
        self._config_cache["symbols"] = new_symbols
        return self.save_config()
