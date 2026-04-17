import os
import yaml
from typing import Dict, Any, List
from returns.result import Result, Success, Failure
from returns.future import FutureResult, future_safe

class ConfigManager:
    """
    설정 파일(config.yaml)의 읽기/쓰기 및 상태를 관리하는 중앙 관리자.
    불변성을 지향하며, Result 패턴으로 I/O 에러를 안전하게 핸들링합니다.
    """
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._config_cache: Dict[str, Any] = {}
        self.load_config() # 동기 로드

    def load_config(self) -> Result[Dict[str, Any], Exception]:
        try:
            if not os.path.exists(self.config_path):
                # 기본 설정 템플릿 생성
                self._config_cache = {"symbols": [], "ws_url": "ws://localhost:8080"}
                self.save_config()
                return Success(self._config_cache)

            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                self._config_cache = data if data else {"symbols": []}
            return Success(self._config_cache)
        except Exception as e:
            return Failure(e)

    def save_config(self) -> Result[bool, Exception]:
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(self._config_cache, f, default_flow_style=False, allow_unicode=True)
            return Success(True)
        except Exception as e:
            return Failure(e)

    def get_symbols(self) -> List[Dict[str, str]]:
        """저장된 종목 리스트 반환 [{'code': '005930', 'name': '삼성전자'}]"""
        return self._config_cache.get("symbols", [])

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
