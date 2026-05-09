import os
import yaml
import logging

class SystemConfig:
    """
    시스템 전역 설정을 관리하는 중앙 설정 클래스.
    'config.yaml' 파일과 동기화되어 여러 프로세스(main, main_live_trader) 간 설정을 공유합니다.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SystemConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = "config.yaml"):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.config_path = config_path
        self.logger = logging.getLogger("SystemConfig")
        
        # 기본값 설정
        self.BYPASS_MARKET_HOURS = False
        
        self.load()
        self._initialized = True

    def load(self):
        """로컬 파일에서 설정을 읽어옵니다."""
        if not os.path.exists(self.config_path):
            self.logger.warning(f"설정 파일이 존재하지 않습니다: {self.config_path}. 기본값을 사용합니다.")
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data:
                    self.BYPASS_MARKET_HOURS = data.get("BYPASS_MARKET_HOURS", False)
                    self.logger.info(f"시스템 설정 로드 완료: BYPASS_MARKET_HOURS={self.BYPASS_MARKET_HOURS}")
        except Exception as e:
            self.logger.error(f"시스템 설정 로드 중 오류 발생: {e}")

    def save(self):
        """현재 설정을 로컬 파일에 저장합니다."""
        try:
            data = {}
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}

            # 기존 데이터에 덮어쓰기
            data["BYPASS_MARKET_HOURS"] = self.BYPASS_MARKET_HOURS

            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                self.logger.info(f"시스템 설정 저장 완료: BYPASS_MARKET_HOURS={self.BYPASS_MARKET_HOURS}")
        except Exception as e:
            self.logger.error(f"시스템 설정 저장 중 오류 발생: {e}")

    def set_bypass_market_hours(self, value: bool):
        """장외 시간 테스트 모드 값을 설정하고 저장합니다."""
        if self.BYPASS_MARKET_HOURS != value:
            self.BYPASS_MARKET_HOURS = value
            self.save()
