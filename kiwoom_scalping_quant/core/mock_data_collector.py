import asyncio
import csv
import logging
from datetime import datetime

class MockDataCollector:
    """
    증권사 웹소켓 대신 CSV 파일을 읽어 실시간 틱을 모사(Mock)하는 수집기.
    """
    def __init__(self, config, data_file="mock_data.csv", speed_multiplier=10.0):
        self.config = config
        self.data_file = data_file
        # 10.0 이면 실제보다 10배 빠르게 틱을 던짐 (1시간치 데이터를 6분만에 테스트)
        self.speed_multiplier = speed_multiplier
        self.on_state_updated_callbacks = []
        self.logger = logging.getLogger("MockDataCollector")
        self._is_running = False

    async def start(self):
        self._is_running = True
        self.logger.info(f"🚀 [MOCK MODE] 가상 수집기 시작 (배속: {self.speed_multiplier}x, 파일: {self.data_file})")
        asyncio.create_task(self._play_data())

    async def stop(self):
        self._is_running = False
        self.logger.info("🛑 [MOCK MODE] 가상 수집기 중지됨.")

    async def subscribe_symbol(self, symbol):
        self.logger.info(f"[MOCK] {symbol} 구독 (가상)")

    async def unsubscribe_symbol(self, symbol):
        pass

    async def _play_data(self):
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                last_time = None

                for row in reader:
                    if not self._is_running:
                        break

                    current_time_str = row['timestamp']
                    symbol = row['symbol']
                    price = float(row['price'])
                    volume = int(row['volume'])

                    current_time = datetime.strptime(current_time_str, "%Y-%m-%d %H:%M:%S")

                    # 배속 재생을 위한 대기 시간 계산
                    if last_time is not None:
                        time_diff = (current_time - last_time).total_seconds()
                        if time_diff > 0:
                            await asyncio.sleep(time_diff / self.speed_multiplier)

                    last_time = current_time

                    # 마스터 브릿지(main.py)를 통해 StrategyManager로 가상 틱 발사!
                    for callback in self.on_state_updated_callbacks:
                        if asyncio.iscoroutinefunction(callback):
                            asyncio.create_task(callback(symbol, {}, price=price, volume=volume, timestamp=current_time_str))
                        else:
                            callback(symbol, {}, price=price, volume=volume, timestamp=current_time_str)

            self.logger.info("✅ [MOCK MODE] 파일의 모든 가상 데이터 재생이 완료되었습니다!")

        except FileNotFoundError:
            self.logger.warning(f"🚨 [MOCK MODE] 재생할 파일이 없습니다: {self.data_file}")
            self.logger.info("프로젝트 루트에 mock_data.csv 파일을 생성해주세요.")
        except Exception as e:
            self.logger.error(f"🚨 [MOCK MODE] 재생 중 에러: {e}")

    @property
    def is_connected(self):
        """UI에서 웹소켓 연결 상태를 물어보면 항상 '정상'이라고 속입니다."""
        return True