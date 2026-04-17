import os
import asyncio
import time
import logging
import aiohttp
from dotenv import load_dotenv

# ---------------------------------------------------------
# 로깅 설정 (시간, 로그 레벨, 메시지 출력)
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("KiwoomAPITest")

# ---------------------------------------------------------
# 환경 변수 로드
# ---------------------------------------------------------
load_dotenv()
APP_KEY = os.getenv("KIWOOM_APP_KEY")
APP_SECRET = os.getenv("KIWOOM_APP_SECRET")
BASE_URL = os.getenv("KIWOOM_BASE_URL", "https://openapi.kiwoom.com")

async def get_access_token(session: aiohttp.ClientSession) -> str:
    """
    Client Credentials 방식을 사용하여 키움증권 REST API 접근 토큰을 발급받습니다.
    """
    url = f"{BASE_URL}/oauth2/tokenP"

    # 키움증권 가이드에 따른 Body 파라미터 구성 (client_credentials)
    payload = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET
    }

    logger.info(f"접근 토큰 발급 요청 중... (URL: {url})")
    start_time = time.time()

    try:
        async with session.post(url, json=payload, timeout=10) as response:
            elapsed = time.time() - start_time
            logger.info(f"응답 코드: {response.status} (소요 시간: {elapsed:.3f}초)")

            # 실패 시 예외를 던지지 않고 에러 내역을 명확히 로깅
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"토큰 발급 실패. 응답 본문: {error_text}")
                return None

            data = await response.json()
            access_token = data.get("access_token")

            if access_token:
                logger.info("토큰 발급 성공!")
                return access_token
            else:
                logger.error(f"응답은 200 OK이나 토큰이 존재하지 않습니다. 본문: {data}")
                return None

    except asyncio.TimeoutError:
        logger.error("토큰 발급 요청 시간 초과 (Timeout).")
        return None
    except Exception as e:
        logger.error(f"토큰 발급 중 예기치 않은 오류 발생: {str(e)}")
        return None

async def inquire_current_price(session: aiohttp.ClientSession, access_token: str, symbol: str):
    """
    발급받은 토큰과 필수 헤더를 사용하여 특정 종목의 현재가를 조회합니다.
    """
    url = f"{BASE_URL}/v1/domestic-stock/quotations/inquire-price"
    params = {"symbol": symbol}

    headers = {
        "Authorization": f"Bearer {access_token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "TEST_PRICE_INQUIRY_001" # 임의의 트랜잭션 ID
    }

    logger.info(f"[{symbol}] 현재가 조회 요청 중... (URL: {url})")
    start_time = time.time()

    try:
        async with session.get(url, headers=headers, params=params, timeout=5) as response:
            elapsed = time.time() - start_time
            logger.info(f"응답 코드: {response.status} (소요 시간: {elapsed:.3f}초)")

            if response.status != 200:
                error_text = await response.text()
                logger.error(f"현재가 조회 실패. 응답 본문: {error_text}")
                return None

            data = await response.json()
            logger.info(f"현재가 조회 성공! 결과:\n{data}")
            return data

    except asyncio.TimeoutError:
        logger.error("현재가 조회 요청 시간 초과 (Timeout).")
        return None
    except Exception as e:
        logger.error(f"현재가 조회 중 예기치 않은 오류 발생: {str(e)}")
        return None

async def main():
    logger.info("=== 키움증권 REST API 통신 테스트 시작 ===")

    if not APP_KEY or not APP_SECRET:
        logger.error("환경 변수 (KIWOOM_APP_KEY, KIWOOM_APP_SECRET)가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        return

    async with aiohttp.ClientSession() as session:
        # 1. 토큰 발급 테스트
        token = await get_access_token(session)

        if not token:
            logger.error("토큰 발급에 실패하여 테스트를 중단합니다.")
            return

        logger.info("-" * 50)

        # 2. 현재가 조회 테스트 (예: 삼성전자 '005930')
        symbol_to_test = "005930"
        await inquire_current_price(session, token, symbol_to_test)

    logger.info("=== 키움증권 REST API 통신 테스트 종료 ===")

if __name__ == "__main__":
    asyncio.run(main())
