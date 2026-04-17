import asyncio
import websockets
import json
import time
import random

async def kiwoom_mock_handler(websocket, path=None):
    """
    키움증권 WebSocket 서버를 흉내내는 Mock 서버입니다.
    """
    print("클라이언트 연결됨.")
    try:
        # 1. 구독 요청 받기
        msg = await websocket.recv()
        print(f"수신된 구독 요청: {msg}")

        # 2. 실시간 가짜 틱 데이터 전송
        base_price = 50000
        while True:
            base_price += random.randint(-100, 100)

            fake_tick = {
                "symbol": "005930",
                "price": base_price,
                "volume": random.randint(1, 100),
                "timestamp": time.time()
            }

            await websocket.send(json.dumps(fake_tick))
            await asyncio.sleep(0.5) # 0.5초마다 틱 데이터 전송

    except websockets.exceptions.ConnectionClosed:
        print("클라이언트 연결 종료됨.")

async def main():
    print("Mock WebSocket 서버를 시작합니다... (ws://localhost:8080/kiwoom)")
    # websockets 11.0.3 호환 방식
    server = await websockets.serve(kiwoom_mock_handler, "localhost", 8080)
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
