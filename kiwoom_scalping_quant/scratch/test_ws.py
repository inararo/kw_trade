import asyncio
import json
import os
import websockets
from dotenv import load_dotenv

# .env 로드
load_dotenv()

async def main():
    token = os.getenv("KIWOOM_ACCESS_TOKEN")
    ws_url = "wss://api.kiwoom.com:10000/api/dostk/websocket"
    
    print(f"Token: {token[:10]}...{token[-10:] if token else 'None'}")
    print(f"WS URL: {ws_url}")
    
    headers = {"authorization": f"Bearer {token}"} if token else {}
    
    try:
        async with websockets.connect(ws_url, extra_headers=headers) as ws:
            print("✅ WebSocket Connected!")
            
            # LOGIN
            await ws.send(json.dumps({"trnm": "LOGIN", "token": token}))
            print("✉️ Sent LOGIN")
            
            # Read first response
            resp = await ws.recv()
            print(f"📩 Recv LOGIN Response: {resp}")
            
            # CNSRLST
            await ws.send(json.dumps({"trnm": "CNSRLST"}))
            print("✉️ Sent CNSRLST")
            
            # Listen to messages
            for _ in range(5):
                msg = await ws.recv()
                print(f"📩 Recv Message: {msg}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
