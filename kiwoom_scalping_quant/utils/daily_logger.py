import os
import csv
from datetime import datetime
import asyncio

def log_universe_snapshot(universe: list, reason: str = "유니버스 갱신"):
    """
    갱신된 유니버스 전체 리스트를 스냅샷 형태로 기록합니다.
    [Snapshot_Time, Rank, Symbol, Name, Price, Change(%), Volume, Reason]
    """
    if not universe:
        return

    now = datetime.now()
    today_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H:%M:%S")
    
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "leaders")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    filename = f"leading_stocks_{today_str}.csv"
    filepath = os.path.join(log_dir, filename)
    
    file_exists = os.path.exists(filepath)
    
    try:
        with open(filepath, mode="a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Snapshot_Time", "Rank", "Symbol", "Name", "Price", "Change(%)", "Volume", "Reason"])
                
            for i, stock in enumerate(universe):
                writer.writerow([
                    time_str, 
                    i + 1, 
                    stock.get("code"), 
                    stock.get("name"), 
                    stock.get("price", 0), 
                    stock.get("flu_rt", 0), 
                    stock.get("volume", 0),
                    reason
                ])
    except Exception as e:
        import logging
        logging.getLogger("DailyLogger").error(f"주도주 스냅샷 로깅 에러: {e}")

