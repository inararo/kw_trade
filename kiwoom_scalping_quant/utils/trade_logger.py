import os
import csv
from datetime import datetime
from pathlib import Path

class TradeHistoryLogger:
    """
    매매 완료 시 매수/매도 이력 및 수익률을 파일에 기록하는 유틸리티입니다.
    """
    def __init__(self, filename="logs/trade_history.csv"):
        self.filepath = Path(filename)
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        # logs 디렉토리가 없으면 생성
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 파일이 없으면 헤더와 함께 생성
        if not self.filepath.exists():
            with open(self.filepath, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "일시", "종목코드", "구분", "수량", "단가", 
                    "총금액", "수익금", "수익률(%)", "비고"
                ])

    def log_trade(self, symbol, side, qty, price, pnl=0, pnl_pct=0.0, note=""):
        """
        매매 내역을 CSV 파일에 추가합니다.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_amount = int(qty * price)
        
        try:
            with open(self.filepath, mode='a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    now, symbol, side, qty, price, 
                    total_amount, int(pnl), round(pnl_pct * 100, 2), note
                ])
        except Exception as e:
            print(f"매매 이력 파일 기록 중 오류 발생: {e}")

# 싱글톤 인스턴스 제공
trade_logger = TradeHistoryLogger()
