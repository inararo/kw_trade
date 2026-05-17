import asyncio
import logging
import os
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal
from typing import List, Dict, Any
from sb3_contrib import MaskablePPO

class MultiThresholdBatchWorker(QThread):
    """
    7가지 임계값 조합을 독립된 스레드에서 순차적으로 실행하는 워커.
    메인 UI 루프와의 충돌을 방지하기 위해 자체 이벤트 루프를 가집니다.
    """
    sig_progress = pyqtSignal(int, int, float) # step, total, dummy
    sig_status = pyqtSignal(str)
    sig_finished = pyqtSignal(dict)
    sig_error = pyqtSignal(str)

    def __init__(self, config_manager, engine, historical_fetcher, universe_manager, token_manager, 
                 influx_client, model_path, start_date, end_date):
        super().__init__()
        # [복원] 스레드 고유의 루프를 사용하기 위해 main_loop 제거
            
        self.config_manager = config_manager
        self.engine = engine
        self.historical_fetcher = historical_fetcher
        self.universe_manager = universe_manager
        self.token_manager = token_manager
        self.influx_client = influx_client
        self.model_path = model_path
        self.start_date = start_date
        self.end_date = end_date
        self.is_running = True
        self.logger = logging.getLogger("MultiThWorker")

    def stop(self):
        self.is_running = False
        if hasattr(self, 'engine') and self.engine:
            self.engine.stop()

    def run(self):
        """별도 스레드에서 자체 이벤트 루프를 생성하여 실행"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._execute_batch())
        except Exception as e:
            self.logger.error(f"워커 실행 중 치명적 에러: {e}")
            self.sig_error.emit(str(e))
        finally:
            self.loop.close()

    async def _execute_batch(self):
        # 1. 기존 설정 백업
        orig_buy_th = self.config_manager.get("ai_buy_threshold", 0.4)
        orig_sell_th = self.config_manager.get("ai_sell_threshold", 0.4)

        threshold_pairs = [
            (0.70, 0.50), (0.75, 0.45), (0.75, 0.50),
            (0.80, 0.50), (0.80, 0.55), (0.85, 0.50), (0.85, 0.55)
        ]

        try:
            self.logger.info("🚀 임계값 순회 배치 작업 시작")
            access_token = self.token_manager.get_token()
            if not access_token:
                self.logger.info("토큰 갱신 시도 중...")
                await self.token_manager.refresh_token()
                access_token = self.token_manager.get_token()
            
            if not access_token:
                self.sig_error.emit("API 토큰 발급 실패")
                return

            self.sig_status.emit("거래량 상위 30개 종목 리스트 가져오는 중...")
            res = await self.universe_manager.fetch_top_30_volume_symbols(access_token)
            
            # [수정] Result 객체 성공 여부 확인 및 언래핑
            from returns.pipeline import is_successful
            if is_successful(res):
                top_30_list = res.unwrap()
                if hasattr(top_30_list, "_inner_value"):
                    top_30_list = top_30_list._inner_value
            else:
                err_val = res.failure() if hasattr(res, "failure") else res
                self.logger.error(f"Top 30 종목 추출 실패: {err_val}")
                self.sig_error.emit(f"Top 30 종목 추출 실패: {err_val}")
                return
            
            if not top_30_list:
                self.sig_error.emit("Top 30 종목 리스트가 비어있습니다.")
                return

            symbols = [s["code"] for s in top_30_list]
            total_pairs = len(threshold_pairs)
            self.logger.info(f"분석 대상 종목({len(symbols)}개): {symbols}")

            # [최적화] 모든 종목의 데이터를 미리 한 번만 수집 (캐싱)
            self.sig_status.emit(f"백테스트용 데이터 사전 수집 중 (30개 종목)...")
            cached_data = {} # {symbol: (env_config, df)}
            
            # 통합 진행률 계산을 위한 상수
            TOTAL_CACHING = 30
            TOTAL_SIMS = total_pairs * 30
            GLOBAL_TOTAL = TOTAL_CACHING + TOTAL_SIMS
            
            from env.trading_env import ScalpingTradingEnv
            from models.agent import TradingAgentWrapper
            model_dim = TradingAgentWrapper.get_model_dimension(self.model_path)
            detected_mode = "advanced" if model_dim >= 200 else "basic"
            all_symbols_list = [s.get("code") for s in self.config_manager.get_symbols()]

            # [복원] 모델 재사용 제거 (안정성 문제로 매번 로드 방식으로 회귀)
            # shared_model = MaskablePPO.load(self.model_path, device="cpu")

            for i, sym in enumerate(symbols):
                if not self.is_running: break
                self.sig_status.emit(f"데이터 준비 중 [{i+1}/30]: {sym}")
                # 글로벌 진행률 송신 (0~30/240)
                self.sig_progress.emit(i + 1, GLOBAL_TOTAL, 0.0)
                
                # [수정] "여러임계값 순회" 기능은 오프라인 모드라도 네트워크 접근을 허용 (사용자 요청)
                data = None
                
                # 1. API 수집 시도 (종료일 기준 하루치)
                self.logger.info(f"[{sym}] 데이터 수집 시도 (API)...")
                stop_ts = f"{self.end_date[:4]}-{self.end_date[4:6]}-{self.end_date[6:8]} 08:00:00"
                result = await self.historical_fetcher.fetch_historical_data(
                    sym, self.end_date, access_token, stop_timestamp=stop_ts, max_pages=5
                )
                
                if is_successful(result):
                    inner_res = result.unwrap()
                    data = inner_res._inner_value if hasattr(inner_res, "_inner_value") else inner_res
                
                # 2. API 실패 시 DB 조회 시도
                if not data or len(data) == 0:
                    self.logger.info(f"[{sym}] API 데이터 없음, DB에서 조회를 시도합니다. (조회일: {self.end_date})")
                    data = await self.influx_client.fetch_data_by_range(sym, self.end_date, self.end_date)
                    if data:
                        self.logger.info(f"[{sym}] DB에서 {len(data)}건의 데이터를 찾았습니다.")
                    else:
                        self.logger.warning(f"[{sym}] DB에도 데이터가 없습니다.")
                
                if data and len(data) > 0:
                    df = pd.DataFrame(data)
                    df['step'] = range(len(df))
                    env_config = {
                        "symbol": sym, "initial_balance": 10000000, "historical_data": data,
                        "mode": "backtest", "feature_mode": detected_mode, "target_dim": model_dim,
                        "all_symbols": all_symbols_list
                    }
                    cached_data[sym] = (env_config, df)
                else:
                    self.logger.warning(f"종목 {sym}: 유효한 데이터를 찾을 수 없습니다 (API/DB 모두 실패).")

            if not cached_data:
                self.logger.error("캐시된 데이터가 하나도 없습니다. 작업을 중단합니다.")
                self.sig_error.emit("백테스트를 위한 데이터를 하나도 수집하지 못했습니다.")
                return

            self.logger.info(f"✅ 데이터 준비 완료 ({len(cached_data)}개 종목). 임계값 순회 시작...")

            # [최적화] 수집된 캐시 데이터를 사용하여 7가지 조합 순회
            for idx, (b_th, s_th) in enumerate(threshold_pairs):
                if not self.is_running: break
                
                current_combo_name = f"b{int(b_th*100)}s{int(s_th*100)}"
                self.logger.info(f"▶️ 조합 {idx+1}/{total_pairs} 실행 중: {current_combo_name}")

                # 시뮬레이션 단계 진행률 콜백 (30~240/240)
                def sim_progress_cb(cur, tot, msg):
                    global_step = TOTAL_CACHING + (idx * 30) + cur
                    self.sig_status.emit(f"[{current_combo_name}] {msg}")
                    self.sig_progress.emit(global_step, GLOBAL_TOTAL, 0.0)

                # [복원] 공유 모델 대신 모델 경로만 전달하여 각 시뮬레이션에서 로드
                results = await self._run_cached_batch(cached_data,
                                                     progress_cb=sim_progress_cb,
                                                     buy_th=b_th, sell_th=s_th)
                self.logger.info(f"   └ 조합 {idx+1} 완료. {len(results)}개 결과 저장 시도...")
                # 결과 저장 시에도 해당 루프의 임계값 명시
                self._save_results(results, buy_th=b_th, sell_th=s_th)

            self.logger.info("🏁 모든 임계값 순회 작업이 종료되었습니다.")
            self.sig_progress.emit(GLOBAL_TOTAL, GLOBAL_TOTAL, 0.0) # 최종 100%
            self.sig_finished.emit({"Batch Count": total_pairs, "Status": "Success"})

        except Exception as e:
            self.logger.error(f"❌ 순회 배치 중 치명적 에러: {e}", exc_info=True)
            self.sig_error.emit(str(e))
        # finally 블록의 설정 복구 로직 삭제 (이제 필요 없음)

    async def _run_cached_batch(self, cached_data, progress_cb=None, buy_th=None, sell_th=None):
        """이미 수집된 데이터를 사용하여 배치를 실행 (API 호출 없음)"""
        from env.trading_env import ScalpingTradingEnv
        from models.agent import TradingAgentWrapper
        
        symbols = list(cached_data.keys())
        
        async def env_builder(sym, start, end):
            # 캐시된 데이터 반환
            env_config, df = cached_data.get(sym, (None, None))
            if env_config is None: return None, None
            return ScalpingTradingEnv(None, None, env_config), df

        def agent_builder(env):
            # [복원] 매번 가중치를 로드 (CPU 강제)
            agent = TradingAgentWrapper(env, {"seq_len": 10}, device="cpu")
            agent.load_weights(self.model_path)
            return agent

        return await self.engine.run_automation_batch(
            agent_builder, env_builder, symbols, self.start_date, self.end_date,
            progress_cb=progress_cb,
            buy_threshold=buy_th,
            sell_threshold=sell_th
        )

    def _save_results(self, results, buy_th=None, sell_th=None):
        """결과 CSV 저장 및 임계값/평균 정보 주입"""
        if not results: return
        
        # [수정] 주입된 임계값이 있으면 사용, 없으면 설정에서 읽음
        b_th = buy_th if buy_th is not None else self.config_manager.get("ai_buy_threshold", 0.4)
        s_th = sell_th if sell_th is not None else self.config_manager.get("ai_sell_threshold", 0.4)
        
        for r in results:
            r["Buy Threshold"] = b_th
            r["Sell Threshold"] = s_th
        
        # [추가] 평균값 계산 및 최상단 삽입
        import pandas as pd
        df_raw = pd.DataFrame(results)
        
        # 숫자 컬럼만 추출하여 평균 계산 (전체 평균 및 거래발생 종목 평균)
        numeric_cols = ["Total Return (%)", "Win Rate (%)", "MDD (%)", "Profit Factor", "Total Trades"]
        avg_values = {}
        active_df = df_raw[df_raw["Total Trades"] > 0] if "Total Trades" in df_raw.columns else df_raw.iloc[0:0]
        for col in numeric_cols:
            if col in df_raw.columns:
                avg_all = round(df_raw[col].mean(), 2)
                if not active_df.empty:
                    avg_active = round(active_df[col].mean(), 2)
                    avg_values[col] = f"{avg_all} ({avg_active})"
                else:
                    avg_values[col] = f"{avg_all} (0.0)"
        
        # 평균 행 생성
        avg_row = {
            "Symbol": "[AVERAGE]",
            "Start Date": results[0].get("Start Date", ""),
            "End Date": results[0].get("End Date", ""),
            "Buy Threshold": b_th,
            "Sell Threshold": s_th
        }
        avg_row.update(avg_values)
        
        # 평균 행을 맨 앞으로 하여 데이터프레임 재구성
        df_final = pd.concat([pd.DataFrame([avg_row]), df_raw], ignore_index=True)

        results_dir = os.path.abspath("./backtest_results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        model_name = os.path.basename(self.model_path).replace('.zip', '')
        th_suffix = f"_b{int(b_th*100)}s{int(s_th*100)}"
        
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"auto_bt_{model_name}_{date_str}{th_suffix}.csv"
        filepath = os.path.join(results_dir, filename)
        
        df_final.to_csv(filepath, index=False, encoding='utf-8-sig')
        self.logger.info(f"✅ 배치 결과 저장 완료(평균 포함): {filepath}")
        self.sig_status.emit(f"결과 저장됨: {filename}")
