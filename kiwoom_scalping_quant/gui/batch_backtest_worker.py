import os
import pandas as pd
import datetime
import logging
import asyncio
import time
import torch
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from typing import List, Dict, Any

class BatchBacktestWorker(QThread):
    """
    다중 모델 x 다중 종목 교차 백테스트를 수행하는 백그라운드 워커.
    """
    # UI 업데이트를 위한 시그널
    # (현재 진행률, 상태 메시지, 현재 PnL)
    sig_progress = pyqtSignal(int, int, float) 
    sig_status = pyqtSignal(str)
    sig_finished = pyqtSignal(dict)    # 결과 요약 정보
    sig_error = pyqtSignal(str)       # 에러 메시지

    def __init__(self, model_paths: List[str], symbols: List[Dict[str, str]], 
                 start_date: str, end_date: str, backtest_engine, influx_client, config_manager):
        super().__init__()
        self.model_paths = model_paths
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.engine = backtest_engine
        self.influx_client = influx_client
        self.config_manager = config_manager
        self.logger = logging.getLogger("BatchBacktestWorker")
        self.is_running = True

    def run(self):
        import traceback
        from influxdb_client import InfluxDBClient  # 동기 클라이언트 임포트
        
        # 설정값 미리 추출 (스레드 안전)
        url = self.config_manager.get("INFLUX_URL", "http://localhost:8086")
        token = self.config_manager.get("INFLUX_TOKEN", "")
        org = self.config_manager.get("INFLUX_ORG", "my-trade")
        bucket = self.config_manager.get("influx_bucket", "stock_data")

        try:
            # 1. 스레드 전용 비동기 루프 (백테스트 엔진 실행용)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # [핵심] 조희는 동기(Sync) 클라이언트를 사용하여 루프 충돌 원천 차단
            sync_client = InfluxDBClient(url=url, token=token, org=org, timeout=300000)
            query_api = sync_client.query_api()

            # 워커 전용 백테스트 엔진
            from core.backtester import BacktestEngine
            from models.agent import TradingAgentWrapper
            from env.trading_env import ScalpingTradingEnv
            from core.backtester import KPICalculator
            
            # [수정] 실행 시점의 임계값 고정 (루프 도중 전역 설정이 바뀌어도 일관성 유지)
            fixed_buy_th = float(self.config_manager.get("ai_buy_threshold", 0.4))
            fixed_sell_th = float(self.config_manager.get("ai_sell_threshold", 0.4))
            self.logger.info(f"일괄 백테스트 시작 임계값: 매수 {fixed_buy_th}, 매도 {fixed_sell_th}")

            worker_engine = BacktestEngine(None, self.config_manager)

            results = []
            total_models = len(self.model_paths)
            total_symbols = len(self.symbols)
            total_tasks = total_models * total_symbols
            completed_tasks = 0

            # --- [외부 루프: 모델] ---
            for m_idx, model_path in enumerate(self.model_paths):
                if not self.is_running: break
                
                model_name = os.path.basename(model_path)
                self.sig_status.emit(f"모델 로딩 중 ({m_idx+1}/{total_models}): {model_name}")
                
                try:
                    model_dim = TradingAgentWrapper.get_model_dimension(model_path)
                    detected_mode = "advanced" if model_dim >= 200 else "basic"
                    all_symbols_list = [s.get("code") for s in self.symbols]
                except Exception as e:
                    self.logger.error(f"모델 정보 읽기 실패 ({model_name}): {e}")
                    completed_tasks += total_symbols
                    continue

                # --- [내부 루프: 종목] ---
                for s_idx, stock in enumerate(self.symbols):
                    if not self.is_running: break
                    
                    symbol = stock.get("code")
                    name = stock.get("name", symbol)
                    clean_symbol = symbol.split('_')[0].strip()
                    
                    self.sig_status.emit(f"[{model_name}] {name} 테스트 중 ({s_idx+1}/{total_symbols})...")
                    self.sig_progress.emit(completed_tasks, total_tasks, 0.0)

                    # --- [동기 방식 데이터 로드] ---
                    try:
                        import datetime
                        s_str = self.start_date.replace("-", "").replace("/", "").strip()
                        e_str = self.end_date.replace("-", "").replace("/", "").strip()
                        s_dt = datetime.datetime.strptime(s_str, "%Y%m%d")
                        e_dt = datetime.datetime.strptime(e_str, "%Y%m%d") + datetime.timedelta(days=1)
                        
                        start_iso = s_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                        stop_iso = e_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

                        query = f'''
                            from(bucket: "{bucket}")
                            |> range(start: {start_iso}, stop: {stop_iso})
                            |> filter(fn: (r) => r["_measurement"] == "historical_data" or r["_measurement"] == "tick_data")
                            |> filter(fn: (r) => r["symbol"] == "{symbol}" or r["symbol"] == "{clean_symbol}")
                            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                            |> sort(columns: ["_time"], desc: false)
                        '''
                        
                        # [Sync Call] 비동기 루프를 타지 않으므로 안전함
                        tables = query_api.query(query, org=org)
                        data_list = []
                        for table in tables:
                            for record in table.records:
                                data_list.append({
                                    "timestamp": record.get_time(),
                                    "open": float(record.values.get("open", 0.0)),
                                    "high": float(record.values.get("high", 0.0)),
                                    "low": float(record.values.get("low", 0.0)),
                                    "price": float(record.values.get("price", 0.0)),
                                    "volume": float(record.values.get("volume", 0.0))
                                })
                    except Exception as fetch_e:
                        self.logger.error(f"데이터 조회 중 예외 발생: {fetch_e}\n{traceback.format_exc()}")
                        data_list = []

                    if not data_list:
                        self.logger.warning(f"데이터 없음: {symbol} ({self.start_date}~{self.end_date})")
                        completed_tasks += 1
                        self.sig_progress.emit(completed_tasks, total_tasks, 0.0)
                        continue

                    df = pd.DataFrame(data_list)
                    df['step'] = range(len(df))

                    # 환경 및 에이전트 설정
                    env_config = {
                        "symbol": symbol, "initial_balance": 10000000, "historical_data": data_list,
                        "mode": "backtest", "feature_mode": detected_mode,
                        "target_dim": model_dim, "all_symbols": all_symbols_list
                    }
                    
                    env = ScalpingTradingEnv(None, None, env_config)
                    agent = TradingAgentWrapper(env, {"seq_len": 10})
                    
                    try:
                        agent.load_weights(model_path)
                        # 백테스트 실행 (엔진은 내부 연산용이므로 loop 사용)
                        trades_df = loop.run_until_complete(worker_engine.run_backtest(
                            agent, env, df, 
                            buy_threshold=fixed_buy_th, 
                            sell_threshold=fixed_sell_th
                        ))
                        
                        buy_actions  = ['Buy40%', 'Buy60%']
                        sell_actions = ['Sell60%', 'Sell40%']
                        kpi = KPICalculator.calculate(trades_df, 10000000)
                        action_counts = trades_df['action'].value_counts(normalize=True) * 100

                        results.append({
                            "테스트 일시": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "모델명": model_name, "종목코드": symbol, "종목명": name,
                            "매수 임계값": fixed_buy_th,
                            "매도 임계값": fixed_sell_th,
                            "시작일": self.start_date, "종료일": self.end_date,
                            "점 거래횟수": len(trades_df[trades_df['action'].isin(buy_actions + sell_actions)]),
                            "승률 (%)": round(kpi.get("Win Rate", 0), 2),
                            "총수익률 (%)": round(kpi.get("Total Return", 0), 2),
                            "MDD (%)": round(kpi.get("MDD", 0), 2),
                            "Profit Factor": round(kpi.get("Profit Factor", 0), 3),
                            # 5-액션 비율
                            "Buy40% Ratio": round(action_counts.get("Buy40%",  0), 2),
                            "Buy60% Ratio": round(action_counts.get("Buy60%",  0), 2),
                            "Sell60% Ratio": round(action_counts.get("Sell60%", 0), 2),
                            "Sell40% Ratio": round(action_counts.get("Sell40%", 0), 2),
                            "Hold Ratio (%)": round(action_counts.get("Hold",   0), 2),
                            # 세부 커운트
                            "Buy40 Count":  kpi.get("Buy40_Count",  0),
                            "Buy60 Count":  kpi.get("Buy60_Count",  0),
                            "Sell60 Count": kpi.get("Sell60_Count", 0),
                            "Sell40 Count": kpi.get("Sell40_Count", 0),
                        })
                    except Exception as e:
                        self.logger.error(f"백테스트 실행 실패 ({model_name} - {symbol}): {e}")

                    completed_tasks += 1
                    # 진행률 갱신 시그널 송신
                    self.sig_progress.emit(completed_tasks, total_tasks, 0.0)

            # 2. 결과 저장 및 클라이언트 종료
            if results:
                csv_path = self.save_to_csv(results)
                self.sig_status.emit(f"모든 작업 완료. 결과 저장됨: {os.path.basename(csv_path)}")
                self.sig_finished.emit({"CSV_Path": csv_path, "Batch Count": len(results)})
                self.sig_progress.emit(total_tasks, total_tasks, 100.0)
            else:
                self.sig_error.emit("백테스트 결과가 비어있습니다. 데이터를 확인해 주세요.")

            sync_client.close()

        except Exception as e:
            self.logger.error(f"Batch Backtest Error: {e}\n{traceback.format_exc()}")
            self.sig_error.emit(f"일괄 백테스트 중 오류 발생: {str(e)}")

    def save_to_csv(self, results: List[Dict]) -> str:
        """결과 리스트를 DataFrame으로 변환 후 CSV 저장"""
        df = pd.DataFrame(results)
        
        # 저장 폴더 생성
        base_dir = "./backtest_results/batch"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_backtest_results_{timestamp}.csv"
        filepath = os.path.join(base_dir, filename)
        
        # 엑셀 한글 깨짐 방지를 위해 utf-8-sig 사용
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        return filepath

    def _predict_with_confidence(self, agent, obs, action_masks=None):
        """에이전트로부터 액션과 해당 액션의 확률(Confidence)을 추출 (SB3 최적화)"""
        try:
            # 1. TradingAgentWrapper인 경우 (우리가 만든 래퍼 클래스)
            if hasattr(agent, 'model') and hasattr(agent, 'predict'):
                result = agent.predict(obs, action_masks=action_masks, return_probs=True)
                if isinstance(result, tuple) and len(result) == 2:
                    action, probs = result
                    confidence = float(probs[action]) if hasattr(probs, '__len__') else float(probs)

                    # [5-액션] 동적 임계값 반영
                    buy_threshold = float(self.config_manager.get("ai_buy_threshold", 0.4))
                    sell_threshold = float(self.config_manager.get("ai_sell_threshold", 0.4))

                    if action in (1, 2) and confidence < buy_threshold:
                        action = 0
                    elif action in (3, 4) and confidence < sell_threshold:
                        action = 0
                    return action, confidence
                else:
                    return int(result), 1.0

            # 2. SB3 모델인 경우 직접 policy 활용
            if hasattr(agent, 'policy'):
                obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    if action_masks is not None and hasattr(agent.policy, "get_distribution"):
                        masks_tensor = torch.as_tensor(action_masks).unsqueeze(0).to(agent.device)
                        latent_pi, _, latent_sde = agent.policy._get_latent(obs_tensor)
                        distribution = agent.policy._get_action_dist_from_latent(latent_pi, latent_sde)
                        distribution.apply_masking(masks_tensor)
                        probs = distribution.distribution.probs.cpu().numpy()[0]
                    else:
                        dist = agent.policy.get_distribution(obs_tensor)
                        probs = dist.distribution.probs.cpu().numpy()[0]
                
                action = int(probs.argmax())
                confidence = float(probs[action])

                # [5-액션] 동적 임계값 반영
                buy_threshold = float(self.config_manager.get("ai_buy_threshold", 0.4))
                sell_threshold = float(self.config_manager.get("ai_sell_threshold", 0.4))

                if action in (1, 2) and confidence < buy_threshold:
                    action = 0
                elif action in (3, 4) and confidence < sell_threshold:
                    action = 0
                return action, confidence
            else:
                # 일반 객체인 경우 (0-d array 언패킹 방지)
                result = agent.predict(obs, action_masks=action_masks)
                if isinstance(result, tuple):
                    return int(result[0]), 1.0
                return int(result), 1.0
        except Exception:
            # 에러 시 기본 추론으로 폴백
            try:
                result = agent.predict(obs, action_masks=action_masks)
                action = result[0] if isinstance(result, tuple) else result
                return int(action), 0.5
            except:
                return 0, 0.0

    def stop(self):
        """작업 중지 플래그 설정"""
        self.is_running = False
