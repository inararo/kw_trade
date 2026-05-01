import sys
import os
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
import torch

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.container import Container
from core.feature_engineer import AdvancedFeatureEngineer
from models.lstm_extractor import OnlineRollingNormalizer
from sb3_contrib import MaskablePPO

async def run_diagnosis(symbol="001440"):
    """
    지정된 모델의 정책 성향을 과거 데이터를 통해 진단합니다.
    """
    container = Container()
    config_manager = container.config_manager()
    fetcher = container.historical_fetcher()
    
    # 1. 모델 로드
    model_path = config_manager.get("active_model_path")
    if not model_path:
        print("❌ active_model_path가 설정되지 않았습니다.")
        return

    # 상대 경로 처리
    if model_path.startswith("./"):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path[2:])

    if not os.path.exists(model_path):
        # .zip 확장자 자동 추가 시도
        if os.path.exists(model_path + ".zip"):
            model_path += ".zip"
        else:
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
            return

    print(f"📡 모델 로드 중: {model_path}")
    try:
        model = MaskablePPO.load(model_path)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 2. 데이터 페칭 (약 500분치)
    print(f"📂 {symbol} 과거 데이터 페칭 시작...")
    # config.yaml 또는 .env에서 토큰 가져오기
    token = config_manager.get("KIWOOM_ACCESS_TOKEN") or os.getenv("KIWOOM_ACCESS_TOKEN")
    
    if not token:
        print("⚠️ KIWOOM_ACCESS_TOKEN이 없습니다. 데이터 페칭이 불가능할 수 있습니다.")
    
    today_str = datetime.now().strftime("%Y%m%d")
    data_result = await fetcher.fetch_historical_data(symbol, today_str, token, max_pages=1)
    
    from returns.pipeline import is_successful
    
    # 1. Result 객체 성공 여부 체크
    if not is_successful(data_result):
        error_msg = data_result.failure()
        print(f"❌ 데이터 페칭 실패: {error_msg}")
        if "Return Code 3" in str(error_msg):
            print("💡 조치: 키움 API 토큰이 만료되었습니다. main.py를 실행하여 토큰을 갱신하세요.")
        return

    # 2. 안전하게 데이터 추출 (Success인 경우에만 실행됨)
    data = data_result.unwrap()
    
    # [방어] 리스트 형태가 아닌 경우 처리
    if not isinstance(data, list):
        if hasattr(data, "_inner_value") and isinstance(data._inner_value, list):
            data = data._inner_value
        else:
            print(f"❌ 데이터 형식이 올바르지 않습니다: {type(data)}")
            return

    # 3. 데이터 길이 체크
    if len(data) < 50:
        print(f"❌ 데이터가 너무 부족합니다. (현재 {len(data)}개)")
        return

    sorted_data = sorted(data, key=lambda x: x["timestamp"])
    print(f"✅ 데이터 로드 완료: {len(sorted_data)}개 캔들")
    
    # 3. 정규화기 및 피처 엔지니어 초기화
    normalizer = OnlineRollingNormalizer(window_size=200)
    seq_len = config_manager.get("seq_len", 10)
    
    # 통계용 변수
    total_steps = 0
    counts = {0: 0, 1: 0, 2: 0}
    buy_confidences = []
    
    # 4. 시뮬레이션 루프 (LiveTradingEngine._run_inference 로직 재현)
    minute_buffer = deque(maxlen=100)
    
    print("🧠 모델 정책 분석 중...")
    for i in range(len(sorted_data)):
        minute_buffer.append(sorted_data[i])
        
        # 최소 데이터 확보 (Feature Engineering용)
        if len(minute_buffer) < 30:
            continue
            
        # 피처 엔지니어링
        try:
            features = AdvancedFeatureEngineer.process_historical_data(list(minute_buffer))
        except Exception as e:
            continue

        if len(features) < seq_len:
            continue
            
        obs_1d = features[-seq_len:].flatten()
        
        # 정규화 (OnlineRollingNormalizer 업데이트 및 정규화)
        obs_normalized = normalizer.normalize(obs_1d)
        
        # 패딩 (Target Dim 동기화)
        target_dim = model.observation_space.shape[0]
        if len(obs_normalized) < target_dim:
            obs_normalized = np.pad(obs_normalized, (0, target_dim - len(obs_normalized)), 'constant')
            
        # 추론 준비 (Batch 차원 추가)
        state_input = np.expand_dims(obs_normalized, axis=0)
        
        # 정책 성향을 보기 위해 마스킹은 모두 True로 설정 (제약 없는 모델의 의도 파악)
        action_masks = np.array([True, True, True])
        
        # 1) 액션 예측
        action, _ = model.predict(state_input, action_masks=action_masks, deterministic=True)
        if isinstance(action, np.ndarray): action = int(action[0])
        
        # 2) 확률(Confidence) 추출 (Raw Probabilities)
        obs_tensor, _ = model.policy.obs_to_tensor(state_input)
        with torch.no_grad():
            distribution = model.policy.get_distribution(obs_tensor)
            # Discrete Action Space의 경우 probs를 직접 추출
            if hasattr(distribution.distribution, 'probs'):
                probs = distribution.distribution.probs.cpu().numpy()[0]
            else:
                # Softmax 적용 (확률 분포가 다를 경우를 대비한 폴백)
                logits = distribution.distribution.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
        hold_prob, buy_prob, sell_prob = probs[0], probs[1], probs[2]
            
        # 통계 누적
        total_steps += 1
        counts[action] += 1
        
        # [신규] 매수 신호 시 상세 로깅 (0.9999 포화 여부 확인용)
        if action == 1: # Buy
            buy_confidences.append(buy_prob)
            # if buy_prob > 0.9:
            print(f"  [!] {sorted_data[i]['timestamp']} | Buy 신뢰도: {buy_prob:.6f} | Probs: [H:{hold_prob:.4f}, B:{buy_prob:.4f}, S:{sell_prob:.4f}]")
        elif action == 2: # Sell
            print(f"  [-] {sorted_data[i]['timestamp']} | Sell 신뢰도: {sell_prob:.6f} | Probs: [H:{hold_prob:.4f}, B:{buy_prob:.4f}, S:{sell_prob:.4f}]")
            
    # 5. 결과 리포트 출력
    print("\n" + "📊 [모델 정책 진단 결과]".center(40))
    print("-" * 40)
    print(f"- 테스트 종목: {symbol}")
    print(f"- 총 테스트 스텝 수: {total_steps}")
    
    if total_steps > 0:
        # 비율 계산
        p_hold = (counts[0] / total_steps) * 100
        p_buy  = (counts[1] / total_steps) * 100
        p_sell = (counts[2] / total_steps) * 100
        
        print(f"- 🛑 Hold (0) 선택 횟수: {counts[0]} ({p_hold:.1f}%)")
        print(f"- 🟢 Buy  (1) 선택 횟수: {counts[1]} ({p_buy:.1f}%)")
        print(f"- 🔴 Sell (2) 선택 횟수: {counts[2]} ({p_sell:.1f}%)")
        
        avg_buy_conf = np.mean(buy_confidences) if buy_confidences else 0.0
        print(f"- 🧠 Buy 평균 신뢰도: {avg_buy_conf:.4f}")
        
        # 진단 평점
        if p_buy > 80:
            print("\n⚠️ [진단] 정책 붕괴 위험! 모델이 지나치게 매수 편향적입니다.")
        elif p_buy < 5:
            print("\n⚠️ [진단] 과소 추론 위험! 모델이 지나치게 소극적입니다.")
        else:
            print("\n✅ [진단] 모델 정책이 비교적 균형 잡혀 있습니다.")
    else:
        print("❌ 유효한 추론 스텝이 없습니다. 데이터 양을 확인하세요.")
    print("-" * 40 + "\n")

if __name__ == "__main__":
    # 인자로 심볼을 받을 수 있게 처리
    target_sym = sys.argv[1] if len(sys.argv) > 1 else "001440"
    asyncio.run(run_diagnosis(target_sym))
