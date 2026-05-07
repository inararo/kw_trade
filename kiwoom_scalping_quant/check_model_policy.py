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

# ─────────────────────────────────────────────────────
# 5-액션 레이블 정의 (trading_env.py와 동기화)
# 0:Hold  1:Buy40%  2:Buy60%  3:Sell60%  4:Sell40%
# ─────────────────────────────────────────────────────
ACTION_LABELS = {
    0: ("🛑", "Hold    "),
    1: ("🟢", "Buy 40% "),
    2: ("💚", "Buy 60% "),
    3: ("🔴", "Sell 60%"),
    4: ("🟠", "Sell 40%"),
}
NUM_ACTIONS = 5

# Confidence 필터 임계값 (라이브 엔진과 동일)
BUY_THRESHOLD  = 0.6
SELL_THRESHOLD = 0.6


def compute_indicators(data: list) -> pd.DataFrame:
    """
    [보조지표 자동 계산] trading_env._compute_indicators()와 완전히 동일한 로직.
    입력 데이터(list of dict)에서 SMA_20 / SMA_60 / RSI_14를 계산합니다.
    """
    prices = pd.Series([float(d.get('price', 0)) for d in data], dtype=np.float64)

    sma20 = prices.rolling(window=20, min_periods=1).mean()
    sma60 = prices.rolling(window=60, min_periods=1).mean()

    # RSI-14 수동 계산 (pandas_ta 미설치 환경 대응)
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
    rs    = gain / (loss + 1e-9)
    rsi14 = 100.0 - (100.0 / (1.0 + rs))

    df = pd.DataFrame({'SMA_20': sma20, 'SMA_60': sma60, 'RSI_14': rsi14})
    df.ffill(inplace=True)
    df.fillna(0.0, inplace=True)
    return df


def get_indicator_obs(indicator_df: pd.DataFrame, idx: int, current_price: float) -> np.ndarray:
    """
    trading_env._get_indicator_obs()와 동일한 스케일링 로직.
    SMA_20/SMA_60: (현재가 - SMA) / 현재가  → clip(-1, 1)
    RSI_14       : RSI / 100               → 0 ~ 1
    """
    idx = min(idx, len(indicator_df) - 1)
    row = indicator_df.iloc[idx]
    p = current_price + 1e-9
    sma20_s = float(np.clip((current_price - row['SMA_20']) / p, -1.0, 1.0))
    sma60_s = float(np.clip((current_price - row['SMA_60']) / p, -1.0, 1.0))
    rsi14_s = float(row['RSI_14']) / 100.0
    return np.array([sma20_s, sma60_s, rsi14_s], dtype=np.float32)


def is_pullback(indicator_df: pd.DataFrame, idx: int) -> bool:
    """눌림목 조건: SMA_20 > SMA_60 AND RSI_14 < 40"""
    idx = min(idx, len(indicator_df) - 1)
    row = indicator_df.iloc[idx]
    return (row['SMA_20'] > row['SMA_60']) and (row['RSI_14'] < 40.0)


async def run_diagnosis(symbol: str = "001440"):
    """
    지정된 모델의 정책 성향을 과거 데이터를 통해 진단합니다.
    - 5-액션 대응
    - 보조지표(SMA_20, SMA_60, RSI_14) 전처리 포함
    - Confidence 필터 전후 결과 분리 출력
    """
    container = Container()
    config_manager = container.config_manager()
    fetcher = container.historical_fetcher()

    # ─── 1. 모델 로드 ───────────────────────────────
    model_path = config_manager.get("active_model_path")
    if not model_path:
        print("❌ active_model_path가 설정되지 않았습니다.")
        return

    if model_path.startswith("./"):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path[2:])

    if not os.path.exists(model_path):
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

    model_action_n = model.action_space.n
    print(f"✅ 모델 로드 완료 | Action Space: Discrete({model_action_n}) | Obs Shape: {model.observation_space.shape}")

    if model_action_n != NUM_ACTIONS:
        print(f"⚠️  경고: 모델의 액션 수({model_action_n})가 스크립트 기준({NUM_ACTIONS})과 다릅니다.")
        print(f"   마스크를 {model_action_n}개로 조정합니다.")

    # ─── 2. 데이터 페칭 ─────────────────────────────
    print(f"\n📂 [{symbol}] 과거 데이터 페칭 시작...")
    token = config_manager.get("KIWOOM_ACCESS_TOKEN") or os.getenv("KIWOOM_ACCESS_TOKEN")
    if not token:
        print("⚠️  KIWOOM_ACCESS_TOKEN 없음. 데이터 페칭이 실패할 수 있습니다.")

    today_str = datetime.now().strftime("%Y%m%d")
    data_result = await fetcher.fetch_historical_data(symbol, today_str, token, max_pages=5)

    from returns.pipeline import is_successful
    if not is_successful(data_result):
        error_msg = data_result.failure()
        print(f"❌ 데이터 페칭 실패: {error_msg}")
        if "Return Code 3" in str(error_msg):
            print("💡 조치: 키움 API 토큰이 만료되었습니다. main.py를 실행하여 갱신하세요.")
        return

    data = data_result.unwrap()
    if not isinstance(data, list):
        if hasattr(data, "_inner_value") and isinstance(data._inner_value, list):
            data = data._inner_value
        else:
            print(f"❌ 데이터 형식이 올바르지 않습니다: {type(data)}")
            return

    if len(data) < 60:
        print(f"❌ 데이터가 너무 부족합니다. (현재 {len(data)}개, 최소 60개 필요)")
        return

    sorted_data = sorted(data, key=lambda x: x["timestamp"])
    print(f"✅ 데이터 로드 완료: {len(sorted_data)}개 캔들")

    # ─── 3. 보조지표 전처리 ─────────────────────────
    # trading_env._compute_indicators()와 완전히 동일한 로직 적용
    print("📐 보조지표(SMA_20 / SMA_60 / RSI_14) 계산 중...")
    indicator_df = compute_indicators(sorted_data)
    print(f"✅ 보조지표 계산 완료 (총 {len(indicator_df)}행, NaN: {indicator_df.isna().sum().sum()}개)")

    # ─── 4. 추론 루프 ───────────────────────────────
    normalizer = OnlineRollingNormalizer(window_size=200)
    seq_len    = config_manager.get("seq_len", 10)
    target_dim = model.observation_space.shape[0]

    # 통계 카운터
    total_steps   = 0
    # 순수 모델 판단 (필터 미적용)
    raw_counts    = {i: 0 for i in range(NUM_ACTIONS)}
    # Confidence 필터 적용 후 판단
    filtered_counts = {i: 0 for i in range(NUM_ACTIONS)}
    # 필터로 인해 Hold로 변경된 횟수
    filter_overrides = 0

    # 신뢰도 기록
    buy_confidences  = []   # action in (1, 2)
    sell_confidences = []   # action in (3, 4)

    minute_buffer = deque(maxlen=100)

    print(f"\n🧠 모델 정책 분석 중... (Confidence 필터: Buy≥{BUY_THRESHOLD}, Sell≥{SELL_THRESHOLD})\n")

    for i, candle in enumerate(sorted_data):
        minute_buffer.append(candle)
        if len(minute_buffer) < 30:
            continue

        # Feature Engineering
        try:
            features = AdvancedFeatureEngineer.process_historical_data(list(minute_buffer))
        except Exception:
            continue

        if len(features) < seq_len:
            continue

        obs_1d = features[-seq_len:].flatten()

        # 정규화
        obs_normalized = normalizer.normalize(obs_1d)

        # 보조지표 obs 추가 (trading_env._get_observation()과 동일한 순서)
        current_price = float(candle.get('price', 0))
        indicator_obs = get_indicator_obs(indicator_df, i, current_price)
        obs_with_indicator = np.concatenate([obs_normalized, indicator_obs])

        # 패딩 (Target Dim 동기화)
        if len(obs_with_indicator) < target_dim:
            obs_with_indicator = np.pad(
                obs_with_indicator, (0, target_dim - len(obs_with_indicator)), 'constant'
            )
        elif len(obs_with_indicator) > target_dim:
            obs_with_indicator = obs_with_indicator[:target_dim]

        state_input = np.expand_dims(obs_with_indicator, axis=0)

        # 마스킹: 모두 True (모델의 순수 의도 파악)
        action_masks = np.array([True] * model_action_n)

        # ─ 액션 예측
        raw_action, _ = model.predict(state_input, action_masks=action_masks, deterministic=True)
        if isinstance(raw_action, np.ndarray):
            raw_action = int(raw_action[0])

        # ─ 확률(Confidence) 추출
        obs_tensor, _ = model.policy.obs_to_tensor(state_input)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor)
            if hasattr(dist.distribution, 'probs'):
                probs = dist.distribution.probs.cpu().numpy()[0]
            else:
                logits = dist.distribution.logits
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # probs 길이를 NUM_ACTIONS에 맞춰 안전하게 조정
        if len(probs) < NUM_ACTIONS:
            probs = np.pad(probs, (0, NUM_ACTIONS - len(probs)))

        total_steps += 1
        raw_counts[raw_action] += 1

        # ─ Confidence 필터 적용
        filtered_action = raw_action
        if raw_action in (1, 2):  # 매수 계열
            conf = float(probs[raw_action])
            buy_confidences.append(conf)
            if conf < BUY_THRESHOLD:
                filtered_action = 0
                filter_overrides += 1
        elif raw_action in (3, 4):  # 매도 계열
            conf = float(probs[raw_action])
            sell_confidences.append(conf)
            if conf < SELL_THRESHOLD:
                filtered_action = 0
                filter_overrides += 1

        filtered_counts[filtered_action] += 1

        # ─ 상세 로그 출력 (매수/매도 신호 시)
        ts_str = str(candle.get('timestamp', ''))[:16]
        p_str  = "  ".join([f"{ACTION_LABELS.get(j, ('?','?'))[1].strip()}: {probs[j]:.4f}"
                             for j in range(min(NUM_ACTIONS, len(probs)))])

        if raw_action in (1, 2):
            conf = float(probs[raw_action])
            override_mark = " ⚡→Hold" if filtered_action == 0 else ""
            pullback_mark = " 🎯눌림목!" if is_pullback(indicator_df, i) else ""
            print(
                f"  {ACTION_LABELS[raw_action][0]} {ts_str} | "
                f"{ACTION_LABELS[raw_action][1].strip()} | "
                f"신뢰도: {conf:.4f}{override_mark}{pullback_mark}\n"
                f"     Probs → [{p_str}]"
            )
        elif raw_action in (3, 4):
            conf = float(probs[raw_action])
            override_mark = " ⚡→Hold" if filtered_action == 0 else ""
            print(
                f"  {ACTION_LABELS[raw_action][0]} {ts_str} | "
                f"{ACTION_LABELS[raw_action][1].strip()} | "
                f"신뢰도: {conf:.4f}{override_mark}\n"
                f"     Probs → [{p_str}]"
            )

    # ─── 5. 결과 리포트 ─────────────────────────────
    sep = "─" * 50
    print(f"\n{'📊 [모델 정책 진단 결과]':^50}")
    print(sep)
    print(f"  테스트 종목   : {symbol}")
    print(f"  총 추론 스텝  : {total_steps}")
    print(f"  Obs Dim       : {target_dim}")
    print(f"  Action Space  : Discrete({model_action_n})")
    print(sep)

    if total_steps > 0:
        print(f"\n  ▶ 순수 모델 판단 (Confidence 필터 미적용)")
        for act_id in range(NUM_ACTIONS):
            cnt = raw_counts[act_id]
            pct = cnt / total_steps * 100
            icon, label = ACTION_LABELS.get(act_id, ("?", f"Action{act_id}"))
            print(f"    {icon} {label} ({act_id}): {cnt:4d}회  ({pct:5.1f}%)")

        print(f"\n  ▶ Confidence 필터 적용 후 판단 (Buy≥{BUY_THRESHOLD}, Sell≥{SELL_THRESHOLD})")
        for act_id in range(NUM_ACTIONS):
            cnt = filtered_counts[act_id]
            pct = cnt / total_steps * 100
            icon, label = ACTION_LABELS.get(act_id, ("?", f"Action{act_id}"))
            print(f"    {icon} {label} ({act_id}): {cnt:4d}회  ({pct:5.1f}%)")

        print(f"\n  ▶ 필터로 Hold 전환된 횟수: {filter_overrides}회 "
              f"({filter_overrides/total_steps*100:.1f}%)")

        avg_buy_conf  = np.mean(buy_confidences)  if buy_confidences  else 0.0
        avg_sell_conf = np.mean(sell_confidences) if sell_confidences else 0.0
        max_buy_conf  = np.max(buy_confidences)   if buy_confidences  else 0.0
        print(f"\n  ▶ 매수 신뢰도 — 평균: {avg_buy_conf:.4f}  최대: {max_buy_conf:.4f} "
              f"(n={len(buy_confidences)})")
        print(f"  ▶ 매도 신뢰도 — 평균: {avg_sell_conf:.4f} "
              f"(n={len(sell_confidences)})")

        # ─ 진단 평점
        raw_buy_pct  = (raw_counts[1] + raw_counts[2]) / total_steps * 100
        raw_sell_pct = (raw_counts[3] + raw_counts[4]) / total_steps * 100
        print(f"\n  ▶ 매수 편향도: {raw_buy_pct:.1f}%  |  매도 편향도: {raw_sell_pct:.1f}%")

        print(f"\n  {'[진단 결론]':^46}")
        if raw_buy_pct > 80:
            print("  ⚠️  정책 붕괴 위험! 모델이 지나치게 매수 편향적입니다.")
            print("       → 학습률 조정 또는 Hold 페널티 완화를 검토하세요.")
        elif raw_buy_pct + raw_sell_pct < 5:
            print("  ⚠️  과소 추론 위험! 모델이 지나치게 소극적(Hold 편향)입니다.")
            print("       → 거래 관련 보상 배율(reward_multiplier)을 높이세요.")
        elif avg_buy_conf > 0.9:
            print("  ⚠️  신뢰도 포화(Saturation) 의심! 매수 확률이 0.9999에 수렴합니다.")
            print("       → 정규화(Normalizer) 또는 클리핑 로직을 점검하세요.")
        else:
            print("  ✅  모델 정책이 비교적 균형 잡혀 있습니다.")
    else:
        print("  ❌ 유효한 추론 스텝이 없습니다. 데이터 양을 확인하세요.")

    print(sep + "\n")


if __name__ == "__main__":
    target_sym = sys.argv[1] if len(sys.argv) > 1 else "001440"
    asyncio.run(run_diagnosis(target_sym))
