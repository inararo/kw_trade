"""
check_model_policy.py
=====================
모델 정책 진단 스크립트.

[핵심 설계 원칙]
 - ScalpingTradingEnv를 직접 인스턴스화하여 env.reset() / env.step()이
   반환하는 obs를 그대로 model.predict()에 전달합니다.
 - 수동 feature 추출·concatenate·정규화 코드가 전혀 없으므로
   훈련 환경과 100% 동일한 obs 보장이 구조적으로 달성됩니다.

실행:
    python check_model_policy.py [종목코드]
    python check_model_policy.py 010170
"""

import sys
import os
import json
import asyncio
import numpy as np
import torch
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.container import Container
from sb3_contrib import MaskablePPO

# ─── 5-액션 정의 (trading_env.py 동기화) ───────────────────────
ACTION_LABELS = {
    0: ("🛑", "Hold    "),
    1: ("🟢", "Buy 40% "),
    2: ("💚", "Buy 60% "),
    3: ("🔴", "Sell 60%"),
    4: ("🟠", "Sell 40%"),
}
NUM_ACTIONS    = 5
BUY_THRESHOLD  = 0.3
SELL_THRESHOLD = 0.3


# ═══════════════════════════════════════════════════════════════
# Universe 복원 유틸리티
# ═══════════════════════════════════════════════════════════════

def load_training_universe(config_manager, model_path: str) -> list:
    """
    훈련 시 trading_env.__init__가 sorted(all_symbols)로 만든
    symbol_to_idx 매핑을 동일하게 재생산합니다.

    우선순위:
    1. {model}.symbols.json 캐시
    2. InfluxDB schema.tagValues (훈련 당시 실제 심볼 목록)
    3. config_manager.get_symbols() fallback
    """
    cache_path = model_path.replace(".zip", "") + ".symbols.json"

    # ── 1. 캐시 파일 ────────────────────────────────────────
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                syms = json.load(f)
            if isinstance(syms, list) and syms:
                print(f"   💾 [Universe] 캐시 로드: {len(syms)}종목")
                return sorted(list(set(syms)))
        except Exception:
            pass

    # ── 2. InfluxDB 전체 심볼 조회 (Sync 클라이언트) ─────────
    try:
        from influxdb_client import InfluxDBClient
        url    = config_manager.get("INFLUX_URL",    "http://localhost:8086")
        token  = config_manager.get("INFLUX_TOKEN",  "")
        org    = config_manager.get("INFLUX_ORG",    "my-trade")
        bucket = config_manager.get("influx_bucket", "stock_data")
        sc = InfluxDBClient(url=url, token=token, org=org, timeout=30000)
        qa = sc.query_api()
        q  = f'import "influxdata/influxdb/schema" schema.tagValues(bucket: "{bucket}", tag: "symbol")'
        raw = []
        for table in qa.query(q, org=org):
            for rec in table.records:
                v = rec.get_value()
                if v and v != "UNKNOWN":
                    raw.append(v.split('_')[0].strip())
        sc.close()
        if raw:
            syms = sorted(list(set(raw)))
            print(f"   🌐 [Universe] InfluxDB {len(syms)}종목 조회 완료")
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(syms, f, ensure_ascii=False)
                print(f"   💾 [Universe] 캐시 저장: {os.path.basename(cache_path)}")
            except Exception:
                pass
            return syms
    except Exception as e:
        print(f"   ⚠️ [Universe] InfluxDB 조회 실패: {e}")

    # ── 3. config_manager fallback ───────────────────────────
    try:
        raw_list = config_manager.get_symbols()
        syms = sorted(list(set(s.get("code", "") for s in raw_list if s.get("code"))))
        if syms:
            print(f"   ⚠️ [Universe] config fallback: {len(syms)}종목")
            return syms
    except Exception:
        pass

    print("   ❌ [Universe] 복원 실패 → One-hot 전체 0으로 진단")
    return []


# ═══════════════════════════════════════════════════════════════
# 확률 추출 헬퍼
# ═══════════════════════════════════════════════════════════════

def get_action_probs(model, obs_input: np.ndarray,
                     action_masks: np.ndarray, n_actions: int) -> np.ndarray:
    """
    PPO 정책망에서 소프트맥스 확률 배열을 직접 추출합니다.
    사용자 요청: model.policy.get_distribution(obs_tensor).distribution.probs
    """
    fallback = np.ones(n_actions) / n_actions
    try:
        # 1. Observation을 텐서로 변환
        obs_t, _ = model.policy.obs_to_tensor(obs_input)
        
        with torch.no_grad():
            # 2. 정책 분포 추출
            distribution = model.policy.get_distribution(obs_t)
            
            # 3. 소프트맥스 확률값 추출 (사용자 요청 방식)
            if hasattr(distribution.distribution, 'probs'):
                probs = distribution.distribution.probs.detach().cpu().numpy()[0]
            else:
                # Logits만 있는 경우 Softmax 적용
                probs = torch.softmax(distribution.distribution.logits, -1).detach().cpu().numpy()[0]
                
        # 차원 맞춤 (Padding)
        if len(probs) < n_actions:
            probs = np.pad(probs, (0, n_actions - len(probs)))
        return probs
    except Exception as e:
        # print(f"DEBUG: 확률 추출 중 오류: {e}")
        return fallback


# ═══════════════════════════════════════════════════════════════
# 메인 진단 루틴
# ═══════════════════════════════════════════════════════════════

async def run_diagnosis(symbol: str = "001440"):
    container      = Container()
    config_manager = container.config_manager()
    fetcher        = container.historical_fetcher()

    # ── 1. 모델 로드 ────────────────────────────────────────
    model_path = config_manager.get("active_model_path", "")
    if not model_path:
        print("❌ active_model_path가 설정되지 않았습니다.")
        return
    if model_path.startswith("./"):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path[2:])
    if not os.path.exists(model_path):
        if os.path.exists(model_path + ".zip"):
            model_path += ".zip"
        else:
            print(f"❌ 모델 파일 없음: {model_path}")
            return

    print(f"📡 모델 로드 중: {os.path.basename(model_path)}")
    try:
        model = MaskablePPO.load(model_path)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}"); return

    model_action_n = model.action_space.n
    target_dim     = model.observation_space.shape[0]
    print(f"✅ 모델 로드 완료 | Action: Discrete({model_action_n}) | Obs Dim: {target_dim}")
    if model_action_n != NUM_ACTIONS:
        print(f"   ⚠️ 액션 수({model_action_n}) ≠ 기준({NUM_ACTIONS})")

    # ── 2. Universe 복원 ────────────────────────────────────
    print("🔍 훈련 Universe 복원 중...")
    all_symbols   = load_training_universe(config_manager, model_path)
    symbol_to_idx = {s: i for i, s in enumerate(all_symbols)}
    symbol_idx    = symbol_to_idx.get(symbol, -1)

    # feature_dim 역산으로 stock_id_dim 계산
    FEAT_DIM   = 5 * 10   # basic: single_feat(5) * window(10) = 50
    PORT_DIM   = 2
    IND_DIM    = 7
    stock_id_dim = max(1, target_dim - FEAT_DIM - PORT_DIM - IND_DIM)

    print(f"   Universe: {len(all_symbols)}종목 | stock_id_dim={stock_id_dim}")
    if symbol_idx >= 0:
        print(f"   One-hot: {symbol} → idx={symbol_idx}"
              + (" ⚠️ idx>stock_id_dim!" if symbol_idx >= stock_id_dim else " ✅"))
    else:
        print(f"   ⚠️ '{symbol}'이 Universe에 없음 → One-hot 전체 0")
        if all_symbols:
            print(f"   💡 Universe 샘플: {all_symbols[:5]} ...")

    # ── 3. 데이터 페칭 ──────────────────────────────────────
    print(f"\n📂 [{symbol}] 과거 데이터 페칭 시작...")
    token       = config_manager.get("KIWOOM_ACCESS_TOKEN") or os.getenv("KIWOOM_ACCESS_TOKEN")
    today_str   = datetime.now().strftime("%Y%m%d")
    data_result = await fetcher.fetch_historical_data(symbol, today_str, token, max_pages=5)

    from returns.pipeline import is_successful
    if not is_successful(data_result):
        print(f"❌ 데이터 페칭 실패: {data_result.failure()}"); return

    data = data_result.unwrap()
    if not isinstance(data, list):
        data = data._inner_value if hasattr(data, "_inner_value") else []
    if len(data) < 60:
        print(f"❌ 데이터 부족 ({len(data)}개, 최소 60개 필요)"); return

    sorted_data = sorted(data, key=lambda x: x["timestamp"])
    print(f"✅ 데이터 로드 완료: {len(sorted_data)}개 캔들")

    # ── 4. ScalpingTradingEnv 인스턴스화 ────────────────────
    # 수동 feature 복제 없이, env 자체가 관측값을 생성하도록 합니다.
    print("\n🏗️  ScalpingTradingEnv 생성 중...")
    from env.trading_env import ScalpingTradingEnv

    env_config = {
        "symbol":           symbol,
        "initial_balance":  10_000_000,
        "historical_data":  sorted_data,
        "mode":             "backtest",   # start_step=0, end_step=len-1 고정
        "feature_mode":     "basic",
        "target_dim":       target_dim,   # 모델 obs dim으로 env 차원 맞춤
        "all_symbols":      all_symbols,  # InfluxDB에서 복원한 전체 Universe
    }
    env = ScalpingTradingEnv(None, None, env_config)
    obs, info = env.reset()   # ← 내부에서 indicators 계산, buffer 초기화, _get_observation() 실행
    print(f"✅ Env 생성 완료 | Obs Shape: {obs.shape}")

    # ── [DEBUG] 첫 obs 값 출력 ──────────────────────────────
    print("\n─── [DEBUG] 첫 번째 obs 배열 값 ───────────────────────")
    print(f"  obs[:10]  = {np.round(obs[:10], 4)}")
    print(f"  obs[-10:] = {np.round(obs[-10:], 4)}")
    print(f"  obs 범위  = [{obs.min():.4f}, {obs.max():.4f}]  "
          f"(절대값 > 50인 원소: {(np.abs(obs) > 50).sum()}개)")
    if (np.abs(obs) > 50).sum() > 0:
        print("  ⚠️  원소값 > 50 감지 — raw 가격이 섞였을 가능성!")
    print("────────────────────────────────────────────────────\n")

    # ── 5. 통계 카운터 ──────────────────────────────────────
    total_steps      = 0
    raw_counts       = {i: 0 for i in range(NUM_ACTIONS)}
    filtered_counts  = {i: 0 for i in range(NUM_ACTIONS)}
    max_probs        = {i: 0.0 for i in range(NUM_ACTIONS)}  # 각 액션별 최대 확률 기록
    filter_overrides = 0
    buy_conf_list    = []
    sell_conf_list   = []
    obs_warn_count   = 0
    probs_accumulator = np.zeros(NUM_ACTIONS, dtype=np.float64)  # 평균 확률 산출용

    print(f"🧠 정책 분석 시작... "
          f"(Buy≥{BUY_THRESHOLD}, Sell≥{SELL_THRESHOLD})\n")
    print("─── [인스펜션] 초반 10스텝 및 유의미한 확률 변화 출력 ─────")

    done = False
    while not done:
        # obs 스케일 방어
        if (np.abs(obs) > 50).sum() > 0:
            obs_warn_count += 1

        # ── 액션 마스킹 ─────────────────────────────────
        action_masks = env.action_masks()                 # env 자체 메서드 사용
        masks_arr    = np.array(action_masks, dtype=bool)
        masks_input  = masks_arr[:model_action_n] if len(masks_arr) >= model_action_n \
                       else np.pad(masks_arr, (0, model_action_n - len(masks_arr)), constant_values=True)

        state_input = np.expand_dims(obs.astype(np.float32), 0)

        # ── 원시 액션 예측 ───────────────────────────────
        raw_action, _ = model.predict(state_input, action_masks=masks_input, deterministic=True)
        raw_action    = int(raw_action[0] if isinstance(raw_action, np.ndarray) else raw_action)

        # ── 확률 추출 ────────────────────────────────
        probs = get_action_probs(model, state_input, masks_input, model_action_n)
        if len(probs) < NUM_ACTIONS:
            probs = np.pad(probs, (0, NUM_ACTIONS - len(probs)))

        total_steps += 1
        raw_counts[raw_action] += 1
        conf = float(probs[raw_action])
        probs_accumulator += probs[:NUM_ACTIONS]
        
        # 최대 확률 갱신
        for k in range(NUM_ACTIONS):
            max_probs[k] = max(max_probs[k], float(probs[k]))

        # ── 인스펜션: 초반 10스텝 또는 매수/매도 확률이 5%를 넘는 경우 출력 ────────────
        # 100% Hold 상황에서 모델이 어떤 액션을 '고민'하는지 확인용
        potential_action = np.argmax(probs[1:]) + 1  # Hold를 제외한 최대 확률 액션
        potential_prob   = probs[potential_action]
        
        if total_steps <= 10 or potential_prob > 0.05:
            p_str = "  ".join(
                f"{ACTION_LABELS.get(j, ('?','?'))[1].strip()}:{probs[j]:.4f}"
                for j in range(NUM_ACTIONS)
            )
            mark = "⭐" if potential_prob > 0.1 else "  "
            print(f"{mark} step={env.current_step:4d} | raw={ACTION_LABELS[raw_action][1].strip()} "
                  f"| [{p_str}]")
            if total_steps == 10:
                print("────────────────────────────────────────────────\n")

        # ── Confidence 필터 ──────────────────────────────
        filtered_action = raw_action
        if raw_action in (1, 2):
            buy_conf_list.append(conf)
            if conf < BUY_THRESHOLD:
                filtered_action  = 0
                filter_overrides += 1
        elif raw_action in (3, 4):
            sell_conf_list.append(conf)
            if conf < SELL_THRESHOLD:
                filtered_action  = 0
                filter_overrides += 1
        filtered_counts[filtered_action] += 1

        # ── 상세 로그 출력 (매수/매도 신호) ─────────────
        if raw_action in (1, 2, 3, 4):
            icon, lbl    = ACTION_LABELS[raw_action]
            override_mrk = " ⚡→Hold"   if filtered_action == 0 else ""
            inds         = env._current_indicators
            pullback_mrk = ""
            curr_price = env._get_current_price()
            if raw_action in (1, 2) and inds:
                vwap = inds.get('VWAP', 0)
                atr14 = inds.get('ATR_14', 0)
                bb_lower = inds.get('BB_LOWER', 0)
                rsi14 = inds.get('RSI_14', 50)
                atr_threshold = curr_price * 0.001
                if curr_price > vwap and atr14 > atr_threshold and (curr_price <= bb_lower or rsi14 < 30):
                    pullback_mrk = " 🎯A급타점!"
            
            p_str = "  ".join(
                f"{ACTION_LABELS.get(j, ('?','?'))[1].strip()}:{probs[j]:.3f}"
                for j in range(min(NUM_ACTIONS, len(probs)))
            )
            cs = env.current_step
            
            print(
                f"  {icon} step={cs:4d} | {lbl.strip()} | 신뢰도:{conf:.4f} | 현재가:{curr_price:,.0f}원"
                f"{override_mrk}{pullback_mrk}\n"
                f"     [{p_str}]"
            )

        # ── 환경 Step (필터 적용 액션으로) ──────────────
        obs, reward, done, truncated, info = env.step(filtered_action)
        done = done or truncated

    # ── 6. 결과 리포트 ──────────────────────────────────────
    sep = "─" * 52
    print(f"\n{'📊 [모델 정책 진단 결과]':^52}")
    print(sep)
    print(f"  테스트 종목    : {symbol}")
    print(f"  총 추론 스텝   : {total_steps}")
    print(f"  Obs Dim        : {target_dim}")
    print(f"  Action Space   : Discrete({model_action_n})")
    print(f"  Obs 스케일 경고: {obs_warn_count}건")
    print(sep)

    if total_steps > 0:
        print(f"\n  ▶ 순수 모델 판단 (필터 미적용)")
        for k in range(NUM_ACTIONS):
            cnt = raw_counts[k]
            pct = cnt / total_steps * 100
            icon, lbl = ACTION_LABELS[k]
            print(f"    {icon} {lbl}({k}): {cnt:4d}회  ({pct:5.1f}%)")

        print(f"\n  ▶ Confidence 필터 적용 후 (Buy≥{BUY_THRESHOLD}, Sell≥{SELL_THRESHOLD})")
        for k in range(NUM_ACTIONS):
            cnt = filtered_counts[k]
            pct = cnt / total_steps * 100
            icon, lbl = ACTION_LABELS[k]
            print(f"    {icon} {lbl}({k}): {cnt:4d}회  ({pct:5.1f}%)")

        print(f"\n  ▶ 필터로 Hold 전환: {filter_overrides}회 "
              f"({filter_overrides/total_steps*100:.1f}%)")

        avg_bc  = np.mean(buy_conf_list)  if buy_conf_list  else 0.0
        max_bc  = np.max(buy_conf_list)   if buy_conf_list  else 0.0
        avg_sc  = np.mean(sell_conf_list) if sell_conf_list else 0.0
        print(f"\n  ▶ 매수 신뢰도 — 평균:{avg_bc:.4f}  최대:{max_bc:.4f} (n={len(buy_conf_list)})")
        print(f"  ▶ 매도 신뢰도 — 평균:{avg_sc:.4f} (n={len(sell_conf_list)})")

        rbp = (raw_counts[1] + raw_counts[2]) / total_steps * 100
        rsp = (raw_counts[3] + raw_counts[4]) / total_steps * 100
        print(f"\n  ▶ 매수 편향도: {rbp:.1f}%  |  매도 편향도: {rsp:.1f}%")

        # ── 전체 평균 확률 분포 (핵심 진단 지표) ──────────────
        avg_probs = probs_accumulator / max(total_steps, 1)
        print(f"\n  ▶ 전체 스텝 평균 및 최대 확률 분포 (Softmax — {total_steps}스텝)")
        prob_entropy = -np.sum(avg_probs * np.log(avg_probs + 1e-12))
        for k in range(NUM_ACTIONS):
            icon, lbl = ACTION_LABELS[k]
            bar_w = int(avg_probs[k] * 40)
            bar   = "█" * bar_w + "░" * (40 - bar_w)
            print(f"    {icon} {lbl}({k}): Avg:{avg_probs[k]*100:6.3f}% | Max:{max_probs[k]*100:6.3f}% |{bar}|")
        print(f"    엔트로피(다양성): {prob_entropy:.4f}"
              + ("  ← 값이 작을수록 특정 액션에 쏠림" if prob_entropy < 0.5 else ""))
        if avg_probs[0] > 0.98:
            print("    🚨 Hold 확률이 98%+ → 모델이 사실상 모든 상황에서 Hold만 선택")
            print("       → 이 모델은 훈련과 현재 obs 분포가 크게 다르거나 재학습이 필요합니다.")
        elif prob_entropy < 0.3:
            print("    ⚠️  정책 엔트로피가 매우 낮습니다 → 단일 액션에 과도하게 집중")

        print(f"\n  [진단 결론]")
        if raw_counts[0] == total_steps:
            print("  🚨 100% Hold — 모델이 모든 상황을 관망으로 판단합니다.")
            print("       훈련 데이터와 진단 데이터의 시장 상황이 크게 다를 수 있습니다.")
        elif rbp > 80:
            print("  ⚠️  정책 붕괴 위험! 매수 편향이 심각합니다.")
            print("       → 학습률 조정 또는 Hold 페널티 완화를 검토하세요.")
        elif rbp + rsp < 5:
            print("  ⚠️  과소 추론 위험! 모델이 지나치게 소극적입니다.")
            print("       → 거래 보상 배율(reward_multiplier)을 높이세요.")
        elif avg_bc > 0.9:
            print("  ⚠️  신뢰도 포화(Saturation) 의심!")
            print("       → 정규화(Normalizer) 또는 클리핑 로직을 점검하세요.")
        else:
            print("  ✅  모델 정책이 비교적 균형 잡혀 있습니다.")
    else:
        print("  ❌ 유효한 추론 스텝이 없습니다.")

    print(sep + "\n")


if __name__ == "__main__":
    target_sym = sys.argv[1] if len(sys.argv) > 1 else "001440"
    asyncio.run(run_diagnosis(target_sym))
