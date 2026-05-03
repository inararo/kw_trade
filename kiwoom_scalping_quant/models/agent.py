import os
import datetime
import numpy as np
from typing import Dict, Any, Optional
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from models.lstm_extractor import LSTMExtractor

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    최고의 보상을 얻었을 때 모델을 저장하고, 샤프 지수 등 커스텀 지표를 TensorBoard에 기록하는 콜백
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Info dict 등에서 커스텀 지표를 추출해 기록할 수 있음
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])

                if self.verbose > 0:
                    print(f"Step: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

            # Example custom logging:
            # self.logger.record("custom/sharpe_ratio", ...)
        return True

class TradingAgentWrapper:
    """
    LSTMExtractor와 MaskablePPO를 결합한 에이전트 래퍼 클래스
    """
    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        self.env = env
        self.config = config
        self.feature_mode = config.get("feature_mode", "basic")
        
        # 모델 저장/로드 시 샘플링 방식에 따른 prefix 생성
        # 예: model_advanced_smart, model_advanced_random, model_basic_smart ...
        sampling_suffix = config.get("model_name_suffix", "random")
        self.model_prefix = f"model_{self.feature_mode}_{sampling_suffix}"
        self.model = None
        self.seq_len = config.get("seq_len", 10)

        self._initialize_model()

    def _initialize_model(self):
        policy_kwargs = dict(
            features_extractor_class=LSTMExtractor,
            features_extractor_kwargs=dict(seq_len=self.seq_len, features_dim=128),
        )

        # Monitor 래퍼 등을 통해 환경 래핑 필요 (생략 가능)

        # [에러 방지] 매 학습마다 유니크한 로그 폴더 생성 (PermissionError 방지)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_log_dir = os.path.join(self.config.get("tensorboard_log", "./tensorboard_logs/"), f"PPO_{timestamp}")

        self.model = MaskablePPO(
            "MlpPolicy",
            self.env,
            policy_kwargs=policy_kwargs,
            learning_rate=self.config.get("learning_rate", 3e-4),
            ent_coef=self.config.get("ent_coef", 0.01),
            clip_range=self.config.get("clip_range", 0.2),
            gamma=self.config.get("gamma", 0.99),
            gae_lambda=self.config.get("gae_lambda", 0.95),
            tensorboard_log=tb_log_dir,
            verbose=1
        )

    def train(self, total_timesteps: int = 100000, callbacks: list = None):
        """
        에이전트 학습을 실행합니다.
        추가 콜백(예: GUI 모니터링, Early Stopping 등)을 리스트로 받아 통합 실행합니다.
        """
        log_dir = self.config.get("model_save_dir", "./saved_models/")
        best_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

        all_callbacks = [best_callback]
        if callbacks:
            all_callbacks.extend(callbacks)

        self.model.learn(total_timesteps=total_timesteps, callback=all_callbacks)

        # 학습 완료 후 최종 가중치 저장 (버전 관리를 위해 타임스탬프와 feature_mode 부여)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        final_path = os.path.join(log_dir, f"{self.model_prefix}_{timestamp}")
        self.model.save(final_path)

    def load_weights(self, path: str):
        """GUI에서 모델을 동적으로 교체하기 위한 메서드"""
        if os.path.exists(path + ".zip") or os.path.exists(path):
            try:
                self.model = MaskablePPO.load(path, env=self.env)
            except ValueError as e:
                # Value Error: 보통 Observation Space Dimension mismatch 시 발생
                raise ValueError(f"Observation Space 차원 불일치 (현재 피처 모드: {self.feature_mode}): {e}")
        else:
            raise FileNotFoundError(f"Model weights not found at {path}")

    @staticmethod
    def get_model_dimension(path: str) -> int:
        """모델 파일을 실제로 에칭하기 전에 관측 차원(Dimension)만 추출합니다."""
        try:
            # env=None으로 로드하여 메타데이터만 확인
            model = MaskablePPO.load(path, env=None)
            if hasattr(model, "observation_space"):
                return model.observation_space.shape[0]
        except Exception:
            pass
        return 0

    def predict(self, state: np.ndarray, action_masks: Optional[np.ndarray] = None, return_probs: bool = False):
        """실시간 틱 데이터에서 다음 행동 추론"""
        action, _states = self.model.predict(state, action_masks=action_masks, deterministic=True)
        
        if return_probs:
            import torch
            # obs를 텐서로 변환
            obs_tensor, _ = self.model.policy.obs_to_tensor(state)
            with torch.no_grad():
                # 정책망으로부터 확률 분포 획득
                distribution = self.model.policy.get_distribution(obs_tensor)
                
                # [보강] MaskablePPO의 경우 action_masks를 수동으로 적용해야 정확한 확률 분포(마스킹 반영)가 나옴
                if action_masks is not None and hasattr(distribution, 'apply_mask'):
                    # action_masks를 텐서로 변환하여 적용 (금지된 액션의 확률을 0으로 만듦)
                    mask_tensor = torch.as_tensor(action_masks).to(obs_tensor.device)
                    distribution.apply_mask(mask_tensor)

                if hasattr(distribution, 'distribution') and hasattr(distribution.distribution, 'probs'):
                    probs = distribution.distribution.probs
                else:
                    # 마스킹되지 않은 일반 분포일 경우
                    probs = distribution.probs
                
            return action, probs.cpu().numpy()[0]

        return action
