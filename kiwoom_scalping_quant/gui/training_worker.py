import os
import time
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from stable_baselines3.common.callbacks import BaseCallback

class TrainingSignals(QObject):
    """QThread와 ViewModel 간 통신을 담당하는 신호 객체"""
    started = pyqtSignal()
    progress_updated = pyqtSignal(int, float, float) # step, reward, loss
    log_msg = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

class GUIMonitorCallback(BaseCallback):
    """
    Stable-Baselines3 훈련 루프 도중 주기적으로 정보를 PyQt GUI로 전송하는 콜백.
    메인 스레드를 블로킹하지 않도록 설계되었습니다.
    """
    def __init__(self, signals: TrainingSignals, update_freq: int = 100, verbose=0):
        super().__init__(verbose)
        self.signals = signals
        self.update_freq = update_freq
        self._is_cancelled = False

    def _on_step(self) -> bool:
        if self._is_cancelled:
            self.signals.log_msg.emit("Training interrupted by user. Early stopping.")
            return False # False 반환 시 SB3 훈련 루프 종료

        if self.n_calls % self.update_freq == 0:
            mean_reward = 0.0
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])

            # 임의의 Loss 추출 (SB3 logger에서 안전하게 접근하기 어려우므로 Mock/추정치)
            # 실전 환경에서는 model.logger.name_to_value 등 활용
            mock_loss = max(0.1, 10.0 / (self.num_timesteps + 1))

            self.signals.progress_updated.emit(self.num_timesteps, mean_reward, mock_loss)

        return True

    def cancel_training(self):
        self._is_cancelled = True

class TrainingWorker(QThread):
    """
    RL Agent의 model.learn()은 동기적으로 실행되며 메인 스레드를 멈추게 하므로,
    QThread 내부에서 실행되도록 분리한 워커.
    """
    def __init__(self, agent, total_timesteps: int, signals: TrainingSignals):
        super().__init__()
        self.agent = agent
        self.total_timesteps = total_timesteps
        self.signals = signals
        self.callback = GUIMonitorCallback(self.signals, update_freq=50)

    def run(self):
        try:
            self.signals.started.emit()
            self.signals.log_msg.emit(f"Starting Maskable PPO Training... (Timesteps: {self.total_timesteps})")

            # Agent 학습 루프 실행 (블로킹 콜, 내부적으로 callback이 시그널 송출)
            self.agent.train(total_timesteps=self.total_timesteps, callbacks=[self.callback])

            self.signals.log_msg.emit("Training completed successfully.")
            self.signals.finished.emit()
        except Exception as e:
            self.signals.error.emit(str(e))

    def stop(self):
        """학습 중지 플래그 전달"""
        if self.callback:
            self.callback.cancel_training()
