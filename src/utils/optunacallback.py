import optuna
from stable_baselines3.common.callbacks import BaseCallback
from loguru import logger
import numpy as np

class OptunaPruneCallback(BaseCallback):
    """
    Callback for Stable-Baselines3 that reports metrics to Optuna and prunes unpromising trials.
    """

    def __init__(self, trial: optuna.trial.Trial, verbose: int = 0):
        """
        :param trial: The Optuna trial object.
        :param verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.trial = trial
        self.episode_idx = 0
        self.step_count = 0
        self.episode_reward = 0

    def _on_step(self) -> bool:
        reward = self.locals['rewards']
        dones = self.locals['dones']

        self.episode_reward += reward.item() if isinstance(reward, np.ndarray) else reward
        if self.locals.get('truncated') is not None:
            truncated = self.locals['truncated']
            if np.any(dones) or np.any(truncated):
                self._on_episode_end()
        else:
            if np.any(dones):
                self._on_episode_end()

        return True


    def _on_training_start(self) -> None:
        """
        Initialize metrics tracking when training starts.
        """
        self.episode_reward = 0
        self.episode_idx = 0

    def _on_episode_end(self) -> None:
        """
        Called at the end of an episode. Reports metrics to Optuna and checks for pruning.
        """

        # Report intermediate value to Optuna every 10 episodes
        # if self.episode_idx % 10 == 0:
        self.trial.report(self.episode_reward, step=self.episode_idx )

        # Check if the trial should be pruned
        if self.trial.should_prune():
            logger.info(f"Trial {self.trial.number} pruned at episode {self.episode_idx }.")
            raise optuna.TrialPruned()

        self.episode_idx  += 1
        self.episode_reward = 0

    def _on_training_end(self) -> None:
        """
        Clean up at the end of training.
        """
        logger.info(f"Trial {self.trial.number} finished after {self.episode_idx} episodes.")