import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from loguru import logger

from src.utils.metrictracker import MetricsTracker


class MetricsTrackerCallback(BaseCallback):
    """
    Stable Baselines callback to track and record the agent's performance using the MetricsTracker.
    Additionally, it runs the evaluate_target_policy function at the end of each episode.
    """

    def __init__(self, tracker: MetricsTracker,
                 agent_id: str = "DDPG", verbose: int = 0):
        """
        Constructor for StableBaselinesMetricsCallback.

        :param tracker: The MetricsTracker instance for tracking policy metrics.
        :param eval_env: The evaluation environment.
        :param agent_id: The identifier for the agent (default: "DQN").
        :param verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.agent_id = agent_id
        self.tracker = tracker
        self.episode_reward = 0
        self.episode_idx = 0  # Track the current episode index
        self.highest_avg_return = float('-inf')

    def _on_training_start(self) -> None:
        """
        Initialize metrics tracking when training starts.
        """
        self.episode_reward = 0
        self.highest_avg_return = float('-inf')
        self.episode_idx = 0

    def _on_step(self) -> bool:
        """
        This method will be called after each call to `env.step()`.
        Tracks the reward received during each step.

        :return: If the callback returns False, training is aborted early.
        """
        reward = self.locals['rewards']
        dones = self.locals['dones']
        self.episode_reward += reward.item() if isinstance(reward, np.ndarray) else reward

        if np.any(dones):
            self._on_episode_end()

        return True

    def _on_episode_end(self) -> None:
        """
        Record the episode's return and check if the agent achieved a new highest average return.
        """
        logger.info(f"Episode {self.episode_idx} finished, {self.episode_reward}")
        # Record the return for the policy
        self.tracker.record_metric("return", agent_id=self.agent_id, episode_idx=self.episode_idx,
                                   value=self.episode_reward)
        # Get the current mean and stddev return for this episode
        current_mean_return, current_std_return = self.tracker.get_mean_std("return",
                                                                            self.agent_id,
                                                                            self.episode_idx)
        if current_mean_return and current_mean_return > self.highest_avg_return:
            self.highest_avg_return = current_mean_return

        # Reset the episode reward
        self.episode_reward = 0
        self.episode_idx += 1  # Increment the episode index

    def _on_training_end(self) -> None:
        # You could add additional logging or save the tracker data to file if needed
        pass