import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback


class MetricsTrackerCallback(BaseCallback):
    """
    Callback to track metrics during training and store them in memory for later analysis.
    """
    def __init__(self, eval_env: gym.Env, verbose: int = 0):
        """
        :param eval_env: Evaluation environment for optional evaluation at episode end.
        :param verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_idx = 0
        self.episode_reward = 0
        self.episode_length = 0

    def _on_training_start(self) -> None:
        """
        Initialize metrics tracking when training starts.
        """
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_idx = 0
        self.episode_reward = 0
        self.episode_length = 0

    def _on_step(self) -> bool:
        """
        Called after each environment step.
        Tracks rewards and episode length, resets metrics at the end of an episode.
        """
        reward = self.locals["rewards"]
        dones = self.locals["dones"]

        # Accumulate reward and increment step count
        self.episode_reward += reward.item() if isinstance(reward, np.ndarray) else reward
        self.episode_length += 1

        # If the episode is done, record metrics
        if np.any(dones):
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_length)
            self._on_episode_end()

        return True

    def _on_episode_end(self) -> None:
        """
        Record the episode's return and length.
        Optionally, evaluate the target policy.
        """
        # Log metrics
        if self.verbose > 0:
            print(f"Episode {self.episode_idx}: Reward = {self.episode_reward}, Length = {self.episode_length}")

        # Optional evaluation step
        if self.eval_env is not None:
            self._evaluate_policy()

        # Reset metrics for the next episode
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_idx += 1

    def _evaluate_policy(self) -> None:
        """
        Evaluate the agent on the evaluation environment.
        """
        obs, _ = self.eval_env.reset()
        eval_reward = 0

        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = self.eval_env.step(action)

            eval_reward += reward

            # If the environment is done, reset it
            if terminated or truncated:
                obs, _ = self.eval_env.reset()
                break

        if self.verbose > 0:
            print(f"Evaluation Reward: {eval_reward}")


    def get_tracked_metrics(self):
        """
        Returns the tracked rewards and episode lengths for external analysis.
        """
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths
        }
