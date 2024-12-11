import threading
from typing import Union, SupportsFloat, Any, Dict, Tuple
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from src.utils.welford import Welford
import pandas as pd

"""
Sourced from RL theoretical course material.
"""

class MetricsTracker:
    """
    Thread-safe object for recording various metrics across multiple runs.
    Tracks the mean and standard deviation of each metric using Welford's algorithm,
    averaged over multiple runs.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # Dictionary of metric histories where key is metric_name, value is agent_id -> episode_idx -> Welford object
        self._metrics_history: Dict[str, Dict[str, Dict[int, Welford]]] = defaultdict(lambda: defaultdict(dict))

        self.register_metric("loss")
        self.register_metric("return")

    def register_metric(self, metric_name: str) -> None:
        """
        Register a new metric to be tracked.

        :param metric_name: The name of the metric to register (e.g., "return", "accuracy").
        """
        with self._lock:
            if metric_name not in self._metrics_history:
                self._metrics_history[metric_name] = defaultdict(dict)

    def record_metric(self, metric_name: str, agent_id: str, episode_idx: int,
                      value: Union[float, int, SupportsFloat]) -> None:
        """
        Record a value for a specific metric, agent, and episode. Uses Welford's algorithm to update mean and variance.

        :param metric_name: The name of the metric (e.g., "return", "accuracy").
        :param agent_id: The identifier of the agent.
        :param episode_idx: The index of the episode.
        :param value: The metric value to record.
        """
        with self._lock:
            if episode_idx not in self._metrics_history[metric_name][agent_id]:
                self._metrics_history[metric_name][agent_id][episode_idx] = Welford()

            self._metrics_history[metric_name][agent_id][episode_idx].update_aggr(float(value))

    def get_mean_std(self, metric_name: str, agent_id: str, episode_idx: int) -> Tuple[Any, Any]:
        """
        Get the latest recorded mean and standard deviation for a specific metric, agent, and episode.

        :param metric_name: The name of the metric.
        :param agent_id: The identifier of the agent.
        :param episode_idx: The episode index.
        :return: The mean and standard deviation for the metric, or (None, None) if no values have been recorded.
        """
        with self._lock:
            if agent_id in self._metrics_history[metric_name] and episode_idx in self._metrics_history[metric_name][
                agent_id]:
                welford = self._metrics_history[metric_name][agent_id][episode_idx]
                mean, var = welford.get_curr_mean_variance()
                return mean, np.sqrt(var)
            else:
                return None, None

    def plot_metric(self, metric_name: str, file_name: str, num_episodes: int = None, x_axis_label="Episodes",
                    y_axis_label="Metric Value", title=None) -> None:
        """
        Plot the average value and standard deviation over episodes for a specific metric across multiple runs.

        :param metric_name: The name of the metric to plot (e.g., "return", "accuracy").
        :param file_name: The file to save the plot.
        :param num_episodes: The number of episodes to plot.
        :param x_axis_label: The label for the x-axis.
        :param y_axis_label: The label for the y-axis.
        :param title: The title of the plot (optional).
        """
        with self._lock:
            if metric_name not in self._metrics_history:
                raise ValueError(f"Metric '{metric_name}' not found")

            fig, ax = plt.subplots(figsize=(10, 8))

            for agent_id, episode_welfords in self._metrics_history[metric_name].items():
                # Sort by episode index to ensure proper plotting order
                sorted_episodes = sorted(episode_welfords.items())

                if num_episodes:
                    sorted_episodes = sorted_episodes[:num_episodes]

                mean_values = [w.get_curr_mean_variance()[0] for _, w in sorted_episodes]
                std_values = [np.sqrt(w.get_curr_mean_variance()[1]) for _, w in sorted_episodes]
                # Apply a smoothing on the data with a running window average over episodes
                window = 10
                mean_values = pd.Series(mean_values).rolling(window=window, min_periods=1).mean()
                std_values = pd.Series(std_values).rolling(window=window, min_periods=1).mean()

                x_values = np.array([ep for ep, _ in sorted_episodes])  # X-axis corresponds to episode indices

                # Plot the mean value for each episode
                ax.plot(x_values, mean_values, label=f'{agent_id} agent')

                # Fill between mean ± standard deviation
                ax.fill_between(x_values,
                                np.array(mean_values) - np.array(std_values),
                                np.array(mean_values) + np.array(std_values),
                                alpha=0.2)

            ax.set_title(title if title else f'{metric_name.capitalize()} History')
            ax.set_xlabel(x_axis_label)
            ax.set_ylabel(y_axis_label)
            ax.legend()
            ax.grid(True)

            plt.tight_layout()
            plt.savefig(file_name)

    def clear_metrics(self) -> None:
        """
        Clear the recorded metrics for all agents and all metrics.
        """
        with self._lock:
            self._metrics_history.clear()

    @property
    def metrics_history(self) -> dict:
        """
        Get the entire history of recorded metrics.

        :return: A dictionary containing the recorded metric values for each agent and episode.
        """
        with self._lock:
            return dict(self._metrics_history)

    def get_avg_return(self, agent_id: str = None, last_n_episodes: int = None) -> float:
        """
        Compute the average return for the specified agent or across all agents.
        
        :param agent_id: The identifier of the agent. If None, calculate for all agents.
        :param last_n_episodes: Number of recent episodes to average over. If None, average over all episodes.
        :return: The average return.
        """
        with self._lock:
            total_return = 0.0
            count = 0
            
            # Check if we are calculating for a specific agent or for all agents
            agents = [agent_id] if agent_id else self._metrics_history["return"].keys()
            
            for agent in agents:
                if agent in self._metrics_history["return"]:
                    # Sort episodes by index
                    episodes = sorted(self._metrics_history["return"][agent].items())
                    
                    # Optionally only consider the last n episodes
                    if last_n_episodes:
                        episodes = episodes[-last_n_episodes:]
                    
                    # Sum returns and count episodes
                    for episode_idx, welford in episodes:
                        mean_return, _ = welford.get_curr_mean_variance()
                        total_return += mean_return
                        count += 1
            
            # Return the average return
            return total_return / count if count > 0 else 0.0

    def save_metrics(self, file_name: str) -> None:
        """
        Save the recorded metrics to a file.

        :param file_name: The file to save the metrics to.
        """
        with self._lock:
            return pd.DataFrame(self._metrics_history).to_csv(file_name, index=False)
