import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from utils.tracker_callback import MetricsTrackerCallback  # Make sure the callback is correctly imported

def create_env(env_name="InvertedDoublePendulum-v4"):
    """
    Creates and returns a monitored Gym environment.
    """
    env = gym.make(env_name)
    env = Monitor(env, filename='data.csv')  # Monitor for logging rewards
    return env


def train_ddpg(env, total_timesteps=100000, model_path="ddpg_inverted_double_pendulum"):
    """
    Trains a DDPG model on the provided environment.
    :param env: The Gym environment.
    :param total_timesteps: Number of timesteps to train.
    :param model_path: Path to save the trained model.
    :return: Trained DDPG model.
    """
    model = DDPG("MlpPolicy", env, verbose=0)
    
    # Create the callback
    metrics_callback = MetricsTrackerCallback(eval_env=env, verbose=1)

    # Train the model with the callback
    model.learn(total_timesteps=total_timesteps, callback=metrics_callback)
    model.save(model_path)
    
    # Optionally, after training, you can access the metrics stored in the callback
    tracked_metrics = metrics_callback.get_tracked_metrics()
    print(f"Tracked metrics (Rewards and Lengths): {tracked_metrics}")
    
    return model, metrics_callback

def plot_rewards(metrics_tracker, title="Rewards over Episodes (DDPG on Inverted Double Pendulum)"):
    """
    Plots the episode rewards and lengths tracked by the MetricsTrackerCallback.
    :param metrics_tracker: The MetricsTrackerCallback object that stores the tracked data.
    :param title: Title of the plot.
    """
    # Retrieve the tracked episode rewards and lengths
    episode_rewards = metrics_tracker.get_tracked_metrics()['episode_rewards']
    episode_lengths = metrics_tracker.get_tracked_metrics()['episode_lengths']

    # Create an array for episode numbers (1, 2, 3, ...)
    episodes = np.arange(1, len(episode_rewards) + 1)

    # Plot the rewards over episodes
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, episode_rewards, label="Episode Rewards", color='b')

    # Optionally, you can plot episode lengths as well
    plt.plot(episodes, episode_lengths, label="Episode Lengths", color='g', linestyle='--')

    # Adding labels and title
    plt.xlabel("Episode")
    plt.ylabel("Reward / Length")
    plt.title(title)
    plt.legend()
    plt.grid()

    # Save the plot as a PNG file
    plt.savefig("rewards.png")
    plt.show()


def main():
    """
    Main function to run the training and evaluation of DDPG on Inverted Double Pendulum.
    """
    # Create environment
    env = create_env()

    # Train DDPG with callback
    model, tracker = train_ddpg(env, total_timesteps=10000)
    print("Training complete!")


    # Plot rewards
    print("Plotting rewards...")
    plot_rewards(tracker)

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
