import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from src.utils.tracker_callback import MetricsTrackerCallback  # Make sure the callback is correctly imported
import pickle
from stable_baselines3.common.noise import NormalActionNoise


def create_env(env_name="InvertedPendulum-v5"):
    """
    Creates and returns a monitored Gym environment.
    """
    env = gym.make(env_name)
    env = Monitor(env, filename='data.csv')  # Monitor for logging rewards
    return env


def train_ddpg(env, total_timesteps=20000, model_path="ddpg_inverted_pendulum"):
    """ 
    Trains a DDPG model on the provided environment.
    :param env: The Gym environment.
    :param total_timesteps: Number of timesteps to train.
    :param model_path: Path to save the trained model.
    :return: Trained DDPG model.
    """
    noise_std = 0.2
    noise = NormalActionNoise(mean=np.zeros(env.action_space.shape), sigma=noise_std * np.ones(env.action_space.shape))
    model = DDPG(
        "MlpPolicy", 
        env, 
        verbose=0,
        learning_rate=1e-3,
        learning_starts=1000,
        buffer_size=200000,
        batch_size=64,
        tau=0.005,
        gamma=0.98,
        train_freq=1,
        gradient_steps=1,
        action_noise=noise,  # You can add noise here if needed
        policy_kwargs=dict(net_arch=[400, 300])
    )
    eval_env = create_env()
    
    # Create the callback
    metrics_callback = MetricsTrackerCallback(eval_env=eval_env, verbose=1)

    # Train the model with the callback
    model.learn(total_timesteps=total_timesteps, callback=metrics_callback)
    model.save(model_path)
    
    return model, metrics_callback

def plot_rewards(metrics_tracker=None, window_size=10, title="Rewards over Episodes (DDPG on Inverted Pendulum)", episode_rewards=None):
    """
    Plots the episode rewards with a running average and its standard deviation.
    :param metrics_tracker: The MetricsTrackerCallback object that stores the tracked data.
    :param window_size: The window size for the moving average and standard deviation (default: 10).
    :param title: Title of the plot.
    """
    if metrics_tracker is not None:
        episode_rewards = metrics_tracker.get_tracked_metrics()['episode_rewards']
        with open('episode_rewards.pkl', 'wb') as f:
            pickle.dump(episode_rewards, f)
    # Calculate the running average (moving average)
    running_avg = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')
    
    # Calculate the standard deviation for the same window size
    running_std = [np.std(episode_rewards[i-window_size+1:i+1]) if i >= window_size-1 else 0 for i in range(len(episode_rewards))]

    episodes = np.arange(1, len(episode_rewards) + 1)

    plt.figure(figsize=(10, 6))
    # plt.plot(episodes, episode_rewards, label="Episode Rewards", color='b', alpha=0.6)
    plt.plot(episodes[window_size-1:], running_avg, label=f"Episode Average (Window = {window_size})", color='b', linewidth=2)
    plt.fill_between(episodes[window_size-1:], 
                     running_avg - running_std[window_size-1:], 
                     running_avg + running_std[window_size-1:], 
                     color='r', alpha=0.2, label="Standard Deviation")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid()

    plt.savefig("rewards_smoothed.png")



def main():
    """
    Main function to run the training and evaluation of DDPG on Inverted Double Pendulum.
    """
    # Create environment
    env = create_env()

    # Train DDPG with callback
    # model, tracker = train_ddpg(env, total_timesteps=50000)
    print("Training complete!")



    # Plot rewards
    print("Plotting rewards...")
    with open('episode_rewards.pkl', 'rb') as f:
        episode_rewards = pickle.load(f)
    plot_rewards(episode_rewards=episode_rewards)

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
