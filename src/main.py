import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from src.utils.tracker_callback import MetricsTrackerCallback  # Make sure the callback is correctly imported
import pickle
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import optuna
from stable_baselines3.common.evaluation import evaluate_policy


def create_env(env_name="Inverted-Pendulum-v5"):
    """
    Creates and returns a monitored Gym environment.
    """
    env = gym.make(env_name)
    env = Monitor(env, filename='data.csv')  # Monitor for logging rewards
    return env


def train_ddpg(env, total_timesteps=50000, model_path="ddpg_inverted_pendulum"):
    """ 
    Trains a DDPG model on the provided environment.
    :param env: The Gym environment.
    :param total_timesteps: Number of timesteps to train.
    :param model_path: Path to save the trained model.
    :return: Trained DDPG model.
    """

    noise_std = 0.2
    noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(env.action_space.shape),
        sigma=noise_std * np.ones(env.action_space.shape)
    )
    model = DDPG(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0005,
        buffer_size=100000,
        batch_size=32,
        tau=0.005,
        gamma=0.99,
        action_noise=noise,
        seed=10
    )

    eval_env = create_env()
    
    # Create the callback
    metrics_callback = MetricsTrackerCallback(eval_env=eval_env, verbose=0)

    # Train the model with the callback
    model.learn(total_timesteps=total_timesteps, callback=metrics_callback)
    model.save(model_path)
    
    return model, metrics_callback


def plot_rewards(metrics_tracker=None, window_size=10, title="Rewards over Episodes (DDPG on Pendulum)", episode_rewards=None):
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



def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning.
    """
    env = create_env()
    
    # Define the hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    buffer_size = trial.suggest_int('buffer_size', 10000, 100000)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    tau = trial.suggest_float('tau', 0.001, 0.01, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.999, log=True)
    noise_std = trial.suggest_float('noise_std', 0.1, 0.3, log=True)
    
    noise = NormalActionNoise(mean=np.zeros(env.action_space.shape), sigma=noise_std * np.ones(env.action_space.shape))
    
    model = DDPG(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        action_noise=noise,
    )
    
    eval_env = create_env()

    metrics_callback = MetricsTrackerCallback(eval_env=eval_env, verbose=0)
    
    model.learn(total_timesteps=50000, callback=[metrics_callback])
    
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    
    return mean_reward


def optimize_ddpg():
    """
    Optimize DDPG hyperparameters using Optuna.
    """
    study = optuna.create_study(direction='maximize', study_name='ddpg_inverted_pendulum', storage='sqlite:///ddpg_inverted_pendulum.db', load_if_exists=True)
    study.optimize(objective, n_trials=100)
    
    print("Best hyperparameters: ", study.best_params)
    return study.best_params


def main():
    """
    Main function to run the training and evaluation of DDPG on Inverted Pendulum.
    """

    best_params = optimize_ddpg()
    print("Optimization complete!")
    print(best_params)
    # Create environment
    # env = create_env()

    # Train DDPG with callback
    # # model, tracker = train_ddpg(env, total_timesteps=50000)
    # print("Training complete!")
    # file = open('episode_rewards.pkl', 'rb')
    # episode_rewards = pickle.load(file)


    # # Plot rewards
    # plot_rewards(metrics_tracker=None, episode_rewards=episode_rewards)

    # # # Close environment
    # env.close()


if __name__ == "__main__":
    main()
