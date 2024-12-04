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


def create_env(env_name="InvertedPendulum-v5"):
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

    noise_std = 0.13
    noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(env.action_space.shape),
        sigma=noise_std * np.ones(env.action_space.shape)
    )
    model = DDPG(
        "MlpPolicy", 
        env, 
        verbose=0,
        learning_rate=1e-3,
        buffer_size=int(1e6),
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        action_noise=noise,
        seed=10
    )

    eval_env = create_env()
    
    # Create the callback
    metrics_callback = MetricsTrackerCallback(eval_env=eval_env, verbose=1)

    # Train the model with the callback
    model.learn(total_timesteps=total_timesteps, callback=metrics_callback)
    model.save(model_path)
    
    return model, metrics_callback

def train_multiple_runs(env_name="InvertedPendulum-v5", total_timesteps=50000, num_runs=5, model_path_prefix="ddpg_run"):
    """
    Train the DDPG agent multiple times and return a list of episode rewards across runs.
    
    :param env_name: Name of the Gym environment.
    :param total_timesteps: Number of timesteps to train for each run.
    :param num_runs: Number of independent training runs.
    :param model_path_prefix: Prefix for saving models from each run.
    :return: List of lists containing episode rewards for each run.
    """
    episode_rewards_list = []

    for run in range(num_runs):
        print(f"Starting run {run + 1}/{num_runs}...")
        env = create_env(env_name)
        model_path = f"{model_path_prefix}_{run}"

        model, metrics_callback = train_ddpg(env, total_timesteps=total_timesteps, model_path=model_path)
        print(f"Run {run + 1} complete!")

        # Collect rewards for this run
        rewards = metrics_callback.get_tracked_metrics()['episode_rewards']
        episode_rewards_list.append(rewards)

        # Cleanup
        env.close()

    return episode_rewards_list


def plot_rewards(metrics_tracker=None, window_size=10, title="Rewards over Episodes (DDPG on Inverted Pendulum)", episode_rewards_list=None):
    if metrics_tracker is not None:
        episode_rewards_list = metrics_tracker.get_tracked_metrics()['episode_rewards']
        with open('episode_rewards.pkl', 'wb') as f:
            pickle.dump(episode_rewards_list, f)
 
    # Convert list of episode rewards to a numpy array
    episode_rewards_array = np.array(episode_rewards_list)
    
    # Calculate the mean and standard deviation across runs
    mean_rewards = np.mean(episode_rewards_array, axis=0)
    std_rewards = np.std(episode_rewards_array, axis=0)
    
    # Calculate the running average (moving average)
    running_avg = np.convolve(mean_rewards, np.ones(window_size) / window_size, mode='valid')
    
    # Calculate the standard deviation for the same window size
    running_std = [np.std(mean_rewards[i-window_size+1:i+1]) if i >= window_size-1 else 0 for i in range(len(mean_rewards))]

    episodes = np.arange(1, len(mean_rewards) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes[window_size-1:], running_avg, label=f"Episode Average (Window = {window_size})", color='b', linewidth=2)
    plt.fill_between(episodes[window_size-1:], 
                    running_avg - std_rewards[window_size-1:], 
                    running_avg + std_rewards[window_size-1:], 
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
    
    noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape), sigma=noise_std * np.ones(env.action_space.shape))
    
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
        seed=10
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
    env_name = "InvertedPendulum-v5"
    num_runs = 3
    total_timesteps = 100000

    # Perform multiple runs
    episode_rewards_list = train_multiple_runs(env_name=env_name, total_timesteps=total_timesteps, num_runs=num_runs)
    print("Training complete for all runs!")

    # Save episode rewards for analysis
    with open('episode_rewards_multiple_runs.pkl', 'wb') as f:
        pickle.dump(episode_rewards_list, f)

    # Plot rewards
    plot_rewards(episode_rewards_list=episode_rewards_list)

if __name__ == "__main__":
    main()
