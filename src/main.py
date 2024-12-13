import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
# from src.utils.tracker_callback import MetricsTrackerCallback  # Make sure the callback is correctly imported
import pickle
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import optuna
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
from src.utils.metrictracker import MetricsTracker
from src.utils.metricktrackercallback import MetricsTrackerCallback


def create_env(env_name="Reacher-v5", seed=None):
    """
    Creates and returns a monitored Gym environment.
    """
    env = gym.make(env_name)
    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    env = Monitor(env, filename='data.csv')  # Monitor for logging rewards
    return env


def train_ddpg(env, tracker=None, total_timesteps=50000, model_path="ddpg_inverted_pendulum", seed=10):
    """ 
    Trains a DDPG model on the provided environment.
    :param env: The Gym environment.
    :param total_timesteps: Number of timesteps to train.
    :param model_path: Path to save the trained model.
    :return: Trained DDPG model.
    """

    noise_std = 0.1
    noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(env.action_space.shape),
        sigma=noise_std * np.ones(env.action_space.shape)
    )
    model = DDPG(
        "MlpPolicy", 
        env, 
        verbose=0,
        learning_rate=1e-4,
        buffer_size=int(1e6),
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        action_noise=noise,
        seed=seed,
        policy_kwargs=dict(net_arch=[400,300]),
        learning_starts=1000
    )

    # Create the callback
    metrics_callback = MetricsTrackerCallback(tracker)

    # Train the model with the callback
    model.learn(total_timesteps=total_timesteps, callback=metrics_callback)
    model.save(model_path)
    
    return model, metrics_callback


def train_multiple_runs(env_name="Reacher-v5", total_timesteps=50000, num_runs=5, model_path_prefix="ddpg_run", tracker=None):
    """
    Train the DDPG agent multiple times and return a list of episode rewards across runs.
    
    :param env_name: Name of the Gym environment.
    :param total_timesteps: Number of timesteps to train for each run. 50,000 by default.
    :param num_runs: Number of independent training runs. 5 by default.
    :param model_path_prefix: Prefix for saving models from each run.
    :return: List of lists containing episode rewards for each run.
    """


    for run in range(num_runs):
        print(f"Starting run {run + 1}/{num_runs}...")
        env = create_env(env_name)
        model_path = f"{model_path_prefix}_{run}"

        model, metrics_callback = train_ddpg(env, total_timesteps=total_timesteps, model_path=model_path, tracker=tracker, seed=run+10)
        print(f"Run {run + 1} complete!")

        # Cleanup
        env.close()

    tracker.plot_metric("return", "target_return.png")




def objective(trial, tracker=None):
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
        verbose=0,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        action_noise=noise,
        seed=10
    )
    
    metrics_callback = MetricsTrackerCallback(tracker)
    
    model.learn(total_timesteps=100000, callback=[metrics_callback])
    
    mean_reward = tracker.get_avg_return()
    print(f"Mean reward: {mean_reward}")
    return mean_reward


def optimize_ddpg(tracker):
    """
    Optimize DDPG hyperparameters using Optuna.
    """
    study = optuna.create_study(direction='maximize', study_name='ddpg_inverted_pendulum', storage='sqlite:///ddpg_inverted_pendulum.db', load_if_exists=True)
    study.optimize(lambda trial: objective(trial, tracker), n_trials=100)
    
    print("Best hyperparameters: ", study.best_params)
    return study.best_params


def main():
    """
    Main function to run the training and evaluation of DDPG on Inverted Pendulum.
    """
    # env_name = "Reacher-v5"
    # num_runs = 5
    # total_timesteps = 1000

    tracker = MetricsTracker()

    optimize_ddpg(tracker)

    # Perform multiple runs
    # train_multiple_runs(env_name=env_name, total_timesteps=total_timesteps, num_runs=num_runs, tracker=tracker)
    # print("Training complete for all runs!")


if __name__ == "__main__":
    main()
