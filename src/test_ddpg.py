from src.agent import DDPG
from src.utils.noise import OUNoise, NormalNoise
from src.utils.metrictracker import MetricsTracker
from src.utils.metricktrackercallback import MetricsTrackerCallback
import gymnasium as gym


def create_env(env_name="InvertedPendulum-v5", seed=None):
    """
    Creates and returns a monitored Gym environment.
    """
    env = gym.make(env_name)
    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def main():
    env = create_env()
    action_space = env.action_space
    num_runs = 5

    # Create a noise object
    noise = OUNoise(action_space=action_space, sigma=0.10)
    tracker = MetricsTracker()
    metrics_callback = MetricsTrackerCallback(tracker, verbose=1)

    model = DDPG(discount_factor=0.99,
                action_space=env.action_space,
                hidden_size=[400, 300],
                input_size=env.observation_space.shape[0],
                tau=0.005,
                critic_lr=0.0005,
                actor_lr=0.0002,
                buffer_size=100000,
                callback=[metrics_callback],
                )


    for run in range(num_runs):
        print(f"Starting run {run + 1}/{num_runs}...")
        model.train(env, num_episodes=1000, batch_size=64, noise=noise, max_steps=1000)
        print(f"Run {run + 1} complete!")
        # Cleanup
        env.close()
    
    tracker.plot_metric("return", "return_test.png")


if __name__ == "__main__":
    main()