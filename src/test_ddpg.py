from src.agent import DDPG
from src.modified_ddpg import AdaptiveDDPG
from src.utils.noise import OUNoise, NormalNoise
from src.utils.metrictracker import MetricsTracker
from src.utils.metricktrackercallback import MetricsTrackerCallback
from src.main import create_env
from loguru import logger


def test_ddpg(model_class, tracker, *args, **kwargs):
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'agent_str'}
    num_runs = 2
    
    metrics_callback = MetricsTrackerCallback(tracker, verbose=0, agent_id=kwargs.get('agent_str', 'DDPG'))
    try:
        for run in range(num_runs):
            env = create_env()
            action_space = env.action_space

            # Create a noise object
            noise = OUNoise(action_space=action_space, sigma=0.05)

            model = model_class(discount_factor=0.99,
                                action_space=env.action_space,
                                hidden_size=[400, 300],
                                input_size=env.observation_space.shape[0],
                                batch_size=64,
                                tau=0.005,
                                critic_lr=1e-3,
                                actor_lr=1e-4,
                                buffer_size=1e6,
                                callback=[metrics_callback],
                                update_interval=(1, "step"),
                                learning_starts=25000,
                                log_dir=f"./runs/{kwargs.get('agent_str', 'DDPG')}_run_{run}",
                                verbose=True,
                                *args, 
                                **filtered_kwargs)
            logger.info(f"Starting run {run + 1}/{num_runs} - {kwargs.get('agent_str', 'DDPG')}")
            model.train(env, num_episodes=2000, noise=noise, max_steps=50)  # Max possible steps in reacher is 50
            logger.info(f"Run {run + 1} - {kwargs.get('agent_str', 'DDPG')} complete!")
            env.close()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        tracker.plot_metric("return", "return_test.png")

    finally:
        logger.info("Saving metrics")
        tracker.plot_metric("return", "return_test.png")
        tracker.save_metrics(metric_name="return", file_name="metrics_test.pkl")

def main():
    vanilla_model = DDPG
    modified_model = AdaptiveDDPG
    tracker = MetricsTracker()

    models = [
        (modified_model, {'alpha': 0.5, 'beta': 0.5, 'tau_min': 0.005, 'tau_max': 0.05, 'agent_str': "AdaptiveDDPG"}),
        (vanilla_model, {'agent_str': "DDPG"}),
    ]
    try:

        for model, kwargs in models:
            test_ddpg(model, tracker, **kwargs)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()