from src.agent import DDPG
from src.modified_ddpg import AdaptiveDDPG
from src.utils.noise import OUNoise, NormalNoise
from src.utils.metrictracker import MetricsTracker
from src.utils.metricktrackercallback import MetricsTrackerCallback
from src.main import create_env
from loguru import logger
import optuna
import json


def test_ddpg(model_class, tracker, learning_starts, update_factor, batch_size, opt=False, sigma=0.05, *args, **kwargs):
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'agent_str'}
    num_runs = 1

    metrics_callback = MetricsTrackerCallback(
        tracker, verbose=0, agent_id=kwargs.get('agent_str', 'DDPG'))
    try:
        for run in range(num_runs):
            env = create_env()
            action_space = env.action_space

            # Create a noise object
            noise = OUNoise(action_space=action_space, sigma=sigma)

            model = model_class(discount_factor=0.99,
                                action_space=env.action_space,
                                hidden_size=[400, 300],
                                input_size=env.observation_space.shape[0],
                                batch_size=batch_size,
                                tau=0.005,
                                critic_lr=1e-3,
                                actor_lr=1e-4,
                                buffer_size=1e6,
                                callback=[metrics_callback],
                                update_interval=(update_factor, 'step'),
                                learning_starts=learning_starts,
                                log_dir=f"./runs/tune/{kwargs.get('agent_str', 'DDPG')}_run_{run}",
                                verbose=True,
                                *args,
                                **filtered_kwargs)
            logger.info(
                f"Starting run {run + 1}/{num_runs} - {kwargs.get('agent_str', 'DDPG')}")
            # Max possible steps in reacher is 50
            model.train(env, num_episodes=2000, noise=noise, max_steps=50)
            logger.info(
                f"Run {run + 1} - {kwargs.get('agent_str', 'DDPG')} complete!")
            env.close()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        tracker.plot_metric("return", "return_test.png")

    if opt:
        return tracker.get_avg_return(agent_id=kwargs.get('agent_str', 'DDPG'), last_n_episodes=1500)


def compare_algs():
    vanilla_model = DDPG
    modified_model = AdaptiveDDPG
    tracker = MetricsTracker()

    models = [
        (modified_model, {'alpha': 0.5, 'beta': 0.5, 'tau_min': 0.005,
         'tau_max': 0.5, 'agent_str': "AdaptiveDDPG"}),
        (vanilla_model, {'agent_str': "DDPG"}),
    ]
    try:
        for model, kwargs in models:
            test_ddpg(model, tracker, **kwargs)
    except Exception as e:
        print(e)
    finally:
        logger.info("Saving metrics")
        tracker.plot_metric("return", "return_test.png")
        tracker.save_metrics(metric_name="return",
                             file_name="metrics_test.pkl")


def objective(trial, tracker, *args, **kwargs):
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    beta = trial.suggest_float("beta", 0.0, 1.0)
    tau_min = trial.suggest_float("tau_min", 0.001, 0.01)
    tau_max = trial.suggest_float("tau_max", tau_min, 1.0)
    sigma = trial.suggest_float("sigma", 0.001, 0.8)
    learning_starts = trial.suggest_int("learning_starts", 1000, 50000)
    update_factor = trial.suggest_int("update_factor", 1, 4)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)

    agent_str = f"AdaptiveDDPG_{trial.number}"
    avg_return = test_ddpg(AdaptiveDDPG, tracker, alpha=alpha, beta=beta, tau_min=tau_min,
                           tau_max=tau_max, agent_str=agent_str, sigma=sigma, opt=True, update_factor=update_factor, learning_starts=learning_starts, batch_size=batch_size)
    return avg_return


def optimize():
    tracker = MetricsTracker()
    try:
        study = optuna.create_study(
            direction="maximize", study_name="DDPG + Mod Tuning", storage="sqlite:///tune.db", load_if_exists=True)
        study.optimize(lambda trial: objective(trial, tracker),
                       n_trials=100, show_progress_bar=True)
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user")
    finally:
        logger.info("Saving metrics")
        tracker.save_metrics(metric_name="return",
                             file_name="metrics_test.pkl")
        tracker.plot_metric("return", "return_test.png")
        with open("optuna_results.json", "w") as f:
            json.dump(study.best_params, f)


def main():
    # compare_algs()
    optimize()


if __name__ == "__main__":
    main()
