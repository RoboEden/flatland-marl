import optuna
import os

from cleanrl_utils.tuner import Tuner


tuner = Tuner(
    script="flatland_ppo_training_torchrl.py",
    metric="stats/arrival_ratio",
    metric_last_n_average_window=50,
    direction="maximize",
    aggregation_type="average",
    target_scores={
        "flatland": None,
    },
    params_fn=lambda trial: {
        "learning-rate": trial.suggest_float("learning-rate", 2.5e-6, 2.5e-3, log=True),
        "num-minibatches": trial.suggest_categorical("num-minibatches", [1, 10, 100]),
        "update-epochs": trial.suggest_categorical("update-epochs", [1, 2, 4, 8]),
        "num-steps": trial.suggest_categorical("num-steps", [50, 100, 200, 500, 1000]),
        "vf-coef": 0.01,
        "ent-coef": 0.01,
        "clip-coef": trial.suggest_float("clip-coef", 0, 0.5),
        "max-grad-norm": trial.suggest_float("max-grad-norm", 0, 1),
        "total-timesteps": 100000,
        "num-envs": 8,
        "num-agents": 2,
        "map-height": 25,
        "map-width": 25,
        "arrival-reward-coef": 5,
        "delay-reward-coef": 0,
        "shortest-path-reward-coef": 0,
        "departure-reward-coef": 0,
        "deadlock-penalty-coef": 5,
        "arrival-delay-penalty-coef": 0,
    },
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    sampler=optuna.samplers.TPESampler(),
    study_name="smaller_sample_space",
)
os.system("poetry env info")
tuner.tune(
    num_trials=100,
    num_seeds=3,
)
