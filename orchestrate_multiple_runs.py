import optuna
import os

from cleanrl_utils.tuner import Tuner


tuner = Tuner(
    script="flatland_ppo_training_workshop.py",
    metric="rewards/mean",
    metric_last_n_average_window=50,
    direction="maximize",
    aggregation_type="average",
    target_scores={
        "flatland": None,
    },
    params_fn=lambda trial: {
        "learning-rate": trial.suggest_float("learning-rate", 2.5e-6, 2.5e-1, log=True),
        "num-minibatches": trial.suggest_categorical("num-minibatches", [1, 10, 100]),
        "update-epochs": trial.suggest_categorical("update-epochs", [1, 2, 4, 8, 16]),
        "num-steps": trial.suggest_categorical("num-steps", [100, 500, 1000, 2000]),
        "vf-coef": trial.suggest_float("vf-coef", 0, 1),
        "max-grad-norm": trial.suggest_float("max-grad-norm", 0, 5),
        "total-timesteps": 100000,
        "num-envs": 1,
        "num-agents": 1
    },
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    sampler=optuna.samplers.TPESampler(),
)
os.system('poetry env info')
tuner.tune(
    num_trials=100,
    num_seeds=3,
)