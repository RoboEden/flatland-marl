# https://flatland.aicrowd.com/challenges/flatland3/envconfig.html
import os
from dataclasses import dataclass

import pandas as pd
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import (
    MalfunctionParameters,
    ParamMalfunctionGen,
)
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from tqdm import tqdm

if __name__ == "__main__":
    eval_list = [
        "n_agents",
        "x_dim",
        "y_dim",
        "n_cities",
        "max_rail_pairs_in_city",
        "n_envs_run",
        "grid_mode",
        "max_rails_between_cities",
        "malfunction_duration_min",
        "malfunction_duration_max",
        "malfunction_interval",
        "speed_ratios",
    ]
    PATH = os.path.dirname(os.path.abspath(__file__))
    parameters_flatland = pd.read_csv(PATH + "/parameters_flatland_round_2_new.csv")
    parameters_flatland[eval_list] = parameters_flatland[eval_list].applymap(
        lambda x: eval(str(x))
    )

    for idx, env_config in tqdm(
        parameters_flatland.iterrows(), total=parameters_flatland.shape[0]
    ):
        env_config = env_config.to_dict()
        if not os.path.exists(os.path.join(PATH, env_config["test_id"])):
            os.mkdir(os.path.join(PATH, env_config["test_id"]))

        malfunction_parameters = MalfunctionParameters(
            malfunction_rate=1 / env_config["malfunction_interval"],
            min_duration=env_config["malfunction_duration_min"],
            max_duration=env_config["malfunction_duration_max"],
        )

        env = RailEnv(
            width=env_config["x_dim"],
            height=env_config["y_dim"],
            rail_generator=sparse_rail_generator(
                max_num_cities=env_config["n_cities"],
                grid_mode=env_config["grid_mode"],
                max_rails_between_cities=env_config["max_rails_between_cities"],
                max_rail_pairs_in_city=env_config["max_rail_pairs_in_city"],
            ),
            line_generator=sparse_line_generator(env_config["speed_ratios"]),
            number_of_agents=env_config["n_agents"],
            malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
            random_seed=env_config["random_seed"],
        )
        env.reset()
        Level_idx = env_config["env_id"]
        RailEnvPersister.save(
            env,
            os.path.join(PATH, env_config["test_id"], f"{Level_idx}.pkl"),
            save_distance_maps=True,
        )
