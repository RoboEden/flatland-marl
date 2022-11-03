from argparse import ArgumentParser
from time import sleep

from flatland.envs.line_generators import SparseLineGen
from flatland.envs.malfunction_generators import (
    MalfunctionParameters,
    ParamMalfunctionGen,
)
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import SparseRailGen
from flatland_cutils import TreeObsForRailEnv as TreeCutils

from eval_env import LocalTestEnvWrapper
from impl_config import FeatureParserConfig as fp
from plfActor import Actor
from utils import VideoWriter, debug_show


def create_random_env():
    return RailEnv(
        number_of_agents=50,
        width=30,
        height=35,
        rail_generator=SparseRailGen(
            max_num_cities=3,
            grid_mode=False,
            max_rails_between_cities=2,
            max_rail_pairs_in_city=2,
        ),
        line_generator=SparseLineGen(
            speed_ratio_map={1.0: 1 / 4, 0.5: 1 / 4, 0.33: 1 / 4, 0.25: 1 / 4}
        ),
        malfunction_generator=ParamMalfunctionGen(
            MalfunctionParameters(
                malfunction_rate=1 / 4500, min_duration=20, max_duration=50
            )
        ),
        obs_builder_object=TreeCutils(fp.num_tree_obs_nodes, fp.tree_pred_path_depth),
    )


def get_model_path(n_agents):
    if n_agents <= 50:
        model_path = "policy/phase-III-50.pt"
    elif n_agents <= 80:
        model_path = "policy/phase-III-80.pt"
    elif n_agents <= 100:
        model_path = "policy/phase-III-100.pt"
    else:
        model_path = "policy/phase-III-200.pt"
    return model_path


def get_args():
    parser = ArgumentParser(
        description="A multi-agnet reinforcement learning solution to flatland3."
    )
    parser.add_argument(
        "--nr",
        "--no-render",
        dest="render",
        action="store_const",
        const=False,
        default=True,
        help="do not display game window",
    )
    parser.add_argument(
        "--fps", type=float, default=30, help="frames per second (default 10)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="the checkpoint file of saved model. If not given, a proper model is chosen according to number of agents.",
    )
    parser.add_argument(
        "--env", default=None, help="path to saved '*.pkl' file of envs"
    )
    parser.add_argument("--save-video", "-s", default=None, help="path to save video")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # create env
    if args.env is None:
        env = create_random_env()
    else:
        env, _ = RailEnvPersister.load_new(args.env)
        env.obs_builder = TreeCutils(fp.num_tree_obs_nodes, fp.tree_pred_path_depth)
    env_wrapper = LocalTestEnvWrapper(env)
    n_agents = env_wrapper.env.number_of_agents

    # load actor
    if args.model is None:
        model_path = get_model_path(n_agents)
    else:
        model_path = args.model
    actor = Actor(model_path)
    print(f"Load actor from {model_path}")

    # create video writer
    if args.save_video is not None:
        video_writer = VideoWriter(args.save_video, args.fps)

    # start step loop
    obs = env_wrapper.reset()
    while True:
        va = env_wrapper.get_valid_actions()
        action = actor.get_actions(obs, va, n_agents)
        obs, all_rewards, done = env_wrapper.step(action)

        if args.render:
            debug_show(env_wrapper.env)
            sleep(1 / args.fps)

        if args.save_video is not None:
            frame = debug_show(env_wrapper.env, mode="rgb_array")
            video_writer.write(frame)

        if done["__all__"]:
            if args.save_video is not None:
                video_writer.close()
                print(f"Write video to {args.save_video}")

            arrival_ratio, total_reward, norm_reward = env_wrapper.final_metric()
            print(f"TOTAL_REW: {total_reward}")
            print(f"NORM_REW: {norm_reward:.4f}")
            print(f"ARR_RATIO: {arrival_ratio*100:.2f}%")
            break
