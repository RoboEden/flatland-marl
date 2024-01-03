import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDict

from typing import Tuple, List, Callable, Mapping, Optional, Any
from numpy.random.mtrand import RandomState

from flatland.envs.line_generators import SparseLineGen
from flatland.envs.malfunction_generators import (
    MalfunctionParameters,
    ParamMalfunctionGen,
)
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import SparseRailGen
from flatland.envs.step_utils.states import TrainState
from flatland.utils.rendertools import AgentRenderVariant, RenderTool
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.core.transition_map import GridTransitionMap
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.timetable_utils import Line
from flatland.envs.agent_utils import EnvAgent

import numpy as np
import pandas as pd
import sys

from IPython.display import clear_output
from matplotlib import pyplot as plt

from flatland_cutils import TreeObsForRailEnv as TreeCutils
from impl_config import FeatureParserConfig as fp
from solution.nn.net_tree_torchrl import embedding_net, actor_net, critic_net

from torchrl.envs.common import EnvBase
from torchrl.envs.gym_like import GymLikeEnv 
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec, UnboundedDiscreteTensorSpec
import torch

from tensordict.tensordict import TensorDict
from tensordict.nn import InteractionType
from torchrl.envs import ParallelEnv, SerialEnv
from torchrl.modules import MultiAgentMLP, ProbabilisticActor
from tensordict.nn import ProbabilisticTensorDictSequential

from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torch.nn import MSELoss, SmoothL1Loss

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

torch.manual_seed(0)
from matplotlib import pyplot as plt
from tqdm import tqdm

from flatland_ppo_training_torchrl import td_torch_rail, RailEnvTd

from solution.utils import VideoWriter
from PIL import Image

import os
import shutil

import time

import re

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--rollout-name", type=str, default=None)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--num-agents", type= int, default = 1,
        help="number of agents in the environment")
    parser.add_argument("--rollout-steps", type=int, default=1000, #for rollout
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument('--max-episode-steps', type=int, default=None)
    parser.add_argument("--pretrained-network-path", type=str, default = None,
        help="path to the pretrained network to be used")
    parser.add_argument("--do-render", action='store_true')
    parser.add_argument('--map-width', type=int, default=30)
    parser.add_argument('--map-height', type=int, default=35)
    parser.add_argument(
        "--fps", type=float, default=2, help="frames per second (default 10)"
    )
    parser.add_argument("--render", action=argparse.BooleanOptionalAction)
    parser.add_argument("--video-path", type=str, default="videos/")
    args = parser.parse_args()
    # fmt: on
    return args
    
class td_torch_rail_render(td_torch_rail):
    def __init__(self, env, saving_directory):
        super().__init__(env)        
        self.env = env
        self.num_agents = env.get_num_agents()
        self._make_spec()
        self.env_renderer = RenderTool(
            self.env,
            agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
            show_debug=False,
            screen_height=600,  # Adjust these parameters to fit your resolution
            screen_width=800,        )
        self.step_nr = 0
        self.saving_directory = saving_directory
        
    def _reset(self, tensordict = None):
        self.env_renderer.reset()
        return self.env.reset()

    def _step(self, tensordict):
        self.env_renderer.render_env(show=True, show_observations=False, show_predictions=True)
        image_rbg = self.env.render(show_predictions=True, screen_height = self.env.height*40, screen_width=self.env.width*40)
        image = Image.fromarray(image_rbg.astype('uint8')).convert('RGB')
        image.save(self.saving_directory + "rollout_video_" + str(self.step_nr) + ".jpg")
        
        self.step_nr += 1
        return self.env.step(tensordict)

if __name__ == "__main__":
    print('starting main')
    os.system('poetry env info')
    print(f'check if in venv: {sys.prefix == sys.base_prefix}')
    print(sys.argv)
    args = parse_args()
    
    saving_directory = "videos/"
    temp_directory = saving_directory + "temp/"
    os.mkdir(temp_directory)
    if args.pretrained_network_path is not None:
        pretrained_network_name = re.sub(".tar", "", args.pretrained_network_path)
        pretrained_network_name = re.sub(".*/", "", pretrained_network_name)
    else:
        pretrained_network_name = ""
    
    if args.rollout_name is None:
        rollout_name = pretrained_network_name + "_rollout_" + str(time.time())
    else:
        rollout_name = pretrained_network_name + args.rollout_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    def make_env():
        td_env = RailEnvTd(
            number_of_agents=args.num_agents,
            width = args.map_width,
            height = args.map_height,
            rail_generator=SparseRailGen(
                max_num_cities=3,
                grid_mode=False,
                max_rails_between_cities=2,
                max_rail_pairs_in_city=2,
            ),
            line_generator=SparseLineGen(
                speed_ratio_map={1.0: 1}
            ),
            malfunction_generator=ParamMalfunctionGen(
                MalfunctionParameters(
                    malfunction_rate=1 / 4500, min_duration=20, max_duration=50)
            ),
            obs_builder_object=TreeCutils(31, 500),
        )
        td_env.reset()
        if args.do_render:
            return td_torch_rail_render(td_env, temp_directory)
        else:
            return td_torch_rail(td_env)

    env = ParallelEnv(1, make_env)
    
    print('set up envs')

    embedding_net = embedding_net()
    common_module = TensorDictModule(
        embedding_net,
        in_keys=[('agents', 'observation')],
        out_keys=[('hidden', 'embedding'), ('hidden', 'att_embedding')],
    )
    
    actor_net = actor_net()
    actor_module = TensorDictModule(
        actor_net,
        in_keys=[('hidden', 'embedding'), ('hidden', 'att_embedding'), ('agents', 'observation', 'valid_actions')],
        out_keys=[('agents', 'logits')]
    )
    
    policy = ProbabilisticActor(
        module = actor_module,
        in_keys = ('agents', 'logits'),
        out_keys = [('agents', 'action')],
        distribution_class = torch.distributions.categorical.Categorical,
        return_log_prob=True,
        log_prob_key=('agents', 'sample_log_prob'),
        cache_dist=True,
        default_interaction_type=InteractionType.RANDOM
    )

    from torchrl.modules import ValueOperator

    critic_net = critic_net()
    critic_net_module = TensorDictModule(
        critic_net,
        in_keys=[('hidden', 'embedding'), ('hidden', 'att_embedding')],
        out_keys=['state_value']
    )
    
    critic_module = ValueOperator(
        critic_net,
        in_keys=[('hidden', 'embedding'), ('hidden', 'att_embedding')],
        out_keys=['state_value']
    )
    from torchrl.modules import ValueOperator
    
    from torchrl.modules import ActorValueOperator
    
    model = ActorValueOperator(
        common_module,
        policy,
        critic_module
    ).to('cuda')
    
    print(f'device of critic module: {critic_module.device}')
    
    if args.pretrained_network_path is not None:
        model_path = args.pretrained_network_path
        assert (model_path.endswith('tar')), 'Network format not known.'
        checkpoint = torch.load(model_path)
        print(f'type of state dict: {type(checkpoint["model_state_dict"])}')
        #print([key for key, _ in checkpoint['model_state_dict']])
        model.load_state_dict(checkpoint['model_state_dict'])
        print('loaded pretrained model from .tar file') 
        
    collector = SyncDataCollector(
        env,
        model,
        device='cpu',
        storing_device='cpu',
        frames_per_batch=args.rollout_steps,
        total_frames=args.rollout_steps,
    )
    start_time = time.time()
    start_rollout = time.time()
    
    results_td = collector.rollout()
    
    os.system("ffmpeg -r 2 -i " + temp_directory + "/rollout_video_%01d.jpg -y " + saving_directory + rollout_name + "_video.mp4")
    shutil.rmtree(temp_directory)
    final_steps = results_td[("next", "done")].squeeze()
    final_reward = results_td[("next", "agents", "reward")].squeeze()[torch.roll(final_steps, 1, -1)].sum(-1).mean()
    print(f'arrival ratio: {results_td[("stats", "arrival_ratio")].squeeze()[torch.roll(final_steps, 1, -1)].mean()}')
    print(f'deadlock ratio: {results_td[("stats", "deadlock_ratio")].squeeze()[torch.roll(final_steps, 1, -1)].mean()}')
    print(f'average total final reward: {final_reward}')