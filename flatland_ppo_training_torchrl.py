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

from time import sleep

import json


import time

from flatland.envs.agent_chains import MotionCheck

n_iters = 10

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=False,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="flatland-rl",
        help="the id of the environment")
    parser.add_argument("--num-agents", type= int, default = 1,
        help="number of agents in the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000, #added zero for overnight
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=1000, #for rollout
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, # try constant learning rate
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=10,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.000000001,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.0000001,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.2,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--pretrained-network-path", type=str, default = "solution/policy/phase-III-50.pt",
        help="path to the pretrained network to be used")
    parser.add_argument("--use-pretrained-network", action=argparse.BooleanOptionalAction)
    parser.add_argument('--no-training', action='store_true')
    parser.add_argument('--use-left-right-map', action='store_true')
    parser.add_argument('--use-line-map', action=argparse.BooleanOptionalAction)
    parser.add_argument('--use-meet-map', action=argparse.BooleanOptionalAction)
    parser.add_argument("--do-render", action='store_true')
    parser.add_argument('--use-arrival-reward', action='store_true')
    parser.add_argument('--freeze-embeddings', action=argparse.BooleanOptionalAction)
    parser.add_argument("--delay-reward-coef", type=float, default=1,
        help="Toggles whether to use the default flatland env reward at each step")
    parser.add_argument("--shortest-path-reward-coef", type=float, default=0,
        help="Toggles whether to use the default flatland env reward at each step")
    parser.add_argument('--departure-reward-coef', type=float, default=0)
    parser.add_argument('--arrival-reward-coef', type=float, default=5)
    parser.add_argument('--deadlock-penalty-coef', type=float, default=0)
    parser.add_argument('--initialize-action-weights', action=argparse.BooleanOptionalAction)
    parser.add_argument('--max-episode-steps', type=int, default=None)
    parser.add_argument('--force-shortest-path', action=argparse.BooleanOptionalAction)
    parser.add_argument("--arrival-reward", type=int, default=1, 
        help="The reward to be returned when the train departs.")
    parser.add_argument('--stepwise', action=argparse.BooleanOptionalAction)
    parser.add_argument('--map-width', type=int, default=30)
    parser.add_argument('--map-height', type=int, default=35)
    parser.add_argument('--curriculum-path', type=str, default=None)
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    print(f'minibatch size: {args.minibatch_size}')
    print(f'batch size: {args.batch_size}')
    # fmt: on
    return args

class RailEnvTd(RailEnv):
    ''' Custom version of default flatland rail env that changes in- and outputs to tensordicts
    
    Methods:
        - obs_to_dt: Return the list of observations given by the envirnoment as a tensordict
        - reset: Extend the default flatland reset method by returning a tensordict
        - step: Extend the default flatland step by accepting and returning a tensordict
        - update_step_reward: Override default flatland update_step_reward, allowing for custom reward functions upon step completion
    '''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_reward_coef = 0
        self.shortest_path_reward_coef = 0
        self.arrival_reward_coef = 0
        self.deadlock_penalty_coef = 0
        self.departure_reward_coef = 0
        self.previous_deadlocked = None
    
    def obs_to_td(self, obs_list):
        ''' Return the observation as a tensordict.'''
        obs_td = TensorDict(
            {
                "agents_attr": torch.tensor(
                    obs_list[0], dtype=torch.float32
                ),
                "node_attr": torch.tensor( # change name from forest in demo plfActor.py
                    obs_list[1][0], dtype=torch.float32
                ),
                "adjacency": torch.tensor(
                    obs_list[1][1], dtype=torch.int64
                ),
                "node_order": torch.tensor(
                    obs_list[1][2], dtype=torch.int64
                ),
                "edge_order": torch.tensor(
                    obs_list[1][3], dtype=torch.int64
                ),
            },
            [self.get_num_agents()],
        )
        return obs_td

    def reset(self, tensordict=None, random_seed = None):
        ''' Extend default flatland reset by returning a tensordict. '''
        # get observation
        #print("entering reset env")
        observations, _ = super().reset(random_seed=random_seed)
        print(f'max episode steps: {self._max_episode_steps}')

        # use default for the moment, give enough time to train
        """ if args.max_episode_steps is not None:
            for agent_i, agent in enumerate(self.agents):
                agent.earliest_departure = 0
                agent.latest_arrival = args.max_episode_steps
            
            self._max_episode_steps = args.max_episode_steps """

        if tensordict is None:
            tensordict_out = TensorDict({}, batch_size=[])
            tensordict_out['agents'] = TensorDict({}, batch_size = [])
        tensordict_out['agents']["observation"] = self.obs_to_td(observations)
        # get valid actions
        (
            _,
            _,
            valid_actions,
        ) = self.obs_builder.get_properties()
        tensordict_out["agents"]["observation"]["valid_actions"] = torch.tensor(
            valid_actions, dtype=torch.bool
        ) 
        self.previous_deadlocked = self.motionCheck.svDeadlocked
        return tensordict_out

    def step(self, tensordict):
        '''Extend default flatland step by returning a tensordict. '''
        #print('entering step function')
        #print("shortest path: {}".format((self.agents[0].get_shortest_path(self.distance_map)[0])))

        actions = {
            handle: action.item()
            for handle, action in enumerate(tensordict['agents']["action"].flatten())
        }
        observations, rewards, done, _ = super().step(actions)
        #print(f'done: {done["__all__"]}')
        (
            _,
            _,
            valid_actions,
        ) = self.obs_builder.get_properties()
        return_td = TensorDict({'agents': TensorDict({}, []), 'stats': TensorDict({}, [])}, batch_size=[])
        #rewards_td = TensorDict({'agents': TensorDict({}, [])}, batch_size=[])
        return_td['agents']["observation"] = self.obs_to_td(observations)
        return_td['agents']["reward"] = torch.tensor(
            [value for _, value in rewards.items()], dtype = torch.float32
        )
        return_td["done"] = torch.tensor(done["__all__"]).type(torch.bool)
        return_td['agents']["observation"]["valid_actions"] = torch.tensor(
            valid_actions, dtype=torch.bool
        )
        n_arrival = 0
        for agent in self.agents:
            if agent.state == TrainState.DONE:
                n_arrival += 1
        return_td[("stats", "arrival_ratio")] = torch.tensor(n_arrival/self.number_of_agents, dtype = torch.float32)
        self.previous_deadlocked = self.motionCheck.svDeadlocked
        n_deadlocked_agents = len(self.previous_deadlocked)
        return_td[("stats", "deadlock_ratio")] = torch.tensor(n_deadlocked_agents/self.number_of_agents, dtype = torch.float32)
        #print(f'rewards: {return_td[("agents", "reward")]}')
        
        return return_td

    def update_step_rewards(self, i_agent):
        agent = self.agents[i_agent]
        delay_reward = shortest_path_reward = arrival_reward = deadlock_penalty = departure_reward = 0
        if self.delay_reward_coef != 0:        
            # we start giving the model rewards as soon as the agent can actually choose actions
            if (agent.earliest_departure <= self._elapsed_steps) and agent.state != TrainState.DONE:
                delay_reward = min(agent.get_current_delay(self._elapsed_steps, self.distance_map), 0) # only give reward if delayed
        
        if self.shortest_path_reward_coef != 0:
            if (agent.earliest_departure <= self._elapsed_steps) and agent.state != TrainState.DONE:
                shortest_path_reward = agent.get_current_delay(self._elapsed_steps, self.distance_map)
        
        if self.arrival_reward_coef != 0:
            if agent.state == TrainState.DONE and agent.state_machine.previous_state != TrainState.DONE: 
                # only give arrival reward once
                arrival_reward = 1

        if self.deadlock_penalty_coef != 0:
            if (agent.position in self.motionCheck.svDeadlocked) and (agent.position not in self.previous_deadlocked):
                deadlock_penalty = 1
            
        if self.departure_reward_coef != 0:
            if (agent.state.is_on_map_state() and agent.state_machine.previous_state.is_off_map_state()):
                departure_reward = 1            
        self.rewards_dict[i_agent] += (self.delay_reward_coef*delay_reward + 
                                       self.shortest_path_reward_coef*shortest_path_reward + 
                                       self.arrival_reward_coef*arrival_reward + 
                                       self.deadlock_penalty_coef*deadlock_penalty + 
                                       self.departure_reward_coef*departure_reward)
    
    def set_reward_coef(self, delay_reward_coef, shortest_path_reward_coef, arrival_reward_coef, deadlock_penalty_coef, departure_reward_coef):
        self.delay_reward_coef = delay_reward_coef
        self.shortest_path_reward_coef = shortest_path_reward_coef
        self.arrival_reward_coef = arrival_reward_coef
        self.deadlock_penalty_coef = deadlock_penalty_coef
        self.departure_reward_coef = departure_reward_coef
        
    
    def _handle_end_reward(self, agent: EnvAgent) -> int:
        if True:
            #print("end of episode status: {}".format(agent.state))
            return 0
        else:
            print(f'agent: {agent}')
            super()._handle_end_reward(self, agent)

class td_torch_rail(EnvBase):
    def __init__(self, env):
        super().__init__()        
        self.env = env
        self.num_agents = env.get_num_agents()
        #print(f'self num agents after init: {self.num_agents}')
        #print('initializing torchrl env')
        #batch_size = torch.Size([1])
        #print('batch size: {}'.format(batch_size))
        self._make_spec()
        """         self.env_renderer = RenderTool(
            self.env,
            agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
            show_debug=False,
            screen_height=600,  # Adjust these parameters to fit your resolution
            screen_width=800,
        ) """
        
    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng
        
    def _make_spec(self):
        print('self.self.num_agents: {}'.format(self.num_agents))
        self.observation_spec = CompositeSpec(
            agents = CompositeSpec(
                observation = CompositeSpec(
                    agents_attr = UnboundedContinuousTensorSpec(
                        shape = [self.num_agents, 83],
                        dtype = torch.float32
                    ),
                    adjacency = UnboundedDiscreteTensorSpec(
                        shape = [self.num_agents, 30, 3],
                        dtype = torch.int64
                    ),
                    node_attr = UnboundedDiscreteTensorSpec(
                        shape = [self.num_agents, 31, 12],
                        dtype = torch.float32
                    ),
                    node_order = UnboundedDiscreteTensorSpec(
                        shape = [self.num_agents, 31],
                        dtype = torch.int64
                    ),
                    edge_order = UnboundedDiscreteTensorSpec(
                        shape = [self.num_agents, 30],
                        dtype = torch.int64
                    ),
                    valid_actions = DiscreteTensorSpec(
                        n = 2, 
                        dtype = torch.bool, 
                        shape = [self.num_agents, 5]),
                shape = [self.num_agents]),
            shape = []
            ),
            stats = CompositeSpec(
                arrival_ratio = UnboundedContinuousTensorSpec(shape = [], dtype = torch.float32),
                deadlock_ratio = UnboundedContinuousTensorSpec(shape = [], dtype = torch.float32),
                shape = []
            ),
        shape = []
        )
        self.action_spec = CompositeSpec(
            agents = CompositeSpec(
                action = DiscreteTensorSpec(
                    n = 5, 
                    shape = [self.num_agents],
                    dtype = torch.int64),
                shape = []),
            shape = []
            )
        
        self.reward_spec = CompositeSpec(
            agents = CompositeSpec(
                reward = UnboundedContinuousTensorSpec(
                    shape = [self.num_agents], 
                    dtype = torch.float32),
                shape = []
            ),
            shape = []
            )
        
        self.done_spec = DiscreteTensorSpec(
            n = 2,
            dtype = torch.bool, 
            shape = [1]
            )


        
    def _reset(self, tensordict = None):
        #print('resetting with _reset function')
        #self.env_renderer.reset()
        return self.env.reset()
    
    def _step(self, tensordict):
        #print('actions: {}'.format(tensordict['action']))
        #actions_dict = {handle: action.item() for handle, action in enumerate(tensordict['action'])}
        #self.env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
        return self.env.step(tensordict)
    

transitions = RailEnvTransitions()
cells = transitions.transition_list
 
empty = cells[0]
 
vertical_straight = cells[1]
horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
 
left_switch_from_south = cells[2]
left_switch_from_west = transitions.rotate_transition(left_switch_from_south, 90)
left_switch_from_north = transitions.rotate_transition(left_switch_from_south, 180)
left_switch_from_east = transitions.rotate_transition(left_switch_from_south, 270)
 
diamond_crossing = cells[3]
 
left_slip_from_south = cells[4]
left_slip_from_west = transitions.rotate_transition(left_slip_from_south, 90)
left_slip_from_north = transitions.rotate_transition(left_slip_from_south, 180)
left_slip_from_east = transitions.rotate_transition(left_slip_from_south, 270)
 
right_double_slip_vertical = cells[5]
right_double_slip_horizontal = transitions.rotate_transition(right_double_slip_vertical, 90)
 
symmetrical_slip_from_south = cells[6]
symmetrical_slip_from_west = transitions.rotate_transition(symmetrical_slip_from_south, 90)
symmetrical_slip_from_north = transitions.rotate_transition(symmetrical_slip_from_south, 180)
symmetrical_slip_from_east = transitions.rotate_transition(symmetrical_slip_from_south, 270)
 
dead_end_from_south = cells[7]
dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
 
right_turn_from_south = cells[8]
right_turn_from_west = transitions.rotate_transition(right_turn_from_south, 90)
right_turn_from_north = transitions.rotate_transition(right_turn_from_south, 180)
right_turn_from_east = transitions.rotate_transition(right_turn_from_south, 270)
 
left_turn_from_south = cells[9]
left_turn_from_west = transitions.rotate_transition(left_turn_from_south, 90)
left_turn_from_north = transitions.rotate_transition(left_turn_from_south, 180)
left_turn_from_east = transitions.rotate_transition(left_turn_from_south, 270)
 
right_switch_from_south = cells[10]
right_switch_from_west = transitions.rotate_transition(right_switch_from_south, 90)
right_switch_from_north = transitions.rotate_transition(right_switch_from_south, 180)
right_switch_from_east = transitions.rotate_transition(right_switch_from_south, 270)


def generate_custom_rail():
    if args.use_left_right_map:
        print("left right loop")
        rail_map = np.array(
            [[empty] + [right_turn_from_south] + [horizontal_straight] + [right_turn_from_west] + [empty]] +
            [[left_turn_from_east] + [right_switch_from_east] + [horizontal_straight] + [left_switch_from_west] + [right_turn_from_west]] +
            [[right_turn_from_east] + [horizontal_straight]*(3) + [left_turn_from_west]],
        dtype=np.uint16)
        train_stations = [
            [((0,2), 0)],
            [((2, 2), 0)],
        ]

        city_positions = [(0, 2), (2, 2)]
        city_orientations = [1, 1]

        agents_hints = {
            "city_positions": city_positions,
            "train_stations": train_stations,
            "city_orientations": city_orientations,
        }

        optionals = {"agents_hints": agents_hints}

        rail = GridTransitionMap(width=rail_map.shape[1], height=rail_map.shape[0], transitions=transitions)
        rail.grid = rail_map
        
        return rail, optionals
    
    if args.use_meet_map:
        print("meet_map")
        rail_map = np.array(
            [[right_turn_from_south] + [horizontal_straight] + [left_turn_from_south] + [empty] * 2 + [right_turn_from_south] + [horizontal_straight] * 3 + [right_turn_from_west] + [empty] * 2 + [right_turn_from_south] + [horizontal_straight] + [right_turn_from_west]] +
            [[right_turn_from_east] + [horizontal_straight] + [right_switch_from_east] + [horizontal_straight] * 2 + [left_switch_from_west] +  [horizontal_straight] * 3 + [right_switch_from_east] + [horizontal_straight] * 2 + [left_switch_from_west] + [horizontal_straight] + [left_turn_from_west]],
        dtype=np.uint16)
        train_stations = [
            [((1,1), 0)],
            [((1, 13), 0)],
        ]

        city_positions = [(1, 1), (1, 13)]
        city_orientations = [1, 1]

        agents_hints = {
            "city_positions": city_positions,
            "train_stations": train_stations,
            "city_orientations": city_orientations,
        }

        optionals = {"agents_hints": agents_hints}

        rail = GridTransitionMap(width=rail_map.shape[1], height=rail_map.shape[0], transitions=transitions)
        rail.grid = rail_map
        
        return rail, optionals

    if args.use_line_map:
        map_width=10
        map_height=5
        rail_map = np.array(
            [[empty]*map_width] +
            [[empty]*map_width] +
            [[dead_end_from_east] + [horizontal_straight]*(map_width-2) + [dead_end_from_west]] +
            [[empty]*map_width] +
            [[empty]*map_width],
        dtype=np.uint16)

        train_stations = [
            [((2, map_width-2), 0)],
            [((2, 0), 0)],
        ]

        city_positions = [(2, map_width-2), (2, 0)]
        city_orientations = [1, 1]

        agents_hints = {
            "city_positions": city_positions,
            "train_stations": train_stations,
            "city_orientations": city_orientations,
        }

        optionals = {"agents_hints": agents_hints}

        rail = GridTransitionMap(width=rail_map.shape[1], height=rail_map.shape[0], transitions=transitions)
        rail.grid = rail_map
        
        return rail, optionals
    
if __name__ == "__main__":
    print('starting main')
    print(f'avail gpus: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}')
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = 'cuda'
    run_name = (
        f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    )
    if args.track:
        print('initializing tracking')

        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )
    print("device used: {}".format(device))
    
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
    
    if args.use_pretrained_network:
        model_path = args.pretrained_network_path
        assert (model_path.endswith('tar')), 'Network format not known.'
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('loaded pretrained model from .tar file') 

    print('collector done')
    from torchrl.data import TensorDictReplayBuffer

    loss_module = ClipPPOLoss(
        actor=ProbabilisticTensorDictSequential(common_module, policy),
        critic=critic_module,
        critic_coef=args.vf_coef,
        clip_epsilon=args.clip_coef,
        entropy_coef=args.ent_coef,
        normalize_advantage=args.norm_adv,  # Important to avoid normalizing across the agent dimension
        loss_critic_type = "smooth_l1",
    )

    loss_module.set_keys(  # We have to tell the loss where to find the keys
        #reward=env.reward_key,
        action=("agents", "action"),
        sample_log_prob=("agents", "sample_log_prob"),
        value=("state_value"),
        # These last 2 keys will be expanded to match the reward shape
        done=("agents", "done"),
        terminated=("agents", "terminated"),
        advantage=("agents", "advantage")
    )
    """     loss_module.make_value_estimator(
    ValueEstimators.GAE, gamma=0.99, lmbda=0.95
    )  # We build GAE
    GAE = loss_module.value_estimator """
    
    optim = torch.optim.Adam(loss_module.parameters(), lr=2.5e-4, weight_decay=1e-5)
    
    pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")

    global_step = 0

    print(f'map height: {args.map_height}')
    from tensordict import LazyStackedTensorDict
    if args.curriculum_path is not None:
        curriculums = open(args.curriculum_path)
        curriculums = json.load(curriculums)
        print(curriculums)
    else: 
        curriculums =[
            {"map_height": args.map_height,
             "map_width": args.map_width,
             "num_agents" : args.num_agents,
             "arrival_reward_coef": args.arrival_reward_coef,
             "delay_reward_coef": args.delay_reward_coef,
             "shortest_path_reward_coef": args.shortest_path_reward_coef,
             "departure_reward_coef": args.departure_reward_coef,
             "deadlock_penalty_coef": args.deadlock_penalty_coef,
             "total_timesteps": args.total_timesteps,
             },
        ]
        
    curriculum = curriculums[0]
    
    print(f'curriculum: {curriculum}')
    start_time = time.time()
    
    for curriculum in curriculums:
        def make_env():
            td_env = RailEnvTd(
                number_of_agents= curriculum["num_agents"],
                width = curriculum["map_width"],
                height = curriculum["map_height"],
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
            
            td_env.set_reward_coef(shortest_path_reward_coef=curriculum["shortest_path_reward_coef"], 
                                   delay_reward_coef=curriculum["delay_reward_coef"],
                                   arrival_reward_coef=curriculum["arrival_reward_coef"], 
                                   deadlock_penalty_coef=curriculum["deadlock_penalty_coef"], 
                                   departure_reward_coef=curriculum["departure_reward_coef"])
            td_env.reset()
            return td_torch_rail(td_env)
        
        env = ParallelEnv(args.num_envs, make_env)
        
        collector = SyncDataCollector(
            env,
            model,
            device='cpu',
            storing_device='cpu',
            frames_per_batch=args.batch_size,
            total_frames=curriculum["total_timesteps"],
        )
        
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                args.batch_size, device=device
            ),  # We store the frames_per_batch collected at each iteration
            sampler=SamplerWithoutReplacement(),
            batch_size=args.minibatch_size,  # We will sample minibatches of this size
        )

        start_rollout = time.time()
        for tensordict_data in collector:
            rollout_duration = time.time() - start_rollout
            training_start = time.time()
            print(f'duration rollout: {time.time()-start_time}')
            global_step += args.batch_size
            print(f'global step: {global_step}')
            #print(f'td rewards: {tensordict_data[("next", "agents", "reward")]}')
            #print(f'actions chosen: {tensordict_data[("agents", "action")]}')
            #print(f'device of tensordict data: {tensordict_data.device}')
            #print(f'example of tensordict data: {tensordict_data}')
            # log data about collector:
            if args.track:
                writer.add_scalar(
                    "rewards/min", tensordict_data[('next', 'agents', 'reward')].min(), global_step
                )
                writer.add_scalar(
                    "rewards/mean", tensordict_data[('next', 'agents', 'reward')].mean(), global_step
                )
                writer.add_scalar(
                    'rewards/max', tensordict_data[('next', 'agents', 'reward')].max(), global_step
                )
                #print("valid action probs: {}".format(rollout_data['valid_actions_probs'].shape))
                mean_logits = torch.sigmoid(tensordict_data[('agents', 'logits')]).mean((0, 1,2))
                #print(f'mean log: {mean_logits}')
                softmax_value = mean_logits
                print(f'softmax vals: {softmax_value}')
                writer.add_scalar("action_probs/do_nothing", softmax_value[0], global_step)
                writer.add_scalar("action_probs/left", softmax_value[1], global_step)
                writer.add_scalar("action_probs/forward", softmax_value[2], global_step)
                writer.add_scalar("action_probs/right",softmax_value[3], global_step)
                writer.add_scalar("action_probs/stop_moving", softmax_value[4], global_step)
                writer.add_scalar("charts/rollout_steps_per_second", args.batch_size/rollout_duration, global_step)
                
                final_steps = tensordict_data[("next", "done")].squeeze()
                writer.add_scalar("charts/total_rollout_duration", rollout_duration, global_step)
                writer.add_scalar("stats/arrival_ratio", tensordict_data[("stats", "arrival_ratio")][torch.roll(final_steps, 1, -1)].mean(), global_step)
                writer.add_scalar("stats/deadlock_ratio", tensordict_data[("stats", "deadlock_ratio")][torch.roll(final_steps, 1, -1)].mean(), global_step)
                        
            tensordict_data = tensordict_data.to('cuda')
            tensordict_data.set(
                ("next", "agents", "reward"),
                tensordict_data.get(("next", "agents", "reward")).mean(-1))

            #do we have n issue with averaging averages?
            #print(f'averaged rewards: {tensordict_data[("next", "agents", "reward")]}')
            
            #are we averaging the rewards over all agents here? what's happening?
            with torch.no_grad():
                #tensordict_data = tensordict_data.flatten()
                """ GAE(
                    tensordict_data,
                    params=loss_module.critic_params,
                    target_params=loss_module.target_critic_params,
                ) """
                lastgaelam = torch.zeros(args.num_envs).to('cuda')
                rollout_lengths = int(args.batch_size/args.num_envs)
                tensordict_data['advantage'] = torch.zeros_like(tensordict_data['state_value']).to('cuda')
                
                #we get the value of the last rollout observation
                next_val = model.get_value_operator()(tensordict_data[('next')][:,rollout_lengths-1])
                #print(f'next vals: {next_val}')
                
                for t in reversed(range(rollout_lengths)):
                    if t == rollout_lengths-1: # for the last one, we get the special values for the last observation
                        nextnonterminal = ~tensordict_data[('next', 'done')][:,t].squeeze()
                        #print(f'shape nextonterminal: {nextnonterminal.shape}')
                        nextvalues = next_val['state_value'].squeeze()
                    else:
                        # for all other rollouts, we get the dones from the current t, and the values from the next t
                        nextnonterminal = ~tensordict_data[('next', 'done')][:,t].squeeze()
                        #print(f'next nonterminal: {nextnonterminal}')
                        nextvalues = tensordict_data[('state_value')][:,t + 1].squeeze()
                    #print(f'nextvalue: {nextvalues}')
                    delta = (
                        tensordict_data[('next', 'agents', 'reward')][:, t]
                        + args.gamma * nextvalues * nextnonterminal
                        - tensordict_data[('state_value')][:, t]
                    )

                    tensordict_data['advantage'][:,t] = lastgaelam = (
                        delta
                        + args.gamma
                        * args.gae_lambda
                        * nextnonterminal
                        * lastgaelam
                    ).flatten() #why do we do flatten here?

                tensordict_data['value_target'] = tensordict_data['advantage'] + tensordict_data["state_value"]
            
            tensordict_data[("agents", "advantage")] = tensordict_data["advantage"].repeat_interleave(curriculum["num_agents"]).reshape(tensordict_data[("agents", "action")].shape).unsqueeze(-1)
            
            #print(f'advantage: {tensordict_data["advantage"]}')
            #print(f'value_target: {tensordict_data["value_target"]}')
            #print(f'state value: {tensordict_data["state_value"]}')

            if args.track:
                writer.add_scalar("internal_values/advantages_max", tensordict_data['advantage'].max(), global_step)
                writer.add_scalar("internal_values/advantages_min", tensordict_data['advantage'].min(), global_step)
            
            data_view = tensordict_data.reshape(-1)
            #print(f'shape of data view: {data_view.shape}')# Flatten the batch size to shuffle data
            replay_buffer.extend(data_view)
            #print(f'device of model: {model.device}')
            for n_epoch in range(args.update_epochs):
                print(f'epoch nr: {n_epoch}')
                for _ in range(args.num_minibatches):
                    subdata = replay_buffer.sample()
                    
                    loss_vals = loss_module(subdata)
                    
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )
                    
                    loss_value.backward()

                    torch.nn.utils.clip_grad_norm_(
                        loss_module.parameters(), args.max_grad_norm
                    )  # Optional

                    optim.step()
                    
                    #for param in policy.parameters():
                        #print(f'param grad in policy: {param.grad}')
                    optim.zero_grad()
            print(f'last loss is: {loss_value}')
            training_duration =  time.time() - training_start
            
            collector.update_policy_weights_()   
            
            if args.track:
                loss_vals = loss_module(subdata)
                    
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                
                loss_value = loss_value/args.num_minibatches
                writer.add_scalar(
                    "charts/learning_rate", optim.param_groups[0]["lr"], global_step
                )
                writer.add_scalar("losses/value_loss", loss_vals["loss_critic"], global_step)
                writer.add_scalar("losses/policy_loss", loss_vals["loss_objective"], global_step)
                writer.add_scalar("losses/entropy", loss_vals["loss_entropy"], global_step)

                writer.add_scalar("losses/total_loss", loss_value, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                writer.add_scalar("charts/training_speed", args.batch_size*args.update_epochs / training_duration, global_step)
                writer.add_scalar("charts/total_training_duration", training_duration, global_step)

                original_data_logprobs = data_view[("agents", "sample_log_prob")].clone()              
                updated_data_logprobs = model(data_view)[("agents", "sample_log_prob")] #yes we overwrite the original data here, but it doesn't matter at this point
                #also the trainign above does not save the log probs, so we do have to calculate them once more unfortunately
                logratio = updated_data_logprobs - original_data_logprobs
                #print(f'logatio: {logratio}')
                approx_kl = ((logratio.exp() - 1) - logratio).mean()
                writer.add_scalar("losses/approx_kl", approx_kl, global_step)
                
                print(f'approx_kl: {approx_kl}')
                if global_step % 50_000 == 0:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                    }, f"model_checkpoints/{run_name}_" + str(global_step) + ".tar")
                
            
            #for param in model.parameters():
                #print(param.data)    
            start_rollout = time.time()
 