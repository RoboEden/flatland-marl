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
from torchrl.envs import ParallelEnv, SerialEnv
from torchrl.modules import MultiAgentMLP, ProbabilisticActor

from torchrl.objectives import ClipPPOLoss, ValueEstimators

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

import time

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
    parser.add_argument("--use-env-reward", type=bool, default=False,
        help="Toggles whether to use the default flatland env reward at each step")
    parser.add_argument('--use-start-reward', action=argparse.BooleanOptionalAction)
    parser.add_argument('--use-delay-reward', action=argparse.BooleanOptionalAction)
    parser.add_argument('--initialize-action-weights', action=argparse.BooleanOptionalAction)
    parser.add_argument('--max-episode-steps', type=int, default=80)
    parser.add_argument('--force-shortest-path', action=argparse.BooleanOptionalAction)
    parser.add_argument("--arrival-reward", type=int, default=1, 
        help="The reward to be returned when the train departs.")
    parser.add_argument('--stepwise', action=argparse.BooleanOptionalAction)
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
        
        try: 
            if True:
                for agent_i, agent in enumerate(self.agents):
                    agent.earliest_departure = 0
                    agent.latest_arrival = args.max_episode_steps
                
                self._max_episode_steps = args.max_episode_steps
        except:
            #print('using default max episode steps')
            pass
        
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
        #tensordict_out["done"] = torch.tensor(False).type(torch.bool) # not done since just initialized
        if False:
            #print('shorestest path action: {}'.format((self.agents[0].get_shortest_path(self.distance_map)[0][1])))
            tensordict_out['observations']['shortest_path_action'] = torch.reshape(torch.tensor(self.agents[0].get_shortest_path(self.distance_map)[0][1]).type(torch.int64), [1])
        return tensordict_out

    def step(self, tensordict):
        '''Extend default flatland step by returning a tensordict. '''
        #print('entering step function')
        #print("shortest path: {}".format((self.agents[0].get_shortest_path(self.distance_map)[0])))
        for handle, action in enumerate(tensordict['agents']['action']):
            #print(handle, action.flatten())
            pass
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
        return_td = TensorDict({'agents': TensorDict({}, [])}, batch_size=[])
        #rewards_td = TensorDict({'agents': TensorDict({}, [])}, batch_size=[])
        return_td['agents']["observation"] = self.obs_to_td(observations)
        return_td['agents']["reward"] = torch.tensor(
            [value for _, value in rewards.items()], dtype = torch.float32
        )
        return_td["done"] = torch.tensor(done["__all__"]).type(torch.bool)
        return_td['agents']["observation"]["valid_actions"] = torch.tensor(
            valid_actions, dtype=torch.bool
        )
        if False:
            print('shorestest path action: {}'.format((self.agents[0].get_shortest_path(self.distance_map)[0][1])))
            observation_td['observations']['shortest_path_action'] = torch.reshape(torch.tensor(self.agents[0].get_shortest_path(self.distance_map)[0][1]).type(torch.int64), [1])
        return return_td

    def update_step_rewards(self, i_agent):
        
        if False:
            print("using env reward")
            reward = None
            agent = self.agents[i_agent]
            # agent done? (arrival_time is not None)
            if agent.state == TrainState.DONE:
                # if agent arrived earlier or on time = 0
                # if agent arrived later = -ve reward based on how late
                reward = min(agent.latest_arrival - agent.arrival_time, 0)

            # Agents not done (arrival_time is None)
            else:
                # CANCELLED check (never departed)
                if agent.state.is_off_map_state():
                    reward = (
                        -1
                        * self.cancellation_factor
                        * (
                            agent.get_travel_time_on_shortest_path(
                                self.distance_map
                            )
                            + self.cancellation_time_buffer
                        )
                    )

                # Departed but never reached
                if agent.state.is_on_map_state():
                    reward = agent.get_current_delay(
                        self._elapsed_steps, self.distance_map
                    )
            self.rewards_dict[i_agent] += reward
        
        if False:
            print("using start reward")
            #print(self._elapsed_steps)
            agent = self.agents[i_agent]
            if agent.state.is_on_map_state() and agent.state_machine.previous_state.is_off_map_state():
                self.rewards_dict[i_agent] = args.start_reward
            #if agent.state.is_off_map_state() and agent.state_machine.previous_state.is_off_map_state() and self._elapsed_steps > 1:
            #    self.rewards_dict[i_agent] = args.start_reward
            
        if False:
            agent = self.agents[i_agent]
            #print("shortest path: {}".format(agent.get_shortest_path(self.distance_map)))
            print('agent delay: {}'.format((agent.get_current_delay(self._elapsed_steps, self.distance_map))))
            #print("agent position: {}".format(agent.position))
            #print("agent target: {}".format(agent.target))
            if agent.state == TrainState.DONE:
                #print("agent position equal targer: {}".format(agent.position == agent.target))
                print("reward for arriving")
                self.rewards_dict[i_agent] = args.arrival_reward
            if agent.get_travel_time_on_shortest_path(self.distance_map) > (self._max_episode_steps-self._elapsed_steps + 2):
                #print(self._max_episode_steps - self._elapsed_steps)
                #print(agent.get_travel_time_on_shortest_path(self.distance_map))
                #print("penalty for wrong choice")
                #print(agent.get_travel_time_on_shortest_path(self.distance_map) - self._elapsed_steps)
                #print(agent.get_current_delay(self._elapsed_steps, self.distance_map))
                self.rewards_dict[i_agent] = agent.get_travel_time_on_shortest_path(self.distance_map) - self._elapsed_steps # not too big penalty, so trains still depart
            #if agent.state==TrainState.READY_TO_DEPART and self._elapsed_steps > 2:
            #    print("penalty for waiting")
            #    self.rewards_dict[i_agent] = -args.arrival_reward
        if True:
            agent=self.agents[i_agent]
            self.rewards_dict[i_agent]=agent.get_current_delay(self._elapsed_steps, self.distance_map)
                
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
        return self.env.reset()
    
    def _step(self, tensordict):
        #print('actions: {}'.format(tensordict['action']))
        #actions_dict = {handle: action.item() for handle, action in enumerate(tensordict['action'])}
        return self.env.step(tensordict)
    
def make_env():
    td_env = RailEnvTd(
        number_of_agents=args.num_agents,
        width = 30,
        height = 35,
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
                malfunction_rate=1 / 4500, min_duration=20, max_duration=50)
        ),
        obs_builder_object=TreeCutils(31, 500),
    )
    td_env.reset()
    return td_torch_rail(td_env)

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

class BaseLineGen(object):
    def __init__(self, speed_ratio_map: Mapping[float, float] = None, seed: int = 1):
        self.speed_ratio_map = speed_ratio_map
        self.seed = seed

    def generate(self, rail: GridTransitionMap, num_agents: int, hints: Any=None, num_resets: int = 0,
        np_random: RandomState = None) -> Line:
        pass

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)
    
if __name__ == "__main__":
    print('starting main')
    print(f'avail gpus: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}')
    args = parse_args()

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
    
    env = ParallelEnv(args.num_envs, make_env)
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
        in_keys=[('hidden', 'embedding'), ('hidden', 'att_embedding')],
        out_keys=[('agents', 'logits')]
    )
    
    policy = ProbabilisticActor(
        module = actor_module,
        in_keys = ('agents', 'logits'),
        out_keys = [('agents', 'action')],
        distribution_class = torch.distributions.categorical.Categorical,
        return_log_prob=True,
        log_prob_key=('agents', 'sample_log_prob')
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
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, eps=1e-5, weight_decay = 1e-5
    )
    
    print('optim done')
    #test_rollout = env.rollout(max_steps= 100, policy = model.get_policy_operator())
    #print('rollout successful')
    
    collector = SyncDataCollector(
        env,
        model,
        device='cpu',
        storing_device='cpu',
        frames_per_batch=args.batch_size,
        total_frames=args.total_timesteps,
    )
    print('collector done')
    from torchrl.data import TensorDictReplayBuffer
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            args.batch_size, device=device
        ),  # We store the frames_per_batch collected at each iteration
        sampler=SamplerWithoutReplacement(),
        batch_size=args.minibatch_size,  # We will sample minibatches of this size
    )
    print('replay buffer done')
    loss_module = ClipPPOLoss(
        actor=model.get_policy_operator(),
        critic=model.get_value_operator(),
        critic_coef=args.vf_coef,
        clip_epsilon=args.clip_coef,
        entropy_coef=args.ent_coef,
        normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
    )
    
    print(f'env action key: {env.action_key}')
    print(f'env reward key: {env.reward_key}')
    loss_module.set_keys(  # We have to tell the loss where to find the keys
        reward=env.reward_key,
        action=env.action_key,
        sample_log_prob=("agents", "sample_log_prob"),
        value=("state_value"),
        # These last 2 keys will be expanded to match the reward shape
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr=2.5e-4, weight_decay=1e-5)
    
    pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")

    episode_reward_mean_list = []
    global_step = 0
    start_time = time.time()
    start_rollout = time.time()
    for tensordict_data in collector:
        rollout_duration = time.time() - start_rollout
        training_start = time.time()
        print(f'duration rollout: {time.time()-start_time}')
        global_step += args.batch_size
        print(f'global step: {global_step}')
        print(f'device of tensordict data: {tensordict_data.device}')
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
            writer.add_scalar("action_probs/do_nothing", tensordict_data[('agents', 'logits')][:,:,:,0].mean(), global_step)
            writer.add_scalar("action_probs/left", tensordict_data[('agents', 'logits')][:,:,:,1].mean(), global_step)
            writer.add_scalar("action_probs/forward", tensordict_data[('agents', 'logits')][:,:,:,2].mean(), global_step)
            writer.add_scalar("action_probs/right", tensordict_data[('agents', 'logits')][:,:,:,3].mean(), global_step)
            writer.add_scalar("action_probs/stop_moving", tensordict_data[('agents', 'logits')][:,:,:,4].mean(), global_step)
            writer.add_scalar("charts/rollout_steps_per_second", args.batch_size/rollout_duration, global_step)
            writer.add_scalar("charts/total_rollout_duration", rollout_duration, global_step)
            
            
        
        tensordict_data = tensordict_data.to(device)
        tensordict_data.set(
            ("next", "agents", "reward"),
            tensordict_data.get(("next", "agents", "reward")).mean(-1))
        
        with torch.no_grad():

            lastgaelam = torch.zeros(args.num_envs).to(device)
            rollout_lengths = int(args.batch_size/args.num_envs)
            tensordict_data['advantage'] = torch.zeros_like(tensordict_data['state_value']).to('cuda')
            #print('shape of advantages: {}'.format(advantages.shape))
            #print('rollout data values shape: {}'.format(rollout_data['values'].shape))
            for t in reversed(range(rollout_lengths)):
                if t == rollout_lengths - 1:
                    nextnonterminal = ~tensordict_data[('next', 'done')][:,t].squeeze()
                    nextvalues = tensordict_data[('state_value')][:,t].squeeze()
                else:
                    nextnonterminal = ~tensordict_data[('next', 'done')][:,t].squeeze()
                    nextvalues = tensordict_data[('state_value')][:,t].squeeze()
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
                ).flatten()

            tensordict_data['returns'] = tensordict_data['advantage'] + tensordict_data["state_value"]
        tensordict_data['value_target'] = tensordict_data['state_value']
        
        if args.track:
            writer.add_scalar("internal_values/advantages_max", tensordict_data['advantage'].max(), global_step)
            writer.add_scalar("internal_values/advantages_min", tensordict_data['advantage'].min(), global_step)
        
        data_view = tensordict_data.reshape(-1)
        print(f'shape of data view: {data_view.shape}')# Flatten the batch size to shuffle data
        replay_buffer.extend(data_view)
        print(f'device of model: {model.device}')
        for n_epoch in range(args.update_epochs):
            print(f'epoch nr: {n_epoch}')
            for _ in range(args.num_minibatches):
                subdata = replay_buffer.sample()
                #print(f'shape of subdata {subdata.shape}')
                #print(f'example of subdata: {subdata}')
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
                optim.zero_grad()
        print(f'last loss is: {loss_value}')
        training_duration =  time.time() - training_start
        if args.track:
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )
            writer.add_scalar("losses/value_loss", loss_vals["loss_critic"], global_step)
            writer.add_scalar("losses/policy_loss", loss_vals["loss_objective"], global_step)
            writer.add_scalar("losses/entropy", loss_vals["loss_entropy"], global_step)

            writer.add_scalar("losses/total_loss", loss_value, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("charts/training_speed", args.batch_size*args.update_epochs / training_duration, global_step)
            writer.add_scalar("charts/total_training_duration", training_duration, global_step)
            
        
                
        collector.update_policy_weights_()

        
        start_rollout = time.time()
        # Logging
 