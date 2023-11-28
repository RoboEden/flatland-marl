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

from IPython.display import clear_output
from matplotlib import pyplot as plt

from flatland_cutils import TreeObsForRailEnv as TreeCutils
from impl_config import FeatureParserConfig as fp
from solution.nn.net_tree import Network_td

from gym.vector import SyncVectorEnv

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
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
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
        #print("reset env")
        observations, _ = super().reset(random_seed=random_seed)
        
        if args.max_episode_steps is not None:
            for agent_i, agent in enumerate(self.agents):
                agent.earliest_departure = 0
                agent.latest_arrival = args.max_episode_steps
            
            self._max_episode_steps = args.max_episode_steps
        else:
            print('using default max episode steps')
        
        if tensordict is None:
            tensordict_out = TensorDict({}, batch_size=[])
        tensordict_out["observations"] = self.obs_to_td(observations)
        # get valid actions
        (
            _,
            _,
            valid_actions,
        ) = self.obs_builder.get_properties()
        tensordict_out["observations"]["valid_actions"] = torch.tensor(
            valid_actions, dtype=torch.bool
        ) 
        #tensordict_out["done"] = torch.tensor(False).type(torch.bool) # not done since just initialized
        if args.force_shortest_path:
            #print('shorestest path action: {}'.format((self.agents[0].get_shortest_path(self.distance_map)[0][1])))
            tensordict_out['observations']['shortest_path_action'] = torch.reshape(torch.tensor(self.agents[0].get_shortest_path(self.distance_map)[0][1]).type(torch.int64), [1])
        return tensordict_out

    def step(self, tensordict):
        '''Extend default flatland step by returning a tensordict. '''
        
        #print("shortest path: {}".format((self.agents[0].get_shortest_path(self.distance_map)[0])))
        actions = {
            handle: action.item()
            for handle, action in enumerate(tensordict["actions"])
        }
        observations, rewards, done, _ = super().step(actions)
        (
            _,
            _,
            valid_actions,
        ) = self.obs_builder.get_properties()
        observation_td = TensorDict({}, batch_size=[])
        rewards_td = TensorDict({}, batch_size=[])
        observation_td["observations"] = self.obs_to_td(observations)
        rewards_td["rewards"] = torch.tensor(
            [value for _, value in rewards.items()]
        )
        observation_td["done"] = torch.tensor(done["__all__"]).type(torch.bool)
        observation_td["observations"]["valid_actions"] = torch.tensor(
            valid_actions, dtype=torch.bool
        )
        if args.force_shortest_path:
            print('shorestest path action: {}'.format((self.agents[0].get_shortest_path(self.distance_map)[0][1])))
            observation_td['observations']['shortest_path_action'] = torch.reshape(torch.tensor(self.agents[0].get_shortest_path(self.distance_map)[0][1]).type(torch.int64), [1])
        return observation_td, rewards_td

    def update_step_rewards(self, i_agent):
        
        if args.use_env_reward:
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
        
        if args.use_start_reward:
            print("using start reward")
            #print(self._elapsed_steps)
            agent = self.agents[i_agent]
            if agent.state.is_on_map_state() and agent.state_machine.previous_state.is_off_map_state():
                self.rewards_dict[i_agent] = args.start_reward
            #if agent.state.is_off_map_state() and agent.state_machine.previous_state.is_off_map_state() and self._elapsed_steps > 1:
            #    self.rewards_dict[i_agent] = args.start_reward
            
        if args.use_arrival_reward:
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
        if args.use_delay_reward:
            agent=self.agents[i_agent]
            self.rewards_dict[i_agent]=agent.get_current_delay(self._elapsed_steps, self.distance_map)
                
    def _handle_end_reward(self, agent: EnvAgent) -> int:
        if args.use_start_reward or args.use_arrival_reward or args.use_delay_reward:
            #print("end of episode status: {}".format(agent.state))
            return 0
        else:
            super()._handle_end_reward(self, agent)
            

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
    
    if args.use_left_right_map:
        map_height = 3
        map_width = 5
    if args.use_meet_map:
        map_height = 2
        map_width = 15
    if args.use_line_map:
        map_height=5
        map_width=10
    
    
    if args.use_left_right_map or args.use_line_map or args.use_meet_map:
        def create_random_env():
            """Create a random railEnv object
            Taken from the flatland-marl demo
            """
            #rail, rail_map, optionals = generate_switch_rail()
            return RailEnvTd(
                number_of_agents=args.num_agents,
                #width=rail_map.shape[1],
                #height=rail_map.shape[0],
                width = map_width,
                height = map_height,
                #rail_generator=rail_from_grid_transition_map(rail, optionals),
                rail_generator=rail_from_grid_transition_map(*generate_custom_rail()),
                line_generator=SparseLineGen(
                    speed_ratio_map={1.0: 1}, #0.5: '', 0.33: 1 / 4, 0.25: 1 / 4},
                    seed = 1
                ),
                malfunction_generator=ParamMalfunctionGen(
                    MalfunctionParameters(
                        malfunction_rate=1 / 4500, min_duration=20, max_duration=50
                    ),
                    
                ),
                obs_builder_object=TreeCutils(
                    fp.num_tree_obs_nodes, fp.tree_pred_path_depth
                ),
            )
    else: 
        def create_random_env():
            return RailEnvTd(
                number_of_agents=args.num_agents,
                width=30,
                height=35,
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
                        malfunction_rate=1 / 4500, min_duration=20, max_duration=50
                    )
                ),
                obs_builder_object=TreeCutils(fp.num_tree_obs_nodes, fp.tree_pred_path_depth),
            )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )
    print("device used: {}".format(device))
    device = torch.device('cpu')
    env = create_random_env()
    
    if args.do_render:
        env_renderer = RenderTool(
            env,
            agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
            show_debug=False,
            screen_height=600,  # Adjust these parameters to fit your resolution
            screen_width=800,
        )
    #env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
    #torch.manual_seed(0) # changed from 1
    network = Network_td()
    print(args.use_pretrained_network)
    if args.use_pretrained_network:
        model_path = "solution/policy/phase-III-50.pt"
        loaded_model = torch.load(args.pretrained_network_path, map_location=torch.device(device))
        network.load_state_dict(loaded_model)
        print("loaded pretrained model")

    td_module = TensorDictModule(
        network,
        in_keys=["observations", "actions"],
        out_keys=["actions", "logprobs", "entropy", "values", "valid_actions_probs", "tree_embedding"],
    ).to(device)
    
    if args.no_training:
        for param in td_module.parameters():
            param.requires_grad = False
            
    if args.freeze_embeddings:
        for param in td_module.attr_embedding.parameters():
            param.requires_grad = False
        for param in td_module.transformer.parameters():
            param.requires_grad = False
        #for param in td_module.critic_net.parameters():
        #    param.requires_grad = False
        for param in td_module.tree_lstm.parameters():
            param.requires_grad = False
            
        print("froze non-action layers")
    
    if args.initialize_action_weights:
        for param in td_module.actor_net:
            if isinstance(param, nn.Linear):
                print("init linear layer")
                param.weight.data.normal_(mean=0.0, std=0.1)
                param.bias.data.zero_()
        for param in td_module.critic_net:
            if isinstance(param, nn.Linear):
                print("init linear layer")
                param.weight.data.normal_(mean=0.0, std=0.1)
                param.bias.data.zero_()
        #td_module.actor_net.apply(td_module._init_weights(td_module.actor_net))
        print("initialized action and critic weights")
        

    optimizer = optim.Adam(
        td_module.parameters(), lr=args.learning_rate, eps=1e-5
    )

    #initialize storage for the rollouts
    observations = TensorDict(
        {
            "agents_attr": torch.zeros(
                args.num_steps, args.num_agents, 83, dtype=torch.float32
            ),
            "node_attr": torch.zeros(
                args.num_steps, args.num_agents, 31, 12, dtype=torch.float32
            ),
            "adjacency": torch.zeros(
                args.num_steps, args.num_agents, 30, 3, dtype=torch.int64
            ),
            "node_order": torch.zeros(
                args.num_steps, args.num_agents, 31, dtype=torch.int64
            ),
            "edge_order": torch.zeros(
                args.num_steps, args.num_agents, 30, dtype=torch.int64
            ),
            "valid_actions": torch.zeros(
                args.num_steps, args.num_agents, *env.action_space, dtype=torch.bool
            ),
            'shortest_path_action': torch.zeros(args.num_steps, args.num_agents, 1, dtype = torch.int64)
        },
        batch_size=(args.num_steps, args.num_agents),
    )

    actions_init = torch.zeros((args.num_steps, args.num_agents), dtype=torch.int64)
    logprobs = torch.zeros((args.num_steps, args.num_agents), dtype=torch.float32)
    entropy = torch.zeros((args.num_steps), dtype = torch.float32)
    rewards = torch.zeros((args.num_steps, args.num_agents), dtype=torch.float32)
    done = torch.zeros((args.num_steps), dtype=torch.bool)
    values = torch.zeros((args.num_steps), dtype=torch.float32)
    valid_actions_probs = torch.zeros((args.num_steps, args.num_agents, 5), dtype=torch.float32)
    tree_embedding = torch.zeros((args.num_steps, args.num_agents, 128), dtype=torch.float32)

    rollout_data = TensorDict(
        {
            "observations": observations,
            "actions": actions_init,
            "logprobs": logprobs,
            'entropy': entropy,
            "rewards": rewards,
            "done": done,
            "values": values,
            "valid_actions_probs": valid_actions_probs,
            'tree_embedding': tree_embedding,
        },
        batch_size=[args.num_steps],
    )

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = env.reset(random_seed = 1)
    rollout_data = rollout_data.to(device)
    num_updates = args.total_timesteps // args.batch_size
    print('total timesteps: {}'.format(args.total_timesteps))
    print('batch size: {}'.format(args.batch_size))
    print("num updates: {}".format(num_updates))

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        rollout_data['actions'] = torch.zeros_like(rollout_data['actions'])
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            #print("logprobs shape: {}".format(next_obs['logprobs'].shape))
            #print("next obs: {}".format(next_obs))
            #print('adjacency list: {}'.format(rollout_data['observations']['adjacency']))
            rollout_data[
                step
            ].update(next_obs, inplace = True)# save for training, also includes the done
            #print("valid actions before step: {}".format(next_obs['observations']['valid_actions']))
            # ALGO LOGIC: action logic
            #print('observations adjacency before step: {}'.format(rollout_data[step]['observations']['adjacency']))
            #exit()
            with torch.no_grad():
                rollout_data[[step]] = (td_module(rollout_data[[step]]))
            #print('observations adjacency after step: {}'.format(rollout_data[step]['observations']['adjacency']))
            #exit()
            #print("actions drawn from net: {}".format(rollout_data[step]['actions']))
            #print("action chosen: {}".format(rollout_data[step]['actions']))
            next_obs, rewards = env.step(rollout_data[step])
            #print("reward received: {}".format(rewards['rewards']))
            #print("rewards: {}".format(rewards['rewards']))
            #print('rewards location where to be saved: {}'.format(rollout_data[step]['rewards']))
            rollout_data[step].update_(rewards) # save the rewards received for actions in current step
            #print("rewards saved: {}".format(rollout_data[step]['rewards']))
            #exit()
            if next_obs['done']:
                #print("resetting env")
                #next_obs.update_(env.reset(random_seed=1)) # only overwriting keys returned by reset, i.e. 'observations'
                next_obs.update_(env.reset())
                if args.do_render:
                    env_renderer.reset()

            if args.do_render:
                env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
                
            if args.stepwise:
                print("valid actions: {}".format(next_obs['observations']['valid_actions']))
                input("Press Enter to continue...")
        
        
        if args.track:
            writer.add_scalar(
                "rewards/min", rollout_data["rewards"].min(), global_step
            )
            writer.add_scalar(
                "rewards/mean", rollout_data["rewards"].mean(), global_step
            )
            writer.add_scalar(
                'rewards/max', rollout_data['rewards'].max(), global_step
            )
            writer.add_scalar(
                "action_freq/forward", (rollout_data['actions'] == 2).sum(), global_step
            )
            writer.add_scalar(
                "action_freq/do_nothing",  (rollout_data['actions'] == 4).sum(), global_step
            )
            #print("valid action probs: {}".format(rollout_data['valid_actions_probs'].shape))
            writer.add_scalar("action_probs/do_nothing", (rollout_data['valid_actions_probs'][:,:,0].mean()), global_step)
            writer.add_scalar("action_probs/left", (rollout_data['valid_actions_probs'][:,:,1].mean()), global_step)
            writer.add_scalar("action_probs/forward", (rollout_data['valid_actions_probs'][:,:,2].mean()), global_step)
            writer.add_scalar("action_probs/right", (rollout_data['valid_actions_probs'][:,:,3].mean()), global_step)
            writer.add_scalar("action_probs/stop_moving", (rollout_data['valid_actions_probs'][:,:,4].mean()), global_step)
            
            writer.add_scalar("action_probs_max/do_nothing", (rollout_data['valid_actions_probs'][:,:,0].max()), global_step)
            writer.add_scalar("action_probs_max/left", (rollout_data['valid_actions_probs'][:,:,1].max()), global_step)
            writer.add_scalar("action_probs_max/forward", (rollout_data['valid_actions_probs'][:,:,2].max()), global_step)
            writer.add_scalar("action_probs_max/right", (rollout_data['valid_actions_probs'][:,:,3].max()), global_step)
            writer.add_scalar("action_probs_max/stop_moving", (rollout_data['valid_actions_probs'][:,:,4].max()), global_step)

        print('rollout actions shape: {}'.format(rollout_data['actions'].flatten().shape))
        print("rollout actions frequency: {}".format(torch.bincount(rollout_data['actions'].flatten())))
        #print('shape of rollout')
        
        if args.num_agents==1 and False:
        
            tree_embedding_df = rollout_data['tree_embedding'].squeeze(1)
            #print('tree embedding df shape: {}'.format(tree_embedding_df.shape))
            tree_embedding_df = pd.DataFrame(tree_embedding_df.cpu().numpy())
            
            valid_actions_df = rollout_data['observations']['valid_actions'].squeeze(1)
            valid_actions_df = pd.DataFrame(valid_actions_df.cpu().numpy())
            #print(valid_actions_df)
            
            chosen_actions_df = rollout_data['actions'].squeeze(1)
            chosen_actions_df=pd.DataFrame(chosen_actions_df.cpu().numpy())
            
            combined_df = pd.concat([tree_embedding_df, valid_actions_df, chosen_actions_df], axis=1)
            combined_df.to_csv("combined_rollout_tree_embedding_" + str(update) + ".csv")
        
        next_obs = next_obs.to(device)
        next_obs['actions'] = torch.ones_like(rollout_data[0]['actions'])
        
        rollout_data['rewards_mean']=rollout_data['rewards'].mean(-1)
        #print('shape of rewards after mean: {}'.format(rollout_data['rewards_mean'].shape))
        
        # Calculate advantages
        with torch.no_grad():
            next_value = td_module(next_obs.unsqueeze(0))["values"]
 
            advantages = torch.zeros_like(rollout_data['rewards_mean']).to(device)
            lastgaelam = torch.zeros(1).to(device)
            
            #print('shape of advantages: {}'.format(advantages.shape))
            #print('rollout data values shape: {}'.format(rollout_data['values'].shape))
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_obs['done'].item()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - rollout_data["done"][t + 1].item()
                    nextvalues = rollout_data["values"][t + 1]
                delta = (
                    rollout_data['rewards_mean'][t]
                    + args.gamma * nextvalues * nextnonterminal
                    - rollout_data["values"][t]
                )

                #print('shape of nextnonterminal: {}'.format(nextnonterminal.shape))
                #()
                advantages[t] = lastgaelam = (
                    delta
                    + args.gamma
                    * args.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                ).flatten()
            returns = advantages + rollout_data["values"]
        if args.track:
            writer.add_scalar("internal_values/advantages_max", advantages.max(), global_step)
            writer.add_scalar("internal_values/advantages_min", advantages.min(), global_step)
        #print(rollout_data['rewards_mean'].shape)
        #print(rollout_data['done'].shape)
        #print(rollout_data['values'].shape)
        #print("rollout data: {}".format(torch.cat((advantages, returns, rollout_data['actions'], rollout_data['rewards_mean'], rollout_data['done'].unsqueeze(-1), rollout_data['values']), dim=1)))
        # Optimizing the policy and value network
        b_inds = np.arange(args.num_steps)
        clipfracs = []
        for epoch in range(args.update_epochs):
            print("currently in epoch: {}".format(epoch))
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                #print("start of mb: {}".format(start))
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                updated_rollout_data = td_module(rollout_data[mb_inds])
                #print("updated logprobs have autograd: {}".format(updated_rollout_data['logprobs'].requires_grad))
                #print("updated values have autograd: {}".format(updated_rollout_data['values'].requires_grad))
                #print('original rollout data logprobs: {}'.format(rollout_data["logprobs"][mb_inds]))
                #print("updated rollout data logprobs: {}".format(updated_rollout_data['logprobs']))
                #print("shape of rollout data logprobs: {}".format(rollout_data['logprobs'][mb_inds].shape))
                #print("shape of updated rollout data logrpobs: {}".format(updated_rollout_data['logprobs'].shape))
                logratio = (
                    updated_rollout_data["logprobs"]
                    - rollout_data["logprobs"][mb_inds]
                )
                #print("shape of logratio: {}".format(logratio.shape))
                ratio = logratio.exp() # not changing at the moment, that makes senses
                #print("prob ration in training: {}".format(ratio))
                #print("ration mean: {}".format(ratio.max()))
                #print("ratio has autograd: {}".format(ratio.requires_grad))
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef)
                        .float()
                        .mean()
                        .item()
                    ]

                mb_advantages = advantages[mb_inds]
                #print('advantages: {}'.format(advantages))
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                #print(mb_advantages.cpu().squeeze(-1).numpy().shape)
                #print(ratio.cpu().squeeze(-1).numpy().shape)
                #print(rollout_data[mb_inds]['actions'].cpu().squeeze(-1).numpy().shape)
                #print("cat advangates: {}".format(torch.cat((mb_advantages, ratio, rollout_data[mb_inds]['actions']), dim=1)))
                #print('mb advantages shape: {}'.format(mb_advantages.shape))

                mb_advantages = mb_advantages.repeat_interleave(args.num_agents).reshape(ratio.shape)
                #print('ratio shape: {}'.format(ratio.shape))
                pg_loss1 = -mb_advantages * ratio
                #print('ratio: {}'.format(ratio))
                #print('pg loss 1: {}'.format(pg_loss1))
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                #print('pg loss 2: {}'.format(pg_loss2))
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                #print('pg loss: {}'.format(pg_loss))
                # Value loss
                newvalue = updated_rollout_data["values"]
                if args.clip_vloss:
                    # not adapted for flatland yet
                    v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                    v_clipped = rollout_data["values"][mb_inds] + torch.clamp(
                        newvalue - rollout_data["values"][mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()
                    # print('v_loss: {}'.format(v_loss))

                entropy_loss = updated_rollout_data["entropy"].mean()
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + v_loss * args.vf_coef
                )

                if not args.no_training:
                    #print("updating")
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        td_module.parameters(), args.max_grad_norm
                    )
                    optimizer.step()
                    #print('loss: {}'.format(loss))

                #for param in td_module.actor_net.parameters():
                    #print(param.names)
                    #print('param weights: {}'.format(param.data))
                    #print('grad: {}'.format(param.grad))

                #for param in td_module.actor_net:
                    #if isinstance(param, nn.Linear):
                        #print("actor layer: {}".format(param.weight.data))



            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break
        
        """         for param in td_module.actor_net.parameters():
            print("shape of param: {}".format(param.shape))
            print("content of param: {}".format(param.data))
            print("param has autogra: {}".format(param.requires_grad))
            print(" param gradients: {}".format(param.grad))
            
        for param in td_module.critic_net.parameters():
            print("critic shape of param: {}".format(param.shape))
            print("content of param: {}".format(param.data))
            print("param has autogra: {}".format(param.requires_grad))
            print(" param gradients: {}".format(param.grad)) """
            
        del next_obs['actions']
        del next_obs['logprobs']
        print('update nr: {} out of'.format(update, num_updates))
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if args.track:
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            loss_total = (
                        pg_loss.item()
                        - args.ent_coef * entropy_loss.item()
                        + v_loss.item() * args.vf_coef
                    )
            writer.add_scalar("loss_share/policy", pg_loss.item()/loss_total, global_step)
            writer.add_scalar("loss_share/value", v_loss.item() * args.vf_coef/loss_total, global_step)
            writer.add_scalar("loss_share/entropy", entropy_loss.item() * args.ent_coef/loss_total, global_step)
            writer.add_scalar("losses/total_loss", loss_total, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
    if args.track:
        writer.close()

