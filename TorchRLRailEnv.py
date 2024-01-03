from typing import Optional

import torch
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_env import RailEnv
from flatland.envs.step_utils.states import TrainState
from tensordict.tensordict import TensorDict
from torchrl.data import (CompositeSpec, DiscreteTensorSpec,
                          UnboundedContinuousTensorSpec,
                          UnboundedDiscreteTensorSpec)
from torchrl.envs.common import EnvBase

class TDRailEnv(RailEnv):
    """Custom version of default flatland rail env that changes in- and outputs to tensordicts

    Methods:
        - obs_to_dt: Return the list of observations given by the envirnoment as a tensordict
        - reset: Extend the default flatland reset method by returning a tensordict
        - step: Extend the default flatland step by accepting and returning a tensordict
        - update_step_reward: Override default flatland update_step_reward, allowing for custom reward functions upon step completion
        - set_reward_coef: Set the coefficients used in the reward calculation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_reward_coef = 0
        self.shortest_path_reward_coef = 0
        self.arrival_reward_coef = 0
        self.deadlock_penalty_coef = 0
        self.departure_reward_coef = 0
        self.arrival_delay_penalty_coef = 0
        self.previous_deadlocked = None

    def obs_to_td(self, obs_list):
        """Return the observation as a tensordict."""
        obs_td = TensorDict(
            {
                "agents_attr": torch.tensor(obs_list[0], dtype=torch.float32),
                "node_attr": torch.tensor(  # change name from forest in demo plfActor.py
                    obs_list[1][0], dtype=torch.float32
                ),
                "adjacency": torch.tensor(obs_list[1][1], dtype=torch.int64),
                "node_order": torch.tensor(obs_list[1][2], dtype=torch.int64),
                "edge_order": torch.tensor(obs_list[1][3], dtype=torch.int64),
            },
            [self.get_num_agents()],
        )
        return obs_td

    def reset(self, tensordict=None, random_seed=None):
        """Extend default flatland reset by returning a tensordict."""
        # get observation
        # print("entering reset env")
        observations, _ = super().reset(random_seed=random_seed)
        print(f"max episode steps: {self._max_episode_steps}")

        # use default for the moment, give enough time to train
        """ if args.max_episode_steps is not None:
            for agent_i, agent in enumerate(self.agents):
                agent.earliest_departure = 0
                agent.latest_arrival = args.max_episode_steps
            
            self._max_episode_steps = args.max_episode_steps """

        if tensordict is None:
            tensordict_out = TensorDict({}, batch_size=[])
            tensordict_out["agents"] = TensorDict({}, batch_size=[])
        tensordict_out["agents"]["observation"] = self.obs_to_td(observations)
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
        """Extend default flatland step by returning a tensordict."""
        # print('entering step function')
        # print("shortest path: {}".format((self.agents[0].get_shortest_path(self.distance_map)[0])))

        actions = {
            handle: action.item()
            for handle, action in enumerate(tensordict["agents"]["action"].flatten())
        }
        observations, rewards, done, _ = super().step(actions)
        # print(f'done: {done["__all__"]}')
        (
            _,
            _,
            valid_actions,
        ) = self.obs_builder.get_properties()
        return_td = TensorDict(
            {"agents": TensorDict({}, []), "stats": TensorDict({}, [])}, batch_size=[]
        )
        # rewards_td = TensorDict({'agents': TensorDict({}, [])}, batch_size=[])
        return_td["agents"]["observation"] = self.obs_to_td(observations)
        return_td["agents"]["reward"] = torch.tensor(
            [value for _, value in rewards.items()], dtype=torch.float32
        )
        return_td["done"] = torch.tensor(done["__all__"]).type(torch.bool)
        return_td["agents"]["observation"]["valid_actions"] = torch.tensor(
            valid_actions, dtype=torch.bool
        )
        n_arrival = 0
        n_on_time_arrival = 0
        for agent in self.agents:
            if agent.state == TrainState.DONE:
                n_arrival += 1
                if agent.arrival_time <= agent.latest_arrival:
                    n_on_time_arrival += 1
        return_td[("stats", "arrival_ratio")] = torch.tensor(
            n_arrival / self.number_of_agents, dtype=torch.float32
        )
        return_td[("stats", "on_time_arrival_ratio")] = torch.tensor(
            n_arrival / self.number_of_agents, dtype=torch.float32
        )
        self.previous_deadlocked = self.motionCheck.svDeadlocked
        n_deadlocked_agents = len(self.previous_deadlocked)
        return_td[("stats", "deadlock_ratio")] = torch.tensor(
            n_deadlocked_agents / self.number_of_agents, dtype=torch.float32
        )
        # print(f'rewards: {return_td[("agents", "reward")]}')

        return return_td

    def update_step_rewards(self, i_agent):
        agent = self.agents[i_agent]
        delay_reward = (
            shortest_path_reward
        ) = (
            arrival_reward
        ) = deadlock_penalty = departure_reward = arrival_delay_penalty = 0
        if self.delay_reward_coef != 0:
            # we start giving the model rewards as soon as the agent can actually choose actions
            if (
                agent.earliest_departure <= self._elapsed_steps
            ) and agent.state != TrainState.DONE:
                delay_reward = min(
                    agent.get_current_delay(self._elapsed_steps, self.distance_map), 0
                )  # only give reward if delayed

        if self.shortest_path_reward_coef != 0:
            if (
                agent.earliest_departure <= self._elapsed_steps
            ) and agent.state != TrainState.DONE:
                shortest_path_reward = agent.get_current_delay(
                    self._elapsed_steps, self.distance_map
                )

        if self.arrival_reward_coef != 0:
            if (
                agent.state == TrainState.DONE
                and agent.state_machine.previous_state != TrainState.DONE
                and self._elapsed_steps <= agent.latest_arrival
            ):
                # only give arrival reward once
                arrival_reward = 1

        if self.deadlock_penalty_coef != 0:
            if (agent.position in self.motionCheck.svDeadlocked) and (
                agent.position not in self.previous_deadlocked
            ):
                deadlock_penalty = -1

        if self.departure_reward_coef != 0:
            if (
                agent.state.is_on_map_state()
                and agent.state_machine.previous_state.is_off_map_state()
            ):
                departure_reward = 1

        if self.arrival_delay_penalty_coef != 0:
            if (
                agent.state == TrainState.DONE
                and agent.state_machine.previous_state != TrainState.DONE
            ):
                arrival_delay_penalty = min(
                    agent.get_current_delay(self._elapsed_steps, self.distance_map), 0
                )

        self.rewards_dict[i_agent] += (
            self.delay_reward_coef * delay_reward
            + self.shortest_path_reward_coef * shortest_path_reward
            + self.arrival_reward_coef * arrival_reward
            + self.deadlock_penalty_coef * deadlock_penalty
            + self.departure_reward_coef * departure_reward
            + self.arrival_delay_penalty_coef * arrival_delay_penalty
        )

    def set_reward_coef(
        self,
        delay_reward_coef,
        shortest_path_reward_coef,
        arrival_reward_coef,
        deadlock_penalty_coef,
        departure_reward_coef,
        arrival_delay_penalty_coef,
    ):
        """Set the coefficients used in reward calculation

        - delay_reward_coef: As soon as agent can depart, give it a reward corresponding to its negative minimal delay at each step.
            E.g.: If the shortest path to the destination is 5 steps, but there are only 2 steps left before the latest arrival
            time, give a reward of -3.
        - shortest_path_reward_coef: As soon as agent can depart, give a reward corresponding to the time steps necessary to reach
            destination on shortest path from current position (check).
        - arrival_reward_coef: Give a reward equal to arrival_reward_coef upon early or punctual arrival at destination.
        - deadlock_penalty_coef: Give a penalty equal to deadlock_penalty_coef for each agent newly stuck in deadlock.
        - departure_reward_coef: Give a reward equal to departure_reward_coef when agent departs.
        - arrival_delay_penalty_coef: Give a penalty equal to arrival_delay_penalty_coef*delay when agent arrives late.
        """
        self.delay_reward_coef = delay_reward_coef
        self.shortest_path_reward_coef = shortest_path_reward_coef
        self.arrival_reward_coef = arrival_reward_coef
        self.deadlock_penalty_coef = deadlock_penalty_coef
        self.departure_reward_coef = departure_reward_coef
        self.arrival_delay_penalty_coef = arrival_delay_penalty_coef

    def _handle_end_reward(self, agent: EnvAgent) -> int:
        if agent.state == TrainState.DONE:
            # print("end of episode status: {}".format(agent.state))
            return 0

        if (
            self.arrival_delay_penalty_coef != 0
        ):  # only return this if we are giving arrival delay penalties anyway
            return min(
                agent.get_current_delay(self._elapsed_steps, self.distance_map), 0
            )

        return 0

class TorchRLRailEnv(EnvBase):  # class should be camel case
    """
    tbd: add docstring
    """

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.num_agents = env.get_num_agents()
        self._make_spec()
        self.rng = None

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _make_spec(self):
        self.observation_spec = CompositeSpec(
            agents=CompositeSpec(
                observation=CompositeSpec(
                    agents_attr=UnboundedContinuousTensorSpec(
                        shape=[self.num_agents, 83], dtype=torch.float32
                    ),
                    adjacency=UnboundedDiscreteTensorSpec(
                        shape=[self.num_agents, 30, 3], dtype=torch.int64
                    ),
                    node_attr=UnboundedDiscreteTensorSpec(
                        shape=[self.num_agents, 31, 12], dtype=torch.float32
                    ),
                    node_order=UnboundedDiscreteTensorSpec(
                        shape=[self.num_agents, 31], dtype=torch.int64
                    ),
                    edge_order=UnboundedDiscreteTensorSpec(
                        shape=[self.num_agents, 30], dtype=torch.int64
                    ),
                    valid_actions=DiscreteTensorSpec(
                        n=2, dtype=torch.bool, shape=[self.num_agents, 5]
                    ),
                    shape=[self.num_agents],
                ),
                shape=[],
            ),
            stats=CompositeSpec(
                arrival_ratio=UnboundedContinuousTensorSpec(
                    shape=[], dtype=torch.float32
                ),
                on_time_arrival_ratio=UnboundedContinuousTensorSpec(
                    shape=[], dtype=torch.float32
                ),
                deadlock_ratio=UnboundedContinuousTensorSpec(
                    shape=[], dtype=torch.float32
                ),
                shape=[],
            ),
            shape=[],
        )
        self.action_spec = CompositeSpec(
            agents=CompositeSpec(
                action=DiscreteTensorSpec(
                    n=5, shape=[self.num_agents], dtype=torch.int64
                ),
                shape=[],
            ),
            shape=[],
        )

        self.reward_spec = CompositeSpec(
            agents=CompositeSpec(
                reward=UnboundedContinuousTensorSpec(
                    shape=[self.num_agents], dtype=torch.float32
                ),
                shape=[],
            ),
            shape=[],
        )

        self.done_spec = DiscreteTensorSpec(n=2, dtype=torch.bool, shape=[1])

    def _reset(self, tensordict=None):
        return self.env.reset()

    def _step(self, tensordict):
        return self.env.step(tensordict)