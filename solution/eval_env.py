from flatland.evaluators.client import FlatlandRemoteClient
from flatland.envs.rail_env import TrainState
from flatland_cutils import TreeObsForRailEnv as TreeCutils

import numpy as np
from impl_config import FeatureParserConfig as fp


class TestEnvWrapper:
    def __init__(self) -> None:
        self.remote_client = FlatlandRemoteClient()
        self.env = None

    def reset(self):
        feature, _ = self.remote_client.env_create(
            TreeCutils(fp.num_tree_obs_nodes, fp.tree_pred_path_depth)
        )
        self.env = self.remote_client.env
        if feature is False:
            return False
        self.env.number_of_agents = len(self.env.agents)
        self.update_obs_properties()
        stdobs = (feature, self.obs_properties)
        obs_list = [self.parse_features(*stdobs)]
        return obs_list

    def action_required(self):
        return {
            i: self.env.action_required(agent)
            for i, agent in enumerate(self.env.agents)
        }

    def parse_actions(self, actions):
        action_required = self.action_required()
        parsed_action = dict()
        for idx, act in actions.items():
            if action_required[idx]:
                parsed_action[idx] = act
        return parsed_action

    def step(self, actions):
        actions = self.parse_actions(actions)
        feature, reward, done, info = self.remote_client.env_step(actions)
        self.update_obs_properties()
        stdobs = (feature, self.obs_properties)
        obs_list = [self.parse_features(*stdobs)]
        return obs_list, reward, done

    def get_valid_actions(self):  # actor id in plf
        valid_actions = self.obs_properties["valid_actions"]
        return valid_actions  # numpy.array

    def submit(self):
        self.remote_client.submit()

    def update_obs_properties(self):
        self.obs_properties = {}
        properties = self.env.obs_builder.get_properties()
        env_config, agents_properties, valid_actions = properties
        self.obs_properties.update(env_config)
        self.obs_properties.update(agents_properties)
        self.obs_properties["valid_actions"] = valid_actions

    def parse_features(self, feature, obs_properties):
        def _fill_feature(items: np.ndarray, max):
            shape = items.shape
            new_items = np.zeros((max, *shape[1:]))
            new_items[: shape[0], ...] = items
            return new_items

        print("type of feature: {}".format(type(feature)))
        print("shape of feature: {}".format(feature))
        feature_list = {}
        feature_list["agent_attr"] = np.array(feature[0])
        feature_list["forest"] = np.array(feature[1][0])
        feature_list["forest"][feature_list["forest"] == np.inf] = -1
        feature_list["adjacency"] = np.array(feature[1][1])
        feature_list["node_order"] = np.array(feature[1][2])
        feature_list["edge_order"] = np.array(feature[1][3])
        feature_list.update(obs_properties)
        return feature_list

    def final_metric(self):
        assert self.env.dones["__all__"]
        env = self.env

        n_arrival = 0
        for a in env.agents:
            if a.position is None and a.state != TrainState.READY_TO_DEPART:
                n_arrival += 1

        arrival_ratio = n_arrival / env.get_num_agents()
        total_reward = sum(list(env.rewards_dict.values()))
        norm_reward = 1 + total_reward / env._max_episode_steps / env.get_num_agents()

        return arrival_ratio, total_reward, norm_reward


class LocalTestEnvWrapper(TestEnvWrapper):
    def __init__(self, env) -> None:
        self.env = env

    def reset(self):
        feature, _ = self.env.reset()
        self.update_obs_properties()
        stdobs = (feature, self.obs_properties)
        obs_list = [self.parse_features(*stdobs)]
        return obs_list

    def step(self, actions):
        actions = self.parse_actions(actions)
        feature, reward, done, info = self.env.step(actions)
        self.update_obs_properties()
        stdobs = (feature, self.obs_properties)
        obs_list = [self.parse_features(*stdobs)]
        return obs_list, reward, done
