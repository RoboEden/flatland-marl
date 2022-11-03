import os
import sys
import numpy as np
import torch

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(PATH))
from nn.net_tree import Network


class Actor:
    def __init__(self, model_path) -> None:
        self.net = Network()
        self.net.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )

    def get_actions(self, obs_list, valid_actions, n_agents):
        # obs = torch.from_numpy(np.array(obs)).float()
        feature = self.get_feature(obs_list)

        logits = self.net(*feature)[0][0]
        logits = logits.squeeze().detach().numpy()
        actions = dict()
        valid_actions = np.array(valid_actions)
        for i in range(n_agents):
            if n_agents == 1:
                actions[i] = self._choose_action(valid_actions[i, :], logits)
            else:
                actions[i] = self._choose_action(valid_actions[i, :], logits[i, :])
        return actions

    def _choose_action(self, valid_actions, logits, soft_or_hard_max="soft"):
        def _softmax(x):
            if x.size != 0:
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()
            else:
                return None

        _logits = _softmax(logits[valid_actions != 0])
        if soft_or_hard_max == "soft":
            if valid_actions.nonzero()[0].size == 0:
                valid_actions = np.ones((1, 5))
            np.random.seed(42)
            action = np.random.choice(valid_actions.nonzero()[0], p=_logits)
        else:
            action = valid_actions.nonzero()[0][np.argmax(_logits)]
        return action

    def get_feature(self, obs_list):
        agents_attr = obs_list[0]["agent_attr"]
        agents_attr = torch.unsqueeze(torch.from_numpy(agents_attr), axis=0).to(
            dtype=torch.float32
        )

        forest = obs_list[0]["forest"]
        forest = torch.unsqueeze(torch.from_numpy(forest), axis=0).to(
            dtype=torch.float32
        )

        adjacency = obs_list[0]["adjacency"]
        adjacency = torch.unsqueeze(torch.from_numpy(adjacency), axis=0).to(
            dtype=torch.int64
        )

        node_order = obs_list[0]["node_order"]
        node_order = torch.unsqueeze(torch.from_numpy(node_order), axis=0).to(
            dtype=torch.int64
        )

        edge_order = obs_list[0]["edge_order"]
        edge_order = torch.unsqueeze(torch.from_numpy(edge_order), axis=0).to(
            dtype=torch.int64
        )

        return agents_attr, forest, adjacency, node_order, edge_order
