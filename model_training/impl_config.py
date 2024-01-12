from typing import Dict


class FeatureParserConfig:
    # Fixed
    action_sz: int = 5
    state_sz: int = 7
    road_type_sz: int = 11
    transitions_sz: int = 4 * 4

    direction_sz: int = 4
    speed_max: float = 1.0
    speed_max_count: int = 10
    max_num_malfunctions: int = 10

    node_sz: int = 12
    num_tree_obs_nodes: int = 1 + 3 * 10
    tree_pred_path_depth: int = 500

    agent_attr: int = 83


class NetworkConfig:
    hidden_sz = 128
    tree_embedding_sz = 128
