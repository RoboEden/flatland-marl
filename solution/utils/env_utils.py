from typing import Dict

import numpy as np
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.step_utils.states import TrainState

direction_diff_to_action = [
    RailEnvActions.MOVE_FORWARD,
    RailEnvActions.MOVE_RIGHT,
    ValueError("Not possible to turn backward."),
    RailEnvActions.MOVE_LEFT,
]


def is_branch_cell(env: RailEnv, position: tuple) -> bool:
    return bin(env.rail.grid[position[0], position[1]]).count("1") > 2


def get_possible_next_cells(
    env: RailEnv, position: tuple, direction: int
) -> Dict[Grid4TransitionsEnum, tuple]:
    """Get possible next cells for a cell.
    Args:
        env: RailEnv instance.
        position: Position of the cell.
        direction: Direction of the cell.
    Returns:
        Dict[Grid4TransitionsEnum, tuple]: Possible next cells.
    """
    possible_transitions: list[bool] = env.rail.get_transitions(*position, direction)
    next_cells = {}
    for direction in Grid4TransitionsEnum:
        if possible_transitions[direction]:
            next_cells[direction] = get_new_position(position, direction)
    return next_cells


def get_possible_actions(env: RailEnv, agent_handle: int) -> Dict[RailEnvActions, Dict]:
    """Get possible actions for an agent given its current state and postion.

    Args:
        env (RailEnv):      rail environment.
        agent_handle (int): agent id.

    Raises:
        ValueError: If the train is in a state we forget to take care of, then raise a ValueError.

    Returns:
        dict[RailEnvActions, dict]: a diction of possible actions and infos.
            'distance': distance to target if the action is taken. inf if the there is no way to target after then action is taken.
            'position': new postion if the action is taken. None stands for off map.
            'direction': new direction if the action is taken.
        {
            action1: {
                ...
            },
            action2: {
                ...
            },
            ...
        }
    """
    agent = env.agents[agent_handle]
    distance_map = env.distance_map.get()
    actions = {}

    if agent.state == TrainState.READY_TO_DEPART:
        distance = distance_map[
            agent_handle,
            agent.initial_position[0],
            agent.initial_position[1],
            agent.initial_direction,
        ]
        actions[RailEnvActions.MOVE_FORWARD] = {
            "distance": distance,
            "position": agent.initial_position,
            "direction": agent.initial_direction,
        }
        actions[RailEnvActions.STOP_MOVING] = {
            "distance": np.inf,
            "position": None,
            "direction": agent.initial_direction,
        }
    elif agent.state.is_on_map_state():
        possible_transitions: list[bool] = env.rail.get_transitions(
            *agent.position, agent.direction
        )
        for direction in Grid4TransitionsEnum:
            if possible_transitions[direction]:
                action = direction_diff_to_action[direction - agent.direction]
                position = get_new_position(agent.position, direction)
                for ag in env.agents:
                    if ag.position == position and ag.direction != direction:
                        break
                else:
                    distance = distance_map[
                        agent_handle, position[0], position[1], direction
                    ]
                    actions[action] = {
                        "distance": distance,
                        "position": position,
                        "direction": direction,
                    }
        actions[RailEnvActions.STOP_MOVING] = {
            "distance": distance_map[
                agent_handle, agent.position[0], agent.position[1], agent.direction
            ],
            "position": agent.position,
            "direction": agent.direction,
        }

        # There is only one cell we can move in, and that cell is not a branch
        # cell, then moving in is optimal action.
        if len(actions) == 2:
            for act, info in actions.items():
                if act != RailEnvActions.STOP_MOVING:
                    break
            # next cell is not a branch
            if bin(env.rail.grid[info["position"]]).count("1") == 2:
                actions.pop(RailEnvActions.STOP_MOVING)
    elif agent.state in [
        TrainState.DONE,
        TrainState.WAITING,
        TrainState.MALFUNCTION_OFF_MAP,
    ]:
        pass  # nothing to do, return empty dict
    else:
        raise ValueError("Unknown state: {}".format(agent.state))

    return actions


def is_conflict(env, pos, d, agent_handle=None):
    # get agent in the cell
    for agent in env.agents:
        not_self = agent.handle != agent_handle
        if not_self and agent.position == pos:
            break
    else:
        # no agent in the cell
        return False
    if agent.direction != d:
        next_cells_of_conflicted_agent = get_possible_next_cells(
            env, agent.position, agent.direction
        )
        # the only possible going direction of the agent is opposite direction
        is_reverse = (d + 2) % 4 in next_cells_of_conflicted_agent
        if len(next_cells_of_conflicted_agent) == 1 and is_reverse:
            return True
    return False


def get_conflict(env: RailEnv, position, direction, target, visited=None):
    def get_conflict_in_branch_cell(env, position, direction, target, visited):
        if is_conflict(env, position, direction):
            return True
        next_cells = get_possible_next_cells(env, position, direction)
        conflict_list = [
            get_conflict(env, position, direction, target, visited)
            for direction, position in next_cells.items()
        ]
        rval = np.all(conflict_list)
        return rval

    if visited is None:
        visited = {}
    if (position, direction) in visited:
        return visited[position, direction]
    # current cell is a branch
    if is_branch_cell(env, position) and (position, direction) not in visited:
        visited[position, direction] = False
        rval = get_conflict_in_branch_cell(env, position, direction, target, visited)
        visited[position, direction] = rval
        return rval

    # current cell
    pos, d = position, direction
    while True:
        if pos == target:
            return False
        if is_conflict(env, pos, d):
            return True
        if is_branch_cell(env, pos) and env.rail.grid[pos] != 0b1000010000100001:
            # return get_conflict(env, pos, d, target, visited)
            break
        next_cells = get_possible_next_cells(env, pos, d)
        # handle a bug in flatland
        if len(next_cells) == 0:
            break
        if len(next_cells) > 1:
            import ipdb

            ipdb.set_trace()
        assert (
            len(next_cells) == 1
        ), f"next_cells =  {next_cells}, trans = {bin(env.rail.grid[pos[0], pos[1]])}, direction= {d}"

        d, pos = next(iter(next_cells.items()))
    return False
