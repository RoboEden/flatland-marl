#include "loader.h"

inline std::tuple<std::map<int, int>, std::map<int, Position>,
                  std::map<int, Grid4Transitions>>
get_possible_actions(const py::array_t<u_int16_t>& rail,
                     const py::array_t<int>& distance_map,
                     const std::vector<Agent>& agents, const Agent& agent,
                     int max_dist_target) {
    std::map<int, int> actions_dist;
    std::map<int, Position> actions_pos;
    std::map<int, Grid4Transitions> actions_direction;
    auto max_distance = [&max_dist_target](int distance) -> int {
        return distance == INT_INFINITY ? max_dist_target : distance;
    };
    static const std::vector<int> direction_diff_to_action = {
        RailEnvActions::MOVE_FORWARD,
        RailEnvActions::MOVE_RIGHT,
        // INT_INFINITY,
        RailEnvActions::MOVE_LEFT,
    };
    int distance = max_distance(agent.dist_target);
    Transitions transitions =
        Transitions(rail.at(agent.position.first, agent.position.second));
    Cell_Transitions possible_transitions;
    RailEnvActions action;

    if (agent.state == State::READY_TO_DEPART) {
        int initial_distance = distance_map.at(
            agent.handle, agent.initial_position.first,
            agent.initial_position.second, agent.initial_direction);
        actions_dist[RailEnvActions::MOVE_FORWARD] =
            max_distance(initial_distance);
        actions_pos[RailEnvActions::MOVE_FORWARD] = agent.initial_position;
        actions_direction[RailEnvActions::MOVE_FORWARD] =
            agent.initial_direction;

        actions_dist[RailEnvActions::STOP_MOVING] = max_dist_target;
        actions_pos[RailEnvActions::STOP_MOVING] = {-1, -1};
        actions_direction[RailEnvActions::STOP_MOVING] =
            agent.initial_direction;
    } else if (agent.is_on_map_state) {
        possible_transitions =
            get_transitions(agent.position, agent.direction, rail);
        for (auto direction : (std::vector<Grid4Transitions>){0, 1, 2, 3}) {
            if (possible_transitions[direction]) {
                action = direction_diff_to_action[direction - agent.direction];
                Position position = get_new_position(agent.position, direction);
                std::vector<bool> overlap;
                for (auto ag : agents) {
                    overlap.push_back(ag.position == agent.position and
                                      ag.direction != agent.direction);
                }
                if (std::none_of(overlap.begin(), overlap.end(),
                                 [](bool v) { return v; })) {
                    actions_dist[action] = distance;
                    actions_pos[action] = position;
                    actions_direction[action] = direction;
                }
            }
        }
        actions_dist[RailEnvActions::STOP_MOVING] = distance;
        actions_pos[RailEnvActions::STOP_MOVING] = agent.position;
        actions_direction[RailEnvActions::STOP_MOVING] = agent.direction;

        // There is only one cell we can move in, and that cell is not a branch
        // cell, then moving in is optimal action.
        if (actions_direction.size() == 2) {
            for (const auto& [act, info] : actions_pos) {
                if (act != RailEnvActions::STOP_MOVING) {
                    break;
                }
            }
            // next cell is not a branch
            if (transitions.count() == 2) {
                actions_direction.erase(RailEnvActions::STOP_MOVING);
                actions_dist.erase(RailEnvActions::STOP_MOVING);
                actions_pos.erase(RailEnvActions::STOP_MOVING);
            }
        }
    } else if (agent.state == State::DONE or agent.state == State::WAITING or
               agent.state == State::MALFUNCTION_OFF_MAP) {
    } else {
        std::stringstream ss;
        ss << "Unknown state: ";
        ss << agent.state;
        throw std::invalid_argument(ss.str());
    }
    return std::make_tuple(actions_dist, actions_pos, actions_direction);
}

inline bool is_branch_cell(const py::array_t<u_int16_t>& rail, Position pos) {
    // std::cout<<pos.first<<" "<<pos.second<<std::endl;
    Transitions t = Transitions(rail.at(pos.first, pos.second));
    return t.count() > 2;
}

inline std::map<int, Position> get_possible_next_cells(
    const py::array_t<u_int16_t>& rail, Position pos,
    Grid4Transitions direction) {
    /*Get possible next cells for a cell.
    Args:
        env: RailEnv instance.
        position: Position of the cell.
        direction: Direction of the cell.
    Returns:
        Dict[Grid4TransitionsEnum, tuple]: Possible next cells.
    */
    Cell_Transitions possible_transitions =
        get_transitions(pos, direction, rail);
    std::map<int, Position> next_cell;
    for (auto direction : (std::vector<Grid4Transitions>){0, 1, 2, 3}) {
        if (possible_transitions[direction]) {
            next_cell[direction] = get_new_position(pos, direction);
        }
    }
    // for(auto cell: next_cell){
    //     std::cout<<"pos: "<<pos.first<<" "<<pos.second;
    //     std::cout<<" "<< std::bitset<16> rail.at(pos.first, pos.second)<<" ";
    //     std::cout<<" cell: "<<cell.first<<" "<<cell.second.first<<"
    //     "<<cell.second.second<<std::endl;
    // }
    return next_cell;
}

inline bool is_conflict(const py::array_t<u_int16_t>& rail, Position pos,
                        Grid4Transitions d, const std::vector<Agent>& agents,
                        int agent_handle = -1) {
    // get agent in the cell
    std::vector<bool> conflict_list;
    for (auto agent : agents) {
        bool not_self = agent.handle != agent_handle;
        conflict_list.push_back(not_self and (agent.position == pos));
    }
    if (std::none_of(conflict_list.begin(), conflict_list.end(),
                     [](bool v) { return v; })) {
        // no agent in the cell
        return false;
    }
    int conflict_handle =
        std::find(conflict_list.begin(), conflict_list.end(), true) -
        conflict_list.begin();

    if (agents[conflict_handle].direction != d) {
        std::map<int, Position> next_cells_of_conflicted_agent =
            get_possible_next_cells(rail, agents[conflict_handle].position,
                                    agents[conflict_handle].direction);
        bool is_reverse = isin_map(next_cells_of_conflicted_agent, (d + 2) % 4);
        if (next_cells_of_conflicted_agent.size() == 1 and is_reverse) {
            return true;
        }
    }
    return false;
}

inline bool get_conflict(const py::array_t<u_int16_t>& rail,
                         const std::vector<Agent>& agents, Position position,
                         Grid4Transitions direction, Position target,
                         std::map<std::tuple<int, int, int>, bool>& visited) {
    auto get_conflict_in_branch_cell =
        [](const py::array_t<u_int16_t>& rail, const std::vector<Agent>& agents,
           Position position, Grid4Transitions direction, Position target,
           std::map<std::tuple<int, int, int>, bool>& visited) -> bool {
        if (is_conflict(rail, position, direction, agents)) {
            return true;
        }
        std::map<int, Position> next_cells =
            get_possible_next_cells(rail, position, direction);
        std::vector<bool> conflict_list;
        for (const auto& [d, pos] : next_cells) {
            conflict_list.push_back(
                get_conflict(rail, agents, pos, d, target, visited));
        }
        bool rval = std::all_of(conflict_list.begin(), conflict_list.end(),
                                [](bool v) { return v; });
        return rval;
    };

    std::tuple<int, int, int> cell = {position.first, position.second,
                                      direction};
    if (isin_map(visited, cell)) {
        return visited[cell];
    }

    // current cell is a branch
    if (is_branch_cell(rail, position) and !isin_map(visited, cell)) {
        visited[cell] = false;
        bool rval = get_conflict_in_branch_cell(rail, agents, position,
                                                direction, target, visited);
        visited[cell] = rval;
        return rval;
    }

    auto [pos, d] = std::make_tuple(position, direction);
    std::map<int, Position> next_cells;
    std::map<int, Position>::iterator itr;
    while (true) {
        if (pos == target) {
            return false;
        }
        if (is_conflict(rail, pos, d, agents)) {
            return true;
        }
        if (is_branch_cell(rail, pos) and
            rail.at(pos.first, pos.second) != 0b1000010000100001) {
            break;
        }
        std::map<int, Position> next_cells =
            get_possible_next_cells(rail, pos, d);
        std::vector<bool> conflict_list;
        if (next_cells.size() == 0) {
            break;
        }

        std::stringstream ss;
        ss << "next_cells error: ";
        ss << "trans = " << position.first << " " << position.second << ", ";
        ss << "direction= " << d;
        assertm(next_cells.size() != 1, ss.str());

        itr = next_cells.begin();
        std::tie(d, pos) = *itr;
    }
    return false;
}

inline std::tuple<std::array<bool, 5>, std::map<int, int>,
                  std::map<int, Position>, std::map<int, Grid4Transitions>>
get_valid_actions(const py::array_t<u_int16_t>& rail,
                  const py::array_t<u_int16_t>& distance_map,
                  const std::vector<Agent>& agents, const Agent& agent,
                  int& max_dist_target) {
    
    clock_t begin = clock();

    auto [actions_dist, actions_pos, actions_direction] = get_possible_actions(
        rail, distance_map, agents, agent, max_dist_target);

    clock_t end = clock();
        printf("Run1 time = %f \n", double(end - begin) / CLOCKS_PER_SEC);

    State state = agent.state;
    std::array<bool, 5> valid_actions = {0, 0, 0, 0, 0};
    if (state == State::MOVING or state == State::STOPPED or
        state == State::READY_TO_DEPART) {
        for (auto [action, next_pos] : actions_pos) {
            Grid4Transitions next_direction = actions_direction[action];
            // READY_TO_DEPART
            if (next_pos.first == -1) {
                next_pos = agent.initial_position;
                next_direction = agent.direction;
            }
            std::map<std::tuple<int, int, int>, bool> visited;
            valid_actions[action] = get_conflict(
                rail, agents, next_pos, next_direction, agent.target, visited);
        }
        if (not std::any_of(valid_actions.begin(), valid_actions.end(),
                            [](bool v) { return v; })) {
            valid_actions[RailEnvActions::DO_NOTHING] = true;
            valid_actions[RailEnvActions::MOVE_LEFT] = true;
            valid_actions[RailEnvActions::MOVE_FORWARD] = true;
            valid_actions[RailEnvActions::MOVE_RIGHT] = true;
            valid_actions[RailEnvActions::STOP_MOVING] = true;
        }
    } else if (state == State::DONE or state == State::WAITING or
               state == State::MALFUNCTION or
               state == State::MALFUNCTION_OFF_MAP) {
        valid_actions[RailEnvActions::DO_NOTHING] = true;
    }
    // clock.tock();
    // std::cout << "2 Run time = " << clock.duration().count() << " ms\n";
    // clock.tick();

    auto result = std::make_tuple(valid_actions, actions_dist, actions_pos,
                                  actions_direction);

    // clock.tock();
    // std::cout << "Run time = " << clock.duration().count() << " ms\n";

    // static std::array<bool, 5> valid_actions = {0, 0, 0, 0, 0};
    // static std::map<int, int> actions_dist = {};
    // static std::map<int, Position> actions_pos = {};
    // static std::map<int, Grid4Transitions> actions_direction = {};

    // auto static result = std::make_tuple(valid_actions, actions_dist,
    // actions_pos, actions_direction);

    
    return result;
};