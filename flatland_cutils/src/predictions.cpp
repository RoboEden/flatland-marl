#include "treeobs.h"

namespace py = pybind11;

// inline int popcount(int i){
//     // return number of '1' bits
//     i = i - ((i >> 1) & 0x55555555);        // add pairs of bits
//     i = (i & 0x33333333) + ((i >> 2) & 0x33333333); // quads
//     i = (i + (i >> 4)) & 0x0F0F0F0F;        // groups of 8
//     return (i * 0x01010101) >> 24;          // horizontal sum of bytes
// }

inline std::set<RailEnvNextAction> get_valid_move_actions_(
    Grid4Transitions agent_direction, Position agent_position,
    py::array_t<u_int16_t>& rail) {
    std::set<RailEnvNextAction> valid_actions;
    auto possible_transitions =
        get_transitions(agent_position, agent_direction, rail);

    int num_transitions = std::accumulate(possible_transitions.begin(), possible_transitions.end(), 0);
    auto is_dead_end = [&](Position rcPos) -> bool {
        int nbits = 0;
        int tmp = *rail.data(rcPos.first, rcPos.second);
        while (tmp > 0) {
            nbits += (tmp & 1);
            tmp = tmp >> 1;
        }
        return nbits == 1;
    };

    RailEnvActions action;
    Position new_position;
    auto agent_directions = [&](std::vector<int> direction_variety)
        -> std::vector<Grid4Transitions> {
        std::vector<Grid4Transitions> agent_directions;
        for (auto d : direction_variety) {
            agent_directions.push_back(
                Grid4Transitions((agent_direction + d + 4) % 4));
        }
        return agent_directions;
    }(std::vector<int>{-1, 0, 1});
    if (is_dead_end(agent_position)) {
        action = RailEnvActions::MOVE_FORWARD;
        Grid4Transitions exit_direction = (agent_direction + 2 + 4) % 4;
        if (possible_transitions[exit_direction]) {
            new_position = get_new_position(agent_position, exit_direction);
            valid_actions.insert(
                RailEnvNextAction(action, new_position, exit_direction));
        }
    } else if (num_transitions == 1) {
        action = RailEnvActions::MOVE_FORWARD;
        for (auto new_direction : agent_directions) {
            if (possible_transitions[new_direction]) {
                new_position = get_new_position(agent_position, new_direction);
                valid_actions.insert(
                    RailEnvNextAction(action, new_position, new_direction));
            }
        }
    } else {
        for (auto new_direction : agent_directions) {
            if (possible_transitions[new_direction]) {
                if (new_direction == agent_direction) {
                    action = RailEnvActions::MOVE_FORWARD;
                } else if (new_direction == (agent_direction + 1 + 4) % 4) {
                    action = RailEnvActions::MOVE_RIGHT;
                } else if (new_direction == (agent_direction - 1 + 4) % 4) {
                    action = RailEnvActions::MOVE_LEFT;
                }
                new_position = get_new_position(agent_position, new_direction);
                valid_actions.insert(
                    RailEnvNextAction(action, new_position, new_direction));
            }
        }
    }
    return valid_actions;
}

inline std::map<int, std::list<std::tuple<Position, Grid4Transitions>>>
get_shortest_paths(py::array_t<float> distance_map, int max_depth,
                   int agent_handle, AgentsLoader& agent_loader,
                   RailLoader& rail_loader) {
    std::map<int, std::list<std::tuple<Position, Grid4Transitions>>>
        shortest_paths;
    auto _shortest_path_for_agent = [&](Agent agent)
        -> std::list<std::tuple<Position, Grid4Transitions>> {
        Position position;
        Grid4Transitions direction = Grid4Transitions(agent.direction.value);
        std::list<std::tuple<Position, Grid4Transitions>> shortest_path;
        if (agent.state.is_off_map_state()) {
            position = agent.initial_position;
        } else if (agent.state.is_on_map_state()) {
            position = agent.position;
        } else if (agent.state.value == State::StateEnum::DONE) {
            position = agent.target;
        } else {
            // shortest_paths[agent.handle] = NULL;
            // std::cout<<"shortest_paths error";
            return shortest_path;
        }
        
        float distance = INFINITY;
        int depth = 0;
        float next_action_distance;

        while ((position != agent.position and max_depth == -1) or
               (depth < max_depth)) {
            auto next_actions =
                get_valid_move_actions_(direction, position, rail_loader.rail);
            RailEnvNextAction best_next_action;
            for (auto next_action : next_actions) {
                next_action_distance = distance_map.at(
                    agent.handle, next_action.next_position.first,
                    next_action.next_position.second,
                    next_action.next_direction);
                if (next_action_distance < distance) {
                    best_next_action = next_action;
                    distance = next_action_distance;
                }
            }
            shortest_path.push_back(std::make_tuple(position, direction));
            depth += 1;

            // if there is no way to continue, the rail must be disconnected!
            // (or distance map is incorrect)
            if (best_next_action.is_null) {                
                return shortest_path;
            }
            position = best_next_action.next_position;
            direction = best_next_action.next_direction;
        }
        if (max_depth != -1 || depth < max_depth) {
            shortest_path.push_back(std::make_tuple(position, direction));
        }
        return shortest_path;
    };
    if (agent_handle != -1) {
        shortest_paths[agent_handle] = _shortest_path_for_agent(agent_loader.agents[agent_handle]);
    } else {
        for (auto agent : agent_loader.agents) {
            shortest_paths[agent.handle] = _shortest_path_for_agent(agent);
        }
    }
    return shortest_paths;
}

std::map<int, std::vector<std::array<int, 5>>>
ShortestPathPredictorForRailEnv::get(int handle, AgentsLoader& agent_loader,
                                     RailLoader& rail_loader) {
    // handle == -1 get all agents
    // std::cout<<"Start"<<std::endl;
    std::vector<Agent>& agents = agent_loader.agents;
    if (handle != -1) {
        agents = {agent_loader.agents[handle]};
    }
    // std::cout<<"Distance_map"<<std::endl;
    py::array_t<float> distance_map = agent_loader.distance_map;

    std::map<int, std::list<std::tuple<Position, Grid4Transitions>>>
        shortest_paths = get_shortest_paths(distance_map, this->max_pred_depth,
                                            -1, agent_loader, rail_loader);

    std::map<int, std::vector<std::array<int, 5>>> prediction_dict;
    Position agent_virtual_position;
    // std::cout<<"agent_virtual_position"<<std::endl;
    for (auto agent : agents) {
        // if(agent.handle == 46){
        //     std::cout<<"START agent: "<<agent.handle<<std::endl;
        // }
        

        if (agent.state.is_off_map_state()) {
            agent_virtual_position = agent.initial_position;
        } else if (agent.state.is_on_map_state()) {
            agent_virtual_position = agent.position;
        } else if (agent.state == State::StateEnum::DONE) {
            agent_virtual_position = agent.target;
        } else {
            std::vector<std::array<int, 5>> prediction;
            for (int i = 0; i < this->max_pred_depth+1; i++) {
                prediction.push_back({i, -1, -1, -1, -1});
            }
            prediction_dict[agent.handle] = prediction;
            continue;
        }

        Grid4Transitions agent_virtual_direction = agent.direction;
        float agent_speed = agent.speed;   
        int times_per_cell = int(1 / agent_speed);
        std::vector<std::array<int, 5>> prediction = {
            {0, agent_virtual_position.first, agent_virtual_position.second,
             agent_virtual_direction, 0}};

        std::list<std::tuple<Position, Grid4Transitions>> shortest_path =
            shortest_paths[agent.handle];

        // if there is a shortest path, remove the initial position
        if (!shortest_path.empty()) {
            shortest_path.pop_front();
        }
        Grid4Transitions new_direction = agent_virtual_direction;
        Position new_position = agent_virtual_position;
        std::set<std::tuple<int, int, int>> visited;

        // if(agent.handle == 46){
        //     std::cout<<"medium agent: "<<agent.handle<<std::endl;
        // }
        for (int idx = 0; idx < this->max_pred_depth + 1; idx++) {
            if (new_position == agent.target || shortest_path.empty()) {
                prediction.push_back({idx, new_position.first,
                                      new_position.second, new_direction,
                                      RailEnvActions::STOP_MOVING});
                visited.insert(std::make_tuple(
                    new_position.first, new_position.second, new_direction));
                continue;
            }

            if (idx % times_per_cell == 0) {
                std::tie(new_position, new_direction) = shortest_path.front();
                shortest_path.pop_front();
            }

            // prediction is ready
            prediction.push_back({idx, new_position.first, new_position.second,
                               new_direction, 0});
            visited.insert(
                {new_position.first, new_position.second, new_direction});
        }

        // TODO: very bady side effects for visualization only: hand the
        // dev_pred_dict back instead of setting on env!
        //    self.env.dev_pred_dict[agent.handle] = visited
        prediction_dict[agent.handle] = prediction;
    }
    return prediction_dict;
}
