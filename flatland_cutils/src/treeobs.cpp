#include "treeobs.h"
// #include "feature_parser.h"
// namespace py = pybind11;

TreeObsForRailEnv::TreeObsForRailEnv(const int _max_nodes,
                                     const int _max_pred_depth)
    : max_nodes(_max_nodes) {
    // this->agents_loader = AgentsLoader();
    this->predictor = ShortestPathPredictorForRailEnv(_max_pred_depth);

    this->observation_dim = 11;
    this->location_has_agent = {};
    this->location_has_agent_direction = {};
    this->location_has_target = {};
}

void TreeObsForRailEnv::set_env(py::object _env) {
    this->railenv = _env;
    this->predictor.set_env(_env);
    // this->agents_loader.set_env(_env);
}
void TreeObsForRailEnv::reset() {
    this->agents_loader.clear();
    this->agents_loader.reset(this->railenv);
    this->agents_loader.update(this->railenv);
    
    this->rail_loader.reset(this->railenv);
}

std::pair<std::vector<std::vector<float>>, Forest> TreeObsForRailEnv::get_many(
    const std::vector<int>& handles) {
    /*
    Called whenever an observation has to be computed for the `env` environment,
    for each agent with handle in the `handles` list.
    */
    // try {

    this->agents_loader.clear();
    this->agents_loader.update(this->railenv);

    assert((!handles.empty(), "Input Error"));
    this->max_prediction_depth = 0;
    this->predicted_pos.clear();
    this->predicted_dir.clear();
    this->predictions.clear();

    this->predictions =
        this->predictor.get(-1, this->agents_loader, this->rail_loader);

    std::vector<Position> pos_list;
    std::vector<int> dir_list;
    for (int t = 0; t < this->predictor.max_pred_depth + 1; t++) {
        pos_list.clear();
        dir_list.clear();
        for (auto a : handles) {
            if (this->predictions[a].empty()) continue;
            std::array<int, 5> predict = this->predictions[a][t];
            pos_list.push_back({predict[1], predict[2]});
            dir_list.push_back(predict[3]);
        }
        this->predicted_pos.insert(
            {{t, coordinate_to_position(this->rail_loader.width, pos_list)}});
        this->predicted_dir.insert({{t, dir_list}});
    }
    this->max_prediction_depth = this->predicted_pos.size();

    this->location_has_agent.clear();
    this->location_has_agent_direction.clear();
    this->location_has_agent_speed.clear();
    this->location_has_agent_malfunction.clear();
    this->location_has_agent_ready_to_depart.clear();
    this->location_has_target.clear();

    for (auto ag : this->agents_loader.agents) {
        if (!ag.state.is_off_map_state() && ag.position.first != -1) {
            this->location_has_agent[ag.position] = 1;
            this->location_has_agent_direction[ag.position] = ag.direction;
            this->location_has_agent_speed[ag.position] = ag.speed;
            this->location_has_agent_malfunction[ag.position] =
                ag.malfunction_down_counter;
        }
        if (ag.state.is_off_map_state() && ag.initial_position.first != -1) {
            if (isin_map(this->location_has_agent_ready_to_depart,
                         ag.initial_position)) {
                this->location_has_agent_ready_to_depart[ag.initial_position] +=
                    1;
            } else {
                this->location_has_agent_ready_to_depart[ag.initial_position] =
                    0;
            }
        }
    }

    Forest forest;
    for (auto handle : handles) {
        auto [tree_obs, adjacency_list, node_order, edge_order] =
            this->get(agents_loader.agents[handle]);
        std::get<0>(forest).push_back(tree_obs);
        std::get<1>(forest).push_back(adjacency_list);
        std::get<2>(forest).push_back(node_order);
        std::get<3>(forest).push_back(edge_order);
    }

    std::pair<std::vector<std::vector<float>>, Forest> feature =
        this->feature_parser.parse(this->agents_loader, this->rail_loader, forest);

    return feature;
}


Node inline scale_node(Node& node, float max_dist, int n_agents) {
    
    // float max_dist = (rail_loader.height + rail_loader.width)* fp::coeff_dist_target;

    // dist_own_target_encountered
    Node new_node;
    new_node[0] = node[0] != INFINITY ? node[0] / (float)max_dist : -1;

    // dist_other_target_encountered
    new_node[1] = node[1] != INFINITY ? node[1] / (float)max_dist : -1;

    // dist_other_agent_encountered
    new_node[2] = node[2] != INFINITY ? node[2] / (float)max_dist : -1;

    // dist_potential_conflict
    new_node[3] = node[3] != INFINITY ? node[3] / (float)max_dist : -1;

    // dist_unusable_switch
    new_node[4] = node[4] != INFINITY ? node[4] / (float)max_dist : -1;

    // dist_to_next_branch
    new_node[5] = node[5] != INFINITY ? node[5] / (float)max_dist : -1;

    // dist_min_to_target
    new_node[6] = node[6] != INFINITY ? node[6] / (float)max_dist : -1;

    // num_agents_same_direction
    new_node[7] = node[7] != -1 ? node[7] / (float)n_agents : -1;

    // num_agents_opposite_direction
    new_node[8] = node[8] != -1 ? node[8] / (float)n_agents : -1;

    // num_agents_malfunctioning
    new_node[9] = node[9] != -1 ? node[9] / (float)n_agents : -1;

    // speed_min_fractional
    new_node[10] = node[10] != -1 ? (float)node[10] : -1;

    // num_agents_ready_to_depart
    new_node[11] = node[11] != -1 ? node[11] / (float)n_agents : -1;
    return new_node;
}

Tree TreeObsForRailEnv::get(const Agent& agent) {
    std::vector<Node> tree_obs;
    std::vector<Adjacency> adjacency_list;
    std::queue<Cell> waiting_queue;
    std::set<std::pair<Position, int>> visited;  // Position, Grid4Transitions

    auto possible_transitions = get_transitions(
        agent.agent_virtual_position, agent.direction, this->rail_loader.rail);
    int num_transitions = std::accumulate(possible_transitions.begin(),
                                          possible_transitions.end(), 0);

    // Here information about the agent itself is stored
    py::array_t distance_map = agents_loader.distance_map;

    // was referring to TreeObsForRailEnv.Node
    // float malfunction_down_counter = agent.malfunction_down_counter;

    Node root{0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              (float)agent.dist_target,
              0.0,
              0.0,
              (float)agent.num_malfunctions,
              agent.speed,
              0.0};
    // std::cout<<"ROOT: ";
    // for(auto n: root){
    //         std::cout<<n<<" ";
    //     }
    //     std::cout<<std::endl;
    int max_dist = agent.dist_target == INT_INFINITY? agents_loader.max_timesteps: agent.dist_target + 1;
    root = scale_node(root, agents_loader.max_timesteps,
               this->agents_loader.n_agents);
    // std::cout<<this->agents_loader.max_timesteps<<this->agents_loader.n_agents<<std::endl;
    // for(auto n: root){
    //         std::cout<<n<<" ";
    // }
    // std::cout<<std::endl;

    tree_obs.push_back(root);

    int orientation = agent.direction;
    int idx_node = 0;

    if (num_transitions == 1) {
        orientation = np_where(possible_transitions, 1)[0];
    }
    for (auto action_direction : std::vector<int>{-1, 0, 1}) {
        int branch_direction = (orientation + action_direction + 4) % 4;
        if (possible_transitions[branch_direction]) {
            Position new_cell = get_new_position(agent.agent_virtual_position,
                                                 branch_direction);
            waiting_queue.push({new_cell, branch_direction, action_direction, 0,
                                1.0, 1, false});

        } else {
            waiting_queue.push({{-1, -1},
                                branch_direction,
                                action_direction,
                                0,
                                1.0,
                                1,
                                true});  // null cell
        }
    }
    while (tree_obs.size() < (std::size_t)this->max_nodes) {
        idx_node = tree_obs.size();
        auto [branch_observation, branch_visited, curr_cell] =
            _explore_branch(idx_node, agent, waiting_queue);
        // std::cout<<"NODE: ";
        // for(auto n: branch_observation){
        //     std::cout<<n<<" ";
        // }
        // std::cout<<std::endl;
        branch_observation =
            scale_node(branch_observation, agents_loader.max_timesteps,
                this->agents_loader.n_agents);
        // std::cout<<this->agents_loader.max_timesteps<<this->agents_loader.n_agents<<std::endl;
        // for(auto n: branch_observation){
        //     std::cout<<n<<" ";
        // }
        // std::cout<<std::endl;

        tree_obs.push_back(branch_observation);
        auto [_1, agent_orientation, action_direction, idx_parent, _2, _3, _4] =
            curr_cell;
        visited.merge(branch_visited);
        if (idx_parent == INVALID_LABEL) {
            idx_node = INVALID_LABEL;
            action_direction = INVALID_LABEL;
        }
        adjacency_list.push_back({idx_parent, idx_node, action_direction});
    }
    // assert(test_adjacency(adjacency_list));

    auto [node_order, edge_order] =
        calculate_evaluation_orders(adjacency_list, tree_obs.size());
    return std::make_tuple(tree_obs, adjacency_list, node_order, edge_order);
}

std::tuple<Node, std::set<std::pair<Position, int>>, Cell>
TreeObsForRailEnv::_explore_branch(int idx_node, const Agent& agent,
                                   std::queue<Cell>& waiting_queue) {
    /*
    Utility function to compute tree-based observations.
    We walk along the branch and collect the information documented in the get()
    function. If there is a branching point a new node is created and each
    possible branch is explored.
    */

    if (waiting_queue.empty()) {
        Node null_observation;
        null_observation = {INFINITY, INFINITY, INFINITY, INFINITY,
                            INFINITY, INFINITY, INFINITY, -1,
                            -1,       -1,       -1,       -1};
        std::set<std::pair<Position, int>> null_visited;
        Cell null_cell = {{-1, -1}, -1, -1, INVALID_LABEL, -1, -1, true};
        return std::make_tuple(null_observation, null_visited, null_cell);
    }

    Cell curr_cell = waiting_queue.front();
    auto [position, direction, action_direction, idx_parent, tot_dist, depth,
          is_null] = curr_cell;

    waiting_queue.pop();
    if (is_null) {
        Node null_observation;
        null_observation = {INFINITY, INFINITY, INFINITY, INFINITY,
                            INFINITY, INFINITY, INFINITY, -1,
                            -1,       -1,       -1,       -1};
        ;
        std::set<std::pair<Position, int>> null_visited;
        return std::make_tuple(null_observation, null_visited, curr_cell);
    }

    // Continue along direction until next switch or
    // until no transitions are possible along the current direction (i.e.,
    // dead-ends) We treat dead-ends as nodes, instead of going back, to avoid
    // loops
    bool exploring = true;
    bool last_is_switch = false;
    bool last_is_dead_end = false;
    bool last_is_terminal = false;  // wrong cell OR cycle;  either way, we
                                    // don't want the agent to land here
    bool last_is_target = false;
    std::set<std::pair<Position, int>> visited;
    float time_per_cell = 1.0 / agent.speed;
    float own_target_encountered = INFINITY;
    float other_agent_encountered = INFINITY;
    float other_target_encountered = INFINITY;
    float potential_conflict = INFINITY;
    float unusable_switch = INFINITY;
    int other_agent_same_direction = 0;
    int other_agent_opposite_direction = 0;
    int malfunctioning_agent = 0;
    float min_fractional_speed = 1.0;
    int num_steps = 1;
    int other_agent_ready_to_depart_encountered = 0;

    int num_transitions;
    Cell_Transitions cell_transitions;
    Transitions transition_bit;

    // int tmp_exploring = 0;
    while (exploring) {
        //////////////////////////////
        //////////////////////////////
        // Modify here to compute any useful data required to build the end
        // node's features. This code is called for each cell visited between
        // the previous branching node and the next switch / target / dead-end.

        if (isin_map(this->location_has_agent, position)) {
            if ((float)tot_dist < other_agent_encountered) {
                other_agent_encountered = (float)tot_dist;
            }

            // Check if any of the observed agents is malfunctioning, store
            // agent with longest duration left
            if (this->location_has_agent_malfunction[position] >
                malfunctioning_agent) {
                malfunctioning_agent =
                    this->location_has_agent_malfunction[position];
            }

            other_agent_ready_to_depart_encountered +=
                isin_map(this->location_has_agent_ready_to_depart, position)
                    ? this->location_has_agent_ready_to_depart[position]
                    : 0;

            if (this->location_has_agent_direction[position] == direction) {
                // Cummulate the number of agents on branch with same direction
                other_agent_same_direction += 1;

                // Check fractional speed of agents
                float current_fractional_speed =
                    this->location_has_agent_speed[position];
                min_fractional_speed =
                    current_fractional_speed < min_fractional_speed
                        ? current_fractional_speed
                        : min_fractional_speed;
            } else {
                // If no agent in the same direction was found all agents in
                // that position are other direction Attention this counts to
                // many agents as a few might be going off on a switch.
                other_agent_opposite_direction +=
                    this->location_has_agent[position];
            }
        }
        // Check number of possible transitions for agent and total number of
        // transitions in cell (type)
        cell_transitions =
            get_transitions(position, direction, this->rail_loader.rail);

        transition_bit =
            (int)this->rail_loader.rail.at(position.first, position.second);
        int total_transitions = transition_bit.count();
        bool crossing_found =
            transition_bit == 0b1000010000100001 ? true : false;

        // Register possible future conflict
        int predicted_time = (int)tot_dist * time_per_cell;
        if (predicted_time < this->max_prediction_depth) {
            std::vector<Position> pos_list = {position};
            int int_position =
                coordinate_to_position(this->rail_loader.width, pos_list)[0];
            if (tot_dist < this->max_prediction_depth) {
                int pre_step =
                    0 > (predicted_time - 1) ? 0 : (predicted_time - 1);
                int post_step =
                    (this->max_prediction_depth - 1) < (predicted_time + 1)
                        ? (this->max_prediction_depth - 1)
                        : (predicted_time + 1);

                // Look for conflicting paths at distance tot_dist
                std::vector<int> curr_possible_conflicting =
                    get_possible_conflicting(predicted_pos, predicted_time,
                                             agent.handle);
                std::vector<int> pre_possible_conflicting =
                    get_possible_conflicting(predicted_pos, pre_step,
                                             agent.handle);
                std::vector<int> post_possible_conflicting =
                    get_possible_conflicting(predicted_pos, post_step,
                                             agent.handle);

                // Look for conflicting paths at distance tot_dist
                if (isin_vec(curr_possible_conflicting, int_position)) {
                    std::vector<int> conflicting_agent =
                        np_where<std::vector<int>>(
                            predicted_pos[predicted_time], int_position);

                    for (auto ca : conflicting_agent) {
                        if ((direction !=
                             this->predicted_dir[predicted_time][ca]) and
                            (cell_transitions[reverse_dir(
                                 this->predicted_dir[predicted_time][ca])] ==
                             1) and
                            (tot_dist < potential_conflict)) {
                            potential_conflict = (float)tot_dist;
                        }
                        if (this->agents_loader.agents[ca].state ==
                                State::DONE and
                            tot_dist < potential_conflict) {
                            potential_conflict = (float)tot_dist;
                        }
                    }
                } else if (isin_vec(pre_possible_conflicting, int_position)) {
                    // Look for conflicting paths at distance num_step-1
                    std::vector<int> conflicting_agent =
                        np_where<std::vector<int>>(predicted_pos[pre_step],
                                                   int_position);
                    for (auto ca : conflicting_agent) {
                        if (direction !=
                                this->predicted_dir[predicted_time][ca] and
                            cell_transitions[reverse_dir(
                                this->predicted_dir[predicted_time][ca])] ==
                                1 and
                            tot_dist < potential_conflict) {
                            potential_conflict = (float)tot_dist;
                        }
                        if (this->agents_loader.agents[ca].state ==
                                State::DONE and
                            tot_dist < potential_conflict) {
                            potential_conflict = (float)tot_dist;
                        }
                    }
                } else if (isin_vec(post_possible_conflicting, int_position)) {
                    // Look for conflicting paths at distance num_step+1
                    std::vector<int> conflicting_agent =
                        np_where<std::vector<int>>(predicted_pos[post_step],
                                                   int_position);
                    for (auto ca : conflicting_agent) {
                        if (direction !=
                                this->predicted_dir[predicted_time][ca] and
                            cell_transitions[reverse_dir(
                                this->predicted_dir[predicted_time][ca])] ==
                                1 and
                            tot_dist < potential_conflict) {
                            potential_conflict = (float)tot_dist;
                        }
                        if (this->agents_loader.agents[ca].state ==
                                State::DONE and
                            tot_dist < potential_conflict) {
                            potential_conflict = (float)tot_dist;
                        }
                    }
                }
            }
        }

        if (isin_map(this->location_has_target, position) and
            position != agent.target) {
            if (tot_dist < other_target_encountered) {
                other_target_encountered = (float)tot_dist;
            }
        }

        if (position == agent.target and tot_dist < own_target_encountered) {
            own_target_encountered = (float)tot_dist;
        }

        // //////////////////////////////
        std::pair<Position, int> curr_cell = {position, direction};
        if (isin_vec(visited, curr_cell)) {
            last_is_terminal = true;
            break;
        }
        visited.insert(curr_cell);

        // If the target node is encountered, pick that as node. Also, no
        // further branching is possible.
        if (position == agent.target) {
            last_is_target = true;
            break;
        }

        // Check if crossing is found --> Not an unusable switch
        if (crossing_found) {
            total_transitions = 2;
        }
        num_transitions = std::accumulate(cell_transitions.begin(),
                                          cell_transitions.end(), 0);

        exploring = false;

        // Detect Switches that can only be used by other agents.
        if (total_transitions > 2 and 2 > num_transitions and
            tot_dist < unusable_switch) {
            unusable_switch = tot_dist;
        }

        if (num_transitions == 1) {
            // Check if dead-end, or if we can go forward along direction
            int nbits = total_transitions;
            if (nbits == 1) {
                // Dead-end!
                last_is_dead_end = true;
            }
            if (not last_is_dead_end) {
                // Keep walking through the tree along `direction`
                exploring = true;
                // convert one-hot encoding to 0,1,2,3
                direction = np_where(cell_transitions, 1)[0];
                position = get_new_position(position, direction);
                num_steps += 1;
                tot_dist += 1;
            }
        } else if (num_transitions > 0) {
            // Switch detected
            last_is_switch = true;
            break;
        } else if (num_transitions == 0) {
            // Wrong cell type, but let's cover it and treat it as a
            // dead-end, just in case
            std::stringstream ss;
            ss << "WRONG CELL TYPE detected in tree-search (0 transitions "
                  "possible) at cell ";
            ss << position.first << " " << position.second << " " << direction;
            throw(std::invalid_argument(ss.str()));
            last_is_terminal = true;
            break;
        }
    }

    // `position` is either a terminal node or a switch

    // /////////////////////////////
    // /////////////////////////////
    // Modify here to append new / different features for each visited cell!
    float dist_to_next_branch;
    float dist_min_to_target;
    if (last_is_target) {
        dist_to_next_branch = tot_dist;
        dist_min_to_target = 0;
    } else if (last_is_terminal) {
        dist_to_next_branch = INFINITY;
        dist_min_to_target = get_dist_target(agent.handle, position, direction,
                                             this->agents_loader.distance_map);
    } else {
        dist_to_next_branch = tot_dist;
        dist_min_to_target = get_dist_target(agent.handle, position, direction,
                                             this->agents_loader.distance_map);
    }

    // TreeObsForRailEnv.Node
    Node branch_observation = {own_target_encountered,
                               other_target_encountered,
                               other_agent_encountered,
                               potential_conflict,
                               unusable_switch,
                               dist_to_next_branch,
                               dist_min_to_target,
                               (float)other_agent_same_direction,
                               (float)other_agent_opposite_direction,
                               (float)malfunctioning_agent,
                               min_fractional_speed,
                               (float)other_agent_ready_to_depart_encountered};

    // ////////////////////////////
    // ////////////////////////////
    // Start from the current orientation, and see which transitions are
    // available; organize them as [left, forward, right, back], relative to the
    // current orientation Get the possible transitions
    Cell_Transitions possible_transitions =
        get_transitions(position, direction, this->rail_loader.rail);

    for (auto action_direction : std::vector<int>{-1, 0, 1}) {
        int branch_direction = (direction + 4 + action_direction) % 4;
        Position new_pos;
        if (last_is_dead_end and
            possible_transitions[reverse_dir(branch_direction)]) {
            // Swap forward and back in case of dead-end, so that an agent can
            // learn that going forward takes it back
            new_pos = get_new_position(position, reverse_dir(branch_direction));
            waiting_queue.push({new_pos, reverse_dir(branch_direction),
                                action_direction, idx_node, tot_dist + 1,
                                depth + 1, false});
        } else if (last_is_switch and possible_transitions[branch_direction]) {
            new_pos = get_new_position(position, branch_direction);
            waiting_queue.push({new_pos, branch_direction, action_direction,
                                idx_node, tot_dist + 1, depth + 1, false});
        } else {
            // no exploring possible, add just cells with infinity
            waiting_queue.push({{-1, -1},
                                branch_direction,
                                action_direction,
                                idx_node,
                                tot_dist + 1,
                                depth + 1,
                                true});  // null cell
        }
    }
    return std::make_tuple(branch_observation, visited, curr_cell);
}

std::tuple<std::map<std::string, int>,
           std::map<std::string, std::vector<double>>,
           std::vector<std::array<bool, 5>>>
TreeObsForRailEnv::get_properties() {
    std::map<std::string, int> env_config;
    std::map<std::string, std::vector<double>> agents_properties;
    std::vector<std::array<bool, 5>> valid_actions;

    env_config["curr_step"] = this->agents_loader.curr_step;
    env_config["n_agents"] = this->agents_loader.n_agents;
    env_config["max_timesteps"] = this->agents_loader.max_timesteps;
    env_config["height"] = this->rail_loader.height;
    env_config["width"] = this->rail_loader.width;


    for (auto ag : this->agents_loader.agents) {
        agents_properties["dist_target"].push_back((double)ag.dist_target);
        agents_properties["deadlocked"].push_back((double)ag.is_deadlocked);
        agents_properties["ready_not_depart"].push_back(
            (double)(ag.state == State::READY_TO_DEPART));
        agents_properties["earliest_departure"].push_back(
            (double)ag.earliest_departure);
        agents_properties["latest_arrival"].push_back(
            (double)ag.latest_arrival);
        agents_properties["speed"].push_back((double)ag.speed);
        valid_actions.push_back(ag.valid_actions);
    }
    return std::make_tuple(env_config, agents_properties, valid_actions);
}

