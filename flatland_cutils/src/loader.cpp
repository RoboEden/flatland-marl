#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "conflict_checker.h"

// namespace py = pybind11;

Agent::Agent(const py::object& agent_env) {
    this->handle = py::int_(agent_env.attr("handle"));
    this->state = State(std::string(agent_env.attr("state").cast<py::str>()));
    this->malfunction_counter_complete =
        py::bool_(agent_env.attr("state_machine")
                      .attr("st_signals")
                      .attr("malfunction_counter_complete"));
    this->moving = py::bool_(agent_env.attr("moving"));
    this->in_malfunction = py::bool_(agent_env.attr("state_machine")
                                         .attr("st_signals")
                                         .attr("in_malfunction"));

    this->earliest_departure = py::int_(agent_env.attr("earliest_departure"));
    this->latest_arrival = py::int_(agent_env.attr("latest_arrival"));
    this->arrival_time = agent_env.attr("arrival_time").is_none()
                             ? -1
                             : int(py::int_(agent_env.attr("arrival_time")));
    this->speed_max_count =
        py::float_(agent_env.attr("speed_counter").attr("max_count"));
    this->speed = py::float_(agent_env.attr("speed_counter").attr("speed"));
    this->speed_is_cell_entry =
        py::bool_(agent_env.attr("speed_counter").attr("is_cell_entry"));
    this->speed_is_cell_exit =
        py::bool_(agent_env.attr("speed_counter").attr("is_cell_exit"));
    this->speed_counter =
        py::int_(agent_env.attr("speed_counter").attr("counter"));

    this->malfunction_counter_complete =
        py::bool_(agent_env.attr("malfunction_handler")
                      .attr("malfunction_counter_complete"));
    this->malfunction_down_counter = py::bool_(
        agent_env.attr("malfunction_handler").attr("malfunction_down_counter"));
    this->num_malfunctions = py::bool_(
        agent_env.attr("malfunction_handler").attr("num_malfunctions"));

    this->initial_direction =
        Grid4Transitions(py::int_(agent_env.attr("initial_direction")));
    this->direction = Grid4Transitions(py::int_(agent_env.attr("direction")));
    this->old_direction =
        agent_env.attr("old_direction").is_none()
            ? Grid4Transitions(this->direction)
            : Grid4Transitions(py::int_(agent_env.attr("old_direction")));

    if (agent_env.attr("position").is_none()) {
        this->position.first = -1;
        this->position.second = -1;
    } else {
        py::tuple pos(agent_env.attr("position"));
        this->position.first = py::int_(pos[0]);
        this->position.second = py::int_(pos[1]);
    }
    py::tuple init_pos(agent_env.attr("initial_position"));
    if (agent_env.attr("old_position").is_none()) {
        this->old_position.first = -1;
        this->old_position.second = -1;
    } else {
        py::tuple pos(agent_env.attr("old_position"));
        this->old_position.first = py::int_(pos[0]);
        this->old_position.second = py::int_(pos[1]);
    }
    this->initial_position.first = py::int_(init_pos[0]);
    this->initial_position.second = py::int_(init_pos[1]);
    py::tuple target_pos(agent_env.attr("target"));
    this->target.first = py::int_(target_pos[0]);
    this->target.second = py::int_(target_pos[1]);

    if (this->state.is_off_map_state()) {
        this->agent_virtual_position = this->initial_position;
    } else if (this->state.is_on_map_state()) {
        this->agent_virtual_position = this->position;
    } else if (this->state == State::StateEnum::DONE) {
        this->agent_virtual_position = this->target;
    } else {
        throw std::invalid_argument("shortest_paths error");
    }

    this->is_malfunction_state = this->state.is_malfunction_state();
    this->is_off_map_state = this->state.is_off_map_state();
    this->is_on_map_state = this->state.is_on_map_state();

    // bool debug = false;
    // if (debug) {
    //     std::cout << "handle: ";
    //     print(std::cout, handle);
    //     std::cout << std::endl;
    //     print(std::cout, state);
    //     print(std::cout, moving);
    //     print(std::cout, in_malfunction);
    //     print(std::cout, earliest_departure);
    //     print(std::cout, latest_arrival);
    //     print(std::cout, arrival_time);
    //     print(std::cout, speed_max_count);
    //     print(std::cout, speed);
    //     print(std::cout, speed_is_cell_entry);
    //     print(std::cout, speed_is_cell_exit);
    //     print(std::cout, speed_counter);
    //     print(std::cout, malfunction_counter_complete);
    //     print(std::cout, malfunction_down_counter);
    //     print(std::cout, num_malfunctions);
    //     print(std::cout, initial_direction);
    //     print(std::cout, direction);
    //     print(std::cout, old_direction);
    //     print(std::cout, position);
    //     print(std::cout, initial_position);
    //     print(std::cout, old_position);
    //     print(std::cout, target);
    //     print(std::cout, agent_virtual_position);
    //     print(std::cout, is_malfunction_state);
    //     print(std::cout, is_off_map_state);
    //     print(std::cout, is_on_map_state);
    //     std::cout << std::endl;
    // }
}

void Agent::update_transitions(py::array_t<u_int16_t>& rail) {
    static const std::vector<int> transition_list = {
        0b0000000000000000,   // empty cell - Case 0
        0b1000000000100000,   // Case 1 - straight
        0b1001001000100000,   // Case 2 - simple switch
        0b1000010000100001,   // Case 3 - diamond drossing
        0b1001011000100001,   // Case 4 - single slip
        0b1100110000110011,   // Case 5 - double slip
        0b0101001000000010,   // Case 6 - symmetrical
        0b0010000000000000,   // Case 7 - dead end
        0b0100000000000010,   // Case 1b (8)  - simple turn right
        0b0001001000000000,   // Case 1c (9)  - simple turn left
        0b1100000000100010};  // Case 2b (10) - simple switch mirrored;
    if (this->position.first == -1) {
        this->transitions = 0;
        this->cell_transitions = {0, 0, 0, 0};
        this->road_type = 0;
        return;
    } else {
        this->transitions =
            rail.at(this->position.first, this->position.second);
        this->cell_transitions =
            get_transitions(this->position, this->direction, rail);
    }
    // std::cout<<this->transitions.to_string()<<std::endl;
    std::vector<int> trans_similar = {
        (int)this->transitions.to_ulong(),
        rotate_transition(this->transitions, 90),
        rotate_transition(this->transitions, 180),
        rotate_transition(this->transitions, 270),
    };
    for (auto t : trans_similar) {
        int idx = std::find(begin(transition_list), end(transition_list), t) -
                  transition_list.begin();
        if ((size_t)idx != transition_list.size()) {
            this->road_type = idx;
            break;
        }
    }
}

void Agent::update_dist_target(py::array_t<float>& distance_map) {
    this->initial_dist_target =
        get_dist_target(this->handle, this->initial_position,
                        (int)this->initial_direction, distance_map);
    float dist;
    if (state == State::DONE) {
        dist = 0;
    } else if (this->is_off_map_state) {
        dist = this->initial_dist_target;
    } else if (this->is_on_map_state) {
        dist = get_dist_target(this->handle, this->position,
                               (int)this->direction, distance_map);
    } else {
        throw std::invalid_argument("Get error distance!");
    }
    this->dist_target = dist;
}

// void AgentsLoader::set_env(const py::object& railenv) {
//     // this->railenv = railenv;
// }

void AgentsLoader::clear() { this->agents.clear(); }

void AgentsLoader::set_deadlock_checker(int n_agents) {
    if (this->deadlock_checker != nullptr) {
        delete this->deadlock_checker;
    }
    this->deadlock_checker = new DeadlockChecker();
    // this->deadlock_checker = std::make_shared<DeadlockChecker>();

    (*this->deadlock_checker).reset(this);
    // (*this->deadlock_checker).reset(std::shared_ptr<AgentsLoader>(this->shared_from_this()));
    std::shared_ptr p_agents =
        std::make_shared<std::vector<Agent>>(this->agents);
    // (*this->deadlock_checker).reset(p_agents, n_agents);
}

AgentsLoader::~AgentsLoader() {
    if (this->deadlock_checker != nullptr) {
        delete this->deadlock_checker;
    }
}

void AgentsLoader::reset(const py::object& railenv) {
    // this->railenv = railenv;
    this->max_timesteps = py::int_(railenv.attr("_max_episode_steps"));
    auto agents_env = py::list(railenv.attr("agents"));
    this->n_agents = agents_env.size();

    this->distance_map = py::array_t<float>(
        railenv.attr("distance_map")
            .attr("distance_map"));  // n_agents, y_dims, x_dims, 4
    this->rail = py::array_t<u_int16_t>(railenv.attr("rail").attr("grid"));

    this->set_deadlock_checker(this->n_agents);
}

void AgentsLoader::update(const py::object& railenv) {
    auto agents_env = py::list(railenv.attr("agents"));
    this->curr_step = py::int_(railenv.attr("_elapsed_steps"));

    for (int i = 0; i < this->n_agents; i++) {
        // clock_t begin2 = clock();

        auto agent_env = agents_env[i];
        Agent agent(agent_env);
        agent.update_dist_target(this->distance_map);
        agent.update_transitions(this->rail);

        // std::tie(agent.valid_actions, agent.actions_dist, agent.actions_pos,
        //  agent.actions_direction) = get_valid_actions(this->rail, this->distance_map,
        // this->agents,agent, this->max_timesteps);

        /* Get Valid Actions*/
        bool use_valid_acions = false;
        std::array<bool, 5> valid_actions = {false, false, false, false, false};
        if(use_valid_acions){
            auto [actions_dist, actions_pos, actions_direction] = get_possible_actions(
            this->rail, this->distance_map, this->agents, agent, this->max_timesteps);
            State state = agent.state;
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
                        this->rail, this->agents, next_pos, next_direction, agent.target, visited);
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
            agent.actions_dist = actions_dist;
            agent.actions_pos = actions_pos;
            agent.actions_direction = actions_direction;
        } else {
            State state = agent.state;
            if (state == State::MOVING or state == State::STOPPED) {
                if (agent.speed_is_cell_entry) {
                    auto cell_transitions = get_transitions(agent.position, agent.direction, this->rail);
                    bool next_cells_has_branch = false;
                    int next_cell_cnt = 0;
                    for (int action = RailEnvActions::MOVE_LEFT; action <= RailEnvActions::MOVE_RIGHT; action++) {
                        int new_direction = (agent.direction + action - 2 + 4) % 4;
                        valid_actions[action] = cell_transitions[new_direction]; 
                        
                        // auto new_position = get_new_position(agent.position, new_direction);
                        // std::cout<<"cell_transitions: "<<cell_transitions[0]<<" "<<cell_transitions[1]<<" "<<cell_transitions[2]<<" "<<cell_transitions[3]<<std::endl;
                        // std::cout<<"position: "<<agent.position.first<<" "<<agent.position.second<<""<<"direction: "<<agent.direction<<std::endl;
                        // std::cout<<"new_position: "<<new_position.first<<" "<<new_position.second<<"new_direction: "<<new_direction<<std::endl;
                        if(valid_actions[action]){
                            next_cell_cnt += 1;
                            if (is_branch_cell(this->rail, get_new_position(agent.position, new_direction))) {
                                next_cells_has_branch = true;
                            };
                        }
                    }
                    // // only when you are on a branch or you are going to enter a branch, you can stop.
                    if (is_branch_cell(this->rail, agent.position) || next_cell_cnt == 1 && next_cells_has_branch) {
                        valid_actions[RailEnvActions::STOP_MOVING] = true;
                    }
                } else {
                    valid_actions[RailEnvActions::DO_NOTHING] = true;
                }
            } else if (state == State::READY_TO_DEPART) {
                valid_actions[RailEnvActions::MOVE_FORWARD] = true;
                valid_actions[RailEnvActions::STOP_MOVING] = true;
            } else if (state == State::DONE or state == State::WAITING or
                    state == State::MALFUNCTION or
                    state == State::MALFUNCTION_OFF_MAP) {
                valid_actions[RailEnvActions::DO_NOTHING] = true;
            } else {
                throw pybind11::value_error("State not implemented");
            }
        }
        agent.valid_actions = valid_actions;
        // END

        this->agents.push_back(agent);

        // clock_t end2 = clock();
        // printf("Run2 time = %f \n", double(end2 - begin2) / CLOCKS_PER_SEC);
    }

    (*this->deadlock_checker).update_deadlocks();
    for (int i = 0; i < this->n_agents; i++) {
        this->agents[i].is_deadlocked =
            this->deadlock_checker->is_deadlocked(i);
    }
}

void RailLoader::reset(py::object railenv) {
    this->height = py::int_(railenv.attr("height"));
    this->width = py::int_(railenv.attr("width"));
    this->rail = py::array_t<u_int16_t>(railenv.attr("rail").attr("grid"));
}
