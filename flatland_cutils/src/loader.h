#include "tool.h"

// using namespace pybind11::literals;

class Agent {
   public:
    int handle;
    State state;
    bool moving;
    bool in_malfunction;

    int earliest_departure;
    int latest_arrival;
    int arrival_time;
    float speed_max_count;
    float speed;
    bool speed_is_cell_entry;
    bool speed_is_cell_exit;
    int speed_counter;

    bool malfunction_counter_complete;
    int malfunction_down_counter;
    int num_malfunctions;

    Grid4Transitions initial_direction;
    Grid4Transitions direction;
    Grid4Transitions old_direction;

    Position position;
    Position initial_position;
    Position old_position;
    Position target;
    Position agent_virtual_position;

    bool is_malfunction_state;
    bool is_off_map_state;
    bool is_on_map_state;
    bool is_deadlocked;

    ///////
    // int max_timesteps;
    // int curr_step;
    // int n_agents;
    float initial_dist_target;
    float dist_target;
    Transitions transitions;
    Cell_Transitions cell_transitions;
    int road_type;
    std::map<int, int> actions_dist;
    std::map<int, Position> actions_pos;
    std::map<int, Grid4Transitions> actions_direction;
    std::array<bool, 5> valid_actions = {false, false, false, false, false};

    Agent(){}
    Agent(const py::object& agent_env);
    void update_dist_target(py::array_t<float>& distance_map);
    void update_transitions(py::array_t<u_int16_t>& rail);
};

class DeadlockChecker;  // forward declaration
class AgentsLoader : public std::enable_shared_from_this<AgentsLoader> {
   public:
    // py::object railenv;
    int n_agents;
    int max_timesteps;
    int curr_step;
    std::vector<Agent> agents;
    py::array_t<float> distance_map;  // n_agents, i_dims, j_dims, 4
    py::array_t<u_int16_t> rail;
    DeadlockChecker* deadlock_checker = nullptr;
    // std::shared_ptr<DeadlockChecker> deadlock_checker;

    AgentsLoader() {}
    // AgentsLoader::operator =
    // void set_env(const py::object& railenv);

    void set_deadlock_checker(int n_agents);
    ~AgentsLoader();
    void reset(const py::object& railenv);
    void clear();
    void update(const py::object& railenv);
};

class RailLoader {
   public:
    int height;
    int width;
    py::array_t<u_int16_t> rail;
    RailLoader() {}
    void reset(py::object railenv);
};

class DeadlockChecker {
   public:
    int n_agents;
    const AgentsLoader* p_agent_loader;
    // std::shared_ptr<AgentsLoader> p_agent_loader;
    // std::shared_ptr<std::vector<Agent>> agents;

    const std::vector<State> ACTIVE_STATE = {State::MOVING, State::STOPPED,
                                             State::MALFUNCTION};
    std::vector<bool> _is_deadlocked;
    std::vector<bool> _old_deadlock;
    std::map<Position, int> agent_positions;
    std::vector<std::vector<int>> dep;
    std::vector<int> checked;

    DeadlockChecker() {}

    void reset(const AgentsLoader* p_agent_loader);
    // void reset(std::shared_ptr<AgentsLoader> p_agent_loader);
    // void reset(std::shared_ptr<std::vector<Agent>> _agents, int n_agents);

    void update_deadlocks();
    bool _check_blocked(int handle);
    void _fix_deps();
    bool is_deadlocked(int handle);
};
