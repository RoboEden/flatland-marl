#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cassert>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <tuple>
#include <utility>

#include "loader.h"

namespace py = pybind11;
using namespace pybind11::literals;

struct RailEnvNextAction {
    typedef std::pair<int, int> RailEnvGridPos;
    RailEnvActions action;
    RailEnvGridPos next_position;
    Grid4Transitions next_direction;
    bool is_null = false;
    RailEnvNextAction() { this->is_null = true; }
    RailEnvNextAction(RailEnvActions _action, RailEnvGridPos _next_position,
                      Grid4Transitions _next_direction)
        : action(_action),
          next_position(_next_position),
          next_direction(_next_direction) {}
    friend bool operator<(const RailEnvNextAction& l,
                          const RailEnvNextAction& r) {
        return std::tie(l.action, l.next_position.first, l.next_position.second,
                        l.next_direction) <
               std::tie(r.action, r.next_position.first, r.next_position.second,
                        r.next_direction);
    }
    friend bool operator==(const RailEnvNextAction& l,
                           const RailEnvNextAction& r) {
        return std::tie(l.action, l.next_position.first, l.next_position.second,
                        l.next_direction) ==
               std::tie(r.action, r.next_position.first, r.next_position.second,
                        r.next_direction);
    }
};

class ShortestPathPredictorForRailEnv {
   private:
    py::object env;

   public:
    int max_pred_depth;
    ShortestPathPredictorForRailEnv(int _max_pred_depth = 20)
        : max_pred_depth(_max_pred_depth) {}
    std::map<int, std::vector<std::array<int, 5>>> get(
        int handle, AgentsLoader& agent_loader, RailLoader& rail_loader);
    void set_env(py::object _env) { this->env = _env; }
    // void test_prediction();
    void reset();
};

class AgentAttrParser {
   private:
    // std::map<std::string, int> onehot_list = {
    //     {"state", fp::state_sz},
    //     {"road_type", fp::road_type_sz},
    //     //   {"num_malfunctions", fp::max_num_malfunctions},
    //     {"initial_direction", fp::direction_sz},
    //     {"direction", fp::direction_sz},
    //     {"old_direction", fp::direction_sz}};
    // std::vector<std::string> bool_list = {
    //     "moving", "deadlocked", "in_malfunction",
    //     "malfunction_counter_complete",
    //     //   "is_near_next_decision",
    //     "speed_is_cell_entry", "speed_is_cell_exit", "is_malfunction_state",
    //     "is_off_map_state", "is_on_map_state"};
    // std::map<std::string, int> vector_list = {
    //     {"transitions", fp::transitions_sz}, {"valid_actions", fp::action_sz},
    //     // {"act_guide_dist", fp::action_sz},
    //     // {"act_guide_dist_diff", fp::action_sz},
    //     // {"act_guide_direction", fp::action_sz}
    // };
    // std::map<std::string, int> scalar_list = {
    //     {"handle", -1},
    //     {"curr_step", -1},
    //     {"earliest_departure", -1},
    //     {"latest_arrival", -1},
    //     {"arrival_time", -1},
    //     {"step_before_late", -1},
    //     // {"total_time_mal", -1},
    //     // {"totalcurr_time_mal", -1},
    //     {"dist_target", fp::max_dist_target},
    //     {"anticipative_return", -1},
    //     {"speed_max_count", fp::speed_max_count},
    //     {"speed", fp::speed_max},
    //     {"speed_counter", fp::speed_max_count},
    //     {"malfunction_down_counter", fp::max_num_malfunctions},
    //     {"initial_dist_target", fp::max_dist_target}};

   public:
    AgentAttrParser() {}

    std::vector<std::vector<float>> parse(const AgentsLoader& agent_loader,
                                          const RailLoader& rail_loader);
    std::vector<float> get_features(const AgentsLoader& agent_loader,
                                    const RailLoader& rail_loader, int handle);
    // int get_dims() {
    //     int n_dims = 0;
    //     for (const auto& [key, value] : this->onehot_list) {
    //         n_dims += value;
    //     }
    //     for (const auto& [key, value] : this->vector_list) {
    //         n_dims += value;
    //     }
    //     n_dims += bool_list.size();
    //     n_dims += scalar_list.size();
    //     return n_dims;
    // };
};

class FeatureParser {
   public:
    AgentAttrParser agent_attr_parser;
    FeatureParser() {}
    // ~FeatureParser(){
    //     delete
    // }
    std::pair<std::vector<std::vector<float>>, Forest> parse(
        const AgentsLoader& obs_builder, const RailLoader& rail_loader,
        Forest forest);
};

// class FeatureParser;
class TreeObsForRailEnv {
   private:
    py::object railenv;
    RailLoader rail_loader;
    ShortestPathPredictorForRailEnv predictor;
    FeatureParser feature_parser;
    // FeatureParser* p_feature_parser = nullptr;

    int max_nodes, max_prediction_depth, observation_dim;
    std::map<Position, bool> location_has_agent;
    std::map<Position, int> location_has_agent_direction;
    std::map<Position, float> location_has_agent_speed;
    std::map<Position, int> location_has_agent_malfunction;
    std::map<Position, int> location_has_agent_ready_to_depart;
    std::map<Position, bool> location_has_target;
    std::unordered_map<int, std::vector<int>> predicted_pos;
    std::unordered_map<int, std::vector<int>> predicted_dir;
    std::map<int, std::vector<std::array<int, 5>>> predictions;
    std::vector<int> pos_list;
    std::vector<int> dir_list;
    std::vector<std::string> tree_explored_actions_char = {"L", "F", "R", "B"};

   public:
    AgentsLoader agents_loader;
    TreeObsForRailEnv(int max_nodes, int _max_pred_depth);
    void set_env(py::object _env);
    void reset();
    std::pair<std::vector<std::vector<float>>, Forest> get_many(
        const std::vector<int>& handles);
    Tree get(const Agent& agent);
    std::tuple<std::map<std::string, int>,
               std::map<std::string, std::vector<double>>,
               std::vector<std::array<bool, 5>>>
    get_properties();
    std::tuple<Node, std::set<std::pair<Position, int>>, Cell> _explore_branch(
        int idx_parent, const Agent& agent, std::queue<Cell>& waiting_queue);
};
