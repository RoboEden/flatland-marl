#include <bits/stdc++.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cmath>
#include <ctime>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

// #include <boost/stacktrace.hpp>
// #include <boost/exception/all.hpp>

// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))
#define getName(VariableName) #VariableName

#define OUTPUT(a) std::cout << #a << ": " << a << '\n'
#define OUTPUT_VEC(vec)        \
    std::cout << #vec << ": "; \
    for (auto v : vec) {        \
        std::cout << v << " "; \
    }                          \
    std::cout << std::endl
#define INT_INFINITY -2147483648
#define INVALID_LABEL -2

// typedef boost::error_info<struct tag_stacktrace,
// boost::stacktrace::stacktrace> traced;
typedef std::pair<int, int> Position;
typedef std::pair<Position, int> Waypoint;

typedef std::bitset<16> Transitions;
typedef std::array<bool, 4> Cell_Transitions;

typedef std::array<float, 12> Node;
typedef std::array<int, 3> Adjacency;
typedef std::tuple<std::vector<Node>, std::vector<Adjacency>, std::vector<int>,
                   std::vector<int>>
    Tree;  // tree, adjacency_list, node_order, edge_order
typedef std::tuple<std::vector<std::vector<Node>>,
                   std::vector<std::vector<Adjacency>>,
                   std::vector<std::vector<int>>, std::vector<std::vector<int>>>
    Forest;
typedef std::tuple<Position, int, int, int, float, int, bool>
    Cell;  // agent_virtual_position, orientation, action_direction, idx_parent,
           // tot_dist, depth, is_null
           // note: action_direction = {-1: left, 0:
           // forward, 1: right}

// FeatureParserConfig
namespace fp {
const int i_max_sz = 158;
const int j_max_sz = 158;

// Fixed
const int action_sz = 5;
const int state_sz = 7;
const int road_type_sz = 11;
const int transitions_sz = 4 * 4;
// const int max_dist_target = (i_max_sz + j_max_sz) * 4;  // estimate
// const int max_dist_target_diff = i_max_sz < j_max_sz ? i_max_sz : j_max_sz;
const int coeff_dist_target = 8;

const int direction_sz = 4;
const float speed_max = 1.0;
const int speed_max_count = 10;
const int max_num_malfunctions = 10;

// const int tree_data_sz = 6;
// const int tree_distance_sz = 1;
// const int tree_agent_data_sz = 4;
// const float node_sz = 11;
// const float tree_obs_depth = 3;
// const float tree_pred_path_depth = 500;
// const float tree_observation_radius = 500;
// const float dist_map_channel_sz = 4;
};

namespace py = pybind11;

template <class DT = std::chrono::milliseconds,
          class ClockT = std::chrono::steady_clock>
class Timer {
    using timep_t = typename ClockT::time_point;
    timep_t _start = ClockT::now(), _end = {};

   public:
    void tick() {
        _end = timep_t{};
        _start = ClockT::now();
    }

    void tock() { _end = ClockT::now(); }

    template <class T = DT>
    auto duration() const {
        // gsl_Expects(_end != timep_t{} && "toc before reporting");
        return std::chrono::duration_cast<T>(_end - _start);
    }
};

struct RailEnvActions {
    enum RailEnvActionsEnum {
        DO_NOTHING = 0,  // implies change of direction in a dead-end!
        MOVE_LEFT = 1,
        MOVE_FORWARD = 2,
        MOVE_RIGHT = 3,
        STOP_MOVING = 4
    };
    RailEnvActionsEnum value;

    constexpr operator RailEnvActionsEnum() const { return this->value; }
    RailEnvActions() {}
    RailEnvActions(RailEnvActionsEnum _value) : value(_value) {}
    RailEnvActions(int _value) {
        this->value = std::map<int, RailEnvActionsEnum>{
            {0, RailEnvActionsEnum::DO_NOTHING},
            {1, RailEnvActionsEnum::MOVE_LEFT},
            {2, RailEnvActionsEnum::MOVE_FORWARD},
            {3, RailEnvActionsEnum::MOVE_RIGHT},
            {4, RailEnvActionsEnum::STOP_MOVING}}[_value];
    }
    RailEnvActions operator=(RailEnvActions other) {
        this->value = other.value;
        return *this;
    }
    friend bool operator<(const RailEnvActions& l, const RailEnvActions& r) {
        return l.value < r.value;
    }
    friend bool operator==(const RailEnvActions& l, const RailEnvActions& r) {
        return l.value == r.value;
    }
    friend bool operator==(const RailEnvActions& l,
                           const RailEnvActionsEnum& value) {
        return l.value == value;
    }
};

struct Grid4Transitions {
    enum Grid4TransitionsEnum {
        NORTH = 0,
        EAST = 1,
        SOUTH = 2,
        WEST = 3,
    };
    Grid4TransitionsEnum value;
    constexpr operator Grid4TransitionsEnum() const { return this->value; }
    Grid4Transitions() {}
    Grid4Transitions(Grid4TransitionsEnum _value) : value(_value) {}
    Grid4Transitions(int _value) {
        this->value = std::map<int, Grid4TransitionsEnum>{
            {0, Grid4TransitionsEnum::NORTH},
            {1, Grid4TransitionsEnum::EAST},
            {2, Grid4TransitionsEnum::SOUTH},
            {3, Grid4TransitionsEnum::WEST},
        }[_value];
    }
    Grid4Transitions operator=(Grid4Transitions other) {
        this->value = other.value;
        return *this;
    }
    friend bool operator==(const Grid4Transitions& l,
                           const Grid4Transitions& r) {
        return l.value == r.value;
    }
    friend bool operator==(const Grid4Transitions& l,
                           const Grid4TransitionsEnum& value) {
        return l.value == value;
    }
    friend bool operator==(const Grid4Transitions& l, const int& value) {
        return l.value == value;
    }
    friend bool operator<(const Grid4Transitions& l,
                          const Grid4Transitions& r) {
        return l.value < r.value;
    }
};

struct State {
    enum StateEnum {
        WAITING = 0,
        READY_TO_DEPART = 1,
        MALFUNCTION_OFF_MAP = 2,
        MOVING = 3,
        STOPPED = 4,
        MALFUNCTION = 5,
        DONE = 6
    };
    constexpr operator StateEnum() const { return this->value; }
    StateEnum value;
    State() {}
    State(StateEnum _value) : value(_value) {}
    State(int _value) {
        this->value =
            std::map<int, StateEnum>{{0, StateEnum::WAITING},
                                     {1, StateEnum::READY_TO_DEPART},
                                     {2, StateEnum::MALFUNCTION_OFF_MAP},
                                     {3, StateEnum::MOVING},
                                     {4, StateEnum::STOPPED},
                                     {5, StateEnum::MALFUNCTION},
                                     {6, StateEnum::DONE}}[_value];
    }
    State(std::string _value) {
        this->value = std::map<std::string, StateEnum>{
            {"TrainState.WAITING", StateEnum::WAITING},
            {"TrainState.READY_TO_DEPART", StateEnum::READY_TO_DEPART},
            {"TrainState.MALFUNCTION_OFF_MAP", StateEnum::MALFUNCTION_OFF_MAP},
            {"TrainState.MOVING", StateEnum::MOVING},
            {"TrainState.STOPPED", StateEnum::STOPPED},
            {"TrainState.MALFUNCTION", StateEnum::MALFUNCTION},
            {"TrainState.DONE", StateEnum::DONE}}[_value];
    }
    bool is_malfunction_state() {
        return false or value == MALFUNCTION or value == MALFUNCTION_OFF_MAP;
    }
    bool is_off_map_state() {
        return false or value == WAITING or value == READY_TO_DEPART or
               value == MALFUNCTION_OFF_MAP;
    }
    bool is_on_map_state() {
        return false or value == MOVING or value == STOPPED or
               value == MALFUNCTION;
    }

    State operator=(State other) {
        this->value = other.value;
        return *this;
    }
    friend bool operator<(const State& l, const State& r) {
        return l.value < r.value;
    }
    friend bool operator==(const State& l, const State& r) {
        return l.value == r.value;
    }
    friend bool operator==(const State& l, const StateEnum& value) {
        return l.value == value;
    }
};

template <typename Type, std::size_t... sizes>
auto concatenate_array(const std::array<Type, sizes>&... arrays) {
    std::array<Type, (sizes + ...)> result;
    std::size_t index{};

    ((std::copy_n(arrays.begin(), sizes, result.begin() + index),
      index += sizes),
     ...);

    return result;
}

template <typename T>
std::vector<T> operator+(std::vector<T> const& x, std::vector<T> const& y) {
    std::vector<T> vec;
    vec.reserve(x.size() + y.size());
    vec.insert(vec.end(), x.begin(), x.end());
    vec.insert(vec.end(), y.begin(), y.end());
    return vec;
}

template <typename T>
std::vector<T>& operator+=(std::vector<T>& x, const std::vector<T>& y) {
    x.reserve(x.size() + y.size());
    x.insert(x.end(), y.begin(), y.end());
    return x;
}

// template <typename T>
// std::ostream& print(std::ostream& out, T const& val) {
//     return (out << val << " ");
// }

// template <typename T1, typename T2>
// std::ostream& print(std::ostream& out, std::pair<T1, T2> const& val) {
//     return (out << "{" << val.first << " " << val.second << "} ");
// }

// template <template <typename, typename...> class TT, typename... Args>
// std::ostream& operator<<(std::ostream& out, TT<Args...> const& cont) {
//     for (auto&& elem : cont) print(out, elem);
//     return out;
// }

inline int rotate_transition(Transitions cell_transition, int rotation) {
    // Rotate the individual bits in each block
    Transitions value = cell_transition;
    rotation = rotation / 90;
    std::string block;
    auto get_trans = [](const Transitions& cell_transition,
                        int orientation) -> std::string {
        Transitions bits = (cell_transition << (orientation)*4);
        return bits.to_string().substr(0, 4);
    };
    auto set_trans = [](Transitions cell_transition, int orientation,
                        std::string new_transitions) -> Transitions {
        Transitions mask =
            (1 << ((4 - orientation) * 4)) - (1 << ((3 - orientation) * 4));
        Transitions negmask = ~mask;

        cell_transition =
            (cell_transition & negmask) |
            (Transitions(new_transitions) << ((3 - orientation) * 4));
        return cell_transition;
    };
    for (int i = 0; i < 4; i++) {
        block = get_trans(value, i);
        block = std::string(
                    block.substr(4 - rotation, block.size() - (4 - rotation))) +
                std::string(block.substr(0, 4 - rotation));
        value = set_trans(value, i, block);
    }

    // Rotate the 4-bits blocks
    value = ((value & Transitions(std::pow(2, (rotation * 4)) - 1))
             << ((4 - rotation) * 4)) |
            (value >> (rotation * 4));
    cell_transition = value;
    return cell_transition.to_ulong();
}

inline Cell_Transitions get_transitions(Position agent_position,
                                        Grid4Transitions agent_direction,
                                        const py::array_t<u_int16_t>& rail) {
    // bool debug = false;
    // if(debug){
    //     std::cout<< "agent_position: ";
    //     print(std::cout, agent_position);
    //     print(std::cout, agent_direction);
    // }

    int cell_transition = rail.at(agent_position.first, agent_position.second);
    Grid4Transitions orientation = Grid4Transitions(agent_direction);
    int bits = (cell_transition >> ((3 - orientation.value) * 4));
    return (Cell_Transitions){(bits >> 3) & 1, (bits >> 2) & 1, (bits >> 1) & 1,
                              (bits)&1};
};

inline Position get_new_position(Position pos, int movement) {
    std::vector<std::pair<int, int>> MOVEMENT_ARRAY = {
        {-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    Position tmp =
        std::pair<int, int>{pos.first + MOVEMENT_ARRAY[movement].first,
                            pos.second + MOVEMENT_ARRAY[movement].second};
    // std::cout<<"Position: "<<pos.first<<" "<<pos.second;
    // std::cout<<" movement: "<<movement<<std::endl;
    // std::cout<<" MOVEMENT_ARRAY: "<<MOVEMENT_ARRAY[movement].first<<"
    // "<<MOVEMENT_ARRAY[movement].second; std::cout<<" result: "<<tmp.first<<"
    // "<<tmp.second<<std::endl;
    return tmp;
}

inline std::vector<int> coordinate_to_position(
    int depth, const std::vector<Position>& coords) {
    /* Converts coordinates to positions::

        [ (0,0) (0,1) ..  (0,w-1)
          (1,0) (1,1)     (1,w-1)
            ...
          (d-1,0) (d-1,1)     (d-1,w-1)
        ]

         -->

        [ 0      d    ..  (w-1)*d
          1      d+1
          ...
          d-1    2d-1     w*d-1
        ]

    Parameters
    ----------
    depth : int
    positions : List[Tuple[int,int]]
    */
    std::vector<int> positions;
    for (auto t : coords) {
        // Set None type coordinates off the grid
        if (t.first == -1) {
            positions.push_back(-1);
        } else {
            positions.push_back(int(t.second * depth + t.first));
        }
    }
    return positions;
}

template <typename Container>
std::vector<int> np_where(Container const& vec, int target) {
    std::vector<int> indices;
    auto it = vec.begin();
    while ((it = std::find_if(it, vec.end(), [&](int const& e) {
                return e == target;
            })) != vec.end()) {
        indices.push_back(std::distance(vec.begin(), it));
        it++;
    }
    return indices;
}

inline int reverse_dir(int direction) { return int((direction + 2 + 4) % 4); }

template <typename Vector, typename FindValue>
inline bool isin_vec(Vector const& vec, FindValue const& target) {
    return std::find(vec.begin(), vec.end(), target) != vec.end();
}

template <typename Map, typename FindValue>
inline bool isin_map(Map const& m, FindValue const& target) {
    return m.find(target) != m.end();
}

inline std::vector<int> get_possible_conflicting(
    std::unordered_map<int, std::vector<int>>& predicted_pos, int time,
    int agent_handle) {
    std::vector<int> possible_conflicting = predicted_pos[time];
    possible_conflicting.erase(possible_conflicting.begin() + agent_handle);
    return possible_conflicting;
}

inline float get_dist_target(int handle, Position pos, int direction,
                             const py::array_t<float>& distance_map) {
    return distance_map.at(handle, pos.first, pos.second, direction);
}

bool inline test_adjacency(const std::vector<Adjacency>& adjacency_list) {
    std::unordered_set<int> parents;
    for (auto r : adjacency_list) {
        parents.insert(r[0]);
    }
    for (auto node : parents) {
        int left = 0;
        int forward = 0;
        int right = 0;
        for (auto r : adjacency_list) {
            if (r[0] == node and r[2] == -1) {
                left++;
            }
            if (r[0] == node and r[2] == 0) {
                forward++;
            }
            if (r[0] == node and r[2] == 1) {
                right++;
            }
        }
        if (left != 1 or forward != 1 or right != 1) {
            return false;
        }
    }
    return true;
}

inline std::tuple<std::vector<int>, std::vector<int>>
calculate_evaluation_orders(const std::vector<Adjacency>& adjacency_list,
                            int tree_size) {
    // https://github.com/unbounce/pytorch-tree-lstm/blob/master/treelstm/util.py#L8
    std::vector<int> node_order(tree_size, 0);
    std::vector<int> edge_order;
    std::unordered_set<int> unevaluated_nodes;

    int order = 0;
    int parent, child;
    for (auto relation : adjacency_list){
        parent = relation[0];
        child = relation[1];
        if(parent != -2){
            unevaluated_nodes.insert(parent);
        }
        if(child != -2){
            unevaluated_nodes.insert(child);
        }
    }
    for (int node_idx = unevaluated_nodes.size(); node_idx < tree_size; node_idx++){
        node_order[node_idx] = INVALID_LABEL;
    }

    std::unordered_set<int> unready_parents;
    while (!unevaluated_nodes.empty()) {
        unready_parents.clear();
        for (auto relation : adjacency_list) {
            parent = relation[0];
            child = relation[1];
            if (unevaluated_nodes.find(child) != unevaluated_nodes.end()) {
                unready_parents.insert(parent);
            }
        }
        for (auto it_node = unevaluated_nodes.begin();
             it_node != unevaluated_nodes.end();) {
            if (unready_parents.find(*it_node) == unready_parents.end()) {
                node_order[*it_node] = order;
                it_node = unevaluated_nodes.erase(it_node);
            } else {
                ++it_node;
            }
        }
        order++;
    }

    for (auto relation : adjacency_list) {
        parent = relation[0];
        if(parent < 0){
            edge_order.push_back(INVALID_LABEL);
        }else{
            edge_order.push_back(node_order[parent]);
        }
        
    }
    return std::make_tuple(node_order, edge_order);
}
