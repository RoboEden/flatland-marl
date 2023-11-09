#include "treeobs.h"

std::vector<float> AgentAttrParser::get_features(
    const AgentsLoader& agent_loader, const RailLoader& rail_loader,
    int handle) {
    const Agent& agent = agent_loader.agents[handle];
    std::vector<float> agent_attr;

    auto insert_onehot = [](std::vector<float>& agent_attr, int length,
                            int pos) {
        for (int i = 0; i < length; i++) {
            if (i == pos) {
                agent_attr.push_back(1.0);
            } else {
                agent_attr.push_back(0.0);
            }
        }
    };
    insert_onehot(agent_attr, fp::state_sz, agent.state);
    insert_onehot(agent_attr, fp::road_type_sz, agent.road_type);
    insert_onehot(agent_attr, fp::max_num_malfunctions, agent.num_malfunctions);
    insert_onehot(agent_attr, fp::direction_sz, agent.initial_direction);
    insert_onehot(agent_attr, fp::direction_sz, agent.direction);
    insert_onehot(agent_attr, fp::direction_sz, agent.old_direction);

    float moving = (float)(agent.state == State::MOVING);
    float deadlocked = (float)(agent.is_deadlocked);
    float in_malfunction = (float)(agent.in_malfunction);
    float malfunction_counter_complete =
        (float)(agent.malfunction_counter_complete);
    float speed_is_cell_entry = (float)(agent.speed_is_cell_entry);
    float speed_is_cell_exit = (float)(agent.speed_is_cell_exit);
    float is_malfunction_state = (float)(agent.is_malfunction_state);
    float is_off_map_state = (float)(agent.is_off_map_state);
    float is_on_map_state = (float)(agent.is_on_map_state);
    agent_attr += {moving,
                   deadlocked,
                   in_malfunction,
                   malfunction_counter_complete,
                   speed_is_cell_entry,
                   speed_is_cell_exit,
                   is_malfunction_state,
                   is_off_map_state,
                   is_on_map_state};

    std::string str = agent.transitions.to_string();
    // OUTPUT(str);
    for (int i = 0; i < str.size(); i++) {
        agent_attr.push_back((float)(str[i] - '0'));
    }

    std::array<bool, 5> va = agent.valid_actions;
    // OUTPUT_VEC(va);
    for (int i = 0; i < va.size(); i++) {
        agent_attr.push_back((float)va[i]);
    }

    auto isValid = [](float x) { return (x >= 1 and x <= -1); };
    float max_dist_target = (rail_loader.height + rail_loader.width)* fp::coeff_dist_target;

    float agent_handle =
        (float)agent.handle / (float)agent_loader.n_agents;
    float curr_step =
        (float)agent_loader.curr_step / (float)agent_loader.max_timesteps;
    float earliest_departure =
        (float)agent.earliest_departure / (float)agent_loader.max_timesteps;
    float latest_arrival =
        (float)agent.latest_arrival / (float)agent_loader.max_timesteps;
    float arrival_time =
        (float)agent.arrival_time / (float)agent_loader.max_timesteps;
    float step_before_late = latest_arrival - curr_step;
    float dist_target = agent.dist_target == INFINITY
                            ? fp::coeff_dist_target
                            : (float)agent.dist_target / max_dist_target;
    float anticipative_return =
        step_before_late < dist_target ? step_before_late : dist_target;
    float speed_max_count = agent.speed_max_count / fp::speed_max_count;
    float speed = agent.speed / fp::speed_max;
    float speed_counter = (float)agent.speed_counter / fp::speed_max_count;
    float malfunction_down_counter =
        (float)agent.malfunction_down_counter / fp::max_num_malfunctions;

    float initial_dist_target =
        agent.initial_dist_target == INFINITY
            ? fp::coeff_dist_target
            : (float)agent.initial_dist_target / max_dist_target;

    agent_attr += {agent_handle,       curr_step,
                   earliest_departure, latest_arrival,
                   arrival_time,       step_before_late,
                   dist_target,        anticipative_return,
                   speed_max_count,    speed,
                   speed_counter,      malfunction_down_counter,
                   initial_dist_target};

    // OUTPUT_VEC(agent_attr);
    return agent_attr;
}

std::vector<std::vector<float>> AgentAttrParser::parse(
    const AgentsLoader& agent_loader, const RailLoader& rail_loader) {
    std::vector<std::vector<float>> agents_attr;
    for (int handle = 0; handle < agent_loader.n_agents; handle++) {
        agents_attr.push_back(
            this->get_features(agent_loader, rail_loader, handle));
    }
    return agents_attr;
}

std::pair<std::vector<std::vector<float>>, Forest> FeatureParser::parse(
    const AgentsLoader& agent_loader, const RailLoader& rail_loader,
    Forest forest) {
    std::vector<std::vector<float>> agent_attr =
        this->agent_attr_parser.parse(agent_loader, rail_loader);
    auto feature = std::make_pair(agent_attr, forest);
    return feature;
}