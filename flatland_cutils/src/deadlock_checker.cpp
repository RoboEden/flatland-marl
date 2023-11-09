#include "loader.h"

void DeadlockChecker::reset(const AgentsLoader* _p_agent_loader){
// void DeadlockChecker::reset(std::shared_ptr<std::vector<Agent>> _p_agents, int _n_agents){
    this->p_agent_loader = _p_agent_loader;
    this->n_agents = _p_agent_loader->n_agents;
    this->_is_deadlocked = std::vector<bool> (this->n_agents, false);
    this->_old_deadlock = std::vector<bool> (this->n_agents, false);
}

void DeadlockChecker::update_deadlocks(){
    this->agent_positions.clear();
    this->dep = std::vector<std::vector<int>> (this->n_agents, std::vector<int>{});

    for(auto ag: this->p_agent_loader->agents){
        this->_old_deadlock[ag.handle] = this->_is_deadlocked[ag.handle];
        if(isin_vec(this->ACTIVE_STATE, ag.state)){
            this->agent_positions[ag.position] = ag.handle;
        }
    }
    this->checked = std::vector<int> (this->n_agents, 0);
    for(auto ag: this->p_agent_loader->agents){
        if(isin_vec(this->ACTIVE_STATE, ag.state) and not this->_is_deadlocked[ag.handle] and not this->checked[ag.handle]){
            this->_check_blocked(ag.handle);
        }
    }
    this->_fix_deps();
}

bool DeadlockChecker::_check_blocked(int handle) {
    const Agent & agent = (this->p_agent_loader->agents)[handle];
    Cell_Transitions cell_transitions = agent.cell_transitions;
    this->checked[handle] = 1;
    int handle_opp_agent;
    for(int direction=0; direction < cell_transitions.size(); direction++){
        bool transition = cell_transitions[direction];
        if(transition == 0){
            continue; // no road
        }

        Position new_position = get_new_position(agent.position, direction);
        handle_opp_agent = isin_map(this->agent_positions, new_position)?  this->agent_positions[new_position]: -1;
        if(handle_opp_agent == -1){
            this->checked[handle] = 2;
            return false; //road is free
        }

        if(this->_is_deadlocked[handle_opp_agent]){
            continue; // road is blocked
        }

        if(this->checked[handle_opp_agent] == 0){
            this->_check_blocked(handle_opp_agent);
        }

        if(this->checked[handle_opp_agent] == 2 and not this->_is_deadlocked[handle_opp_agent]){
            this->checked[handle] = 2;
            return false; // road may become free
        }

        this->dep[handle].push_back(handle_opp_agent);

        continue; // road is blocked. cycle
    }

    if(this->dep[handle].empty()){
        this->checked[handle] = 2;
        if(std::accumulate(cell_transitions.begin(), cell_transitions.end(), 0) == 0){
            return false; // dead-end is not deadlock
        }
        this->_is_deadlocked[handle] = true;
        return true;
    }
    return false;
}

void DeadlockChecker::_fix_deps(){
    bool any_changes = true;
    int cnt;
    // might be slow, but in practice won't # TODO can be optimized
    while (any_changes){
        any_changes = false;
        for(auto ag: this->p_agent_loader->agents){
            if(this->checked[ag.handle] == 1){
                cnt = 0;
                for(auto opp_handle: this->dep[ag.handle]){
                    if(this->checked[opp_handle] == 2){
                        if(this->_is_deadlocked[opp_handle]){
                            cnt += 1;
                        }else{
                            this->checked[ag.handle] = 2;
                            any_changes = true;
                        }
                    }
                }
                if(cnt == this->dep[ag.handle].size()){
                    this->checked[ag.handle] = 2;
                    this->_is_deadlocked[ag.handle] = true;
                    any_changes = true;
                }
            }
        }
    }
    for(auto ag: this->p_agent_loader->agents){
        if(this->checked[ag.handle] == 1){
            this->_is_deadlocked[ag.handle] = true;
            this->checked[ag.handle] = 2;
        }
    }
}

bool DeadlockChecker::is_deadlocked(int handle){
    return this->_is_deadlocked[handle];
}