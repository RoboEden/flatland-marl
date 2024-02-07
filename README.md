# Installation

The poetry set-up should take care of most things, including creating a new virtual environment. Installation needs to be done in WSL, as the c-utils cannot be installed otherwise. Clone the repository and initialize it by running

```shell
poetry install
```

The installation of the C-utils obersvation generator does not work via poetry, therefore you have to run it manually with

```shell
poetry run pip install ./flatland_cutils
```

# The TorchRL Flatland Environment

# Training

## TreeLSTM
To train flatland, run flatland_ppo_training_torchrl.py.

## Other Architecturs

## Curriculums

### Reward Structures

| Name                  | Definition                                                                                                                                                                                                                                              | Equivalent in Flatland TreeLSTM paper |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------|
| departure_reward      | Gives the defined reward once if the agent switches from off-map state to on-map state                                                                                                                                                                  | Departure reward                      |
| arrival_reward        | Gives the defined reward once if the agent arrives at its destination on time.                                                                                                                                                                          | arrival reward                        |
| delay_reward          | Once the train is allowed to depart, at each step give the minimal delay (if the agent were to follow the shortest path) it would have at the destination.                                                                                              | Environmental Reward                  |
| shortest_path_reward  | Once the train is allowed to depart, at each step give the difference between travel time on the shortest path and available  time (positive if the train would arrive early on the shortest path, and equal to delay reward if it were to arrive late) | none                                  |
| deadlock_penalty      | Gives the defined value as negative penalty for each agent newly in a deadlock.                                                                                                                                                                         | deadlock penalty                      |
| arrival_delay_penalty | Equal in value to the delay reward, but only returned once upon the agents arrival at the destination or end of episode.                                                                                                                                | none                                  |
Note that penalties are defined as negative rewards, i.e. a deadlock penalty of 2.5 will result in a reward of -2.5 upon deadlock. 

### Hyperparameter Training

The repo contains a script for hyperparameter optimization 

# Sources

The C-Utils observation generator and the LSTM network implementation stems from https://github.com/RoboEden/flatland-marl.
This project aimed to create an open-source training setup for the above network.