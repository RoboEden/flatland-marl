# Background

The code in this repo was developed as part of career starter data science project at SBB Cargo, with the aim of gaining a general understanding of reinforcement learning by engineering the training script for the paper "Multi-Agent Path Finding via Tree LSTM". Its main contributions are:

* working training scripts for a TreeLSTM in flatland
* adaption of the flatland environment for TorchRL, making training scripts easier to write
* working prototype using tree-based transformers as introduced in this (https://www.microsoft.com/en-us/research/publication/novel-positional-encodings-to-enable-tree-based-transformers/) paper to process flatland observations

# Sources and Credit

Several people in SBB and the Flatland community gave valuable input to this project, notably my supervisor Philipp Germann, Adrian Egli and Matthias Minder from SBB and Jeremy Watson from Flatland. (Any bugs or errors are of course entirely mine (Emanuel Zwyssig)).

The C-Utils observation generator and the LSTM network implementation stems from https://github.com/RoboEden/flatland-marl, which this is a fork of.

# Installation

The poetry set-up should take care of most things, including creating a new virtual environment (if you don't have poetry installed, see [here](https://python-poetry.org/docs/)). Installation needs to be done in WSL, as the c-utils cannot be installed otherwise. Clone the repository and initialize it by running

```shell
poetry install
```

If the flatland installation didn't work directly, do it manually by running

```shell
poetry run pip install flatland-rl
```

The installation of the C-utils obersvation generator does not work via poetry, therefore you have to run it manually with

```shell
poetry run pip install ./flatland_cutils
```

# Training

To train flatland, run flatland_ppo_training_torchrl.py. There are many ready-to-use run commands for different experiments in the /run_commands folder. There are many hyperparameters to chose from, most of which are standard for the algorithm used. Below are brief explanations of the most special cases.

## Reward Structures

In order to try different rewards and reproduce the curriculum learning used in the original paper, rewards are calculated as a linear combination of different components, the weight of each determined by a coefficient. Furthermore, rewards are determined in a curriculum-json file. For examples, see the /curriculums folder. 

| Name                  | Definition                                                                                                                                                                                                                                              | Equivalent in Flatland TreeLSTM paper |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------|
| departure_reward      | Gives the defined reward once if the agent switches from off-map state to on-map state                                                                                                                                                                  | Departure reward                      |
| arrival_reward        | Gives the defined reward once if the agent arrives at its destination on time.                                                                                                                                                                          | arrival reward                        |
| delay_reward          | Once the train is allowed to depart, at each step give the minimal delay (if the agent were to follow the shortest path) it would have at the destination.                                                                                              | Environmental Reward                  |
| shortest_path_reward  | Once the train is allowed to depart, at each step give the difference between travel time on the shortest path and available  time (positive if the train would arrive early on the shortest path, and equal to delay reward if it were to arrive late) | none                                  |
| deadlock_penalty      | Gives the defined value as negative penalty for each agent newly in a deadlock.                                                                                                                                                                         | deadlock penalty                      |
| arrival_delay_penalty | Equal in value to the delay reward, but only returned once upon the agents arrival at the destination or end of episode.                                                                                                                                | none                                  |


Note that penalties are defined as negative rewards, i.e. a deadlock penalty of 2.5 will result in a reward of -2.5 upon deadlock. 

## Hyperparameter Training

The repo contains a script for hyperparameter optimization under /hyperparameter_searches.

# Flatland in TorchRL

The adaptions necessary for flatland to be used as a TorchRL environment are contained in the folder /flatland_torchrl and can be used as a stand-alone component.

# Model Comparisons

To compare different models to the pre-trained model from the original paper, the script torchrl_rollout_demo.py allows using both model architectures for a rollout (including the possibility to render the rollouts). 

# Notes

A couple of my notes are in notes.pdf. These are just my personal working notes, and I include them in case they might be useful to someone, whitout claim to completeness or correctness.

# Future of this Repo/Expectations

As this repo was developed during a limited-time project, it will not be maintained or further developed, and questions will be answered sporadically at best.

# License

Code released under the MIT license.