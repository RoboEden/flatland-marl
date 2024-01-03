# Installation

The poetry set-up should take care of most things, including creating a new virtual environment. Installation needs to be done in WSL, as the c-utils cannot be installed otherwise. Clone the repository and initialize it by running

```shell
poetry install
```

The installation of the C-utils obersvation generator does not work via poetry, therefore you have to run it manually with

```shell
poetry run pip install ./flatland_cutils
```

## Training

To train flatland, run flatland_ppo_training_torchrl.py.

# Sources

The C-Utils observation generator and the LSTM network implementation stems from https://github.com/RoboEden/flatland-marl.
This project aimed to create an open-source training setup for the above network.