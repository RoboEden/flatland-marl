# Notes Modifications Emanuel

## Installation

The poetry set-up should take care of most things, including creating a new virtual environment. Installation needs to be done in WSL, as the c-utils cannot be installed otherwise. Clone the repository and initialize it by running

```shell
poetry install
```

The installation of the C-utils obersvation generator does not work via poetry, therefore you have to run it manually with

```shell
poetry run pip install ./flatland_cutils
```

## Running demo

Change the current directory to ./solutions.
Run the demo script. 

## Running the training

Run tree_lstm_ppo_training_workshop.py script, setting any any runtime arguments as desired.

# Flatland-MARL
This is a multi-agent reinforcement learning solution to [Flatland3 challenge](https://www.aicrowd.com/challenges/flatland-3). The solution itself is elaborated on in our paper. 

> Jiang, Yuhao, Kunjie Zhang, Qimai Li, Jiaxin Chen, and Xiaolong Zhu. "[Multi-Agent Path Finding via Tree LSTM](https://arxiv.org/abs/2210.12933)." arXiv preprint arXiv:2210.12933 (2022).

# Install Dependencies

### Clone the repository.
```shell
$ git clone http://gitlab.parametrix.cn/parametrix/challenge/flatland-marl.git
$ cd flatland-marl
```

The code is tested with Python 3.7 and is expected to also work with higher versions of Python. If you are using conda, you can create a new environment with the following command (optional) :
```shell
$ conda create -n flatland-marl python=3.7
$ conda activate flatland-marl 
```

### Install flatland
We found a bug in flatland environment that may lead to performance drop for RL solutions, so we cloned `flatland-rl-3.0.15` and fixed the bug. The bug-free one is provided in folder `flatland-rl/`. Please install this version of flatland.
```shell
$ cd flatland-rl
$ pip install .
$ cd ..
```

### Install other requirements
```shell
$ pip install -r requirements.txt
```

### Build flatland_cutils
`flatland_cutils` is a feature parsing package designed to substitute the built-in `flatland.envs.observations.TreeObsForRailEnv`. Our feature parser is developed in c++ language, which is much faster than the built-in `TreeObsForRailEnv`.
```shell
$ cd flatland_cutils
$ pip install .
$ cd ..
```

### For WSL2 users only
The game rendering relies on `OpenGL`. If you are wsl2 user, it is very likely that you don't have OpenGL installed. Please install it.
```shell
$ sudo apt-get update
$ sudo apt-get install freeglut3-dev
```



# Usage

## Quick demo
Run our solution in random environments:
```shell
$ cd solution/
$ python demo.py
```

In a terminal without GUI, you may disable real-time rendering and save the replay as a video.
```shell
$ python demo.py --no-render --save-video replay.mp4
```

## Test as Flatland3 challenge round 2

#### Generate test cases
We provide a script to generate test cases with the same configuration as [Flatland3 challenge round 2](https://flatland.aicrowd.com/challenges/flatland3/envconfig.html). The generation may take several minutes.
```shell
$ cd solution/
$ python debug-environments/generate_test_cases.py
```

#### Test in a specific case
```shell
$ python demo.py --env debug-environments/Test_3/Level_0.pkl
```

## Run the whole Flatland3 challenge round 2

#### Install redis
The Flatland3 challenge is evaluated in Client/Server architecture, which relies on redis. Please go to https://redis.io/docs/getting-started/ and follow the instructions to install redis.

#### Install ffmpeg
We relies on ffmpeg to generate replay videos.
```shell
$ sudo apt-get install ffmpeg
```

#### Start flatland-evaluator
First, start redis-sever.
```shell
$ sudo service redis-server start
```

Then start the flatland-evaluator server.
```shell
$ redis-cli flushall
$ FLATLAND_OVERALL_TIMEOUT=999999 flatland-evaluator --tests ./debug-environments/ --shuffle False --disable_timeouts true
```

Open another terminal, and run our solution.
```shell
$ cd solution
$ python remote_test.py --save-videos
```
Replays are saved in `solution/replay/`.


<!-- 
## Our results

| Test Stage |     Model     | #agents | Map Size  | #cities| Arrival%| Normalized<br>Reward|
|:----------:|:------------- | -------:|:---------:| ------:| -------:| -------------------:|
|  Test_00   | Phase-III-50  |       7 |  30 x 30  |      2 |    94.3 |                .957 |
|  Test_01   | Phase-III-50  |      10 |  30 x 30  |      2 |    92.0 |                .947 |
|  Test_02   | Phase-III-50  |      20 |  30 x 30  |      3 |    87.0 |                .934 |
|  Test_03   | Phase-III-50  |      50 |  30 x 35  |      3 |    86.2 |                .922 |
|  Test_04   | Phase-III-80  |      80 |  35 x 30  |      5 |    62.6 |                .812 |
|  Test_05   | Phase-III-80  |      80 |  45 x 35  |      7 |    62.9 |                .824 |
|  Test_06   | Phase-III-80  |      80 |  40 x 60  |      9 |    70.6 |                .859 |
|  Test_07   | Phase-III-80  |      80 |  60 x 40  |     13 |    65.4 |                .833 |
|  Test_08   | Phase-III-80  |      80 |  60 x 60  |     17 |    74.3 |                .877 |
|  Test_09   | Phase-III-100 |     100 |  80 x 120 |     21 |    59.7 |                .795 |
|  Test_10   | Phase-III-100 |     100 | 100 x 80  |     25 |    57.6 |                .779 |
|  Test_11   | Phase-III-200 |     200 | 100 x 100 |     29 |    52.8 |                .790 |
|  Test_12   | Phase-III-200 |     200 | 150 x 150 |     33 |    57.3 |                .777 |
|  Test_13   | Phase-III-200 |     400 | 150 x 150 |     37 |    34.9 |                .704 |
|  Test_14   | Phase-III-200 |     425 | 158 x 158 |     41 |    39.3 |                .721 |
 -->

# Bibtex
If you use this repo in your research, please cite our paper.
```bib
@article{jiang2022multi,
  title={Multi-Agent Path Finding via Tree LSTM},
  author={Jiang, Yuhao and Zhang, Kunjie and Li, Qimai and Chen, Jiaxin and Zhu, Xiaolong},
  journal={arXiv preprint arXiv:2210.12933},
  year={2022}
}
```