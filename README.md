# Flatland-MARL
This is a multi-agent reinforcement learning solution to [Flatland3 challenge](https://www.aicrowd.com/challenges/flatland-3). The solution itself is elaborated in our paper. 

> Jiang, Yuhao, Kunjie Zhang, Qimai Li, Jiaxin Chen, and Xiaolong Zhu. "[Multi-Agent Path Finding via Tree LSTM](https://arxiv.org/abs/2210.12933)." arXiv preprint arXiv:2210.12933 (2022).

# Install Dependencies
### Clone the repository.
```shell
$ git clone http://gitlab.parametrix.cn/parametrix/challenge/flatland-marl.git
$ cd flatland-marl
```

The code is tested with Python 3.7, and is expected to also work with higher version of Python. If you are using conda, you can create a new environment with the following command (optional) :
```shell
$ conda create -n flatland-marl python=3.7
$ conda activate flatland-marl 
```

### Install `flatland-rl`. 
We found a bug in flatland environment that may lead to performance drop for RL solutions, so we cloned `flatland-rl-3.0.15` and fixed the bug. The bug-free one is provided in folder `flatland/`. Please install this version of `flatland-rl`.
```shell
$ cd flatland
$ pip install .
$ cd ..
```

### Install other requirements
```shell
$ pip install -r requirements.txt
```

### Build flatland_cutils
`flatland_cutils` is a feature parsing package designed to substitute the built-in `flatland.envs.observations.TreeObsForRailEnv`. Our feature parser is developed in c++ language, so it is much faster than built-in `TreeObsForRailEnv`.
```shell
$ cd flatland_cutils
$ pip install .
$ cd ..
```

### For WSL2 users only
The game rendering relies on `OpenGL`. If you are wsl2 users, it is very likely that you don't have OpenGL installed. 
```shell
$ sudo apt-get update
$ sudo apt-get install freeglut3-dev
```



# Usage

### Quick demo
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

### Generate test cases
We provide a script to generate test cases with the same configuration as [Flatland3 challenge round 2](https://flatland.aicrowd.com/challenges/flatland3/envconfig.html). The generation may take several minutes.
```shell
$ cd solution/
$ python debug-environments/generate_test_cases.py
```

### Test in a specific case
```shell
$ python demo.py --env debug-environments/Test_3/Level_0.pkl
```

### Run the whole Flatland3 challenge round 2

#### Install redis
The Flatland3 challenge is evaluated in Client/Server architecture, which relies on redis. Please go to https://redis.io/docs/getting-started/ and follow the instructions to install redis.

#### Install ffmpeg
We relies on ffmpeg to generate replay videos.
```shell
$ sudo apt-get install ffmpeg
```

#### Start flatland-evaluator
First start redis-sever.
```shell
$ sudo service redis-server start
```
Then start flatland-evaluator server.
```shell
$ redis-cli flushall
$ FLATLAND_OVERALL_TIMEOUT=999999 flatland-evaluator --tests ./debug-environments/ --shuffle False --disable_timeouts true
```
Open another terminal, and run our solution.
```
$ python remote_test.py --save-videos
```
Replays are saved in `solution/replay/`.


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