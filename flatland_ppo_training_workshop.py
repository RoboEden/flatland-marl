# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
import random
import time
from distutils.util import strtobool

from tensordict.tensordict import TensorDict

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from flatland.envs.line_generators import SparseLineGen
from flatland.envs.malfunction_generators import (
    MalfunctionParameters,
    ParamMalfunctionGen,
)
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import SparseRailGen
from flatland_cutils import TreeObsForRailEnv as TreeCutils

from eval_env import LocalTestEnvWrapper
from impl_config import FeatureParserConfig as fp
#from plfActor import Actor
#from utils import VideoWriter, debug_show

from solution.nn.net_tree import Network

from solution.utils import VideoWriter, debug_show

from PIL import Image
import numpy as np
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=300000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=1000,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=20,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

def create_random_env():
    return RailEnv(
        number_of_agents=50,
        width=30,
        height=35,
        rail_generator=SparseRailGen(
            max_num_cities=3,
            grid_mode=False,
            max_rails_between_cities=2,
            max_rail_pairs_in_city=2,
        ),
        line_generator=SparseLineGen(
            speed_ratio_map={1.0: 1 / 4, 0.5: 1 / 4, 0.33: 1 / 4, 0.25: 1 / 4}
        ),
        malfunction_generator=ParamMalfunctionGen(
            MalfunctionParameters(
                malfunction_rate=1 / 4500, min_duration=20, max_duration=50
            )
        ),
        obs_builder_object=TreeCutils(fp.num_tree_obs_nodes, fp.tree_pred_path_depth),
    )


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer



class Agent(Network):
    
    def get_feature(self, obs_list):
        agents_attr = obs_list[0]["agent_attr"]
        agents_attr = torch.unsqueeze(torch.from_numpy(agents_attr), axis=0).to(
            dtype=torch.float32
        )

        forest = obs_list[0]["forest"]
        forest = torch.unsqueeze(torch.from_numpy(forest), axis=0).to(
            dtype=torch.float32
        )

        adjacency = obs_list[0]["adjacency"]
        adjacency = torch.unsqueeze(torch.from_numpy(adjacency), axis=0).to(
            dtype=torch.int64
        )

        node_order = obs_list[0]["node_order"]
        node_order = torch.unsqueeze(torch.from_numpy(node_order), axis=0).to(
            dtype=torch.int64
        )

        edge_order = obs_list[0]["edge_order"]
        edge_order = torch.unsqueeze(torch.from_numpy(edge_order), axis=0).to(
            dtype=torch.int64
        )

        #print('node order after get_features: {}'.format(node_order.shape))
        return agents_attr, forest, adjacency, node_order, edge_order
    
    def get_value(self, x):
        agents_attr, forest, adjacency, node_order, edge_order = self.get_feature(x)
        embedding, att_embedding = self.get_embedding(agents_attr, forest, adjacency, node_order, edge_order)
        return self.critic(embedding, att_embedding)

    def get_action_and_value(self, x, n_agents, actions =None):
        #print(x[0]['agent_attr'])
        agents_attr, forest, adjacency, node_order, edge_order = self.get_feature(x)
        embedding, att_embedding = self.get_embedding(agents_attr, forest, adjacency, node_order, edge_order)
        logits = self.actor(embedding, att_embedding)
        logits = logits.squeeze().detach() #.numpy()  
        
        valid_actions = x[0]['valid_actions']
        
        # define distribution over all actions for the moment
        # might be an idea to only do it for the available options
        #print('tpye of logits: {}'.format(type(logits)))
        probs = Categorical(logits=logits)
        logits = logits.numpy()
        if actions is None:
            #actions = dict()
            actions = torch.zeros(n_agents)
            valid_actions = np.array(valid_actions)
            for i in range(n_agents):
                if n_agents == 1:
                    actions[i] = self._choose_action(valid_actions[i, :], logits)
                else:
                    #print(logits[i, :])
                    actions[i] = self._choose_action(valid_actions[i, :], logits[i, :])
                    #print(actions[i])
        #actions_dict = {handle: action for handle, action in enumerate(actions)}
        #print('action before return: {}'.format(actions))
        return actions, probs.log_prob(actions), probs.entropy(), self.critic(embedding, att_embedding)
    
    def _choose_action(self, valid_actions, logits, soft_or_hard_max="soft"):
        def _softmax(x):
            if x.size != 0:
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()
            else:
                return None

        _logits = _softmax(logits[valid_actions != 0])
        if soft_or_hard_max == "soft":
            if valid_actions.nonzero()[0].size == 0:
                valid_actions = np.ones((1, 5))
            np.random.seed(42)
            action = np.random.choice(valid_actions.nonzero()[0], p=_logits)
        else:
            action = valid_actions.nonzero()[0][np.argmax(_logits)]
        return action
    
    
def observation_to_tensordict(obs, num_agents):
    obs_td = TensorDict({'agents_attr' : obs[0]['agent_attr'],
                           'node_attr': obs[0]['forest'],
                           'adjacency' : obs[0]['adjacency'],
                           'node_order' : obs[0]['node_order'],
                           'edge_order' : obs[0]['edge_order']},
                          [num_agents])
    return obs_td

def observation_from_tensordict(obs_td, num_agents):
    obs = dict({
        'agent_attr' :
            # convert to dict here
    })

def actions_to_dict(actions):
    return {handle: action for handle, action in enumerate(actions)}

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    #envs = gym.vector.SyncVectorEnv(
    #    [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    #)
    
    #envs = [create_random_env()]
    env = create_random_env()
    #envs = [LocalTestEnvWrapper(env) for env in envs]
    env = LocalTestEnvWrapper(env)
    # just create one flatland env here, don't bother with sync vector env
    #assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    #agent = Agent(envs).to(device)
    #optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    agent = Agent()
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    #obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    num_agents = 50
    num_steps = 1000
    num_actions = 5
    observations = TensorDict({'agents_attr' : torch.zeros(num_steps, num_agents, 83),
                                'node_attr': torch.zeros(num_steps, num_agents, 31, 12),
                                'adjacency': torch.zeros(num_steps, num_agents, 30, 3),
                                'node_order': torch.zeros(num_steps, num_agents, 31),
                                'edge_order': torch.zeros(num_steps, num_agents, 30)}, 
                              batch_size = [num_steps, num_agents])

    observations[0]['agents_attr']
    actions_init = torch.zeros((num_steps, num_agents))
    
    logprobs = torch.zeros((num_steps, num_agents))
    
    rewards = torch.zeros((num_steps, num_agents))
    
    dones = torch.zeros((num_steps))
    
    values = torch.zeros((num_steps, num_agents))
    
    rollout_data = TensorDict({'observations' : observations,
                               'actions' : actions_init,
                               'logprobs' : logprobs,
                               'rewards' : rewards,
                               'dones' : dones,
                               'values' : values},
                              batch_size = [num_steps])
    #actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    #logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    #rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    #dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    #values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = env.reset()
    #print('type of next obs: {}'.format(type(next_obs)))
    """     next_obs = TensorDict({'agents_attr' : next_obs[0]['agent_attr'],
                           'node_attr': next_obs[0]['forest'],
                           'adjacency' : next_obs[0]['adjacency'],
                           'node_order' : next_obs[0]['node_order'],
                           'edge_order' : next_obs[0]['edge_order']},
                          [num_agents]) """
    #next_obs = observation_to_tensordict(next_obs, num_agents)
    #print(rollout_data[0]['observations'])
    #print(next_obs['agents_attr'][0])
    
    #next_obs = torch.Tensor(envs.reset()).to(device)
    #next_done = torch.zeros(args.num_envs)
    next_done = 0
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates +1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, num_steps):
            global_step += 1 * args.num_envs
            rollout_data['observations'][step] = observation_to_tensordict(next_obs, num_agents)
            #print(rollout_data['observations'][step])
            #exit()
            rollout_data['dones'][step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                actions, logprob, _, value = agent.get_action_and_value(next_obs, num_agents)
                #values[step] = value.flatten()
                rollout_data['values'][step] = value.flatten()
            #actions[step] = action
            #logprobs[step] = logprob
            #print('actions before assignment to rollout_data: {}'.format(actions))
            rollout_data['actions'][step] = actions
            rollout_data['logprobs'][step] = logprob
            #print('assigned rollout data actions: {}'.format(rollout_data[step]['actions']))
            #exit()
            # TRY NOT TO MODIFY: execute the game and log data.
            #next_obs, reward, done, info = envs.step(action.cpu().numpy())
            if next_done:
                next_obs = env.reset()
                reward = env.env.rewards_dict
                done = 0
            else:
                next_obs, reward, done = env.step(actions_to_dict(actions))
                done = done['__all__'] # only done if all agents are done
            
            #print('done: {}'.format(done))
            reward = torch.tensor([value for _, value in reward.items()])
            #done = torch.tensor([value for _, value in done.items()])
            #print(type(reward))
            #print(reward)
            #rewards[step] = torch.tensor(reward).to(device).view(-1)
            #print('reward at step {}: {}'.format(step, reward))
            rollout_data['rewards'][step] = reward
            
            #next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            next_done = done
            print('current step / done: {} / {}'.format(step, done))

            
            
            """             for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break """
        #print('actions: {}'.format(rollout_data[12]['actions']))
        #print(rollout_data['logprobs'])
        #print('rewards: {}'.format(rewards))
        # rewards are currently empty 
        print('average reward: {}'.format(rollout_data['rewards'].mean()))
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)# .to(device)
            lastgaelam = torch.zeros(num_agents)
            for t in reversed(range(num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    #nextnonterminal = 1.0 - dones[t + 1]
                    nextnonterminal = 1.0 - rollout_data['dones'][t + 1]
                    #nextvalues = values[t + 1]
                    nextvalues = rollout_data['values'][t + 1]
                #delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                #advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                delta = rollout_data['rewards'][t] + args.gamma * nextvalues * nextnonterminal - rollout_data['values'][t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        #print('advantages are:{}'.format(advantages))
        # flatten the batch
        print('advantages: {}'.format(advantages))
        #b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        #b_logprobs = logprobs.reshape(-1)
        #b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        #b_advantages = advantages.reshape(-1)
        #b_returns = returns.reshape(-1)
        #b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(num_steps)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # here we'll have to convert the obs from the rollout_data back to the just-dict
                # see if passing actions along for get_action_and_value works
                # the rest should be fairly similar
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
