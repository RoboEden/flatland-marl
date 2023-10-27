# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy


# try gettign the TreeLSTM to run with CleanRL

# Issues: 
import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch

from solution.nn.net_tree import Network

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


from solution.nn.TreeLSTM import TreeLSTM

from argparse import ArgumentParser
from time import sleep
import pandas as pd

from flatland.envs.line_generators import SparseLineGen
from flatland.envs.malfunction_generators import (
    MalfunctionParameters,
    ParamMalfunctionGen,
)
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import SparseRailGen
from flatland_cutils import TreeObsForRailEnv as TreeCutils

from impl_config import FeatureParserConfig as fp
from impl_config import NetworkConfig as ns

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



#### Load env

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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Transformer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.att_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
        )

    def forward(self, input):
        batch_size, n_agents, embedding_size = input.shape
        input = input.permute(1, 0, 2)  # agents, batch, embedding
        input = input.view(n_agents, -1, embedding_size)
        output, _ = self.attention(input, input, input)

        input = input.view(n_agents, -1, embedding_size)
        input = input.permute(1, 0, 2)
        output = output.view(n_agents, -1, embedding_size)
        output = output.permute(1, 0, 2)
        output = self.att_mlp(torch.cat([input, output], dim=-1))
        return output


class Network(nn.Module):
    """
    Feature:  cat(agents_attr_embedding, tree_embedding)
    structure: mlp
    """

    def __init__(self):
        super(Network, self).__init__()
        self.tree_lstm = TreeLSTM(fp.node_sz, ns.tree_embedding_sz)
        self.attr_embedding = nn.Sequential(
            nn.Linear(fp.agent_attr, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(2 * ns.hidden_sz, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(2 * ns.hidden_sz, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(2 * ns.hidden_sz, ns.hidden_sz),
            nn.GELU(),
        )
        self.transformer = nn.Sequential(
            Transformer(ns.hidden_sz + ns.tree_embedding_sz, 4),
            Transformer(ns.hidden_sz + ns.tree_embedding_sz, 4),
            Transformer(ns.hidden_sz + ns.tree_embedding_sz, 4),
        )
        self.actor_net = nn.Sequential(
            nn.Linear(ns.hidden_sz * 2 + ns.tree_embedding_sz * 2, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(ns.hidden_sz * 2, ns.hidden_sz),
            nn.GELU(),
            nn.Linear(ns.hidden_sz, fp.action_sz),
        )
        self.critic_net = nn.Sequential(
            nn.Linear(ns.hidden_sz * 2 + ns.tree_embedding_sz * 2, ns.hidden_sz * 2),
            nn.GELU(),
            nn.Linear(ns.hidden_sz * 2, ns.hidden_sz),
            nn.GELU(),
            nn.Linear(ns.hidden_sz, 1),
        )

    # @torchsnooper.snoop()
    def get_embedding(self, agents_attr, forest, adjacency, node_order, edge_order):
        batch_size, n_agents, num_nodes, _ = forest.shape
        #print('shape of forest: {}'.format(forest.shape))
        device = next(self.parameters()).device
        adjacency = self.modify_adjacency(adjacency, device)
        #print('batch size, n_agents, num_nodes: {}, {}, {}'.format(batch_size, n_agents, num_nodes))
        tree_embedding = self.tree_lstm(forest, adjacency, node_order, edge_order)
        #print('shape of the tree_embedding before flatten: {}'.format(tree_embedding.shape))
        tree_embedding = tree_embedding.unflatten(0, (batch_size, n_agents, num_nodes))
        #print('shape of the tree embedding after uflatten: {}'.format(tree_embedding.shape))
        tree_embedding = tree_embedding[:, :, 0, :]
        #print('shape of tree embedding after selection: {}'.format(tree_embedding.shape))


        agent_attr_embedding = self.attr_embedding(agents_attr)
        embedding = torch.cat([agent_attr_embedding, tree_embedding], dim=2)

        ## attention
        att_embedding = self.transformer(embedding)

        #worker_action = torch.zeros((batch_size, n_agents, 5), device=device)
        #worker_action[:, :n_agents, :] = self.actor(embedding, att_embedding)
        #critic_value = self.critic(embedding, att_embedding)
        return embedding, att_embedding  # (batch size, 1)

    def get_action_logit(self, embedding, att_embedding):
        actions = torch.cat([embedding, att_embedding], dim=-1)
        actions = self.actor_net(actions)
        return self.actor_net(actions)
    
    def get_values(self, embedding, att_embedding):
        output = torch.cat([embedding, att_embedding], dim=-1)
        critic_value = self.network.critic_net(output)
        critic_value = critic_value.mean(1).view(-1)
        return critic_value
    
    def modify_adjacency(self, adjacency, _device):
        batch_size, n_agents, num_edges, _ = adjacency.shape
        num_nodes = num_edges + 1
        id_tree = torch.arange(0, batch_size * n_agents, device=_device)
        id_nodes = id_tree.view(batch_size, n_agents, 1)
        print('adjacency before modification: {}'.format(adjacency))
        print('adjacencyentires with minus 2: {}'.format(adjacency[adjacency == -2]))        
        adjacency[adjacency == -2] = (
            -batch_size * n_agents * num_nodes
        )  # node_idx == -2 invalid node
        # apparently we usually don't have invalid nodes

        adjacency[..., 0] += id_nodes * num_nodes
        adjacency[..., 1] += id_nodes * num_nodes
        print('adjacency with negative values: {}'.format(adjacency[adjacency < 0]))
        # see ifthis makes a difference
        #adjacency[adjacency < 0] = -2
        print('modified adjacency: {}'.format(adjacency))
        return adjacency
    

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = Network()

    def get_value(self, x):
        features = self.get_feature(x)
        
        return self.network()

    def get_action_and_value(self, x, action=None):
        features = self.get_features(x)
        embedding, att_embedding = self.network.get_embedding(features)
        
        logits = self.network.get_action_logit(embedding, att_embedding)
        logits = logits.squeeze().detach().numpy()
        probs = Categorical(logits = logits)
        
        values = self.network.get_values()
        if actions is None:
            actions = probs.sample()

        return actions, probs.log_prob(actions), probs.entropy(), self.network.critic_net(embedding, att_embedding)


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
    env = create_random_env()
    

    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + env.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + env.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    exit()
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        #print('advantages are:{}'.format(advantages))
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

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
