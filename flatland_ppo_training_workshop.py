# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
import random
import time
from distutils.util import strtobool

from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModule

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from matplotlib import pyplot as plt

from flatland.envs.line_generators import SparseLineGen
from flatland.envs.malfunction_generators import (
    MalfunctionParameters,
    ParamMalfunctionGen,
)
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import SparseRailGen
from flatland_cutils import TreeObsForRailEnv as TreeCutils
from flatland.envs.step_utils.states import TrainState

import PIL
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from IPython.display import clear_output

from eval_env import LocalTestEnvWrapper
from impl_config import FeatureParserConfig as fp
#from plfActor import Actor
#from utils import VideoWriter, debug_show

from solution.nn.net_tree import Network_td

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
    parser.add_argument("--total-timesteps", type=int, default=100000,
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
    parser.add_argument("--vf-coef", type=float, default=0.01,
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

num_agents = 1

class RailEnv_td(RailEnv):
    def obs_to_td(self, observations):
        #print('dim of node attr at output of env: {}'.format(torch.tensor(observations[1][0]).unsqueeze(0).shape))
        obs_td = TensorDict({
            'agents_attr' : torch.tensor(observations[0], dtype = torch.float32),#.unsqueeze(0),
            'node_attr' : torch.tensor(observations[1][0], dtype = torch.float32),#.unsqueeze(0),
            'adjacency' : torch.tensor(observations[1][1], dtype = torch.int64),#.unsqueeze(0),
            'node_order' : torch.tensor(observations[1][2], dtype = torch.int64),#.unsqueeze(0),
            'edge_order' : torch.tensor(observations[1][3], dtype = torch.int64),#.unsqueeze(0)
        },
        [self.get_num_agents()])
        return obs_td
    
    def reset(self, tensordict = None):
        observations, info = super().reset()
        #print('node order from env: {}'.format(observations[1][2]))
        if tensordict is None:
            tensordict_out = TensorDict({}, batch_size = [])
        tensordict_out['observations'] = self.obs_to_td(observations)
        #print('node_order after obs parsing: {}'.format(tensordict_out['observations']['node_order']))
        env_config, agents_properties, valid_actions = self.obs_builder.get_properties()
        #print('valid actions: {}'.format(torch.Tensor(valid_actions)))
        tensordict_out['observations']['valid_actions'] = torch.tensor(valid_actions, dtype = torch.bool)#.unsqueeze(0)
        tensordict_out['done'] =  torch.tensor(False).type(torch.bool)
        return tensordict_out
    
    def step(self, tensordict):
        #print('doing step')
        actions = {handle: action.item() for handle, action in enumerate(tensordict['actions'])}
        #print('action dict passed to env: {}'.format(actions))
        observations, rewards, done, info = super().step(actions)
        env_config, agents_properties, valid_actions = self.obs_builder.get_properties()
        #print('valid actions from env: {}'.format(valid_actions))
        tensordict_out = TensorDict({}, batch_size = [])
        tensordict_out['observations'] = self.obs_to_td(observations)
        tensordict_out['rewards'] = torch.tensor([value for _, value in rewards.items()])
        #tensordict_out['done'] =  torch.tensor([value for _, value in done.items()][:-1])
        tensordict_out['done'] =  torch.tensor(done['__all__']).type(torch.bool)
        tensordict_out['observations']['valid_actions'] = torch.tensor(valid_actions, dtype = torch.bool)
        return tensordict_out


def create_random_env():
    ''' Create a random railEnv object
        Taken from the flatland-marl demo
    '''
    
    return RailEnv_td(
        number_of_agents=num_agents,
        width=30, # try smaller environment for hopefully faster learning
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
    ''' Initialize layers orthogonally
    Used in the CleanRL PPO version, not used yet in this script.
    '''
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

""" class Agent(Network):
    '''Class for the neural network we want to train. Inherits from the original neural network.
    
    Methods:
    get_feature
    get_feature_to_td
    get_value
    get_action_and_value
    _choose_action'''
    def __init__(self, network):
        self.td_network = TensorDictModule(network, in_keys = ['observations', 'actions'], 
                                           out_keys = ["actions", 'log_probs', 'entropy', "value"]) 
       
    def get_value(self, obs_td):
        #agents_attr, forest, adjacency, node_order, edge_order = self.get_feature(x)
        embedding_td = self.get_embedding(obs_td)
        return self.critic(embedding_td)

    def forward(self, obs_td, actions =None):
        embedding_td = self.get_embedding(obs_td)
        logits = self.actor(embedding_td)
        logits = logits.squeeze().detach() #.numpy()  
        
        #valid_actions = x[0]['valid_actions']
        
        # define distribution over all actions for the moment
        # might be an idea to only do it for the available options
        #print('tpye of logits: {}'.format(type(logits)))
        probs = Categorical(logits=logits)
        logits = logits.numpy()
        if actions is None:
            valid_actions = x[0]['valid_actions']
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
        return action """

def observation_to_tensordict(obs, num_agents):
    '''Returns a tensordict containings the entries of obs'''
    obs_td = TensorDict({'agents_attr' : obs[0]['agent_attr'],
                           'node_attr': obs[0]['forest'],
                           'adjacency' : obs[0]['adjacency'],
                           'node_order' : obs[0]['node_order'],
                           'edge_order' : obs[0]['edge_order']},
                          [num_agents])
    return obs_td

def observation_from_tensordict(obs_td):
    '''Returns a list containing a dict of the entries in obs_td'''
    obs = dict({
        'agent_attr' : obs_td['agents_attr'].numpy(),
        'forest' : obs_td['node_attr'].numpy(),
        'adjacency' : obs_td['adjacency'].numpy(),
        'node_order' : obs_td['node_order'].numpy(),
        'edge_order' : obs_td['edge_order'].numpy()
            # convert to dict here
    })
    return [obs]

def actions_to_dict(actions):
    '''Returns a dict of the actions for actions given as tensor'''
    return {handle: action for handle, action in enumerate(actions)}

class custom_reward_rail_env(RailEnv):
    '''Draft for custom reward function, not yet working'''
    def update_step_rewards(self, i_agent):
        agent = self.agents[i_agent]
        if agent.state == TrainState.DONE:
            reward = 50
            self.rewards_dict[i_agent] += reward


if __name__ == "__main__":
    args = parse_args()
    
    total_rewards = []
    
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
    print('device: {}'.format(device))
    # env setup
    #envs = gym.vector.SyncVectorEnv(
    #    [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    #)
    
    env = create_random_env()
    env_renderer = RenderTool(env,
                              agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
                              show_debug=False,
                              screen_height=600,  # Adjust these parameters to fit your resolution
                              screen_width=800)


    # just create one flatland env here, don't bother with sync vector env
    #assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    #agent = Agent(envs).to(device)
    #optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    agent = Network_td()
    td_module = TensorDictModule(Network_td(), in_keys = ['observations', 'actions'], out_keys = ['actions', 'logprobs', 'entropy', 'values']).to(device)

    optimizer = optim.Adam(td_module.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    #obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    num_steps = args.num_steps
    num_actions = 5
    observations = TensorDict({'agents_attr' : torch.zeros(num_steps, num_agents, 83, dtype = torch.float32),
                                'node_attr': torch.zeros(num_steps, num_agents, 31, 12, dtype = torch.float32),
                                'adjacency': torch.zeros(num_steps, num_agents, 30, 3, dtype = torch.int64),
                                'node_order': torch.zeros(num_steps, num_agents, 31, dtype = torch.int64),
                                'edge_order': torch.zeros(num_steps, num_agents, 30, dtype = torch.int64),
                                'valid_actions' : torch.zeros(num_steps, num_agents, num_actions, dtype = torch.bool)}, 
                              batch_size = [num_steps, num_agents])

    actions_init = torch.zeros((num_steps, num_agents), dtype = torch.int64)
    logprobs = torch.zeros((num_steps, num_agents), dtype = torch.float32)
    rewards = torch.zeros((num_steps, num_agents), dtype = torch.float32)
    done = torch.zeros((num_steps), dtype = torch.bool)
    values = torch.zeros((num_steps, num_agents))
    
    rollout_data = TensorDict({'observations' : observations,
                               'actions' : actions_init,
                               'logprobs' : logprobs,
                               'rewards' : rewards,
                               'done' : done,
                               'values' : values},
                              batch_size = [num_steps])
  
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = env.reset()
    rollout_data = rollout_data.to(device)


    #next_done = False
    num_updates = args.total_timesteps // args.batch_size
    
    for update in range(1, num_updates +1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, num_steps):
            global_step += 1 * args.num_envs
            # currently we still change between tensordict and non-tensordict representation of data
            # goal will be to get it all to tensordict, possibly define network as tensormodule
            #print('next obs before assignment: {}'.format(next_obs))
            rollout_data[step] = next_obs# something goes wrong when we save the rollout data
            # ALGO LOGIC: action logic
            with torch.no_grad():
                td_output = td_module(rollout_data[[step]])
                #rollout_data[[step]] = td_module(rollout_data[[step]])
                #print('module output: {}'.format(td_output))

                rollout_data[[step]] = td_output
                #print('rollout data 0: {}'.format(rollout_data[[step]]))                
                
                #values[step] = value.flatten()
                #rollout_data['values'][step] = value.flatten()

            #rollout_data['actions'][step] = actions
            #rollout_data['logprobs'][step] = logprob

            # manually reset the environment if it is done
            # currently we only train for one specific track layout
            if next_obs['done']:
                next_obs = env.reset()
                env_renderer.reset()
                reward = env.rewards_dict
                done = False
            else:
                next_obs = env.step(rollout_data[step]).to(device)

            #env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
            #print('agent attr: {}'.format(next_obs['observations']['agents_attr']))
            #next_done = rollout_data[step]['dones']
            #print('next_done: {}'.format(next_done))
                #done = done['__all__'] # only done if all agents are done
            
            # reward = torch.tensor([value for _, value in reward.items()])
            #rollout_data['rewards'][step] = reward
            #next_done = done

            """             for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break """
        #print('actions: {}'.format(rollout_data['actions']))
        #print('valid_actions: {}'.format(rollout_data['observations']['valid_actions']))
        writer.add_scalar('rewards/min', rollout_data['rewards'].min(), global_step)
        writer.add_scalar('rewards/mean', rollout_data['rewards'].mean(), global_step)

        # Calculate advantages
        with torch.no_grad():
            next_obs_value = next_obs.unsqueeze(0)
            next_obs_value['actions'] = torch.ones(1).to(device)
            next_value = td_module(next_obs.unsqueeze(0))
            next_value = next_value['values']
            #print('got next values: {}'.format(next_value))
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = torch.zeros(num_agents).to(device)
            for t in reversed(range(num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_obs['done'].item()
                    nextvalues = next_value
                else:
                    #nextnonterminal = 1.0 - dones[t + 1]
                    nextnonterminal = 1.0 - rollout_data['done'][t + 1].item()
                    #nextvalues = values[t + 1]
                    nextvalues = rollout_data['values'][t + 1]
                #delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                #advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                delta = rollout_data['rewards'][t] + args.gamma * nextvalues * nextnonterminal - rollout_data['values'][t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + rollout_data['values']

        # Optimizing the policy and value network
        b_inds = np.arange(num_steps)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                """                 updated_rollout_data = TensorDict({
                    'newlogprob' : torch.zeros(args.minibatch_size, num_agents),
                    'entropy' : torch.zeros(args.minibatch_size, num_agents),
                    'newvalue' : torch.zeros(args.minibatch_size, num_agents)
                }, batch_size = [args.minibatch_size, num_agents]) """
                #print('minibatch: {}'.format(rollout_data[mb_inds]))
                updated_rollout_data = td_module(rollout_data[mb_inds])
                #print('got minibatch data')
                # get the logprob, entropy and value for current network
                # probably there is a prettier way to do this
                """                 for i, mb_ind in enumerate(mb_inds):
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(x = observation_from_tensordict(rollout_data[mb_ind]['observations']), 
                                                                                  n_agents = num_agents,
                                                                                  actions = rollout_data[mb_ind]['actions'])
                    updated_rollout_data['newlogprob'][i] = newlogprob
                    updated_rollout_data['entropy'][i] = entropy
                    updated_rollout_data['newvalue'][i] = newvalue """
                
                logratio = updated_rollout_data['logprobs'] - rollout_data['logprobs'][mb_inds]
                ratio = logratio.exp()
                #print('ration: {}'.format(ratio))
                #print('ratio shape: {}'.format(ratio.shape))

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = updated_rollout_data['values']
                if args.clip_vloss:
                    # not adapted for flatland yet
                    v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                    v_clipped = rollout_data['values'][mb_inds] + torch.clamp(
                        newvalue - rollout_data['values'][mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()
                    #print('v_loss: {}'.format(v_loss))

                entropy_loss = updated_rollout_data['entropy'].mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                print('current_loss: {}'.format(loss))

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        #y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        #var_y = np.var(y_true)
        #explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        #writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        #writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        #writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        #writer.add_scalar("losses/explained_variance", explained_var, global_step)
        #print("SPS:", int(global_step / (time.time() - start_time)))
        #writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    #envs.close()
    writer.close()
    #reward_df = pd.DataFrame(total_rewards)
    #reward_df.to_csv('rewards_of_training.csv')
    #plt.plot(total_rewards)
    
