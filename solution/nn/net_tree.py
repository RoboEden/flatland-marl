import torch
import torch.nn as nn
from impl_config import FeatureParserConfig as fp
from impl_config import NetworkConfig as ns
from tensordict.tensordict import TensorDict
from torch.distributions.categorical import Categorical

from .TreeLSTM import TreeLSTM


class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Transformer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        # for param in self.attention.parameters():
        #    print(param.na)
        #    print(isinstance(param, nn.Linear))
        print(f"embed dim: {embed_dim}")
        print(f"num_heads: {num_heads}")
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


class Network_td(nn.Module):
    """
    Feature:  cat(agents_attr_embedding, tree_embedding)
    structure: mlp
    """

    def __init__(self):
        super(Network_td, self).__init__()
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
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.1)
            print("initialized weights")
            if module.bias is not None:
                module.bias.data.zero_()

    # @torchsnooper.snoop()
    def get_embedding(
        input_td,
    ):  # get_embedding(self, agents_attr, forest, adjacency, node_order, edge_order):
        batch_size, n_agents, num_nodes, _ = forest.shape
        # print('agents_attr shape: {}'.format(agents_attr.shape))
        # print('shape of forest: {}'.format(forest.shape))
        # print('shape of adjacency: {}'.format(adjacency.shape))
        # print('node order shape: {}'.format(node_order.shape))
        # print('edge order shape: {}'.format(edge_order.shape))
        device = next(self.parameters()).device
        adjacency = self.modify_adjacency(adjacency, device)
        # print('batch size, n_agents, num_nodes: {}, {}, {}'.format(batch_size, n_agents, num_nodes))
        tree_embedding = self.tree_lstm(forest, adjacency, node_order, edge_order)
        # print('shape of the tree_embedding before flatten: {}'.format(tree_embedding.shape))
        tree_embedding = tree_embedding.unflatten(0, (batch_size, n_agents, num_nodes))
        # print('shape of the tree embedding after uflatten: {}'.format(tree_embedding.shape))
        tree_embedding = tree_embedding[:, :, 0, :]
        # print('shape of tree embedding after selection: {}'.format(tree_embedding.shape))

        agent_attr_embedding = self.attr_embedding(agents_attr)
        embedding = torch.cat([agent_attr_embedding, tree_embedding], dim=2)

        ## attention
        att_embedding = self.transformer(embedding)
        # print(f'shape of embedding:{embedding.shape}')

        return embedding, att_embedding

    def forward(self, obs_td: TensorDict, valid_actions) -> torch.Tensor:
        # agents_attr, forest, adjacency, node_order, edge_order
        # print('shape of node attr: {}'.format(obs_td['node_attr'].shape))
        batch_size, n_agents, num_nodes, _ = obs_td["node_attr"].shape
        device = next(self.parameters()).device
        # print('adjacency before modification: {}'.format(obs_td['adjacency']))
        adjacency = obs_td["adjacency"].clone()
        # print('max adjacency before modification: {}'.format(obs_td['adjacency'].max()))
        # adjacency = self.modify_adjacency(obs_td['adjacency'], device).clone()
        adjacency = self.modify_adjacency(adjacency, device)
        # print('adjacency after modification: {}'.format(adjacency))
        # print('max adjacency node index: {}'.format(adjacency.max()))
        # print('node order before tree call: {}'.format(obs_td['node_order']))
        tree_embedding = self.tree_lstm(
            obs_td["node_attr"], adjacency, obs_td["node_order"], obs_td["edge_order"]
        )
        tree_embedding = tree_embedding.unflatten(0, (batch_size, n_agents, num_nodes))
        tree_embedding = tree_embedding[:, :, 0, :]
        # print('tree embedding: {}'.format(tree_embedding))
        # print('tree_embedding shape: {}'.format(tree_embedding.shape))

        agent_attr_embedding = self.attr_embedding(obs_td["agents_attr"])
        embedding = torch.cat([agent_attr_embedding, tree_embedding], dim=2)

        ## attention
        att_embedding = self.transformer(embedding)
        # print(f'shape of embedding:{embedding.shape}')

        # print("att embedding: {}".format(att_embedding))
        # embedding_td = self.get_embedding(obs_td)
        logits = self.actor(embedding, att_embedding)
        # print('embedding: {}'.format(embedding))
        # print('shape of logits before detach: {}'.format(logits.shape))
        # print("logits: {}".format(logits))

        # logits = logits.squeeze().detach() #.numpy()
        # print('shape of logits after detach: {}'.format(logits.shape))

        # print('logits: {}'.format(logits))
        # valid_actions = x[0]['valid_actions']

        # define distribution over all actions for the moment
        # might be an idea to only do it for the available options
        # print('logits grad before inf: {}'.format(logits.requires_grad))
        # valid_actions = obs_td['valid_actions']
        logits_copy = logits.clone()
        # print('logits before inf: {}'.format(logits))
        logits[~valid_actions] = float("-inf")
        # print('logits require grad: {}'.format(logits.requires_grad))
        # print('logits after inf: {}'.format(logits))
        probs = Categorical(logits=logits)
        # print('probs of dist: {}'.format(probs.probs))
        # print('valid actions: {}'.format(valid_actions))
        # print('chosen actions in rollout: {}'.format(actions))
        # print("batch shape of probs: {}".format(probs.batch_shape))
        # print('event shape of probs: {}'.format(probs.event_shape))
        # print('shape of entropy: {}'.format(probs.entropy))
        # print("shape of logits: {}".format(logits.shape))
        # print("logits taken from dist: {}".format(probs.logits))
        # logits = logits.numpy()
        # check if we already assigned actions or need to draw
        # print("drawing new action")
        # valid_actions = obs_td['valid_actions']
        # actions = dict()
        # print('valid_actions: {}'.format(valid_actions))
        # probs_valid_actions = probs.probs
        # print("probs valid actions from probs dist: {}".format(probs_valid_actions))
        # probs_valid_actions = torch.reshape(probs_valid_actions, valid_actions.shape)
        # print('probs valid actions before zero: {}'.format(probs_valid_actions))

        # avoid this for stability reasons
        # probs_valid_actions[~valid_actions] = 0

        # if not torch.count_nonzero(probs_valid_actions):
        #    probs_valid_actions[valid_actions] = 0.1
        # print('probs valid actions after zero: {}'.format(probs_valid_actions))
        # probs_valid_actions = Categorical(probs = probs_valid_actions)
        # print('probs valid actions: {}'.format(probs.probs))
        try:
            actions = probs.sample()
        except Exception as e:
            print(f"logits returned by network: {logits_copy}")
            print("probs of dist: {}".format(probs.probs))
            print("logits taken from dist: {}".format(probs.logits))
            print("valid_actions: {}".format(valid_actions))
            for param in self.actor_net.parameters():
                print("param weights: {}".format(param.data))
                print("grad: {}".format(param.grad))
            exit()

        # actions_dict = {handle: action for handle, action in enumerate(actions)}
        # print('action before return: {}'.format(actions))
        # actions = torch.tensor(actions, dtype = torch.int64)
        # print('probs valid actions: {}'.format(probs.probs))
        # print('logits valid actions: {}'.format(probs.logits))
        actions.type(torch.int64)
        # print('actions shape: {}'.format(actions.shape))
        # logprobs = torch.tensor(probs.log_prob(actions), dtype = torch.float32)
        entropy = probs.entropy().type(
            torch.float32
        )  # try without this.reshape(batch_size)
        values = self.critic(embedding, att_embedding).type(torch.float32)
        # print('got to before assigning td')
        # td_out = TensorDict({'actions': actions, 'logprobs' : logprobs, 'entropy' : entropy,'values' : values}, batch_size= [])
        # print('after assigning td')
        # print('actions shape: {}'.format(actions.shape))
        # print('logprobs shape: {}'.format(probs.log_prob(actions).shape))
        # print('entropy shape: {}'.format(entropy.shape))
        # print('values shape: {}'.format(values.shape))
        # print(probs_valid_actions.probs)
        # print(logprobs)
        # print("logprobs full: {}".format(probs.logits))
        # print("log probs only of chosen action: {}".format(probs.log_prob(actions)))
        # print("probs batch shape: {}".format(probs.batch_shape))
        # print("probs event shape: {}".format(probs.event_shape))
        # print("actions shape in net: {}".format(actions.shape))
        # print("log probs in net before return no squeeze: {}".format(probs.log_prob(actions).shape))

        # print('actions drawn: {}'.format(actions))
        # actions=obs_td['shortest_path_action']
        # print('actions shortest path: {}'.format(actions.squeeze(-1)))

        # print('log probs from model: {}'.format(probs.log_prob(actions)))
        # print('log probs no squeeze shape: {}'.format(probs.log_prob(actions).shape))
        # print('log probs actions squeeze shape: {}'.format(probs.log_prob(actions.squeeze(-1)).unsqueeze(-1).shape))
        # print("log probs in net before return: {}".format(probs.log_prob(actions.squeeze(-1)).unsqueeze(-1)))
        # return actions, probs.log_prob(actions.squeeze(-1)).unsqueeze(-1), entropy, values, probs.probs, tree_embedding
        assert (not torch.isnan(probs.log_prob(actions)).any(), "NA in action logits")
        return logits
        # return self.actor(embedding, att_embedding), self.critic(embedding, att_embedding)

    def actor(self, embedding, att_embedding):
        worker_action = torch.cat([embedding, att_embedding], dim=-1)
        # print('worker action before net: {}'.format(worker_action))
        worker_action = self.actor_net(worker_action)
        # print('worker action in actor func: {}'.format(worker_action))
        return worker_action

    # @torchsnooper.snoop()
    def critic(self, embedding, att_embedding):
        output = torch.cat([embedding, att_embedding], dim=-1)
        critic_value = self.critic_net(output)
        critic_value = critic_value.mean(1).view(-1)  # what exactly are we doing here?
        return critic_value

    def modify_adjacency(self, adjacency, _device):
        adjacency = adjacency.clone()  # to avoid side effects
        batch_size, n_agents, num_edges, _ = adjacency.shape
        # print('adjacency shape: {}'.format(adjacency.shape))
        # print('modify adjacency batch size: {}'.format(batch_size))
        # print('modify adjacency n agents: {}'.format(n_agents))
        # print('modify adjacency num edges: {}'.format(num_edges))
        num_nodes = num_edges + 1
        id_tree = torch.arange(0, batch_size * n_agents, device=_device)

        # print('id tree: {}'.format(id_tree))
        # print('id tree shape: {}'.format(id_tree.shape))
        id_nodes = id_tree.view(batch_size, n_agents, 1)
        # print('id nodes: {}'.format(id_nodes))
        # print('id nodes shape: {}'.format(id_nodes.shape))
        # print('adjacency for first batch before modification: {}'.format(adjacency[0]))
        # print('adjacency before modification: {}'.format(adjacency))
        # print('adjacencyentires with minus 2: {}'.format(adjacency[adjacency == -2]))
        adjacency[adjacency == -2] = (
            -batch_size * n_agents * num_nodes
        )  # node_idx == -2 invalid node
        # apparently we usually don't have invalid nodes
        # print('product shape: {}'.format((id_nodes * num_nodes).shape))
        # print('product: {}'.format((id_nodes * num_nodes)))

        # print('max adjacency before add: {}'.format(torch.amax(adjacency, dim=(1,2,3))))
        adjacency[..., 0] += id_nodes * num_nodes
        adjacency[..., 1] += id_nodes * num_nodes
        # print('max adjacency after add: {}'.format(torch.amax(adjacency, dim=(1,2,3))))
        # print('added adjacency: {}'.format(adjacency))
        # print('adjacency min: {}'.format(adjacency[adjacency>0].min()))
        # print('adjacency too big: {}'.format(adjacency[adjacency > 6200].sort()))
        # print('adjacency with negative values: {}'.format(adjacency[adjacency < 0]))
        # see ifthis makes a difference
        adjacency[adjacency < 0] = -2
        # print('modified adjacency: {}'.format(adjacency))
        return adjacency


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
    def forward(self, agents_attr, forest, adjacency, node_order, edge_order):
        batch_size, n_agents, num_nodes, _ = forest.shape
        device = next(self.parameters()).device
        adjacency = self.modify_adjacency(adjacency, device)

        tree_embedding = self.tree_lstm(forest, adjacency, node_order, edge_order)
        tree_embedding = tree_embedding.unflatten(0, (batch_size, n_agents, num_nodes))[
            :, :, 0, :
        ]

        agent_attr_embedding = self.attr_embedding(agents_attr)
        embedding = torch.cat([agent_attr_embedding, tree_embedding], dim=2)

        ## attention
        att_embedding = self.transformer(embedding)

        worker_action = torch.zeros((batch_size, n_agents, 5), device=device)
        worker_action[:, :n_agents, :] = self.actor(embedding, att_embedding)
        critic_value = self.critic(embedding, att_embedding)
        return [worker_action], critic_value  # (batch size, 1)

    def actor(self, embedding, att_embedding):
        worker_action = torch.cat([embedding, att_embedding], dim=-1)
        worker_action = self.actor_net(worker_action)
        return worker_action

    # @torchsnooper.snoop()
    def critic(self, embedding, att_embedding):
        output = torch.cat([embedding, att_embedding], dim=-1)
        critic_value = self.critic_net(output)
        critic_value = critic_value.mean(1).view(-1)
        return critic_value

    def modify_adjacency(self, adjacency, _device):
        batch_size, n_agents, num_edges, _ = adjacency.shape
        num_nodes = num_edges + 1
        id_tree = torch.arange(0, batch_size * n_agents, device=_device)
        id_nodes = id_tree.view(batch_size, n_agents, 1)
        adjacency[adjacency == -2] = (
            -batch_size * n_agents * num_nodes
        )  # node_idx == -2 invalid node
        adjacency[..., 0] += id_nodes * num_nodes
        adjacency[..., 1] += id_nodes * num_nodes
        adjacency[adjacency < 0] = -2
        return adjacency
