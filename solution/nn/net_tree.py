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
        print(f'embed dim: {embed_dim}')
        print(f'num_heads: {num_heads}')
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

    # @torchsnooper.snoop()
    def get_embedding(input_td): #get_embedding(self, agents_attr, forest, adjacency, node_order, edge_order):
        batch_size, n_agents, num_nodes, _ = forest.shape
        #print('agents_attr shape: {}'.format(agents_attr.shape))
        #print('shape of forest: {}'.format(forest.shape))
        #print('shape of adjacency: {}'.format(adjacency.shape))
        #print('node order shape: {}'.format(node_order.shape))
        #print('edge order shape: {}'.format(edge_order.shape))
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
        #print(f'shape of embedding:{embedding.shape}')

        return embedding, att_embedding
    
    def forward(self, obs_td, actions):
        #agents_attr, forest, adjacency, node_order, edge_order
        #print('shape of node attr: {}'.format(obs_td['node_attr'].shape))
        batch_size, n_agents, num_nodes, _ = obs_td['node_attr'].shape
        device = next(self.parameters()).device
        adjacency = self.modify_adjacency(obs_td['adjacency'], device)
        #print('node order before tree call: {}'.format(obs_td['node_order']))
        tree_embedding = self.tree_lstm(obs_td['node_attr'], 
                                        adjacency, 
                                        obs_td['node_order'], 
                                        obs_td['edge_order'])
        tree_embedding = tree_embedding.unflatten(0, (batch_size, n_agents, num_nodes))
        tree_embedding = tree_embedding[:, :, 0, :]

        agent_attr_embedding = self.attr_embedding(obs_td['agents_attr'])
        embedding = torch.cat([agent_attr_embedding, tree_embedding], dim=2)

        ## attention
        att_embedding = self.transformer(embedding)
        #print(f'shape of embedding:{embedding.shape}')
        
        
        #embedding_td = self.get_embedding(obs_td)
        logits = self.actor(embedding, att_embedding)
        logits = logits.squeeze().detach() #.numpy()  
        
        #valid_actions = x[0]['valid_actions']
        
        # define distribution over all actions for the moment
        # might be an idea to only do it for the available options
        #print('tpye of logits: {}'.format(type(logits)))
        probs = Categorical(logits=logits)
        #logits = logits.numpy()
        if not torch.count_nonzero(actions): # check if we already assigned actions or need to draw
            valid_actions = obs_td['valid_actions']
            #actions = dict()
            #print('valid_actions: {}'.format(valid_actions))
            probs_valid_actions = probs.probs
            probs_valid_actions = torch.reshape(probs_valid_actions, valid_actions.shape)
            #print('probs valid actions: {}'.format(probs_valid_actions))
            probs_valid_actions[~valid_actions] = 0
            #print('probs valid actions: {}'.format(probs_valid_actions))
            #print('probs valid actions withzeros: {}'.format(probs_valid_actions))
            probs_valid_actions = Categorical(probs = probs_valid_actions)
            actions = probs_valid_actions.sample()       
            #print('sampled actions: {}'.format(actions))
            """             valid_actions = np.array(valid_actions)
            for i in range(n_agents):
                if n_agents == 1:
                    actions[i] = self._choose_action(valid_actions[i, :], logits)
                else:
                    #print(logits[i, :])
                    actions[i] = self._choose_action(valid_actions[i, :], logits[i, :]) """
                    #print(actions[i])
        #actions_dict = {handle: action for handle, action in enumerate(actions)}
        #print('action before return: {}'.format(actions))
        #actions = torch.tensor(actions, dtype = torch.int64)
        actions.type(torch.int64)
        logprobs = torch.tensor(probs.log_prob(actions), dtype = torch.float32)
        entropy = probs.entropy().type(torch.float32).reshape(batch_size)
        values = self.critic(embedding, att_embedding).type(torch.float32)
        #print('got to before assigning td')
        td_out = TensorDict({'actions': actions, 'logprobs' : logprobs, 'entropy' : entropy,'values' : values}, batch_size= [])
        #print('after assigning td')
        #print('actions shape: {}'.format(actions.shape))
        #print('logprobs shape: {}'.format(logprobs.shape))
        #print('entropy shape: {}'.format(entropy.shape))
        #print('values shape: {}'.format(values.shape))
        return actions, logprobs, entropy, values
        #return self.actor(embedding, att_embedding), self.critic(embedding, att_embedding)
        
    """     def forward(self, agents_attr, forest, adjacency, node_order, edge_order):
        batch_size, n_agents, num_nodes, _ = forest.shape
        device = next(self.parameters()).device
        embedding, att_embedding = self.get_embedding(agents_attr, forest, adjacency, node_order, edge_order)
        worker_action = torch.zeros((batch_size, n_agents, 5), device=device)
        worker_action[:, :n_agents, :] = self.actor(embedding, att_embedding)
        critic_value = self.critic(embedding, att_embedding)
        return [worker_action], critic_value  # (batch size, 1) """
    
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
        #print('adjacency before modification: {}'.format(adjacency))
        #print('adjacencyentires with minus 2: {}'.format(adjacency[adjacency == -2]))        
        adjacency[adjacency == -2] = (
            -batch_size * n_agents * num_nodes
        )  # node_idx == -2 invalid node
        # apparently we usually don't have invalid nodes

        adjacency[..., 0] += id_nodes * num_nodes
        adjacency[..., 1] += id_nodes * num_nodes
        #print('adjacency with negative values: {}'.format(adjacency[adjacency < 0]))
        # see ifthis makes a difference
        #adjacency[adjacency < 0] = -2
        #print('modified adjacency: {}'.format(adjacency))
        return adjacency
