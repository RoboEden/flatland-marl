import torch
import torch.nn as nn
from impl_config import FeatureParserConfig as fp
from impl_config import NetworkConfig as ns
from tensordict.tensordict import TensorDict
from torch.distributions.categorical import Categorical
import tensordict
from .TreeLSTM import TreeLSTM

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Transformer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        #for param in self.attention.parameters():
        #    print(param.na)
        #    print(isinstance(param, nn.Linear))
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

class embedding_net(nn.Module):
    """
    Feature:  cat(agents_attr_embedding, tree_embedding)
    structure: mlp
    """

    def __init__(self):
        super(embedding_net, self).__init__()
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
        self.apply(self._init_weights)
        
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.1)
            print("initialized weights")
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, obs_td):

        batch_size, n_agents, num_nodes, _ = obs_td['node_attr'].shape
        #print(f'node attr shape: {obs_td["node_attr"].shape}')
        device = obs_td.device
        
        adjacency = obs_td['adjacency'].clone()
        adjacency = self.modify_adjacency(adjacency, device)
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
        #print(f'embedding: {embedding.shape}')
        return embedding, att_embedding
    
    def modify_adjacency(self, adjacency, _device):
        adjacency = adjacency.clone() # to avoid side effects
        batch_size, n_agents, num_edges, _ = adjacency.shape
        num_nodes = num_edges + 1
        id_tree = torch.arange(0, batch_size * n_agents, device=_device)

        #print('id tree: {}'.format(id_tree))
        #print('id tree shape: {}'.format(id_tree.shape))
        id_nodes = id_tree.view(batch_size, n_agents, 1)    
        # try non masked version
        
        adjacency[adjacency == -2] = (
            -batch_size * n_agents * num_nodes
        )  
        adjacency[..., 0] += id_nodes * num_nodes
        adjacency[..., 1] += id_nodes * num_nodes
        adjacency[adjacency < 0] = -2
        #print('modified adjacency: {}'.format(adjacency))
        return adjacency  
        

class actor_net(nn.Module):
    def __init__(self):
        super(actor_net, self).__init__()
        self.actor_net = nn.Sequential(
        nn.Linear(ns.hidden_sz * 2 + ns.tree_embedding_sz * 2, 2 * ns.hidden_sz),
        nn.GELU(),
        nn.Linear(ns.hidden_sz * 2, ns.hidden_sz),
        nn.GELU(),
        nn.Linear(ns.hidden_sz, fp.action_sz),
        )
        self.apply(self._init_weights)
    
    def forward(self, embedding, att_embedding, valid_actions):
        #print('in actor func')
        worker_action = torch.cat([embedding, att_embedding], dim=-1)
        worker_action = self.actor_net(worker_action)
        #print(f'worker action: {worker_action}')
        #print(f'valid action: {valid_actions}')
        worker_action[~valid_actions] = float('-inf')
        #print(f'modified worker actions: {worker_action}')
        return worker_action
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.1)
            print("initialized weights")
            if module.bias is not None:
                module.bias.data.zero_()

class critic_net(nn.Module):
    def __init__(self):
        super(critic_net, self).__init__()
        self.critic_net = nn.Sequential(
            nn.Linear(ns.hidden_sz * 2 + ns.tree_embedding_sz * 2, ns.hidden_sz * 2),
            nn.GELU(),
            nn.Linear(ns.hidden_sz * 2, ns.hidden_sz),
            nn.GELU(),
            nn.Linear(ns.hidden_sz, 1),
        )
        self.apply(self._init_weights)
    
    def forward(self, embedding, att_embedding):
        #print('in critic fun')
        output = torch.cat([embedding, att_embedding], dim=-1)
        critic_value = self.critic_net(output)
        critic_value = critic_value.mean(1).view(-1) # what exactly are we doing here?
        return critic_value
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.1)
            print("initialized weights")
            if module.bias is not None:
                module.bias.data.zero_()