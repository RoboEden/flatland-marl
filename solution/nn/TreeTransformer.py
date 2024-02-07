import torch
import torch.nn as nn
import numpy as np
import itertools
import networkx as nx

from torch.nn import TransformerEncoder, TransformerEncoderLayer


# import torchsnooper


class TreeTransformer(nn.Module):
    """
    https://github.com/unbounce/pytorch-tree-lstm/blob/master/treelstm/tree_lstm.py#L10
    """

    def __init__(self, in_features, hidden_features, out_features, n_nodes) -> None:
        """TreeLSTM class initializer
        Takes in int sizes of in_features and out_features and sets up model Linear network layers.
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.n_nodes = n_nodes

        # bias terms are only on the W layers for efficiency
        self.input_linear = torch.nn.Linear(self.in_features, self.hidden_features)
        self.output_linear = torch.nn.Linear(self.hidden_features * self.n_nodes, self.out_features)
        #self.output_linear = torch.nn.Linear(self.hidden_features, self.out_features)
        self.transformer_encoder_layer = TransformerEncoderLayer(d_model = self.hidden_features, nhead = 4, batch_first=True, 
                                                                 dim_feedforward=self.hidden_features,
                                                                 dropout=0)
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, num_layers=2)
        self.apply(self._init_weights)
        self.input_linear.weight.data.normal_(mean=0.0, std=0.001)
        self.input_linear.bias.data.zero_()
        self.output_linear.weight.data.normal_(mean=0.0, std=0.001)
        self.output_linear.bias.data.zero_()
        
        self.norm = torch.nn.LayerNorm(self.out_features)
        
        #self.batch_norm = torch.nn.BatchNorm1d(self.out_features)
        
        #self.apply(self._init_weights)
        #self.maxpool = torch.nn.MaxPool2d
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.001)
            print("initialized transformer weights")
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, forest, adjacency, node_order, edge_order) -> torch.Tensor:
        """_summary_

        Parameters
        ----------
        forest : _type_
            _description_
        adjacency : _type_
            _description_
        node_order : _type_
            _description_
        edge_order : _type_
            _description_
        """
        batch_size, n_agents, n_nodes, n_attributes = forest.shape
        embedded_forest = self.input_linear(forest)
        positional_encoding = self.get_positional_encoding(forest, adjacency, node_order, edge_order, self.hidden_features) # in the desired internal model dimension
        #print(f'positional encoding in forward call: {positional_encoding}')
        #print(f'positional encoding shape in forward call: {positional_encoding.shape}')
        #print(f'position-embedded forest: {embedded_forest + positional_encoding}')
        #input = torch.concat((embedded_forest, positional_encoding), -1)
        input = embedded_forest + positional_encoding
        input = input.reshape((batch_size*n_agents, n_nodes, self.hidden_features))
        idx = torch.randperm(n_nodes)
        input = input[:,idx]        
        output = self.transformer_encoder(input)#.reshape(batch_size, n_agents, -1)
        #output = self.norm(output)
        #output = input
        output = output.reshape((batch_size * n_agents, n_nodes * self.hidden_features))
        output =self.output_linear(output).reshape((batch_size, n_agents, -1))
        #output = self.batch_norm(output)
        #output = output.reshape(batch_size, n_agents, self.hidden_features)
        #output = output.
        #print(f'output shape: {output.shape}')
        #print(f'output of linear: {self.output_linear(output).shape}')
        #print(f'output: {output}')
        #print(f'output shape: {output.shape}')
        output = self.norm(output)
        
        return output

    def get_positional_encoding(self, forest, adjacency: torch.Tensor, node_order: torch.Tensor, edge_order: torch.Tensor, output_dim) -> torch.Tensor:
        #print(f'forest shape: {forest.shape}')
        batch_size, n_agents, n_nodes, n_attributes = forest.shape
        current_device = forest.device

        positional_encoding = torch.zeros(batch_size, n_agents, n_nodes, output_dim, device=current_device).view(-1, output_dim)
        current_nodes = adjacency[:,:,0,0].flatten()
        #adjacency = torch.cat(((-1*torch.ones(batch_size, n_agents, 1, 3)), adjacency), 2)
        adjacency_flat = torch.cat(((-1*torch.ones(batch_size, n_agents, 1, 3, device=current_device)), adjacency), 2).view(-1, 3).type(torch.int64)
        #print(f'positional encoding shape: {positional_encoding.shape}')
        #print(f'max value in adjacency: {adjacency.max()}')
        
        #print(f'first current nodes: {current_nodes}')
        current_nodes = adjacency_flat[torch.isin(adjacency_flat[:,0], current_nodes)][:,1]
        #print(f'first child nodes: {current_nodes}')
        parent_nodes = torch.tensor([])
        current_depth = 0

        while True:
            root_node = torch.zeros(batch_size,n_agents,1,output_dim).to(current_device)
            #print(f'current nodes: {current_nodes}')
            parent_nodes = adjacency_flat[current_nodes, 0]
            #print(f'parent nodes: {parent_nodes}')

            index_tensor = torch.tensor([current_depth*3, current_depth*3+1, current_depth*3+2])
            index_tensor = index_tensor.repeat(len(current_nodes)//3)
            #print(f'index tensor: {index_tensor}')
            positional_encoding[current_nodes] = positional_encoding[parent_nodes]
            positional_encoding[current_nodes, index_tensor] = 1
            

            
            current_nodes = adjacency[torch.isin(adjacency[:,:,:,0], current_nodes)][:,1]
            current_depth += 1
            
            if torch.numel(current_nodes) == 0:
                break
            
            continue
            if False:
                print(f'shape of current nodes: {current_nodes.shape}')
                parent_nodes = adjacency[:,:,:,0][torch.isin(adjacency[:,:,:,(1)], current_nodes)]
                print(f'parent nodes: {parent_nodes}')
                parent_mask = torch.nonzero(torch.isin(adjacency[:,:,:,1], parent_nodes), as_tuple=True)
                print(f'parent mask: {parent_mask}')
                print(f'adjacency filter shape: {parent_nodes.shape}')
                print(f'pos enc filtered on parent mask: {positional_encoding[parent_mask].shape}')
                print(f'positional encoding expanded by 3: {positional_encoding[parent_mask].repeat(1, 3).shape}')


                #print(f'node mask unsqueezed: {node_mask.unsqueeze(-1).shape}')
                #parent_mask_expanded = parent_mask.unsqueeze(-1).expand(-1,-1,-1,output_dim)
                #parent_mask_expanded = torch.cat((root_node, parent_mask_expanded), 2).type(torch.bool)
                #print(f'parent mask expanded shape: {parent_mask_expanded.shape}')
                #print(f'parent mask expanded selection: {positional_encoding[parent_mask_expanded].shape}')
                
                
                
                if False:
                    positional_encoding.flatten(start_dim = 0, end_dim = 2)[current_nodes] = positional_encoding.flatten(start_dim = 0, end_dim = 2)[parents_of_current_nodes]
                
                #root_node = torch.zeros(batch_size,n_agents,1,output_dim).to(current_device)
                #print(f'root node device: {root_node.device}')
                
                node_mask = torch.isin(adjacency[:,:,:,0], current_nodes)
                #print(f'node mask shape: {node_mask.shape}')
                #print(f'node mask unsqueezed: {node_mask.unsqueeze(-1).shape}')
                node_mask_expanded = node_mask.unsqueeze(-1).expand(-1,-1,-1,output_dim)
                #print(f'node mask device: {node_mask_expanded.device}')
                node_mask_expanded = torch.cat((root_node, node_mask_expanded), 2).type(torch.bool)
                depth_mask = torch.zeros_like(positional_encoding).to(current_device)
                depth_mask[:,:,:,(current_depth*3,current_depth*3+1, current_depth*3+2)] = 1
                
                index_tensor = torch.tensor([[1,0,0], [0,1,0], [0,0,1]]).repeat(batch_size, n_agents, n_nodes//3, output_dim)[:,:,:,:output_dim].to(current_device)
                total_mask = torch.minimum(node_mask_expanded, depth_mask).type(torch.bool)
                index_tensor = torch.cat((root_node, index_tensor), 2).type(torch.bool)
                

                total_mask = torch.minimum(total_mask, index_tensor).type(torch.bool)
                
                print(f'pos encoding node mask selection: {positional_encoding[node_mask_expanded].shape}')
                # copy results from parents to children
                print(f'node mask: {node_mask_expanded.shape}')
                #print(f'shape of pos en filtered on node max: {positional_encoding[node_mask_expanded]}')
                #print(f'shape of pos en filtered on parent mask: {positional_encoding[parent_mask]}')
                #positional_encoding[node_mask_expanded] = positional_encoding[parent_mask].repeat(1, 3)
                positional_encoding[total_mask] = 1
                
                
                
                
                
                if False:
                    print(f'node_mask_expanded: {node_mask_expanded.shape}')
                    
                    print(f'shape of node mask: {node_mask.shape}')
                    print(f'shape of adjacency: {adjacency.shape}')
                    print(f'masked adjacency shape: {adjacency[node_mask].shape}')

                current_nodes = adjacency[node_mask][:,1]
                current_depth += 1
                

        
        if False:
            max_tree_depth = edge_order.max().type(torch.int64)
            positional_encoding = torch.zeros(batch_size, n_agents, n_nodes, output_dim)
            node_order = torch.cat((torch.ones(1, 2, 1) * max_tree_depth, edge_order), -1)
            print(f'node order before subtract: {node_order}')
            node_order = max_tree_depth - node_order
            node_order = node_order.type(torch.int64)
            print(f'adjacency list: {adjacency}')
            print(f'adjacency list shape: {adjacency.shape}')
            print(f'max tree depth: {max_tree_depth}')
            position_index = adjacency.clone()
            x = torch.tensor([0, 1, 2]).repeat((10))
            position_index[:,:,:,2] = x
            print(f'shape of edge order: {edge_order.shape}')
            print(f' order of first tree: {node_order[0,0,:]}')
            print(f'adj for tree: {adjacency[:,:,:,0:2]}')
            tree_graph = nx.from_edgelist(adjacency[:,:,:,0:2])
            print(f'tree graph adj list: {tree_graph.edges}')
            nx.draw_networkx(tree_graph)
            print(f'drew tree')
            exit()
            for node_depth in (range(max_tree_depth)):
                break
                node_mask = node_order == node_depth
                depth_mask = torch.zeros_like(positional_encoding)
                depth_mask[:,:,:,(node_depth*3,node_depth*3+1, node_depth*3+2)] = 1
                node_mask_expanded = node_mask.unsqueeze(-1).expand(-1,-1,-1,output_dim)
                print(f'position_index: {position_index}')
                index_tensor = torch.tensor([[1,0,0], [0,1,0], [0,0,1]]).repeat(1, 2, 10, 510)[:,:,:,:500]
                root_node = torch.zeros(1,2,1,500)
                total_mask = torch.minimum(node_mask_expanded, depth_mask).type(torch.bool)
                index_tensor = torch.cat((root_node, index_tensor), 2).type(torch.bool)
                

                total_mask = torch.minimum(total_mask, index_tensor).type(torch.bool)
                
                torch.set_printoptions(profile="full")
                print(f'node mask of first entry: {node_mask_expanded[0,0,0:12, 0:12]}')
                print(f'index tensor of first entry: {index_tensor[0,0,0:12, 0:12]}')
                print(f'depth mask of first entry: {depth_mask[0,0,0:12, 0:12]}')
                print(f'total mask of first entry: {total_mask[0,0,0:12, 0:12]}')
                torch.set_printoptions(profile="default")

                positional_encoding[total_mask] = 1
                #print(f'adjacency of depth: {position_index[edge_mask]}')
                print(f'filtered positional encoding: {positional_encoding[total_mask]}')
                continue
            
            
            
            
            
            print(f'node depth: {node_depth}')
            index_list = torch.range(0, n_nodes)
            position_indices = torch.zeros_like(positional_encoding).flatten()
            index_list_summed = (index_list*(batch_size*n_agents*n_nodes) + index_list).type(torch.int64)
            print(f'index_list_summed: {index_list_summed}')
            print(f'index list: {index_list}')
            position_indices[index_list_summed] = 1
            position_indices = position_indices.reshape_as(positional_encoding).type(torch.int64)
            #position_index[:,:,:2] += 1
            print(f'position index: {position_indices}')
            #positional_encoding[:,:,:, (node_depth*3,node_depth*3+1, node_depth*3+2)][:,:,:,0] = 1
            #index_list = [0,1,2,3] + [0]*(n_nodes-4)
            index_list = tuple(range(n_nodes))
            index_list = (0)
            print(f'index list: {index_list}')
            #positional_encoding[:,:,:, index_list] = 1
            positional_encoding[position_indices] =1
            print(f'shape of smaller selection: {positional_encoding[:,:,:, (node_depth*3,node_depth*3+1, node_depth*3+2)].shape}')
            
            print(f'filled positional encoding: {positional_encoding}')
            print(f'positional encoding of first node: {positional_encoding[0,0,0:8, 0:5]}')
            torch.set_printoptions(profile="full")
            print(f'further positional encodings: {positional_encoding[0,0,:,0:18]}')
            torch.set_printoptions(profile="default")
            print(f'positinal encoding all zero: {torch.all(positional_encoding==0)}')
        positional_encoding = positional_encoding.view(batch_size, n_agents, n_nodes, output_dim)
        
        torch.set_printoptions(profile="full")
        #print(f'positional encoding: {positional_encoding[0,0,:,0:18]}')
        torch.set_printoptions(profile="default")
        return positional_encoding
        
    # @torchsnooper.snoop()
    def _run_lstm(
        self,
        iteration: int,
        h: torch.Tensor,
        c: torch.Tensor,
        features: torch.Tensor,
        node_order: torch.Tensor,
        adjacency_list: torch.Tensor,
        edge_order: torch.Tensor,
    ):
        """Helper function to evaluate all tree nodes currently able to be evaluated."""
        # N is the number of nodes in the tree
        # n is the number of nodes to be evaluated on in the current iteration
        # E is the number of edges in the tree
        # e is the number of edges to be evaluated on in the current iteration
        # F is the number of features in each node
        # M is the number of hidden neurons in the network

        # node_order is a tensor of size N x 1
        # edge_order is a tensor of size E x 1
        # features is a tensor of size N x F
        # adjacency_list is a tensor of size E x 2

        # node_mask is a tensor of size N x 1
        node_mask = node_order == iteration
        # edge_mask is a tensor of size E x 1
        edge_mask = edge_order == iteration

        # x is a tensor of size n x F
        x = features[node_mask, :]

        # At iteration 0 none of the nodes should have children
        # Otherwise, select the child nodes needed for current iteration
        # and sum over their hidden states
        if iteration == 0:
            iou = self.W_iou(x)
        else:
            # adjacency_list is a tensor of size e x 2
            adjacency_list = adjacency_list[edge_mask, :]

            # parent_indexes and child_indexes are tensors of size e x 1
            # parent_indexes and child_indexes contain the integer indexes needed to index into
            # the feature and hidden state arrays to retrieve the data for those parent/child nodes.
            parent_indexes = adjacency_list[:, 0]
            #print('parent_indexes: {}'.format(parent_indexes))
            child_indexes = adjacency_list[:, 1]
            #print('child_indexes: {}'.format(child_indexes))
            # child_h and child_c are tensors of size e x 1
            child_h = h[child_indexes, :]
            #print('size of child_h: {}'.format(child_h.shape))
            child_c = c[child_indexes, :]
            #print('size of child_c: {}'.format(child_c.shape))

            # # Add child hidden states to parent offset locations
            # _, child_counts = torch.unique_consecutive(parent_indexes, return_counts=True)
            # child_counts = tuple(child_counts)

            # parent_children = torch.split(child_h, child_counts)
            # parent_list = [item.sum(0) for item in parent_children]

            # h_sum = torch.stack(parent_list)

            i_dims = child_h.shape[0] // 3
            #print('i_dims: {}'.format(i_dims))
            child_h_merge = child_h.unflatten(0, (i_dims, 3)).flatten(start_dim=1)
            #print('child_h_merge_before_flatten: {}'.format(child_h_merge))
            #child_h_merge = child_h_merge.flatten(start_dim=1)
            #print('child h merge dim: {}'.format(child_h_merge.shape))
            # h_reduce = self.W_h(child_h_merge)
            #print('x dim: {}'.format(x.shape))
            iou = self.W_iou(x) + self.U_iou(child_h_merge)

        # i, o and u are tensors of size n x M
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        # At iteration 0 none of the nodes should have children
        # Otherwise, calculate the forget states for each parent node and child node
        # and sum over the child memory cell states
        if iteration == 0:
            c[node_mask, :] = i * u
        else:
            # f is a tensor of size e x M
            #print('shape of features:{}'.format(features.shape))
            #print('shape of parents indexes: {}'.format(parent_indexes.shape))
            #print('shape of child indexes: {}'.format(child_indexes.shape))
            f = self.W_f(features[parent_indexes, :]) + self.U_f(child_h)
            #print('shape of f: {}'.format(f.shape))            
            f = torch.sigmoid(f)

            # fc is a tensor of size e x M
            fc = f * child_c

            # Add the calculated f values to the parent's memory cell state
            # parent_children = torch.split(fc, child_counts)
            # parent_list = [item.sum(0) for item in parent_children]

            # c_sum = torch.stack(parent_list)
            #print('fc before flattening: {}'.format(fc))
            #print('fc_shape before flatten):{}'.format(fc.shape))
            fc = fc.unflatten(0, (fc.shape[0] // 3, 3)).flatten(start_dim=1) # this is where we must 
            #print('fc_shape after flatten:{}'.format(fc.shape))
            c_reduce = self.W_c(fc)
            #print('shape of node mask: {}'.format(node_mask.shape))
            #print('max value of node m')
            #print('shape of c: {}'.format(c.shape))
            c[node_mask, :] = i * u + c_reduce # here we calculate the sum
            #print('vector x: {}'.format(x))
            #print('subset of feature vector: {}'.format(features[parent_indexes, :]))
            #print('size of vector x: {}'.format(x.shape))
            #print('size of subset of feature vector: {}'.format(features[parent_indexes, :].shape))
        h[node_mask, :] = o * torch.tanh(c[node_mask])
