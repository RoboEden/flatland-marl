import torch
import torch.nn as nn
import numpy as np
import itertools


# import torchsnooper


class TreeTransformer(nn.Module):
    """
    https://github.com/unbounce/pytorch-tree-lstm/blob/master/treelstm/tree_lstm.py#L10
    """

    def __init__(self, in_features, hidden_features, out_features) -> None:
        """TreeLSTM class initializer
        Takes in int sizes of in_features and out_features and sets up model Linear network layers.
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        # bias terms are only on the W layers for efficiency
        self.input_linear = torch.nn.Linear(self.in_features, self.hidden_features)

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
        embedded_forest = self.input_linear(forest)
        positional_encoding = self.get_positional_encoding(forest, adjacency, node_order, edge_order) # in the desired internal model dimension
        
        output = transformer(embedded_forest + positional_encoding)
        return output

    def get_positional_encoding(self, forest, adjacency: torch.Tensor, node_order: torch.Tensor, edge_order: torch.Tensor) -> torch.Tensor:
        print(f'forest shape: {forest.shape}')
        batch_size, n_agents, n_nodes, n_attributes = forest.shape
        positional_encoding = torch.zeros(batch_size, n_agents, n_nodes, self.hidden_features)
        max_tree_depth = node_order.max()
        print(f'adjacency list: {adjacency}')
        print(f'adjacency list shape: {adjacency.shape}')
        print(f'max tree depth: {max_tree_depth}')
        position_index = adjacency.clone()
        x = torch.tensor([0, 1, 2]).repeat((10))
        position_index[:,:,:,2] = x
        for node_depth in range(max_tree_depth):
            node_mask = node_order == node_depth
            depth_mask = torch.zeros_like(positional_encoding)
            depth_mask[:,:,:,(node_depth*3,node_depth*3+1, node_depth*3+2)] = 1
            node_mask_expanded = node_mask.unsqueeze(-1).expand(-1,-1,-1,self.hidden_features)
            print(f'position_index: {position_index}')
            index_tensor = torch.tensor([[1,0,0], [0,1,0], [0,0,1]]).repeat(1, 2, 10, 510)[:,:,:,:500]
            root_node = torch.zeros(1,2,1,500)
            total_mask = torch.minimum(node_mask_expanded, depth_mask).type(torch.bool)
            index_tensor = torch.cat((root_node, index_tensor), 2).type(torch.bool)
            

            total_mask = torch.minimum(total_mask, index_tensor).type(torch.bool)
            

            
            print(f'node_mask expanded shape: {node_mask_expanded}')
            print(f'depth mask: {depth_mask}')
            print(f'total mask: {total_mask}')
            #edge_mask = node_mask[:,:,:-1]
            print(f'node_maks: {node_mask}')
            print(f'node mask shape: {node_mask.shape}')
            print(f'positional encoding shape: {positional_encoding.shape}')
            index_tensor = torch.tensor([0,2])
            print(f'filtered for positions: {positional_encoding[node_mask][:,(node_depth*3,node_depth*3+1, node_depth*3+2)].flatten()[index_tensor]}')
            #positional_encoding[node_mask][:,(node_depth*3,node_depth*3+1, node_depth*3+2)].flatten()[index_tensor] = 1
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
        print(f'positional encoding of first node: {positional_encoding[0,0,5]}')
        print(f'positinal encoding all zero: {torch.all(positional_encoding==0)}')
        
        
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
