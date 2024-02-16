import torch
import torch.nn as nn
import numpy as np
import itertools
import networkx as nx

from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TreeTransformer(nn.Module):
    """Module to generate fixed-size embeddings of trees with node attributes"""

    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, n_nodes: int
    ) -> None:
        """Initialize tree transformer module.

        Parameters
        ----------
        in_features : int
            Size of the node attributes of each node
        hidden_features : int
            Size of hidden features used in the module
        out_features : int
            Size of output (gives vector of length out_features for each tree)
        n_nodes : int
            Number of nodes in each input tree
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.n_nodes = n_nodes

        self.input_linear = torch.nn.Linear(self.in_features, self.hidden_features)
        self.output_linear = torch.nn.Linear(
            self.hidden_features * self.n_nodes, self.out_features
        )
        self.transformer_encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden_features,
            nhead=4,
            batch_first=True,
            dim_feedforward=self.hidden_features,
            dropout=0,
        )
        self.transformer_encoder = TransformerEncoder(
            self.transformer_encoder_layer, num_layers=2
        )
        self.apply(self._init_weights)
        self.input_linear.weight.data.normal_(mean=0.0, std=0.001)
        self.input_linear.bias.data.zero_()
        self.output_linear.weight.data.normal_(mean=0.0, std=0.001)
        self.output_linear.bias.data.zero_()
        self.norm = torch.nn.LayerNorm(self.out_features)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.001)
            print("initialized transformer weights")
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, forest: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Calculate the transformer tree embedding

        Parameters
        ----------
        forest : tensor
            Node attributes of each node, of shape batch_size x n_agents x n_nodes x n_attributes
        adjacency : tensor
            Adjacency list for each tree.

        Returns
        ----------
        Fixed-size vector for each tree.
        """
        batch_size, n_agents, n_nodes, n_attributes = forest.shape
        embedded_forest = self.input_linear(forest)
        positional_encoding = self.get_positional_encoding(
            forest, adjacency, self.hidden_features
        )
        input = embedded_forest + positional_encoding
        input = input.reshape((batch_size * n_agents, n_nodes, self.hidden_features))
        idx = torch.randperm(n_nodes)
        input = input[:, idx]
        output = self.transformer_encoder(input)
        output = output.reshape((batch_size * n_agents, n_nodes * self.hidden_features))
        output = self.output_linear(output).reshape((batch_size, n_agents, -1))
        output = self.norm(output)

        return output

    def get_positional_encoding(
        self, forest, adjacency: torch.Tensor, output_dim
    ) -> torch.Tensor:
        """Return a positional encoding for the each tree node based on
        https://www.microsoft.com/en-us/research/publication/novel-positional-encodings-to-enable-tree-based-transformers/
        """

        batch_size, n_agents, n_nodes, n_attributes = forest.shape
        current_device = forest.device

        positional_encoding = torch.zeros(
            batch_size, n_agents, n_nodes, output_dim, device=current_device
        ).view(-1, output_dim)
        current_nodes = adjacency[:, :, 0, 0].flatten()
        adjacency_flat = (
            torch.cat(
                (
                    (
                        -1
                        * torch.ones(batch_size, n_agents, 1, 3, device=current_device)
                    ),
                    adjacency,
                ),
                2,
            )
            .view(-1, 3)
            .type(torch.int64)
        )
        current_nodes = adjacency_flat[torch.isin(adjacency_flat[:, 0], current_nodes)][
            :, 1
        ]
        parent_nodes = torch.tensor([])
        current_depth = 0

        while True:
            root_node = torch.zeros(batch_size, n_agents, 1, output_dim).to(
                current_device
            )
            parent_nodes = adjacency_flat[current_nodes, 0]

            index_tensor = torch.tensor(
                [current_depth * 3, current_depth * 3 + 1, current_depth * 3 + 2]
            )
            index_tensor = index_tensor.repeat(len(current_nodes) // 3)
            positional_encoding[current_nodes] = positional_encoding[parent_nodes]
            positional_encoding[current_nodes, index_tensor] = 1

            current_nodes = adjacency[torch.isin(adjacency[:, :, :, 0], current_nodes)][
                :, 1
            ]
            current_depth += 1

            if torch.numel(current_nodes) == 0:
                break
            continue

        positional_encoding = positional_encoding.view(
            batch_size, n_agents, n_nodes, output_dim
        )

        return positional_encoding
