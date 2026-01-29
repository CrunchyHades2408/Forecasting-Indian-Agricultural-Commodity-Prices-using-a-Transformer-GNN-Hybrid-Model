import torch
import torch.nn as nn

from .transformer_encoder import TransformerEncoder
from .gnn_encoder import GNNEncoder


class FusionModel(nn.Module):
    def __init__(self, ts_input_dim, gnn_input_dim):
        super().__init__()

        self.transformer = TransformerEncoder(ts_input_dim)
        self.gnn = GNNEncoder(gnn_input_dim)

        fusion_dim = (
            self.transformer.output_dim +
            self.gnn.output_dim
        )

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_seq, node_features, adj, node_idx):
        """
        x_seq: (batch, seq_len, features)
        node_features: (num_nodes, gnn_features)
        adj: adjacency matrix (num_nodes, num_nodes)
        node_idx: node index for each sample
        """
        ts_repr = self.transformer(x_seq)
        gnn_repr_all = self.gnn(node_features, adj)
        gnn_repr = gnn_repr_all[node_idx]

        fused = torch.cat([ts_repr, gnn_repr], dim=1)
        return self.regressor(fused).squeeze()
