import torch
import torch.nn as nn

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.relu = nn.ReLU()
        self.output_dim = hidden_dim

    def forward(self, x, adj):
        """
        x: Node features (num_nodes, input_dim)
        adj: Adjacency matrix (num_nodes, num_nodes)
        """
        h = torch.matmul(adj, x)
        h = self.fc1(h)
        h = self.relu(h)

        h = torch.matmul(adj, h)
        h = self.fc2(h)

        return h
