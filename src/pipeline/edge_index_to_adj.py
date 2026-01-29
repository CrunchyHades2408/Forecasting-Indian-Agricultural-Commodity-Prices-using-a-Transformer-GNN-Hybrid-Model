import numpy as np
from pathlib import Path

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
GRAPH_DIR = BASE_DIR / "data" / "processed" / "graph"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
edge_index = np.load(GRAPH_DIR / "edge_index.npy")
node_features = np.load(GRAPH_DIR / "node_feature.npy")

num_nodes = node_features.shape[0]

# --------------------------------------------------
# CREATE ADJACENCY MATRIX
# --------------------------------------------------
adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

for i in range(edge_index.shape[1]):
    src = int(edge_index[0, i])
    dst = int(edge_index[1, i])
    adj[src, dst] = 1.0

# Add self-loops
np.fill_diagonal(adj, 1.0)

# --------------------------------------------------
# SAVE OUTPUT
# --------------------------------------------------
np.save(GRAPH_DIR / "adjacency_matrix.npy", adj)

print("adjacency_matrix.npy created successfully")
print("Shape:", adj.shape)
