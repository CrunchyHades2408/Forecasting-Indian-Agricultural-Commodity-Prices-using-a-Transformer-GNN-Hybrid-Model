import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"
INPUT_PATH = DATA_DIR / "final_scaled_dataset.csv"
OUTPUT_DIR = DATA_DIR / "graph"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(INPUT_PATH)
df["Date"] = pd.to_datetime(df["Date"])

# --------------------------------------------------
# DEFINE NODE ID (UNIQUE MARKET)
# --------------------------------------------------
df["Node_ID"] = (
    df["Commodity"].astype(str) + "|" +
    df["State"].astype(str) + "|" +
    df["Market"].astype(str)
)

node_list = df["Node_ID"].unique()
node_id_map = {node: idx for idx, node in enumerate(node_list)}

# --------------------------------------------------
# CREATE NODE FEATURES
# --------------------------------------------------
NODE_FEATURE_COLS = [
    "Modal_Price_Scaled",
    "Temp_Avg",
    "Rainfall",
    "Sentiment_Score"
]

node_features = []

for node in node_list:
    node_df = df[df["Node_ID"] == node]

    # Aggregate features (mean over time)
    features = node_df[NODE_FEATURE_COLS].mean().values
    node_features.append(features)

node_features = np.array(node_features)

# --------------------------------------------------
# BUILD EDGES
# RULES:
# 1. Same State → connect
# 2. Same Commodity → connect
# --------------------------------------------------
edges = set()

# Group nodes by State
for _, group in df.groupby("State"):
    nodes = group["Node_ID"].unique()
    for u, v in combinations(nodes, 2):
        edges.add((node_id_map[u], node_id_map[v]))
        edges.add((node_id_map[v], node_id_map[u]))

# Group nodes by Commodity
for _, group in df.groupby("Commodity"):
    nodes = group["Node_ID"].unique()
    for u, v in combinations(nodes, 2):
        edges.add((node_id_map[u], node_id_map[v]))
        edges.add((node_id_map[v], node_id_map[u]))

# --------------------------------------------------
# CONVERT EDGES TO EDGE_INDEX FORMAT
# --------------------------------------------------
edge_index = np.array(list(edges)).T  # shape: (2, num_edges)

# --------------------------------------------------
# SAVE OUTPUTS
# --------------------------------------------------
np.save(OUTPUT_DIR / "node_features.npy", node_features)
np.save(OUTPUT_DIR / "edge_index.npy", edge_index)
np.save(OUTPUT_DIR / "node_id_map.npy", node_list)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------
print("Market graph construction completed")
print(f"Number of nodes: {len(node_list)}")
print(f"Number of edges: {edge_index.shape[1]}")
print(f"Saved to: {OUTPUT_DIR}")
