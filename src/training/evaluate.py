# ==================================================
# FIX PYTHON PATH
# ==================================================
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

# ==================================================
# IMPORTS
# ==================================================
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import joblib

from models.fusion_model import FusionModel

# ==================================================
# PATHS
# ==================================================
DATA_DIR = BASE_DIR / "data" / "processed"
SEQ_DIR = DATA_DIR / "sequences"
SPLIT_DIR = DATA_DIR / "splits"
GRAPH_DIR = DATA_DIR / "graph"
SCALER_DIR = DATA_DIR / "scalers"

MODEL_PATH = BASE_DIR / "results" / "models" / "fusion_model.pth"
RESULTS_DIR = BASE_DIR / "results" / "metrics"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# DEVICE
# ==================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ==================================================
# LOAD TEST DATA
# ==================================================
X_test = np.load(SPLIT_DIR / "X_test.npy")
y_test = np.load(SPLIT_DIR / "y_test.npy")

# ==================================================
# LOAD GRAPH DATA
# ==================================================
node_features = np.load(GRAPH_DIR / "node_feature.npy")
adjacency_matrix = np.load(GRAPH_DIR / "adjacency_matrix.npy")

# ==================================================
# LOAD SCALER
# ==================================================
target_scaler = joblib.load(SCALER_DIR / "target_scaler.pkl")

# ==================================================
# TORCH TENSORS
# ==================================================
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

node_features = torch.tensor(node_features, dtype=torch.float32).to(DEVICE)
adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32).to(DEVICE)

# Simplified node index (baseline)
node_idx_test = torch.zeros(len(X_test), dtype=torch.long).to(DEVICE)

# ==================================================
# LOAD MODEL
# ==================================================
ts_input_dim = X_test.shape[2]
gnn_input_dim = node_features.shape[1]

model = FusionModel(
    ts_input_dim=ts_input_dim,
    gnn_input_dim=gnn_input_dim
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==================================================
# INFERENCE
# ==================================================
with torch.no_grad():
    y_pred_scaled = model(
        X_test,
        node_features,
        adjacency_matrix,
        node_idx_test
    )

# ==================================================
# INVERSE SCALE
# ==================================================
y_pred_scaled = y_pred_scaled.cpu().numpy().reshape(-1, 1)
y_test_scaled = y_test.cpu().numpy().reshape(-1, 1)

y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()
y_true = target_scaler.inverse_transform(y_test_scaled).flatten()

# ==================================================
# METRICS
# ==================================================
mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

# ==================================================
# SAVE METRICS
# ==================================================
with open(RESULTS_DIR / "test_metrics.txt", "w") as f:
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")

# ==================================================
# SAVE PREDICTIONS
# ==================================================
results_df = pd.DataFrame({
    "Actual_Price": y_true,
    "Predicted_Price": y_pred
})

results_df.to_csv(
    RESULTS_DIR / "predictions_vs_actual.csv",
    index=False
)

# ==================================================
# OUTPUT
# ==================================================
print("\nEvaluation completed")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print("Results saved to:", RESULTS_DIR)

