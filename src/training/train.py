# ==================================================
# FIX PYTHON PATH (IMPORTANT)
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
from torch.utils.data import DataLoader, TensorDataset

from models.fusion_model import FusionModel

# ==================================================
# PATHS
# ==================================================
DATA_DIR = BASE_DIR / "data" / "processed"
SEQ_DIR = DATA_DIR / "sequences"
SPLIT_DIR = DATA_DIR / "splits"
GRAPH_DIR = DATA_DIR / "graph"
MODEL_DIR = BASE_DIR / "results" / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# TRAINING CONFIG
# ==================================================
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ==================================================
# LOAD TEMPORAL DATA
# ==================================================
X_train = np.load(SPLIT_DIR / "X_train.npy")
y_train = np.load(SPLIT_DIR / "y_train.npy")

X_val = np.load(SPLIT_DIR / "X_val.npy")
y_val = np.load(SPLIT_DIR / "y_val.npy")

# ==================================================
# LOAD GRAPH DATA
# ==================================================
node_features = np.load(GRAPH_DIR / "node_feature.npy")
adjacency_matrix = np.load(GRAPH_DIR / "adjacency_matrix.npy")

# ==================================================
# CONVERT TO TORCH TENSORS
# ==================================================
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

node_features = torch.tensor(node_features, dtype=torch.float32).to(DEVICE)
adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32).to(DEVICE)

# ==================================================
# NODE INDEX (SIMPLIFIED – ACCEPTABLE FOR MINOR)
# ==================================================
node_idx_train = torch.zeros(len(X_train), dtype=torch.long).to(DEVICE)
node_idx_val = torch.zeros(len(X_val), dtype=torch.long).to(DEVICE)

# ==================================================
# DATA LOADERS
# ==================================================
train_dataset = TensorDataset(X_train, y_train, node_idx_train)
val_dataset = TensorDataset(X_val, y_val, node_idx_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==================================================
# MODEL INITIALIZATION
# ==================================================
ts_input_dim = X_train.shape[2]
gnn_input_dim = node_features.shape[1]

model = FusionModel(
    ts_input_dim=ts_input_dim,
    gnn_input_dim=gnn_input_dim
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# ==================================================
# EARLY STOPPING SETUP (NEW)
# ==================================================
best_val_loss = float("inf")
patience = 3
patience_counter = 0
# ==================================================
# TRAINING LOOP
# ==================================================
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for x_seq, y_true, node_idx in train_loader:
        x_seq = x_seq.to(DEVICE)
        y_true = y_true.to(DEVICE)
        node_idx = node_idx.to(DEVICE)

        optimizer.zero_grad()

        y_pred = model(
            x_seq,
            node_features,
            adjacency_matrix,
            node_idx
        )

        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x_seq, y_true, node_idx in val_loader:
            x_seq = x_seq.to(DEVICE)
            y_true = y_true.to(DEVICE)
            node_idx = node_idx.to(DEVICE)

            y_pred = model(
                x_seq,
                node_features,
                adjacency_matrix,
                node_idx
            )

            loss = criterion(y_pred, y_true)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    # ---------------- EARLY STOPPING ----------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        # Save best model
        torch.save(model.state_dict(), MODEL_DIR / "fusion_model.pth")
        print(" Best model saved")

    else:
        patience_counter += 1
        print(f" No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print("\nEarly stopping triggered")
            break


# ==================================================
# SAVE MODEL
# ==================================================
#MODEL_PATH = MODEL_DIR / "fusion_model.pth"
#torch.save(model.state_dict(), MODEL_PATH)

print("\nTraining completed successfully")
#print("Model saved at:", MODEL_PATH)
