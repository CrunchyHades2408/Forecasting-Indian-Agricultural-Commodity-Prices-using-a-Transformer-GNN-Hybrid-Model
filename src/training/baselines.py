# ==================================================
# NAIVE BASELINE (PERSISTENCE MODEL - INVERSE SCALED)
# ==================================================
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

SPLIT_DIR = BASE_DIR / "data" / "processed" / "splits"
SCALER_DIR = BASE_DIR / "data" / "processed" / "scalers"
RESULTS_DIR = BASE_DIR / "results" / "metrics"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD TEST TARGETS (SCALED)
# --------------------------------------------------
y_test_scaled = np.load(SPLIT_DIR / "y_test.npy")

# --------------------------------------------------
# LOAD TARGET SCALER
# --------------------------------------------------
target_scaler = joblib.load(SCALER_DIR / "target_scaler.pkl")

# --------------------------------------------------
# NAIVE PREDICTION (y[t] = y[t-1])
# --------------------------------------------------
y_true_scaled = y_test_scaled[1:].reshape(-1, 1)
y_pred_scaled = y_test_scaled[:-1].reshape(-1, 1)

# --------------------------------------------------
# INVERSE SCALE TO REAL PRICE UNITS
# --------------------------------------------------
y_true = target_scaler.inverse_transform(y_true_scaled).flatten()
y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()

# --------------------------------------------------
# METRICS (REAL UNITS)
# --------------------------------------------------
mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------
output_path = RESULTS_DIR / "baseline_naive.txt"

with open(output_path, "w") as f:
    f.write("Naive Baseline (Persistence Model)\n")
    f.write("Metrics computed in real price units\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")

# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------
print("Naive Baseline Results (Inverse Scaled)")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"Saved to: {output_path}")
