import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"
INPUT_PATH = DATA_DIR / "final_merged_dataset.csv"
OUTPUT_PATH = DATA_DIR / "final_scaled_dataset.csv"
SCALER_DIR = DATA_DIR / "scalers"

SCALER_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(INPUT_PATH)

# Ensure datetime
df["Date"] = pd.to_datetime(df["Date"])

# --------------------------------------------------
# DEFINE FEATURES
# --------------------------------------------------
TARGET_COL = "Modal_Price"

NUMERIC_FEATURES = [
    "Temp_Avg",
    "Rainfall",
    "Sentiment_Score"
]

# --------------------------------------------------
# INITIALIZE SCALERS
# --------------------------------------------------
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

# --------------------------------------------------
# SCALE INPUT FEATURES
# --------------------------------------------------
df[NUMERIC_FEATURES] = feature_scaler.fit_transform(
    df[NUMERIC_FEATURES]
)

# --------------------------------------------------
# SCALE TARGET SEPARATELY (OPTIONAL BUT SAVED)
# --------------------------------------------------
df["Modal_Price_Scaled"] = target_scaler.fit_transform(
    df[[TARGET_COL]]
)

# --------------------------------------------------
# SAVE SCALERS
# --------------------------------------------------
joblib.dump(feature_scaler, SCALER_DIR / "feature_scaler.pkl")
joblib.dump(target_scaler, SCALER_DIR / "target_scaler.pkl")

# --------------------------------------------------
# SAVE SCALED DATASET
# --------------------------------------------------
df.to_csv(OUTPUT_PATH, index=False)

print("Feature scaling completed successfully")
print(f"Scaled dataset saved to: {OUTPUT_PATH}")
print(f"Scalers saved to: {SCALER_DIR}")

print("\nPreview:")
print(df.head())
