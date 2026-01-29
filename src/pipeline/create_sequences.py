import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"
INPUT_PATH = DATA_DIR / "final_scaled_dataset.csv"
OUTPUT_DIR = DATA_DIR / "sequences"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------
SEQUENCE_LENGTH = 14   # 14-day lookback window

FEATURE_COLS = [
    "Temp_Avg",
    "Rainfall",
    "Sentiment_Score"
]

TARGET_COL = "Modal_Price_Scaled"

GROUP_COLS = ["Commodity", "State", "Market"]


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(INPUT_PATH)
df["Date"] = pd.to_datetime(df["Date"])

# Sort for time-series consistency
df = df.sort_values(GROUP_COLS + ["Date"])

# --------------------------------------------------
# ADD LAGGED PRICE (TIER-2 CHANGE)
# --------------------------------------------------
df["Price_Lag"] = (
    df.groupby(GROUP_COLS)["Modal_Price_Scaled"]
      .shift(1)
)

# Drop first row of each group (no lag available)
df = df.dropna(subset=["Price_Lag"]).reset_index(drop=True)
# --------------------------------------------------
# CREATE SEQUENCES
# --------------------------------------------------
X_sequences = []
y_targets = []
meta_info = []  # optional: keep track of market identity

for group_key, group_df in df.groupby(GROUP_COLS):
    group_df = group_df.sort_values("Date")

    feature_values = group_df[FEATURE_COLS].values
    target_values = group_df[TARGET_COL].values

    if len(group_df) <= SEQUENCE_LENGTH:
        continue

    for i in range(len(group_df) - SEQUENCE_LENGTH):
        X_sequences.append(
            feature_values[i : i + SEQUENCE_LENGTH]
        )
        y_targets.append(
            target_values[i + SEQUENCE_LENGTH]
        )
        meta_info.append(group_key)

# --------------------------------------------------
# CONVERT TO NUMPY ARRAYS
# --------------------------------------------------
X_sequences = np.array(X_sequences)
y_targets = np.array(y_targets)

# --------------------------------------------------
# SAVE OUTPUTS
# --------------------------------------------------
np.save(OUTPUT_DIR / "X_sequences.npy", X_sequences)
np.save(OUTPUT_DIR / "y_targets.npy", y_targets)
np.save(OUTPUT_DIR / "meta_info.npy", np.array(meta_info, dtype=object))

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------
print("Sequence creation completed")
print(f"Sequence length: {SEQUENCE_LENGTH}")
print(f"X shape: {X_sequences.shape}")
print(f"y shape: {y_targets.shape}")
print(f"Saved to: {OUTPUT_DIR}")
