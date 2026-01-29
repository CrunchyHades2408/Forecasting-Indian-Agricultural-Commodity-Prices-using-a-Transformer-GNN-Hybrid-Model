import numpy as np
from pathlib import Path

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "sequences"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD SEQUENCES
# --------------------------------------------------
X = np.load(DATA_DIR / "X_sequences.npy")
y = np.load(DATA_DIR / "y_targets.npy")

print("Loaded sequences:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# --------------------------------------------------
# TEMPORAL SPLIT RATIOS
# --------------------------------------------------
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0

n_samples = X.shape[0]

train_end = int(n_samples * TRAIN_RATIO)
val_end = train_end + int(n_samples * VAL_RATIO)

# --------------------------------------------------
# SPLIT DATA (NO SHUFFLING!)
# --------------------------------------------------
X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]

# --------------------------------------------------
# SAVE SPLITS
# --------------------------------------------------
np.save(OUTPUT_DIR / "X_train.npy", X_train)
np.save(OUTPUT_DIR / "y_train.npy", y_train)

np.save(OUTPUT_DIR / "X_val.npy", X_val)
np.save(OUTPUT_DIR / "y_val.npy", y_val)

np.save(OUTPUT_DIR / "X_test.npy", X_test)
np.save(OUTPUT_DIR / "y_test.npy", y_test)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------
print("\nTemporal split completed successfully")
print(f"Train samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Saved to: {OUTPUT_DIR}")
