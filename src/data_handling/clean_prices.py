import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PROJECT ROOT (MINOR/)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PRICES_DIR = BASE_DIR / "data" / "raw" / "prices"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "prices_cleaned.csv"

print("Looking for price CSVs in:", RAW_PRICES_DIR)

dfs = []

for file in RAW_PRICES_DIR.rglob("*.csv"):
    print("Reading:", file)

    df = pd.read_csv(file)

    # --------------------------------------------------
    # STANDARDIZE COLUMN NAMES (YOUR SCHEMA)
    # --------------------------------------------------
    df = df.rename(columns={
        "t": "Date",
        "cmdty": "Commodity",
        "market_name": "Market",
        "state_name": "State",
        "p_modal": "Modal_Price"
    })

    # --------------------------------------------------
    # HARD CHECK: Date must exist now
    # --------------------------------------------------
    if "Date" not in df.columns:
        print(f"⚠️ Skipping {file.name} (no Date column)")
        continue

    # --------------------------------------------------
    # PARSE DATE STRING (YYYY-MM-DD)
    # --------------------------------------------------
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # --------------------------------------------------
    # CLEAN PRICE
    # --------------------------------------------------
    df["Modal_Price"] = pd.to_numeric(df["Modal_Price"], errors="coerce")

    df = df.dropna(subset=["Date", "Modal_Price"])

    # --------------------------------------------------
    # KEEP REQUIRED COLUMNS
    # --------------------------------------------------
    df = df[["Date", "Commodity", "State", "Market", "Modal_Price"]]

    dfs.append(df)

# --------------------------------------------------
# SAFETY CHECK
# --------------------------------------------------
if not dfs:
    raise RuntimeError("No valid price CSV files found after processing.")

# --------------------------------------------------
# MERGE ALL PRICES
# --------------------------------------------------
final_df = pd.concat(dfs, ignore_index=True)

final_df = final_df.drop_duplicates(
    subset=["Date", "Commodity", "State", "Market"]
)

final_df = final_df.sort_values("Date")

# --------------------------------------------------
# SAVE OUTPUT
# --------------------------------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
final_df.to_csv(OUTPUT_PATH, index=False)

print("\nprices_cleaned.csv created successfully")
print(final_df.head())
print("Total rows:", len(final_df))
