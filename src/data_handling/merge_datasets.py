import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_PATH = DATA_DIR / "final_merged_dataset.csv"

# --------------------------------------------------
# LOAD REQUIRED DATASETS
# --------------------------------------------------
prices = pd.read_csv(DATA_DIR / "prices_cleaned.csv")
weather = pd.read_csv(DATA_DIR / "state_level_weather.csv")
sentiment = pd.read_csv(DATA_DIR / "sentiment_processed.csv")

# --------------------------------------------------
# LOAD POLICY (OPTIONAL)
# --------------------------------------------------
policy_path = DATA_DIR / "policy_events.csv"
use_policy = policy_path.exists()

if use_policy:
    policy = pd.read_csv(policy_path)
    print("Policy data loaded")
else:
    policy = None
    print("Policy data NOT found — proceeding without policy")

# --------------------------------------------------
# DATE PARSING
# --------------------------------------------------
for df in [prices, weather, sentiment]:
    df["Date"] = pd.to_datetime(df["Date"])

if use_policy:
    policy["Date"] = pd.to_datetime(policy["Date"])

# --------------------------------------------------
# MERGE WEATHER (Date + State)
# --------------------------------------------------
merged = prices.merge(
    weather,
    on=["Date", "State"],
    how="left"
)

# --------------------------------------------------
# MERGE SENTIMENT (Date + Commodity)
# --------------------------------------------------
merged = merged.merge(
    sentiment,
    on=["Date", "Commodity"],
    how="left"
)

# --------------------------------------------------
# MERGE POLICY (ONLY IF PRESENT)
# --------------------------------------------------
if use_policy:
    merged = merged.merge(
        policy,
        on=["Date", "Commodity"],
        how="left"
    )

# --------------------------------------------------
# HANDLE MISSING VALUES
# --------------------------------------------------
merged["Temp_Avg"] = merged["Temp_Avg"].fillna(method="ffill")
merged["Rainfall"] = merged["Rainfall"].fillna(0)

merged["Sentiment_Score"] = merged["Sentiment_Score"].fillna(0)

if use_policy:
    for col in policy.columns:
        if col not in ["Date", "Commodity"]:
            merged[col] = merged[col].fillna(0)

# --------------------------------------------------
# SORT & SAVE
# --------------------------------------------------
merged = merged.sort_values(["Commodity", "State", "Market", "Date"])

merged.to_csv(OUTPUT_PATH, index=False)

print("\nfinal_merged_dataset.csv created successfully")
print(merged.head())
print("Total rows:", len(merged))
