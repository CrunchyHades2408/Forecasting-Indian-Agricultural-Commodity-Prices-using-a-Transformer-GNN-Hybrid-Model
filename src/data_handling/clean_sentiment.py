import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "raw" / "sentiment" / "sentiment_processed.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "sentiment_processed.csv"

df = pd.read_csv(INPUT_PATH)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Aggregate daily sentiment per commodity
df = (
    df.groupby(["Date", "Commodity"], as_index=False)
      .agg({"Sentiment_Score": "mean"})
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("sentiment_processed.csv cleaned and saved")
print(df.head())
