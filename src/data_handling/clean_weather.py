import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "raw" / "weather" / "state_level_weather.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "state_level_weather.csv"

df = pd.read_csv(INPUT_PATH)

# Standardize column names (if needed)
df = df.rename(columns={
    "time": "Date",
    "tavg": "Temp_Avg",
    "prcp": "Rainfall"
})

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Keep required columns
df = df[["Date", "State", "Temp_Avg", "Rainfall"]]

# Handle missing values
df["Temp_Avg"] = df["Temp_Avg"].fillna(method="ffill")
df["Rainfall"] = df["Rainfall"].fillna(0)

df = df.dropna(subset=["Date", "State"])

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("state_level_weather.csv cleaned and saved")
print(df.head())
