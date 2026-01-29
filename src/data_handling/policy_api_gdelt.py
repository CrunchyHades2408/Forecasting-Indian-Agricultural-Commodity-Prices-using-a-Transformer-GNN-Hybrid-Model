import requests
import pandas as pd

policy_keywords = {
    "Wheat": ["wheat MSP India", "wheat export ban India"],
    "Rice": ["rice MSP India", "rice export ban India"],
    "Maize": ["maize MSP India", "maize government policy India"]
}

def fetch_policy_news(keyword):
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": keyword,
        "mode": "artlist",
        "maxrecords": 50,
        "format": "json",
        "country": "IN"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("articles", [])
    except:
        pass
    return []

records = []

for commodity, keywords in policy_keywords.items():
    for kw in keywords:
        print(f"Fetching policy news for: {kw}")
        articles = fetch_policy_news(kw)

        for art in articles:
            records.append({
                "Date": art.get("seendate", None),
                "Commodity": commodity,
                "Keyword": kw,
                "Title": art.get("title", "")
            })

# -----------------------------
# CREATE DATAFRAME SAFELY
# -----------------------------
df = pd.DataFrame(records)

# ✅ SAFETY CHECK
if df.empty:
    print("No policy news returned from API.")
    df = pd.DataFrame(columns=["Date", "Commodity", "Keyword", "Title"])

# -----------------------------
# CLEAN DATE COLUMN
# -----------------------------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df.sort_values("Date")

# -----------------------------
# SAVE RAW POLICY NEWS
# -----------------------------
df.to_csv("policy_news_raw.csv", index=False)

print("Policy news dataset saved as policy_news_raw.csv")
print(df.head())

