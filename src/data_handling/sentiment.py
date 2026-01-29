import feedparser
import pandas as pd
from urllib.parse import quote_plus
from datetime import datetime

# -----------------------------
# KEYWORDS
# -----------------------------
keywords = {
    "Rice": [
        "Rice price India",
        "Rice MSP India",
        "Rice crop India",
        "Rice export ban India"
    ],
    "Wheat": [
        "Wheat price India",
        "Wheat MSP India",
        "Wheat export ban India",
        "Wheat crop India"
    ],
    "Maize": [
        "Maize price India",
        "Maize crop India",
        "Maize demand India",
        "Maize export India"
    ]
}

# -----------------------------
# FUNCTION TO FETCH NEWS
# -----------------------------
def fetch_news(keyword, commodity):
    encoded_keyword = quote_plus(keyword)  # FIX HERE
    url = (
        f"https://news.google.com/rss/search?"
        f"q={encoded_keyword}&hl=en-IN&gl=IN&ceid=IN:en"
    )

    feed = feedparser.parse(url)

    records = []
    for entry in feed.entries:
        records.append({
            "Date": entry.get("published", ""),
            "Commodity": commodity,
            "Keyword": keyword,
            "Text": entry.title,
            "Source": entry.source.title if "source" in entry else "Google News"
        })

    return records

# -----------------------------
# MAIN EXECUTION
# -----------------------------
all_news = []

for commodity, keyword_list in keywords.items():
    for keyword in keyword_list:
        print(f"Fetching news for: {keyword}")
        news = fetch_news(keyword, commodity)
        all_news.extend(news)

df = pd.DataFrame(all_news)

# Clean Date
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df.sort_values("Date")

# Save
df.to_csv("sentiment_news_raw.csv", index=False)

print("\nSentiment dataset saved as sentiment_news_raw.csv")
print(df.head())
