import pandas as pd
from transformers import pipeline

# -----------------------------
# LOAD RAW NEWS DATA
# -----------------------------
df = pd.read_csv(
    r"C:\Users\ASUS\OneDrive\Desktop\Minor\data\raw\sentiment\sentiment_news_raw.csv"
)
# Keep only required columns
df = df[['Date', 'Commodity', 'Text']]

# -----------------------------
# LOAD SENTIMENT MODEL
# -----------------------------
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# -----------------------------
# FUNCTION TO MAP LABEL → SCORE
# -----------------------------
def get_sentiment_score(text):
    try:
        result = sentiment_pipeline(text[:512])[0]  # BERT limit
        score = result['score']
        return score if result['label'] == 'POSITIVE' else -score
    except:
        return 0.0  # safe fallback

# -----------------------------
# APPLY SENTIMENT ANALYSIS
# -----------------------------
print("Converting headlines to sentiment scores...")

df['Sentiment_Score'] = df['Text'].apply(get_sentiment_score)

# -----------------------------
# CLEAN DATE
# -----------------------------
df['Date'] = pd.to_datetime(df['Date']).dt.date

# -----------------------------
# AGGREGATE DAILY SENTIMENT
# -----------------------------
daily_sentiment = (
    df.groupby(['Date', 'Commodity'])['Sentiment_Score']
      .mean()
      .reset_index()
)

# -----------------------------
# SAVE OUTPUT
# -----------------------------
daily_sentiment.to_csv("sentiment_processed.csv", index=False)

print("Sentiment processing completed.")
print(daily_sentiment.head())
