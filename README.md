Review
# 🌾 Agricultural Commodity Price Forecasting
### A Hybrid Transformer + GNN Deep Learning System for Indian Agri-Markets
---
## 📌 Overview
This project forecasts agricultural commodity prices (Rice, Wheat, Maize) across Indian state-level markets by fusing **temporal patterns** with **spatial market relationships**. It combines:
- **Transformer Encoder** — captures 14-day temporal sequences of weather and market sentiment
- **Graph Neural Network (GNN) Encoder** — models spatial relationships between markets within the same state
- **Fusion Model** — concatenates both representations and regresses the next-day modal price
The model was developed as a Minor Project and is accompanied by a full IEEE-format paper, architecture document, and functional specification.
---
## 🏗️ Architecture
```
Raw Data Sources
    ├── Commodity Prices (Agmarknet / WRIS CSVs)
    ├── State-level Weather (temperature, rainfall)
    ├── Sentiment (Google News RSS — 12 keyword feeds)
    └── Policy Events (GDELT API — MSP / export ban news)
          │
          ▼
    Data Cleaning & Merging
          │
          ▼
    Feature Scaling (StandardScaler)
          │
          ▼
    ┌──────────────────────┐   ┌─────────────────────────┐
    │  Temporal Sequences  │   │    Market Graph           │
    │  (14-day lookback)   │   │  (adjacency by state)     │
    │  [Temp, Rain, Sent]  │   │  node feat = avg price    │
    └────────┬─────────────┘   └───────────┬──────────────┘
             │                             │
             ▼                             ▼
    TransformerEncoder (d=64)     GNNEncoder (2-layer, h=64)
             │                             │
             └──────────── cat ────────────┘
                               │
                         Regressor MLP
                       (128 → 64 → 1)
                               │
                    Predicted Modal Price
```
---
## 📁 Project Structure
```
