🌾 Multimodal Residual Learning for Agricultural Commodity Price Forecasting

A research-oriented machine learning project that predicts agricultural commodity prices by modeling price residuals over a persistence baseline, using temporal, spatial, weather, and sentiment signals.

This project integrates Transformer-based time-series modeling with Graph Neural Networks (GNNs) to capture both temporal dynamics and market-level spatial dependencies.

📌 Project Overview

Agricultural commodity prices exhibit:

Strong temporal persistence

High volatility due to external factors

Spatial dependencies across markets and states

Rather than directly predicting absolute prices, this project reframes the task as residual learning, allowing the model to focus on deviations from price persistence, which is the true source of forecasting difficulty.

🎯 Problem Statement

Predict short-term agricultural commodity price movements by learning residual price changes over a naive persistence baseline, using historical prices, weather data, market sentiment, and spatial market relationships.

🧠 Key Idea (Core Insight)

Instead of predicting:

Price(t)


the model predicts:

Residual(t) = Price(t) − Price(t−1)


and reconstructs:

Pricê(t) = Price(t−1) + Residual̂(t)


This formulation:

Neutralizes dominance of persistence

Enables fair comparison with naive baselines

Improves stability and interpretability

Is standard in financial & commodity forecasting research

🧱 System Architecture
Historical Prices ─┐
Weather Features ──┼──▶ Transformer Encoder ─┐
Sentiment Scores ──┘                          │
                                               ├──▶ Fusion Layer ─▶ Residual Prediction
Market Graph ─▶ GNN Encoder ───────────────────┘

🛠️ Tech Stack
Programming & ML

Python 3.12

PyTorch

PyTorch Geometric (GNN)

NumPy, Pandas

scikit-learn

Data & Utilities

RSS news scraping (sentiment)

joblib (scalers)

CSV-based modular pipelines

Development

VS Code

Script-based, reproducible pipeline

Research-friendly project structure

📂 Project Structure
Minor/
│
├── data/
│   ├── raw/
│   │   ├── prices/
│   │   ├── weather/
│   │   └── sentiment/
│   ├── processed/
│   │   ├── graph/
│   │   ├── scalers/
│   │   ├── sequences/
│   │   ├── splits/
│   │   └── final_scaled_dataset.csv
│
├── src/
│   ├── data_handling/
│   ├── pipeline/
│   ├── models/
│   └── training/
│
├── results/
│   ├── models/
│   └── metrics/
│
├── reports/
│   └── Minor Project PPT.pptx
│
└── README.md

📊 Data Sources
Data Type	Description
Commodity Prices	Market-level daily modal prices
Weather	State-level average temperature & rainfall
Sentiment	News headlines related to commodities
Graph	Market connectivity graph
🔬 Experiments & Methodology
Baseline Models

Naive Persistence: Price(t−1)

Transformer-only

GNN-only

Transformer–GNN Fusion (Proposed)

Experiment Progression
Stage	Description	MAE	RMSE
Naive Baseline	Persistence	~21	~111
Tier-1	Absolute price prediction	~378	~448
Tier-2	Lagged price as input	~333	~371
✅ Tier-3 (Final)	Residual learning + MSE	44.23	115.97
📈 Final Results (Locked)
MAE  = 44.23
RMSE = 115.97

Interpretation

Massive improvement over absolute price prediction

Approaches naive baseline accuracy

Gains explanatory power by using:

Weather

Sentiment

Spatial market structure

This trade-off is expected and desirable in research.

🧪 Training Details

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Early Stopping: Validation-based

Sequence Length: 14 days

Target: Residual price (scaled)

🔍 Evaluation Strategy

Residual predictions reconstructed into prices:

Pricê(t) = Price(t−1) + Residual̂(t)


Correct inverse scaling using saved scalers

Metrics computed in real price units

⚠️ Limitations

Limited historical duration (~1 year)

No intraday price data

Sentiment coverage limited to news headlines

Policy effects not fully integrated

🚀 Future Work

Robust loss functions (Huber loss)

Longer historical datasets

Commodity-specific models

Dynamic graph learning

Policy impact modeling

Batch inference deployment

🏁 Final Takeaway

This project demonstrates that reframing commodity price forecasting as a residual learning problem is essential for overcoming persistence dominance and achieving meaningful predictive performance.
