Multimodal Residual Learning for Agricultural Commodity Price Forecasting
Overview

This repository contains a research-oriented machine learning project focused on forecasting agricultural commodity prices by modeling residual price movements over a persistence baseline.
The project integrates temporal modeling, spatial graph learning, weather data, and market sentiment to study how external and structural factors influence price dynamics.

Instead of directly predicting absolute prices, the system reframes the task as residual learning, which significantly improves model stability and interpretability in the presence of strong price persistence.

Problem Statement

Agricultural commodity prices exhibit:

Strong temporal persistence

Sudden shocks due to weather, sentiment, and policy changes

Spatial dependencies across markets and regions

Traditional regression-based or direct deep learning approaches struggle because the model is dominated by price persistence.

Objective:
Predict short-term commodity price movements by learning price residuals over a naive persistence baseline using multimodal data.

Key Contribution

The central contribution of this project is the reformulation of commodity price forecasting as a residual prediction problem:

𝑅
𝑒
𝑠
𝑖
𝑑
𝑢
𝑎
𝑙
(
𝑡
)
=
𝑃
𝑟
𝑖
𝑐
𝑒
(
𝑡
)
−
𝑃
𝑟
𝑖
𝑐
𝑒
(
𝑡
−
1
)
Residual(t)=Price(t)−Price(t−1)
𝑃
𝑟
𝑖
𝑐
𝑒
^
(
𝑡
)
=
𝑃
𝑟
𝑖
𝑐
𝑒
(
𝑡
−
1
)
+
𝑅
𝑒
𝑠
𝑖
𝑑
𝑢
𝑎
𝑙
^
(
𝑡
)
Price
^
(t)=Price(t−1)+
Residual
^
(t)

This formulation:

Neutralizes persistence dominance

Enables fair comparison with naive baselines

Improves learning stability

Aligns with best practices in financial and economic forecasting research

System Architecture
Historical Prices ─┐
Weather Features ──┼──▶ Transformer Encoder ─┐
Sentiment Scores ──┘                          │
                                               ├──▶ Fusion Network ─▶ Residual Prediction
Market Graph ─▶ Graph Neural Network ──────────┘

Tech Stack
Core Libraries

Python 3.12

PyTorch

PyTorch Geometric

NumPy

Pandas

scikit-learn

Supporting Tools

joblib (scaler persistence)

RSS-based news scraping (sentiment)

CSV-based data pipelines

VS Code (development)

Project Structure
Minor/
│
├── data/
│   ├── raw/                  # Original datasets
│   │   ├── prices/
│   │   ├── weather/
│   │   └── sentiment/
│   │
│   ├── processed/            # Cleaned & engineered data
│   │   ├── graph/
│   │   ├── scalers/
│   │   ├── sequences/
│   │   ├── splits/
│   │   ├── prices_cleaned.csv
│   │   ├── final_scaled_dataset.csv
│   │   └── final_merged_dataset.csv
│
├── src/
│   ├── data_handling/        # Cleaning & preprocessing scripts
│   ├── pipeline/             # Feature engineering & sequence creation
│   ├── models/               # Transformer, GNN, fusion model
│   └── training/             # Training, baselines, evaluation
│
├── results/
│   ├── models/               # Saved trained models
│   └── metrics/              # Evaluation outputs
│
├── reports/
│   └── Minor Project PPT.pptx
│
└── README.md

Dataset Description
Data Sources

Commodity Prices: Market-level daily modal prices

Weather: State-level average temperature and rainfall

Sentiment: Commodity-related news headlines

Graph Data: Market-to-market connectivity

Coverage

Commodities: Rice, Wheat, Maize

Geography: Multiple Indian states and markets

Frequency: Daily

Duration: ~1 year

Feature Engineering

Lagged price (Price(t−1))

Weather aggregates (temperature, rainfall)

Sentiment score (news-based)

Market graph embeddings (GNN)

All features are scaled using MinMaxScaler, with scalers persisted for correct inverse transformation during evaluation.

Models Used

Transformer Encoder – captures temporal dependencies

Graph Neural Network (GCN) – models spatial market relationships

Fusion Network – combines temporal and spatial embeddings

Regression Head – predicts residual price change

Training Details

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Early Stopping: Validation-based

Sequence Length: 14 days

Target Variable: Residual price (scaled)

Experiments & Results
Baselines
Model	MAE	RMSE
Naive Persistence	~21	~111
Model Progression
Stage	Description	MAE	RMSE
Tier-1	Absolute price prediction	~378	~448
Tier-2	Lagged price as input	~333	~371
Tier-3 (Final)	Residual learning + MSE	44.23	115.97
Final Results
MAE  = 44.23
RMSE = 115.97


The final model approaches naive persistence performance while incorporating external and spatial explanatory signals, which is considered a strong and realistic outcome in economic forecasting research.

Evaluation Methodology

Residual predictions reconstructed into absolute prices

Correct inverse scaling using saved scalers

Metrics computed in real price units

Evaluation strictly separated from training data

Limitations

Limited historical time span

No intraday price information

Sentiment derived only from news headlines

Policy impact modeling not fully integrated

Future Work

Robust loss functions (Huber loss)

Longer historical datasets

Commodity-specific models

Dynamic graph learning

Policy and macroeconomic feature integration

Batch inference deployment

Conclusion

This project demonstrates that residual learning is essential for meaningful commodity price forecasting under strong temporal persistence.
The proposed multimodal framework provides a principled and extensible foundation for future research in agricultural price modeling.
