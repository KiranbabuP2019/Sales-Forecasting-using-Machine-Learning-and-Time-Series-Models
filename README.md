# 🛍️ Sales Forecasting Project – Wiseanalytics Internship Assignment

## 📦 Overview
This project predicts daily product sales across stores in Ecuador using historical data, holidays, promotions, oil prices, and more.

## 📁 Contents
- `Sales_Forecasting_Project.ipynb` – Full notebook with code, visualizations, and insights.
- `data/` – Folder containing CSVs (train, test, oil, holidays, stores, transactions).
- `models/` – (Optional) Saved model files (Random Forest, XGBoost, etc.)
- `plots/` – (Optional) Forecast and feature importance visualizations.

## 🚀 How to Run
1. Install dependencies:  
   `pip install -r requirements.txt`

2. Open the notebook:  
   `jupyter notebook Sales_Forecasting_Project.ipynb`

3. Run all cells to train and evaluate models.

## 🧠 Summary of Results
- **Best Model:** XGBoost
- **Key Features:** Promotions, holidays, recent sales trends
- **Business Insight:** Use forecasts for inventory planning and event-based promotions.

## 📊 Evaluation Metrics (RMSE)
| Model         | RMSE   |
|---------------|--------|
| Naïve         | 439.62 |
| ARIMA         | 12190.0|
| Random Forest | 706.61 |
| XGBoost       | 692.22 |
| LSTM          | 939.41 |
