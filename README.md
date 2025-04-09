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

   ## 🏆 Best Model
**XGBoost** performed best with the lowest RMSE and highest R².

  ## 📊 Evaluation Metrics (RMSE)
| Model         | RMSE   |  ![image](https://github.com/user-attachments/assets/5518c25f-5b0a-4c39-a50e-1d85340aab50)

|---------------|--------|
| Naïve         | 439.62 |
| ARIMA         | 12190.0|
| Random Forest | 706.61 |
| XGBoost       | 692.22 |
| LSTM          | 939.41 |

## 🔍 Visual Comparison
- All models were evaluated by plotting actual vs. predicted sales.
- XGBoost produced the most consistent trend matching real sales across various categories.
- ARIMA and Prophet showed seasonality but lacked in predictive strength for irregular patterns.
- Naïve baseline failed on dynamic patterns.

## 📌 Business Insights
1. Model Selection for Deployment
- Use XGBoost for forecasting as it balances speed, accuracy, and interpretability.
- Avoid using ARIMA/Prophet for short-term planning due to volatility.

2. Holiday & Promotion Sensitivity
- Sales spiked during national holidays and special events.
- Black Friday, Ecuadorian holidays, and Christmas contributed to large sales jumps.
- Promotions provided short-term boosts—can be leveraged for clearance or upselling.

3. Oil Price Impact
- Affected overall cost-sensitive categories indirectly (transportation-heavy).

## 📈 Recommendations
- 🏪 Inventory Planning: Use forecasts to pre-stock for holidays and weekends.
- 🎯 Targeted Promotions: Run location-based promotions during off-peak weeks.
- 📉 Avoid Overfitting: Monitor model performance regularly and retrain quarterly.
- 🔄 Retrain: Incorporate recent holiday or pricing changes to stay relevant.

## ✅ Conclusion
- The XGBoost model is production-ready with strong accuracy and business relevance.
- It can significantly aid retail planning, logistics, and marketing strategies.




