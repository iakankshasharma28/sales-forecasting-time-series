# sales_forecasting_project/scripts/03_model_prophet.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.prophet_model import train_prophet_model, forecast_prophet, plot_prophet_forecast, plot_prophet_components

# Load and prepare data
df = pd.read_csv("sales_forecasting_project/data/sales_data.csv")
df.rename(columns={"Date": "ds", "Sales": "y"}, inplace=True)
df['ds'] = pd.to_datetime(df['ds'])

# Train & forecast
model = train_prophet_model(df)
forecast = forecast_prophet(model, periods=12)

# Plot forecast
plot_prophet_forecast(model, forecast, "sales_forecasting_project/reports/figures/prophet_forecast.png")
plot_prophet_components(model, forecast, "sales_forecasting_project/reports/figures/prophet_components.png")
