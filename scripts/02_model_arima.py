# sales_forecasting_project/scripts/02_model_arima.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.arima_model import train_arima_model, forecast_arima, plot_arima_forecast

# Load data
df = pd.read_csv("sales_forecasting_project/data/sales_data.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# Train ARIMA model
model_fit = train_arima_model(df['Sales'], order=(1, 1, 1))

# Forecast next 12 months
forecast = forecast_arima(model_fit, steps=12)

# Plot and save forecast
plot_arima_forecast(df['Sales'], forecast, "sales_forecasting_project/reports/figures/arima_forecast.png")
