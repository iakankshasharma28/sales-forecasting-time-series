# sales_forecasting_project/src/arima_model.py

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import os

def train_arima_model(series, order=(1, 1, 1)):
    """Train ARIMA model and return fitted model."""
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    print("✅ ARIMA model trained")
    return model_fit

def forecast_arima(model_fit, steps=12):
    """Forecast future values."""
    forecast = model_fit.forecast(steps=steps)
    return forecast

def plot_arima_forecast(series, forecast, save_path):
    """Plot ARIMA forecast."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    forecast_index = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=len(forecast), freq='MS')

    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series, label='Historical', linewidth=2.5)
    plt.plot(forecast_index, forecast, label='ARIMA Forecast', color='red', linewidth=2.5)
    plt.title('📉 ARIMA Sales Forecast (Next 12 Months)', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ ARIMA forecast plot saved to {save_path}")
