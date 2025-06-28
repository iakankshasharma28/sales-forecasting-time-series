# sales_forecasting_project/src/prophet_model.py

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os

def train_prophet_model(df):
    """Train and return a fitted Prophet model."""
    model = Prophet()
    model.fit(df)
    print("✅ Prophet model trained")
    return model

def forecast_prophet(model, periods=12):
    """Generate future forecast."""
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)
    return forecast

def plot_prophet_forecast(model, forecast, save_path):
    """Save forecast plot."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = model.plot(forecast)
    fig.set_size_inches(12, 6)
    plt.title("🔮 Prophet Sales Forecast", fontsize=16)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close()
    print(f"✅ Prophet forecast plot saved to {save_path}")

def plot_prophet_components(model, forecast, save_path):
    """Save trend and seasonality components."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig2 = model.plot_components(forecast)
    fig2.set_size_inches(12, 8)
    plt.tight_layout()
    fig2.savefig(save_path)
    plt.close()
    print(f"✅ Prophet components plot saved to {save_path}")
