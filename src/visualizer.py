# sales_forecasting_project/src/visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import os

sns.set_theme(style="whitegrid")

def plot_time_series(df, save_path):
    """Line plot of sales data."""

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df.index, y=df['Sales'], color="#1f77b4", linewidth=2.5)
    plt.title("📈 Monthly Sales Trend (2018–2022)", fontsize=16, weight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Sales", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Time series plot saved to {save_path}")

def plot_seasonality_decomposition(df, save_path):
    """Seasonality and trend decomposition plot."""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    result = seasonal_decompose(df['Sales'], model='additive', period=12)
    fig = result.plot()
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Decomposition plot saved to {save_path}")
