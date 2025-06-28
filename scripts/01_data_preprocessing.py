# sales_forecasting_project/scripts/01_data_preprocessing.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.data_loader import generate_sales_data
from src.visualizer import plot_time_series, plot_seasonality_decomposition

# Load or generate data
df = generate_sales_data("sales_forecasting_project/data/sales_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Visualizations
plot_time_series(df, "sales_forecasting_project/reports/figures/sales_trend.png")
plot_seasonality_decomposition(df, "sales_forecasting_project/reports/figures/sales_decomposition.png")
