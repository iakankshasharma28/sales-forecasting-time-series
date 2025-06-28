# sales_forecasting_project/src/data_loader.py

import pandas as pd
import numpy as np
import os

def generate_sales_data(path="sales_forecasting_project/data/sales_data.csv"):
    """Generates and saves synthetic monthly sales data."""

    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    date_range = pd.date_range(start="2018-01-01", end="2022-12-01", freq="MS")
    np.random.seed(42)
    sales = np.random.normal(loc=200, scale=20, size=len(date_range)) + np.linspace(0, 50, len(date_range))

    df = pd.DataFrame({'Date': date_range, 'Sales': sales})
    df.to_csv(path, index=False)

    print(f"✅ Sales data saved to {path}")
    return df
