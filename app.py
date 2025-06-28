# app.py

import streamlit as st
import pandas as pd
from src.arima_model import train_arima_model, forecast_arima
from src.prophet_model import train_prophet_model, forecast_prophet

# --------------------------- 🎨 PAGE SETUP ----------------------------------
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------- 🌈 CUSTOM CSS ----------------------------------
st.markdown("""
    <style>
        .main {
            background-color: #1F1D36;
        }
        .card {
            padding: 20px;
            border-radius: 15px;
            background-color: #2D2B55;
            color: #F5F5F5;
            margin-bottom: 10px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.25);
        }
        .card-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .card-value {
            font-size: 24px;
            font-weight: bold;
            color: #7F5AF0;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------- 🧾 SIDEBAR ----------------------------------
st.sidebar.title("📁 Upload Sales Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV with 'Date' and 'Sales' columns", type=["csv"])

forecast_steps = st.sidebar.slider("📅 Forecast months", 6, 36, 12)
model_choice = st.sidebar.radio("🧠 Choose Forecast Model", ["ARIMA", "Prophet"])
st.sidebar.markdown("---")
st.sidebar.markdown("💡 Tip: Use monthly data with 'Date' and 'Sales' columns.")

# --------------------------- 🧠 LOAD DATA ----------------------------------
st.title("🔮Sales Forecasting Using TimeSeries")
st.caption("Upload sales data, select a model, and generate forecasts using ARIMA or Prophet.")

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [col.strip().lower() for col in df.columns]
    df.rename(columns={"date": "Date", "sales": "Sales"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df = df[["Date", "Sales"]]
    df.set_index("Date", inplace=True)
    return df

if uploaded_file:
    df = load_data(uploaded_file)

    st.markdown("## 👋 Welcome!")
    
    # ----------------------- 💳 METRIC CARDS -------------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class="card">
                <div class="card-title">📊 Total Records</div>
                <div class="card-value">{len(df)}</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="card">
                <div class="card-title">📅 Date Range</div>
                <div class="card-value">{df.index.min().date()} → {df.index.max().date()}</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="card">
                <div class="card-title">💰 Last Recorded Sales</div>
                <div class="card-value">{round(df['Sales'].iloc[-1], 2)}</div>
            </div>
        """, unsafe_allow_html=True)

    st.subheader("📈 Sales Data Overview")
    st.line_chart(df)

    # ----------------------- 📊 FORECAST -----------------------------------
    st.markdown("### 🔍 Forecast Results")

    tab1, tab2 = st.tabs(["📉 Forecast Plot", "📎 Forecast Data"])

    if model_choice == "ARIMA":
        model_fit = train_arima_model(df["Sales"], order=(1, 1, 1))
        forecast = forecast_arima(model_fit, steps=forecast_steps)
        forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
        forecast_df = pd.DataFrame({"Forecast": forecast}, index=forecast_index)

    elif model_choice == "Prophet":
        df_prophet = df.reset_index().rename(columns={"Date": "ds", "Sales": "y"})
        model = train_prophet_model(df_prophet)
        forecast_full = forecast_prophet(model, periods=forecast_steps)
        forecast_df = forecast_full[["ds", "yhat"]].tail(forecast_steps).set_index("ds")
        forecast_df.columns = ["Forecast"]

    with tab1:
        st.line_chart(forecast_df)

    with tab2:
        st.dataframe(forecast_df, use_container_width=True)

else:
    st.info("👈 Upload a CSV file with monthly sales data to get started.")


# --------------------------- 🔻 FOOTER ----------------------------------
st.markdown("""
    <hr style="margin-top: 40px; margin-bottom: 10px;">
    <div style='text-align: center; padding: 10px; font-size: 14px; color: #AAAAAA;'>
        © 2025 Akanksha Sharma | 📧 akankshasharma2808@gmail.com
    </div>
""", unsafe_allow_html=True)
