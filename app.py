import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

st.title("📈 Stock Price Predictor (LSTM)")

# --- User Input ---
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, INFY.NS):", "AAPL").strip().upper()

# --- Choose Prediction Length ---
forecast_days = st.slider("📆 How many days to predict?", min_value=1, max_value=15, value=7)

# --- Helper to fetch enough data ---
def fetch_enough_data(ticker, end, required_days=60, max_lookback_days=180):
    lookback = 90
    while lookback <= max_lookback_days:
        start = end - dt.timedelta(days=lookback)
        st.write(f"🔄 Trying {lookback} days back...")
        df = yf.download(ticker, start=start, end=end, progress=False)
        st.write(f"📅 Rows fetched: {df.shape[0]}")
        if df.shape[0] >= required_days:
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        lookback += 30
    return pd.DataFrame()

# --- Prediction ---
if st.button("Predict"):
    end = dt.date.today()
    df = fetch_enough_data(ticker, end)

    if df.empty or df.shape[0] < 60:
        st.error("❌ Not enough data to make prediction. Try a different stock or later date.")
        st.stop()

    st.success(f"✅ Stock data loaded! {df.shape[0]} trading days fetched.")

    # --- Extract Close price ---
    try:
        if isinstance(df.columns, pd.MultiIndex):
            data = pd.DataFrame(df['Close'][ticker])
        else:
            data = pd.DataFrame(df['Close'])
    except KeyError:
        st.error("❌ Could not find 'Close' prices for this ticker.")
        st.stop()

    data = data.dropna()
    st.line_chart(data)

    # --- Prepare input for prediction ---
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    if len(scaled_data) < 60:
        st.error("❌ Not enough clean rows after scaling.")
        st.stop()

    input_seq = scaled_data[-60:]

    try:
        model = load_model('lstm_model.h5')
    except:
        st.error("❌ Model file 'lstm_model.h5' not found.")
        st.stop()

    # --- Generate predictions ---
    predictions = []
    for _ in range(forecast_days):
        X = input_seq.reshape(1, 60, 1)
        pred_scaled = model.predict(X, verbose=0)[0][0]
        pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
        predictions.append(pred_price)
        input_seq = np.append(input_seq[1:], [[pred_scaled]], axis=0)

    # --- Forecast Dates ---
    future_dates = []
    next_day = df.index[-1].date()
    while len(future_dates) < forecast_days:
        next_day += dt.timedelta(days=1)
        if next_day.weekday() < 5:
            future_dates.append(next_day)


    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': predictions})
    forecast_df.set_index('Date', inplace=True)

    # --- Display Output ---
    st.subheader(f"📊 {forecast_days}-Day Predicted Prices from Today")
    st.line_chart(forecast_df)
    st.dataframe(forecast_df.style.format({"Predicted Close": "{:.2f}"}))

    # --- 📥 Download Forecast as CSV ---
    csv = forecast_df.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download forecast as CSV",
        data=csv,
        file_name=f"{ticker}_forecast.csv",
        mime='text/csv',
    )

