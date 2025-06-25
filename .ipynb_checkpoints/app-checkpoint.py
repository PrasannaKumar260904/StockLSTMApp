import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model


st.title("📈 7-Day Stock Price Predictor (LSTM)")

ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, INFY.NS):", "AAPL")

if st.button("Predict"):
    end = dt.date.today()
start = end - dt.timedelta(days=90)  # Fetch past 90 days to get at least 60 trading days
df = yf.download(ticker, start=start, end=end)

if df.empty or df.shape[0] < 60:
    st.error("Not enough data to make prediction. Try a different stock or later date.")
else:
    st.success("Stock data loaded successfully!")
    data = df[['Close']]
    st.line_chart(data)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    input_seq = scaled_data[-60:]  # Last 60 trading days

    model = load_model('lstm_model.h5')

    predictions = []
    for i in range(7):
        X = input_seq.reshape(1, 60, 1)
        pred_scaled = model.predict(X, verbose=0)[0][0]
        pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
        predictions.append(pred_price)
        input_seq = np.append(input_seq[1:], [[pred_scaled]], axis=0)

    # Generate next 7 **calendar days**, skipping weekends if needed
    future_dates = []
    next_day = df.index[-1].date()
    while len(future_dates) < 7:
        next_day += dt.timedelta(days=1)
        if next_day.weekday() < 5:  # Only weekdays (Mon-Fri)
            future_dates.append(next_day)

    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': predictions})
    forecast_df.set_index('Date', inplace=True)

    st.subheader("📊 7-Day Predicted Prices from Today")
    st.line_chart(forecast_df)
    st.dataframe(forecast_df.style.format({"Predicted Close": "{:.2f}"}))
