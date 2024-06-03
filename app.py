import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm

# Function to load and preprocess data
@st.cache
def load_data():
    data = pd.read_csv('dataset.csv', index_col='Date', parse_dates=True)
    return data

# Function to fit ARIMA model using auto_arima
def fit_auto_arima(data):
    model = pm.auto_arima(data, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    return model

# Main Streamlit app
def main():
    st.title("Automatic ARIMA Predictions")

    # Load data
    data = load_data()
    st.write("### Dataset")
    st.write(data.head())

    # Fit the model using auto_arima
    st.write("Fitting ARIMA model...")
    model = fit_auto_arima(data['value_column'])
    st.write("### ARIMA Model Summary")
    st.text(model.summary())

    # Get model parameters
    p, d, q = model.order
    st.write("### Model Parameters")
    st.write(f"p: {p}, d: {d}, q: {q}")

    # Predictions
    n_periods = 10  # Number of periods for prediction
    forecast = model.predict(n_periods=n_periods)
    forecast_index = pd.date_range(start=data.index[-1], periods=n_periods + 1, freq='D')[1:]  # Adjust frequency as needed
    forecast_series = pd.Series(forecast, index=forecast_index)
    st.write("### Forecasted Values")
    forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast})
    st.write(forecast_df)

    # Metrics
    actuals = data['value_column'][-n_periods:]
    mse = mean_squared_error(actuals, forecast[:len(actuals)])
    mape = mean_absolute_percentage_error(actuals, forecast[:len(actuals)])
    st.write("### Model Metrics")
    st.write(f"MSE: {mse}")
    st.write(f"MAPE: {mape}")

    # Actual vs Predicted Plot
    st.write("### Actual vs Predicted Values")
    plt.figure(figsize=(10, 6))
    plt.plot(data['value_column'], label='Actual')
    plt.plot(forecast_series, label='Forecast', linestyle='--')
    plt.legend()
    st.pyplot(plt)

    # P-value from Ljung-Box test
    ljung_box_result = acorr_ljungbox(model.resid(), lags=[10], return_df=True)
    p_value = ljung_box_result['lb_pvalue'].iloc[-1]
    st.write("### Ljung-Box Test")
    st.write(f"P-value: {p_value}")

if __name__ == "__main__":
    main()
