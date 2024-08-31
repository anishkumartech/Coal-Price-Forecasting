

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from scipy import stats

# Load the forecast data
forecast_coal_df = pd.read_excel(r"C:\Users\ak427\Downloads\Regression\coal_forecast_et.xlsx", index_col=0)

actual_data = pd.read_excel(r"C:\Users\ak427\Downloads\Regression\numeric_data.xlsx", index_col=0, parse_dates=True)

# Define confidence interval function
def calculate_confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + confidence) / 2., len(data)-1)
    return mean - margin, mean + margin

st.title("Coal Price Forecasting")

# Sidebar for user input
st.sidebar.header("User Input")
confidence_level = st.sidebar.slider("Select Confidence Interval", 0.90, 0.95, value=0.90, step=0.01)
start_date = st.sidebar.date_input("Start Date", value=actual_data.index.min())
end_date = st.sidebar.date_input("End Date", value=actual_data.index.max())
update_button = st.sidebar.button("Update Forecast")

# Select target column for visualization
target_column = st.selectbox("Select target column", options=forecast_coal_df.columns)

if target_column:
    st.subheader(f"Forecast vs Actual for {target_column}")

    if update_button:
        # Filter data based on the selected date range
        train_data = actual_data.loc[start_date:end_date, target_column]
        train_forecast = forecast_coal_df.loc[start_date:end_date, target_column].dropna()
        
        # Align data
        common_index = train_forecast.index.intersection(train_data.index)
        train_data = train_data.loc[common_index]
        train_forecast = train_forecast.loc[common_index]

        test_data = actual_data.loc[start_date:end_date, target_column]
        test_forecast = forecast_coal_df.loc[start_date:end_date, target_column].dropna()

        # Align data
        common_index_test = test_forecast.index.intersection(test_data.index)
        test_data = test_data.loc[common_index_test]
        test_forecast = test_forecast.loc[common_index_test]

        # Calculate confidence intervals
        train_ci = calculate_confidence_interval(train_forecast, confidence=confidence_level)
        test_ci = calculate_confidence_interval(test_forecast, confidence=confidence_level)

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(train_data.index, train_data, label='Actual Train Data')
        plt.plot(test_data.index, test_data, label='Actual Test Data')
        plt.plot(train_forecast.index, train_forecast, label='Forecast Train Data')
        plt.plot(test_forecast.index, test_forecast, label='Forecast Test Data')
        plt.fill_between(train_forecast.index, train_ci[0], train_ci[1], color='green', alpha=0.2, label=f'Train {int(confidence_level*100)}% CI')
        plt.fill_between(test_forecast.index, test_ci[0], test_ci[1], color='Blue', alpha=0.2, label=f'Test {int(confidence_level*100)}% CI')
        plt.title(f'Forecast vs Actual with {int(confidence_level*100)}% Confidence Intervals for {target_column}')
        plt.legend()
        st.pyplot(plt)

        # Show MAPE
        mape_train = mean_absolute_percentage_error(train_data, train_forecast)
        mape_test = mean_absolute_percentage_error(test_data, test_forecast)
        st.write(f"MAPE for Training Data: {mape_train:.2f}")
        st.write(f"MAPE for Testing Data: {mape_test:.2f}")
    else:
        st.info("Click the 'Update Forecast' button to generate the forecast with the selected date range and confidence interval.")

