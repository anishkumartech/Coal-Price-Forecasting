# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from scipy import stats

# Load the forecast data
forecast_train_df = pd.read_excel(r"C:\Users\ak427\Downloads\Regression\target_forecast_train_et.xlsx", index_col=0)
forecast_test_df = pd.read_excel(r"C:\Users\ak427\Downloads\Regression\target_forecast_test_et.xlsx", index_col=0)
actual_data = pd.read_excel(r"C:\Users\ak427\Downloads\Regression\numeric_data.xlsx", index_col=0, parse_dates=True)

# Define confidence interval function
def calculate_confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + confidence) / 2., len(data)-1)
    return mean - margin, mean + margin

st.title("Coal Price Forecasting")

# Select target column for visualization
target_column = st.selectbox("Select target column", options=forecast_train_df.columns)

if target_column:
    st.subheader(f"Forecast vs Actual for {target_column}")

    # Get the range of available dates
    min_date = actual_data.index.min()
    max_date = actual_data.index.max()

    # Input date range with dynamic default values
    start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("Start date must be before end date.")
    else:
        if st.button("Show Forecast"):
            # Filter data based on date range
            filtered_data = actual_data.loc[start_date:end_date, target_column]
            filtered_train_forecast = forecast_train_df.loc[start_date:end_date, target_column].dropna()
            filtered_test_forecast = forecast_test_df.loc[start_date:end_date, target_column].dropna()

            if filtered_data.empty or filtered_train_forecast.empty or filtered_test_forecast.empty:
                st.warning("No data available for the selected date range.")
            else:
                # Calculate confidence intervals
                confidence_level = st.selectbox("Select Confidence Interval", [0.90, 0.95])
                train_ci = calculate_confidence_interval(filtered_train_forecast, confidence=confidence_level)
                test_ci = calculate_confidence_interval(filtered_test_forecast, confidence=confidence_level)

                # Plotting
                plt.figure(figsize=(12, 6))
                plt.plot(filtered_data.index, filtered_data, label='Actual Data')
                plt.plot(filtered_train_forecast.index, filtered_train_forecast, label='Forecast Data')
                plt.fill_between(filtered_train_forecast.index, train_ci[0], train_ci[1], color='gray', alpha=0.2, label=f'{int(confidence_level*100)}% CI')
                plt.title(f'Forecast vs Actual with {int(confidence_level*100)}% Confidence Intervals for {target_column}')
                plt.legend()
                st.pyplot(plt)

                # Show MAPE
                mape = mean_absolute_percentage_error(filtered_data, filtered_train_forecast)
                st.write(f"MAPE: {mape:.2f}")
