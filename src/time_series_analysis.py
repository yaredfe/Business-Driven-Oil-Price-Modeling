# Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns

# Load and preprocess data
def load_data(file_path):
    """Loads the Brent oil price data from a CSV file."""
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return df

# Step 1: Initial Visualization of the Data
def plot_initial_data(df):
    """Plots the initial time series data for visualization."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['Price'], color='blue', label='Brent Oil Price')
    plt.title("Brent Oil Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

# Step 2: Check for Stationarity using the ADF Test
def adf_test(timeseries):
    """Performs the Augmented Dickey-Fuller test and prints the results."""
    print("Results of Augmented Dickey-Fuller Test:")
    adf_result = adfuller(timeseries)
    labels = ['ADF Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    results = pd.Series(adf_result[0:4], index=labels)
    for key, value in adf_result[4].items():
        results[f'Critical Value ({key})'] = value
    print(results)

# Step 3: Apply Differencing if Necessary
def make_stationary(df):
    """Applies differencing to the time series if it's not stationary."""
    df['Price_diff'] = df['Price'].diff().dropna()
    adf_test(df['Price_diff'].dropna())  # Check stationarity of the differenced series
    return df

# Plot Differenced Series
def plot_differenced_data(df):
    """Plots the differenced series to visualize stationarity."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['Price_diff'], color='purple', label='Differenced Brent Oil Price')
    plt.title("Differenced Brent Oil Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Differenced Price")
    plt.legend()
    plt.show()

# Step 4: Decompose the Time Series into Components
def decompose_series(df, period=365):
    """Decomposes the time series into trend, seasonal, and residual components."""
    decomposition = seasonal_decompose(df['Price'], model='additive', period=period)
    decomposition.plot()
    plt.show()

# Step 5: Plot Autocorrelation and Partial Autocorrelation
def plot_acf_pacf(df):
    """Plots the ACF and PACF of the differenced time series."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sm.graphics.tsa.plot_acf(df['Price_diff'].dropna(), ax=axes[0], title="ACF of Differenced Series")
    sm.graphics.tsa.plot_pacf(df['Price_diff'].dropna(), ax=axes[1], title="PACF of Differenced Series")
    plt.show()