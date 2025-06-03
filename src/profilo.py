import yfinance as yf
import pandas as pd

def fetch_data(tickers, start_date, end_date):
    """
    Fetches daily adjusted closing prices for the given tickers and date range.
    Returns a DataFrame with dates as index and tickers as columns.
    """
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    data = data.dropna()
    return data



def calculate_monthly_returns(price_data):
    """
    Resamples daily price data to monthly frequency and computes monthly returns.
    """
    # Get last price of each month
    monthly_prices = price_data.resample('M').last()
    # Compute percentage change month over month
    monthly_returns = monthly_prices.pct_change().dropna()
    return monthly_returns



import numpy as np

def compute_portfolio_metrics(price_data, risk_free_rate=0.02):
    """
    Assumes equal initial investment in each stock. Computes portfolio CAGR, volatility, and Sharpe.
    Returns (annual_return, annual_volatility, sharpe_ratio).
    """
    # Initialize equal investment per stock
    initial_prices = price_data.iloc[0]
    n_stocks = price_data.shape[1]
    allocation = 1.0  # invest $1 in each stock
    shares = allocation / initial_prices

    # Portfolio daily value = sum of each share times its price
    daily_values = (price_data * shares).sum(axis=1)
    # Compute daily returns
    daily_returns = daily_values.pct_change().dropna()

    # Time in years for CAGR
    years = (daily_values.index[-1] - daily_values.index[0]).days / 365.25
    # Compute CAGR
    annual_return = (daily_values.iloc[-1] / daily_values.iloc[0])**(1/years) - 1

    # Annualized volatility (std of daily returns * sqrt(252))
    annual_vol = daily_returns.std() * np.sqrt(252)

    # Sharpe ratio: (annual return - Rf) / annual volatility
    sharpe_ratio = (annual_return - risk_free_rate) / annual_vol

    return annual_return, annual_vol, sharpe_ratio



import matplotlib.pyplot as plt

def plot_portfolio_and_components(price_data):
    """
    Plots the total portfolio value and normalized performance of each stock.
    """
    # Compute portfolio value (assuming $1 per stock as before)
    initial_prices = price_data.iloc[0]
    shares = 1.0 / initial_prices
    portfolio_values = (price_data * shares).sum(axis=1)

    # Plot total portfolio value
    plt.figure(figsize=(8,5))
    plt.plot(portfolio_values.index, portfolio_values, label='Portfolio Value')
    plt.title('Total Portfolio Value (2019–2024)')
    plt.xlabel('Date'); plt.ylabel('Value ($)')
    plt.legend(); plt.grid(True)
    plt.show()

    # Normalize each stock to 1 at start for comparison
    norm_prices = price_data / initial_prices
    plt.figure(figsize=(8,5))
    for ticker in norm_prices.columns:
        plt.plot(norm_prices.index, norm_prices[ticker], label=ticker)
    plt.title('Normalized Stock Performance (Start=1)')
    plt.xlabel('Date'); plt.ylabel('Normalized Price')
    plt.legend(); plt.grid(True)
    plt.show()

