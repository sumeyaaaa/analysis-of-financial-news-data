import matplotlib.pyplot as plt
import pandas as pd

def plot_bollinger_bands(df, column='Close', window=20, num_std=2, stock_name='Stock'):
    """
    Plots Bollinger Bands for a given stock DataFrame and column.
    
    Args:
        df (pd.DataFrame): DataFrame containing the stock data.
        column (str): Column to use for Bollinger Bands (default 'Close').
        window (int): Rolling window size (default 20).
        num_std (int): Number of standard deviations for bands (default 2).
        stock_name (str): Name for the plot title/legend.
    """
    # Calculate rolling mean and std
    rolling_mean = df[column].rolling(window=window).mean()
    rolling_std = df[column].rolling(window=window).std()

    # Calculate upper and lower Bollinger Bands
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)

    # Plot
    plt.figure(figsize=(16, 5))
    plt.plot(df['Date'], df[column], label=f'{stock_name} Close', color='black')
    plt.plot(df['Date'], rolling_mean, label='Rolling Mean', color='blue', linewidth=1)
    plt.plot(df['Date'], upper_band, label='Upper Band', color='green', linestyle='--')
    plt.plot(df['Date'], lower_band, label='Lower Band', color='red', linestyle='--')
    plt.fill_between(df['Date'], lower_band, upper_band, color='gray', alpha=0.1)
    plt.title(f'{stock_name} Close Price with Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    plt.show()
