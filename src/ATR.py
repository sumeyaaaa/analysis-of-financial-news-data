import matplotlib.pyplot as plt
import pandas as pd

def calculate_atr(df, window=14):
    """
    Calculates the Average True Range (ATR) for a stock DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with columns 'High', 'Low', and 'Close'.
        window (int): Rolling window size for ATR (default 14).
    
    Returns:
        pd.Series: ATR values.
    """
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def plot_atr(df, window=14, stock_name='Stock'):
    """
    Plots the ATR indicator.
    
    Args:
        df (pd.DataFrame): DataFrame with 'High', 'Low', 'Close', and 'Date' columns.
        window (int): Window for ATR.
        stock_name (str): Name for plot.
    """
    atr = calculate_atr(df, window=window)
    plt.figure(figsize=(16, 4))
    plt.plot(df['Date'], atr, color='purple', label=f'ATR {window}')
    plt.title(f'{stock_name} Average True Range (ATR {window})')
    plt.xlabel('Date')
    plt.ylabel('ATR')
    plt.legend()
    plt.tight_layout()
    plt.show()
