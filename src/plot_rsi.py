import matplotlib.pyplot as plt
import talib
import pandas as pd

def plot_rsi(
    df,
    price_col='Close',
    ticker_name='STOCK',
    rsi_period=14,
    color_line='purple',
    color_overbought='red',
    color_oversold='green'
):
    """
    Plots the RSI for any stock DataFrame and price column.

    Parameters:
        df (pd.DataFrame): DataFrame with at least 'Date' and the price_col.
        price_col (str): Column to use for RSI calculation (e.g. 'Close').
        ticker_name (str): Ticker for plot title.
        rsi_period (int): RSI calculation period.
        color_line (str): Color for the RSI line.
        color_overbought (str): Color for the 70 overbought line.
        color_oversold (str): Color for the 30 oversold line.
    """
    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])

    # Compute RSI if not already
    rsi_col = f'RSI_{rsi_period}'
    if rsi_col not in df.columns:
        df[rsi_col] = talib.RSI(df[price_col], timeperiod=rsi_period)

    plt.figure(figsize=(16, 3))
    plt.plot(df['Date'], df[rsi_col], label=f'RSI {rsi_period}', color=color_line)
    plt.axhline(70, color=color_overbought, linestyle='--', alpha=0.5)
    plt.axhline(30, color=color_oversold, linestyle='--', alpha=0.5)
    plt.title(f"{ticker_name.upper()} Relative Strength Index (RSI)")
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.tight_layout()
    plt.show()
