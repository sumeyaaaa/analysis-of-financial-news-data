import matplotlib.pyplot as plt
import talib
import pandas as pd

def plot_macd(
    df,
    price_col='Close',
    ticker_name='STOCK',
    fastperiod=12,
    slowperiod=26,
    signalperiod=9,
    color_macd='blue',
    color_signal='red',
    color_hist='gray'
):
    """
    Plots the MACD, Signal line, and Histogram for any stock DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with at least 'Date' and the price_col.
        price_col (str): Column for MACD calculation (e.g., 'Close').
        ticker_name (str): Ticker for plot title.
        fastperiod (int): Fast EMA period.
        slowperiod (int): Slow EMA period.
        signalperiod (int): Signal line EMA period.
        color_macd (str): Color for MACD line.
        color_signal (str): Color for Signal line.
        color_hist (str): Color for MACD Histogram.
    """
    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])

    macd_col = f'MACD_{fastperiod}_{slowperiod}_{signalperiod}'
    signal_col = f'MACD_signal_{fastperiod}_{slowperiod}_{signalperiod}'
    hist_col = f'MACD_hist_{fastperiod}_{slowperiod}_{signalperiod}'

    # Only calculate if not present
    if macd_col not in df.columns or signal_col not in df.columns or hist_col not in df.columns:
        macd, signal, hist = talib.MACD(
            df[price_col],
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod
        )
        df[macd_col] = macd
        df[signal_col] = signal
        df[hist_col] = hist

    plt.figure(figsize=(16, 4))
    plt.plot(df['Date'], df[macd_col], label='MACD', color=color_macd)
    plt.plot(df['Date'], df[signal_col], label='Signal Line', color=color_signal)
    plt.bar(df['Date'], df[hist_col], label='MACD Histogram', color=color_hist, alpha=0.4)
    plt.title(f'{ticker_name.upper()} MACD')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.tight_layout()
    plt.show()
