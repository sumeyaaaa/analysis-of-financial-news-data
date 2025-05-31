import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import talib

def plot_ma(
    df, 
    price_col, 
    ticker_name='STOCK', 
    sma_period=20, 
    ema_period=20, 
    color_main='yellow', 
    color_sma='blue', 
    color_ema='red'
):
    """
    Plots the price, SMA, and EMA for any stock DataFrame with actual dates on the x-axis.
    """
    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])

    # Compute SMA and EMA
    df[f'SMA_{sma_period}'] = talib.SMA(df[price_col], timeperiod=sma_period)
    df[f'EMA_{ema_period}'] = talib.EMA(df[price_col], timeperiod=ema_period)

    plt.figure(figsize=(16, 6))
    plt.plot(df['Date'], df[price_col], label=f'{price_col} Price', color=color_main, linewidth=1)
    plt.plot(df['Date'], df[f'SMA_{sma_period}'], label=f'SMA {sma_period}', color=color_sma)
    plt.plot(df['Date'], df[f'EMA_{ema_period}'], label=f'EMA {ema_period}', color=color_ema)
    
    # Title and labels
    title = f"{ticker_name.upper()} {price_col.title()} Price with SMA and EMA"
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    
    # Use auto date locator and formatter for smart date ticks
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
