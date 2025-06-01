import yfinance as yf
import pandas as pd
from tqdm import tqdm

import yfinance as yf
import pandas as pd
import yfinance as yf

# Suppose news_df is already loaded and cleaned as you posted
# Download prices for all tickers in your news_df

def fetch_yf_data_for_stocks(news_df):
    all_price_data = []
    failed_tickers = []
    tickers = news_df['stock'].unique()
    for ticker in tickers:
        dates = pd.to_datetime(news_df.loc[news_df['stock'] == ticker, 'date'])
        start = dates.min().strftime('%Y-%m-%d')
        end = (dates.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            print(f"No data for {ticker}")
            failed_tickers.append(ticker)
            continue
        df = df.reset_index()
        df['stock'] = ticker
        all_price_data.append(df)
    if all_price_data:
        df_all = pd.concat(all_price_data, ignore_index=True)
        df_all = df_all.rename(columns={'Date': 'date'})
        df_all['date'] = pd.to_datetime(df_all['date']).dt.strftime('%Y-%m-%d')
        return df_all, failed_tickers
    else:
        return pd.DataFrame(), failed_tickers


