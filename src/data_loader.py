import pandas as pd
from pathlib import Path

import pandas as pd

def load_news_data(path=r'C:\Users\ABC\Desktop\10Acadamy\week1\analysis-of-financial-news-data\data\raw_analyst_ratings.csv\raw_analyst_ratings.csv'):
    df = pd.read_csv(path)

    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df['date'] = df['date'].dt.strftime('%Y-%m-%d')


    print(f"✅ Total: {len(df)} | Valid dates: {df['date'].notna().sum()}")

    return df



def load_stock_data(symbol, folder='data/yfinance_data/'):
    path = Path(folder) / f"{symbol}.csv"
    df = pd.read_csv(path, parse_dates=['date'])
    df['symbol'] = symbol
    df['date'] = pd.to_datetime(df['date'])
    return df
