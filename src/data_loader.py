import pandas as pd
from pathlib import Path
from dateutil.parser import parse



def load_news_data(path):
    df = pd.read_csv(path)
    # First try fast, vectorized pandas parsing
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    # Find rows that failed parsing
    missing_mask = df['date_parsed'].isna()
    print(f"✅ Pandas valid dates: {(~missing_mask).sum()} / {len(df)}")
    
    # Try dateutil on the missing ones (slower, but robust)
    def try_dateutil(x):
        try:
            return parse(str(x), fuzzy=True)
        except Exception:
            return pd.NaT

    df.loc[missing_mask, 'date_parsed'] = df.loc[missing_mask, 'date'].apply(try_dateutil)
    # Count again after dateutil
    valid_final = df['date_parsed'].notna().sum()
    print(f"✅ Total valid after dateutil: {valid_final} / {len(df)}")
    
    # Standardize format: YYYY-MM-DD HH:MM:SS
    df['date'] = pd.to_datetime(df['date_parsed'], errors='coerce', utc=True).dt.strftime('%Y-%m-%d %H:%M:%S')
    df.drop(columns=['date_parsed'], inplace=True)
    return df

def load_stock_data(path):
    df = pd.read_csv(path)
    # First try fast, vectorized pandas parsing
    df['Date_parsed'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
    # Find rows that failed parsing
    missing_mask = df['Date_parsed'].isna()
    print(f"✅ Pandas valid dates: {(~missing_mask).sum()} / {len(df)}")
    
    # Try dateutil on the missing ones (slower, but robust)
    def try_dateutil(x):
        try:
            return parse(str(x), fuzzy=True)
        except Exception:
            return pd.NaT

    df.loc[missing_mask, 'Date_parsed'] = df.loc[missing_mask, 'Date'].apply(try_dateutil)
    # Count again after dateutil
    valid_final = df['Date_parsed'].notna().sum()
    print(f"✅ Total valid after dateutil: {valid_final} / {len(df)}")
    
    # Standardize format: YYYY-MM-DD HH:MM:SS
    df['Date'] = pd.to_datetime(df['Date_parsed'], errors='coerce', utc=True).dt.strftime('%Y-%m-%d %H:%M:%S')
    df.drop(columns=['Date_parsed'], inplace=True)
    return df
