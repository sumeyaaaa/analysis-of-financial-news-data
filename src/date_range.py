import pandas as pd
def print_selected_stock_date_ranges(df, stock_col='stock', date_col='date', tickers=None):
    """
    Prints the min and max dates for each specified ticker in a DataFrame with many stocks.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing stock news or prices.
        stock_col (str): The column with the stock ticker symbols.
        date_col (str): The column with the dates.
        tickers (list): List of ticker symbols to include.
    """
    if tickers is None:
        tickers = df[stock_col].unique()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    for ticker in tickers:
        subset = df[df[stock_col] == ticker]
        if not subset.empty:
            min_date = subset[date_col].min()
            max_date = subset[date_col].max()
            print(f"{ticker}: {min_date.date()} to {max_date.date()}")
        else:
            print(f"{ticker}: No data found.")

def print_all_date_ranges(dfs, date_col='Date'):
    """
    Prints the min and max dates for each ticker DataFrame in the given dict.
    
    Parameters:
        dfs (dict): Dictionary like {'AAPL': df_aapl, ...}
        date_col (str): The date column name (default: 'Date')
    """
    for ticker, df in dfs.items():
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        print(f"{ticker}: {min_date.date()} to {max_date.date()}")