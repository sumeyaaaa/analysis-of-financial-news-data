import pandas as pd

def find_missing_dates(
    df, 
    date_col='Date', 
    start_date=None, 
    end_date=None, 
    freq='D'
):
    """
    Returns a list of missing dates within a specified range and frequency.

    Parameters:
        df (pd.DataFrame): DataFrame with the date column.
        date_col (str): Name of the date column.
        start_date (str or Timestamp): Start date for the range. If None, uses min date in df.
        end_date (str or Timestamp): End date for the range. If None, uses max date in df.
        freq (str): Frequency string ('D' for daily, 'B' for business day, etc.)

    Returns:
        pd.DatetimeIndex: Missing dates within the range at the specified frequency.
    """
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # Set default start/end
    if start_date is None:
        start_date = df[date_col].min()
    if end_date is None:
        end_date = df[date_col].max()

    # Build the full range
    full_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    missing_dates = full_range.difference(df[date_col].drop_duplicates())
    return missing_dates
import yfinance as yf
import pandas as pd

def download_missing_yfinance_data(missing_dates, ticker_symbol='AAPL', show_empty=False):
    """
    Checks yfinance for trading data on each date in missing_dates for the given ticker.

    Parameters:
        missing_dates (list): List of dates as strings (e.g. ['2020-03-14', ...])
        ticker_symbol (str): The ticker symbol to check (default 'AAPL')
        show_empty (bool): If True, prints 'No data for ...' for empty dates too.
    
    Returns:
        dict: Dictionary of {date: DataFrame} for dates with data.
    """
    ticker = yf.Ticker(ticker_symbol)
    found_data = {}
    for date in missing_dates:
        df = ticker.history(start=date, end=pd.to_datetime(date) + pd.Timedelta(days=1))
        if not df.empty:
            print(f"Data found for {date}:")
            print(df)
            found_data[date] = df
        else:
            if show_empty:
                print(f"No data for {date}")
    return found_data

