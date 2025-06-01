def normalize_date(date):
    """
    Ensure the date is in 'YYYY-MM-DD' string format.
    Accepts string, datetime, pandas Timestamp, etc.
    """
    import pandas as pd
    if isinstance(date, str):
        # Try parsing just in case it's not normalized
        date = pd.to_datetime(date)
    elif not isinstance(date, pd.Timestamp):
        # Convert datetime.date or datetime.datetime to Timestamp
        date = pd.Timestamp(date)
    return date.strftime('%Y-%m-%d')
