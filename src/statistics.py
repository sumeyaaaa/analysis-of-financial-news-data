import pandas as pd
from scipy import stats

def most_active_publishers(df, col, top_n):
    """
    Returns a Series of the most active publishers by article count.
    """
    return df[col].value_counts().head(top_n)
 
def mean_median_mode_stats(df, column):
    """
    Calculates mean, median, and mode for a headline length column.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Name of the column to analyze.

    Returns:
        dict: Dictionary with mean, median, and mode.
    """
    data = df[column].dropna()  # Exclude NaNs

    mean = data.mean()
    median = data.median()
    # Mode can be multimodal; scipy.stats.mode returns the smallest mode by default in v1.11+.
    mode = data.mode().iloc[0] if not data.mode().empty else None

    return {
        'mean': mean,
        'median': median,
        'mode': mode
    }


def article_counts_by_period(df, date_col='date', period='D'):
    """
    Groups article counts by date, week, or month.
    
    Args:
        df (pd.DataFrame): DataFrame with a date column.
        date_col (str): Name of the date column.
        period (str): Resampling period ('D'=day, 'W'=week, 'M'=month, etc.).
    Returns:
        pd.Series: Article counts indexed by period.
    """
    # Ensure date_col is datetime
    date_series = pd.to_datetime(df[date_col], errors='coerce')
    return date_series.value_counts().sort_index().resample(period).sum()


def article_counts_by_weekday(df, date_col='date'):
    """
    Returns a Series with counts of articles per weekday.
    """
    weekdays = pd.to_datetime(df[date_col], errors='coerce').dt.day_name()
    return weekdays.value_counts().reindex(
        ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    )

def article_counts_by_calendar_month(df, date_col='date'):
    """
    Aggregates the number of articles by calendar month (Jan-Dec), across all years.

    Args:
        df (pd.DataFrame): DataFrame containing the articles.
        date_col (str): Name of the date column.

    Returns:
        pd.Series: Series with month names as index and article counts as values.
    """
    # Ensure the date column is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Extract the month number and month name
    df['month'] = df[date_col].dt.month
    df['month_name'] = df[date_col].dt.strftime('%b')  # 'Jan', 'Feb', ...

    # Group by month number (to keep order), then sum counts
    counts = df.groupby('month').size().reindex(range(1, 13), fill_value=0)
    # Convert month numbers to month names for index
    counts.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return counts
