import pandas as pd
import pandas as pd
from dateutil import parser

# src/utils/date_parser.py

import pandas as pd
from dateutil import parser

def robust_parse(date_str):
    """
    Tries to parse a date string using dateutil.parser with fuzzy matching.
    Returns pandas NaT if parsing fails.
    """
    try:
        return parser.parse(date_str, fuzzy=True)
    except Exception:
        return pd.NaT

def parse_dates_column(df, column_name, new_column='parsed_date'):
    """
    Converts a date column with mixed formats to a uniform datetime column.
    
    Args:
        df (pd.DataFrame): DataFrame containing the date column.
        column_name (str): Name of the column to parse.
        new_column (str): Name for the new parsed date column.
        
    Returns:
        pd.DataFrame: DataFrame with an added or replaced parsed date column.
    """
    # First pass: vectorized pandas parsing
    df[new_column] = pd.to_datetime(df[column_name], errors='coerce')
    
    # Second pass: only for failures, try robust_parse
    mask = df[new_column].isna()
    df.loc[mask, new_column] = df.loc[mask, column_name].apply(robust_parse)
    
    return df


def check_nulls(df):
    """
    Returns a DataFrame with count and percentage of missing values in each column.
    """
    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    null_counts = df.isnull().sum()
    null_percent = (null_counts / len(df)) * 100

    null_summary = pd.DataFrame({
        'null_count': null_counts,
        'null_percent': null_percent.round(2)
    })

    return null_summary[null_summary['null_count'] > 0].sort_values(by='null_count', ascending=False)

def clean_news_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the financial news data.
    
    Steps:
    - Remove rows with missing essential values(if it is essential column)
    - Convert date column to datetime(must)
    - Strip whitespace from text fields
    - Remove duplicates
    """
    # Drop rows with missing headlines or dates
    df = df.dropna(subset=['headline', 'date'])

    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Strip whitespace from headline and publisher
    df['headline'] = df['headline'].str.strip()
    df['publisher'] = df['publisher'].str.strip()

    # Lowercase for consistency (optional)
    df['headline'] = df['headline'].str.lower()
    df['publisher'] = df['publisher'].str.lower()

    # Remove duplicates
    df = df.drop_duplicates()

    return df


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean historical stock data (AAPL, TSLA, etc.)
    
    Steps:
    - Convert 'date' column to datetime
    - Remove rows with missing prices
    - Sort by date
    """
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'close', 'open'])

    df = df.sort_values('date')
    df = df.drop_duplicates()

    return df
