import matplotlib.pyplot as plt
import pandas as pd
def plot_histogram(df, column, bins=30, color='navy', title=None, xlabel=None, ylabel='Frequency', figsize=(10,4), alpha=0.7):
    """
    Plots a histogram for any specified column in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Name of the column to plot.
        bins (int): Number of bins for the histogram.
        color (str): Color of the histogram bars.
        title (str): Title for the plot.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label (default 'Frequency').
        figsize (tuple): Figure size.
        alpha (float): Transparency.
    """
    plt.figure(figsize=figsize)
    df[column].plot(kind='hist', bins=bins, alpha=alpha, color=color)
    plt.title(title or f'Distribution of {column}')
    plt.xlabel(xlabel or column)
    plt.ylabel(ylabel)
    plt.show()
def plot_time_trend(
    series: pd.Series,
    title,
    xlabel,
    ylabel
) -> None:
    """
    Plots a time series trend of article counts.

    Args:
        series (pd.Series): Index must be datetime, values are counts.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
    """
    plt.figure(figsize=(12, 4))
    series.plot(kind='line', marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_bar(
    series,
    title="Bar Plot",
    xlabel=None,
    ylabel="Count",
    color="steelblue",
    figsize=(10, 5),
    rotation=45,
    top_n=None,
    horizontal=False
):
    """
    Plots a bar plot for a pandas Series (or DataFrame column group/count).
    
    Args:
        series (pd.Series): Data to plot (index: categories, values: counts).
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        color (str): Bar color.
        figsize (tuple): Figure size.
        rotation (int): X-axis label rotation.
        top_n (int or None): Show only the top N items if set.
        horizontal (bool): Plot horizontal bar chart if True.
    """
    data = series.head(top_n) if top_n is not None else series
    plt.figure(figsize=figsize)
    if horizontal:
        data.plot(kind="barh", color=color)
        plt.xlabel(ylabel)
        plt.ylabel(xlabel if xlabel else data.index.name or "Category")
    else:
        data.plot(kind="bar", color=color)
        plt.xlabel(xlabel if xlabel else data.index.name or "Category")
        plt.ylabel(ylabel)
        plt.xticks(rotation=rotation, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.show()

import pandas as pd

def get_rolling_counts(
    df, 
    date_col='date', 
    window=7, 
    groupby_period='D'
):
    """
    Returns a rolling mean of article counts grouped by date (or another period).
    
    Args:
        df: DataFrame with a date column
        date_col: Name of the date column (default: 'date')
        window: Rolling window size (default: 7)
        groupby_period: Pandas offset alias ('D' for day, 'M' for month, etc.)
    
    Returns:
        pd.Series: Rolling mean of counts, indexed by date/period
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    grouped = df.groupby(pd.Grouper(key=date_col, freq=groupby_period)).size()
    rolling_avg = grouped.rolling(window=window, min_periods=1).mean()
    return rolling_avg
