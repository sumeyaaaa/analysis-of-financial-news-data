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