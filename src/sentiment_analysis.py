import pandas as pd
from datetime import timedelta
from typing import List, Optional
from tqdm import tqdm

def get_closest_value(series: pd.Series, target_date: pd.Timestamp, before: bool = True) -> Optional[float]:
    """
    Get the closest value to the target_date.
    If before=True, look for the closest date <= target_date.
    If before=False, look for the closest date >= target_date.
    """
    if before:
        candidates = series[series.index <= target_date]
        if not candidates.empty:
            return candidates.iloc[-1]
    else:
        candidates = series[series.index >= target_date]
        if not candidates.empty:
            return candidates.iloc[0]
    return None

def compute_volatility(close_series: pd.Series) -> Optional[float]:
    """
    Compute volatility (std of percent change) in close price series.
    """
    if close_series.empty or len(close_series) < 2:
        return None
    return close_series.pct_change().std() * 100

def analyze_news_volatility(
    df: pd.DataFrame, 
    date_col: str = "date", 
    stock_col: str = "stock", 
    close_col: str = "Close", 
    volume_col: str = "Volume", 
    sentiment_col: str = "finbert_sentiment",
    window: int = 5
) -> pd.DataFrame:
    """
    For each news event, calculate volatility and relate it to sentiment.
    Returns a DataFrame with sentiment, volatility, and stock info.
    """
    results = []
    df[date_col] = pd.to_datetime(df[date_col])
    # Ensure data is sorted for lookups
    for ticker in tqdm(df[stock_col].unique()):
        ticker_data = df[df[stock_col] == ticker].sort_values(date_col).copy()
        ticker_data = ticker_data.set_index(date_col)
        for idx, row in df[df[stock_col] == ticker].iterrows():
            news_date = row[date_col]
            # ±window-day window
            window_slice = ticker_data.loc[
                (ticker_data.index >= news_date - timedelta(days=window)) &
                (ticker_data.index <= news_date + timedelta(days=window))
            ]
            volatility = compute_volatility(window_slice[close_col])
            results.append({
                "stock": ticker,
                "date": news_date,
                "headline": row['headline'],
                "sentiment": row[sentiment_col],
                "volatility_%": volatility,
            })
    return pd.DataFrame(results)
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

def plot_daily_sentiment(
    df: pd.DataFrame,
    date_col: str = "date",
    sentiment_col: str = "finbert_sentiment",
    stock_col: str = None,
    stock_filter: str = None,
    figsize=(14, 5),
    title: str = "Average Daily Sentiment",
    ylabel: str = "Average Sentiment"
):
    """
    Plots average daily sentiment from news data (categorical: positive/negative/neutral).
    Maps sentiment to numeric values: positive=1, neutral=0, negative=-1.
    """
    mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
    plot_df = df.copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])

    # Standardize sentiment values
    plot_df[sentiment_col] = plot_df[sentiment_col].str.lower().str.strip()
    print("Unique sentiment values BEFORE mapping:", plot_df[sentiment_col].unique())

    plot_df[sentiment_col] = plot_df[sentiment_col].map(mapping)
    print("Unique sentiment values AFTER mapping:", plot_df[sentiment_col].unique())

    plot_df = plot_df.dropna(subset=[sentiment_col])

    if stock_col and stock_filter:
        plot_df[stock_col] = plot_df[stock_col].str.upper().str.strip()
        print("Unique stock values:", plot_df[stock_col].unique())
        plot_df = plot_df[plot_df[stock_col] == stock_filter]
        print("Filtered shape:", plot_df.shape)

    daily_sentiment = (
        plot_df.groupby(date_col)[sentiment_col]
        .mean()
        .reset_index()
        .sort_values(date_col)
    )

    plt.figure(figsize=figsize)
    plt.plot(daily_sentiment[date_col], daily_sentiment[sentiment_col], marker='o', linestyle='-')
    plt.title(title if not stock_filter else f"{title}: {stock_filter}")
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

    return daily_sentiment
def plot_weekly_sentiment(
    df,
    date_col='date',
    sentiment_col='finbert_sentiment',
    stock_col=None,
    stock_filter=None,
    figsize=(14, 5),
    title="Average Weekly Sentiment",
    ylabel="Average Sentiment"
):
    mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
    plot_df = df.copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    plot_df[sentiment_col] = plot_df[sentiment_col].str.lower().str.strip().map(mapping)
    plot_df = plot_df.dropna(subset=[sentiment_col])

    if stock_col and stock_filter:
        plot_df[stock_col] = plot_df[stock_col].str.upper().str.strip()
        plot_df = plot_df[plot_df[stock_col] == stock_filter]

    plot_df['period'] = plot_df[date_col].dt.to_period('W')
    weekly_sentiment = (
        plot_df.groupby('period')[sentiment_col]
        .mean()
        .reset_index()
        .sort_values('period')
    )

    plt.figure(figsize=figsize)
    plt.plot(weekly_sentiment['period'].dt.start_time, weekly_sentiment[sentiment_col], marker='o')
    plt.title(title if not stock_filter else f"{title}: {stock_filter}")
    plt.xlabel("Week")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

    return weekly_sentiment
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

def plot_stock_and_sentiment_merged(
    df_merged: pd.DataFrame,
    stock_col: str = "stock",
    date_col: str = "date",
    close_col: str = "Close",
    return_col: str = "Daily_Return",
    abs_col: str = "Abs_Return",
    sentiment_col: str = "sentiment",
    figsize=(16, 8)
):
    """
    For each unique stock in the merged dataset, plot returns/volatility and daily average sentiment.
    Assumes merged DataFrame has all necessary columns (one row per date/news).
    """
    mapping = {'positive': 1, 'neutral': 0, 'negative': -1}

    tickers = df_merged[stock_col].unique()
    for ticker in tickers:
        print(f"\nPlotting for stock: {ticker}")

        df = df_merged[df_merged[stock_col] == ticker].copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # If not already present, calculate returns
        if return_col not in df.columns:
            df = df.sort_values(date_col)
            df[return_col] = df[close_col].pct_change()
        if abs_col not in df.columns:
            df[abs_col] = df[return_col].abs()

        # Map sentiment to numeric for averaging
        if not pd.api.types.is_numeric_dtype(df[sentiment_col]):
            df[sentiment_col] = df[sentiment_col].str.lower().str.strip().map(mapping)

        # Daily average sentiment
        daily_sentiment = (
            df.groupby(date_col)[sentiment_col]
            .mean()
            .reset_index()
            .sort_values(date_col)
        )

        # Plot
        fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # Returns & Volatility
        axs[0].plot(df[date_col], df[return_col], label='Daily Return', color='tab:blue')
        axs[0].plot(df[date_col], df[abs_col], label='Abs Return (Volatility)', color='tab:orange', alpha=0.6)
        axs[0].axhline(0, color='gray', linestyle='--', linewidth=1)
        axs[0].set_title(f'{ticker} | Returns & Volatility vs. Sentiment')
        axs[0].set_ylabel('Return')
        axs[0].legend()

        # Sentiment
        axs[1].plot(daily_sentiment[date_col], daily_sentiment[sentiment_col], marker='o', linestyle='-', color='tab:green')
        axs[1].axhline(0, color='gray', linestyle='--', linewidth=1)
        axs[1].set_ylabel('Avg. Daily Sentiment')
        axs[1].set_xlabel('Date')

        plt.tight_layout()
        plt.show()
import plotly.graph_objs as go
import pandas as pd

import pandas as pd
import plotly.graph_objs as go
import numpy as np

def plot_stock_returns_with_sentiment(
    df, 
    stock_col='stock', 
    ticker='AAPL', 
    date_col='date', 
    close_col='Close', 
    volume_col='Volume',
    sentiment_col='finbert_sentiment',
    window_short=20,
    window_long=50
):
    # Filter for the stock
    df = df[df[stock_col] == ticker].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Daily return
    df['daily_return'] = df[close_col].pct_change()
    # Rolling SMAs
    df['SMA_20'] = df[close_col].rolling(window_short).mean()
    df['SMA_50'] = df[close_col].rolling(window_long).mean()

    # Sentiment mapping for color
    sent_color = df[sentiment_col].map({'positive':'green', 'neutral':'yellow', 'negative':'red'})
    sent_text = df[sentiment_col].fillna('neutral')

    # Make main trace (daily return)
    return_trace = go.Scatter(
        x=df[date_col],
        y=df['daily_return'],
        mode='lines+markers',
        marker=dict(
            size=7,
            color=sent_color,
            line=dict(width=0.5, color='black')
        ),
        name='Daily Return',
        text=[
            f'Date: {d.strftime("%Y-%m-%d")}<br>Return: {r:.2%}<br>Sentiment: {s}<br>Volume: {v:,}'
            for d, r, s, v in zip(df[date_col], df['daily_return'], sent_text, df[volume_col])
        ],
        hoverinfo='text'
    )

    # SMA lines (over returns, for trend context)
    sma20_trace = go.Scatter(
        x=df[date_col], y=df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='blue', width=2)
    )
    sma50_trace = go.Scatter(
        x=df[date_col], y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange', width=2)
    )

    # Volume as a bar at bottom, using secondary y-axis
    volume_trace = go.Bar(
        x=df[date_col], y=df[volume_col],
        name='Volume', yaxis='y2', marker_color='purple', opacity=0.3,
        hoverinfo='skip'
    )

    # Layout: 2 y-axes (left: daily return/SMA, right: volume)
    layout = go.Layout(
        title=f"{ticker} Daily Return, SMA, and Volume",
        xaxis=dict(title='Date', showgrid=True),
        yaxis=dict(title='Daily Return / SMA', showgrid=True),
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False,
            showticklabels=True,
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
        height=500,
        margin=dict(l=60, r=60, t=60, b=40)
    )

    fig = go.Figure(data=[return_trace, sma20_trace, sma50_trace, volume_trace], layout=layout)
    fig.show()


# Example usage:
# plot_stock_with_sentiment_plotly(df_merged, stock_col='stock', ticker='AAPL', date_col='date', close_col='Close', volume_col='Volume', sentiment_col='finbert_sentiment')
import pandas as pd
import plotly.graph_objs as go
from collections import Counter


import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go

class StockAnalyzer:
    def __init__(self, ticker, merged_df):
        self.ticker = ticker
        self.df = merged_df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df[self.df['stock'] == ticker].sort_values('date')

        # Sentiment scoring (optional, kept for possible further features)
        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        self.df['sentiment_score'] = self.df['finbert_sentiment'].str.lower().map(sentiment_map)

        # Aggregate daily sentiment (mean if multiple news per day)
        daily_sentiment = self.df.groupby('date')['sentiment_score'].mean().reset_index()
        daily_sentiment.rename(columns={'sentiment_score': 'avg_sentiment_score'}, inplace=True)

        # ------- Aggregate by publisher for hover text -------
        # For each day, get publisher-level sentiment mapping
        pub_agg = (
            self.df.groupby(['date', 'publisher'])['sentiment_score']
            .mean().unstack(fill_value=float('nan'))
        )
        pub_agg.reset_index(inplace=True)
        pub_agg = pub_agg.sort_values('date')

        # Build hover text for each date
        hover_texts = []
        for idx, row in pub_agg.iterrows():
            date = row['date']
            publishers = [col for col in pub_agg.columns if col != 'date']
            lines = []
            for pub in publishers:
                val = row[pub]
                if pd.isnull(val): continue
                lines.append(f"{pub}: {val:.2f}")
            hover_texts.append("<br>".join(lines))
        pub_agg['hover_text'] = hover_texts

        # Merge publisher hover into daily sentiment
        daily_sentiment = pd.merge(daily_sentiment, pub_agg[['date', 'hover_text']], on='date', how='left')

        # Merge with OHLCV data
        ohlc = self.df[['date', 'Open', 'High', 'Low', 'Close', 'Volume']].drop_duplicates()
        self.daily = pd.merge(ohlc, daily_sentiment, on='date', how='left').sort_values('date')

        # Calculate SMAs
        self.daily['SMA_20'] = self.daily['Close'].rolling(window=20).mean()
        self.daily['SMA_50'] = self.daily['Close'].rolling(window=50).mean()

    def plot_candlestick_with_sma_volume_sentiment(self):
        df = self.daily.copy()
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.04, row_heights=[0.6, 0.2, 0.2],
            subplot_titles=[
                f"{self.ticker} OHLC Candlestick with SMA 20 & SMA 50",
                "Volume",
                "Daily Sentiment Score (Averaged)"
            ]
        )

        # --- Candlestick (OHLC) ---
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Candlestick',
                showlegend=True
            ),
            row=1, col=1
        )

        # SMA 20
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['SMA_20'],
                name='SMA 20', line=dict(color='blue')
            ),
            row=1, col=1
        )

        # SMA 50
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['SMA_50'],
                name='SMA 50', line=dict(color='red')
            ),
            row=1, col=1
        )

        # --- Volume bar ---
        fig.add_trace(
            go.Bar(
                x=df['date'], y=df['Volume'],
                name='Volume', marker_color='purple', opacity=0.3
            ),
            row=2, col=1
        )

        # --- Sentiment score line with custom hover ---
        custom_text = [
            f"Avg sentiment: {row['avg_sentiment_score']:.2f}<br>{row['hover_text'] or ''}"
            for _, row in df.iterrows()
        ]
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['avg_sentiment_score'],
                name='Avg Sentiment Score', line=dict(color='orange', width=2),
                mode='lines+markers',
                text=custom_text,  # <-- this is the custom hover
                hoverinfo='text'
            ),
            row=3, col=1
        )

        fig.update_layout(
            yaxis_title='Price',
            xaxis_title='Date',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.1, xanchor='right', x=1),
            hovermode='x unified',
            height=900
        )
        fig.update_yaxes(title_text='Volume', row=2, col=1)
        fig.update_yaxes(title_text='Sentiment', row=3, col=1, range=[-1, 1])
        fig.show()
