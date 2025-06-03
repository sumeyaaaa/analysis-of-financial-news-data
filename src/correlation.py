import pandas as pd

def sentiment_to_numeric(sentiment):
    """Map sentiment labels to numeric values."""
    mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
    return sentiment.str.lower().map(mapping)

def compute_daily_sentiment_correlation(df, stock_col='stock', date_col='date', sentiment_col='finbert_sentiment', return_col='daily_return'):
    """
    For each stock, computes the correlation between average daily sentiment and daily return.
    Returns a DataFrame with the results.
    """
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Map sentiment to numeric
    df['sentiment_numeric'] = sentiment_to_numeric(df[sentiment_col])
    
    results = []
    for ticker in df[stock_col].unique():
        stock_df = df[df[stock_col] == ticker]
        # Average sentiment per day
        daily_sentiment = stock_df.groupby(date_col)['sentiment_numeric'].mean()
        # Daily return per day
        daily_return = stock_df.groupby(date_col)[return_col].mean()
        # Combine into single DataFrame
        merged = pd.concat([daily_sentiment, daily_return], axis=1).dropna()
        merged.columns = ['avg_sentiment', 'avg_return']
        # Calculate correlation
        corr = merged['avg_sentiment'].corr(merged['avg_return'])
        results.append({
            'stock': ticker,
            'correlation_sentiment_return': corr,
            'n_days': merged.shape[0]
        })
    return pd.DataFrame(results)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np      
def plot_correlation_bar(corr_df, corr_col='correlation_sentiment_return', stock_col='stock'):
    """Bar plot showing correlation per stock."""
    # Map sentiment to numeric

    plt.figure(figsize=(8,5))
    # Remove palette, or use a seaborn palette string like "coolwarm"
    sns.barplot(
        x=stock_col, y=corr_col, 
        data=corr_df, 
        palette="coolwarm"  # just use palette name, or remove this line for default
    )
    plt.axhline(0, color='gray', linestyle='--')
    for i, v in enumerate(corr_df[corr_col]):
        plt.text(i, v, f"{v:.2f}", ha='center', va='bottom' if v>0 else 'top', fontsize=10)
    plt.title("Correlation: Daily Sentiment vs Daily Return (per Stock)")
    plt.ylabel("Pearson Correlation")
    plt.xlabel("Stock")
    plt.tight_layout()
    plt.show()

def plot_scatter_sentiment_vs_return(
    df, 
    stock_name, 
    stock_col='stock', 
    date_col='date', 
    sentiment_col='finbert_sentiment', 
    return_col='daily_return'
):
    """Scatter plot for a single stock (sentiment vs return, by day)."""
    df = df.copy()
    # Clean up and map sentiment
    df['sentiment_clean'] = df[sentiment_col].astype(str).str.strip().str.lower()
    print("Unique sentiment values (after clean):", df['sentiment_clean'].unique())
    mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_numeric'] = df['sentiment_clean'].map(mapping)
    print("Unique numeric sentiment values:", df['sentiment_numeric'].unique())

    stock_df = df[df[stock_col] == stock_name]
    print("Stock rows:", stock_df.shape)
    
    # Drop NAs (both cols)
    stock_df = stock_df.dropna(subset=['sentiment_numeric', return_col])
    print("After dropping NA:", stock_df.shape)

    # Group and aggregate
    daily = (
        stock_df
        .groupby(date_col)
        .agg({'sentiment_numeric': 'mean', return_col: 'mean'})
        .dropna()
        .rename(columns={'sentiment_numeric': 'avg_sentiment', return_col: 'avg_return'})
    )
    print("Grouped daily shape:", daily.shape)
    print(daily.head())

    if daily.empty:
        print(f"No valid data for {stock_name} after processing.")
        return

    plt.figure(figsize=(7,5))
    sns.scatterplot(x='avg_sentiment', y='avg_return', data=daily)
    plt.title(f"{stock_name}: Daily Avg Sentiment vs Daily Return")
    plt.xlabel("Average Daily Sentiment")
    plt.ylabel("Daily Return")
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.show()
def compute_daily_sentiment_correlation_by_publisher(
    df, 
    stock_col='stock', 
    date_col='date', 
    sentiment_col='finbert_sentiment', 
    return_col='daily_return',
    publisher_col='publisher'
):
    """
    For each stock and publisher, computes the correlation between average daily sentiment and daily return.
    Returns a DataFrame with the results.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df['sentiment_numeric'] = sentiment_to_numeric(df[sentiment_col])

    results = []
    for publisher in df[publisher_col].unique():
        pub_df = df[df[publisher_col] == publisher]
        for ticker in pub_df[stock_col].unique():
            stock_df = pub_df[pub_df[stock_col] == ticker]
            daily_sentiment = stock_df.groupby(date_col)['sentiment_numeric'].mean()
            daily_return = stock_df.groupby(date_col)[return_col].mean()
            merged = pd.concat([daily_sentiment, daily_return], axis=1).dropna()
            merged.columns = ['avg_sentiment', 'avg_return']
            corr = merged['avg_sentiment'].corr(merged['avg_return'])
            results.append({
                'publisher': publisher,
                'stock': ticker,
                'correlation_sentiment_return': corr,
                'n_days': merged.shape[0]
            })
    return pd.DataFrame(results)

def compute_daily_sentiment_correlation_by_publisher(
    df, 
    stock_col='stock', 
    date_col='date', 
    sentiment_col='finbert_sentiment', 
    return_col='daily_return',
    publisher_col='publisher'
):
    df[date_col] = pd.to_datetime(df[date_col])
    df['sentiment_numeric'] = sentiment_to_numeric(df[sentiment_col])

    results = []
    for publisher in df[publisher_col].unique():
        pub_df = df[df[publisher_col] == publisher]
        for ticker in pub_df[stock_col].unique():
            stock_df = pub_df[pub_df[stock_col] == ticker]
            daily_sentiment = stock_df.groupby(date_col)['sentiment_numeric'].mean()
            daily_return = stock_df.groupby(date_col)[return_col].mean()
            merged = pd.concat([daily_sentiment, daily_return], axis=1).dropna()
            merged.columns = ['avg_sentiment', 'avg_return']
            corr = merged['avg_sentiment'].corr(merged['avg_return'])
            article_count = stock_df.shape[0]  # Number of articles for this publisher-stock
            results.append({
                'publisher': publisher,
                'stock': ticker,
                'correlation_sentiment_return': corr,
                'n_days': merged.shape[0],
                'article_count': article_count
            })
    return pd.DataFrame(results)


import plotly.express as px


def plotly_correlation_bar_by_stock(
    corr_pub_df, stock, corr_col='correlation_sentiment_return', publisher_col='publisher', article_count_col='article_count', raw_df=None, stock_col='stock'
):
    """
    Plotly bar plot: correlation per publisher for a specific stock.
    Only plots if at least one correlation is < -0.3 or > 0.3.
    Hover shows publisher, correlation, and count.
    Also shows the total article count for the selected stock in the title.
    """
    data = corr_pub_df[corr_pub_df['stock'] == stock]
    data = data[(data[corr_col] < -0.3) | (data[corr_col] > 0.3)]
    if data.empty:
        print(f"No strong correlations for stock '{stock}'.")
        return

    # Calculate total article count for this stock
    if raw_df is not None and stock_col in raw_df.columns:
        total_articles = raw_df[raw_df[stock_col] == stock].shape[0]
        title_text = f"Correlation: Daily Sentiment vs Daily Return ({stock}) [|corr| > 0.3]<br>Total articles for {stock}: {total_articles}"
    elif article_count_col in data.columns:
        # fallback: sum article counts in filtered df (may double-count across publishers)
        total_articles = data[article_count_col].sum()
        title_text = f"Correlation: Daily Sentiment vs Daily Return ({stock}) [|corr| > 0.3]<br>(sum of publisher counts = {total_articles})"
    else:
        title_text = f"Correlation: Daily Sentiment vs Daily Return ({stock}) [|corr| > 0.3]"

    fig = px.bar(
        data,
        x=publisher_col,
        y=corr_col,
        text=data[corr_col].round(2),
        color=corr_col,
        color_continuous_scale='RdBu',
        title=title_text,
        labels={corr_col: "Pearson Correlation", publisher_col: "Publisher"},
        hover_data={
            publisher_col: True,
            corr_col: ':.2f',
            article_count_col: True
        }
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_traces(textposition='outside')
    fig.update_layout(
        yaxis_title="Pearson Correlation",
        xaxis_title="Publisher",
        coloraxis_colorbar=dict(title="Correlation"),
        showlegend=False,
        margin=dict(l=60, r=30, t=60, b=60)
    )
    fig.show()

# Usage:
# plotly_correlation_bar_by_stock(corr_pub_df, stock="AAPL", raw_df=your_full_dataframe)





