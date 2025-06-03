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
