import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def get_top_ngrams(df, text_col='headline', ngram_range=(1,1), top_n=20, stop_words='english'):
    """
    Returns a list of the most common n-grams (words or phrases) in the text column.

    Args:
        df (pd.DataFrame): Your data.
        text_col (str): Column name containing the text.
        ngram_range (tuple): (min_n, max_n) for n-grams.
        top_n (int): How many to return.
        stop_words (str or list): Stopwords to remove.
    Returns:
        List of (ngram, count) tuples.
    """
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words)
    counts = vectorizer.fit_transform(df[text_col].dropna())
    sum_counts = counts.sum(axis=0)
    ngram_freq = [(ngram, sum_counts[0, idx]) for ngram, idx in vectorizer.vocabulary_.items()]
    sorted_ngrams = sorted(ngram_freq, key=lambda x: x[1], reverse=True)
    return sorted_ngrams[:top_n]

from sklearn.decomposition import LatentDirichletAllocation

def extract_topics(df, text_col='headline', n_topics=5, n_top_words=8, stop_words='english'):
    """
    Uses LDA to extract main topics from text data.
    Returns a list of topics, each as a list of top words.
    """
    vectorizer = CountVectorizer(stop_words=stop_words)
    dtm = vectorizer.fit_transform(df[text_col].dropna())
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    words = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[-n_top_words:][::-1]]
        topics.append(top_words)
    return topics
import re

def tag_headline_topics(headline, topic_dict):
    """
    Returns the first matching topic for a headline based on keyword presence.
    """
    headline = str(headline).lower()
    for topic, keywords in topic_dict.items():
        for kw in keywords:
            # Use word boundaries for precision
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            if re.search(pattern, headline):
                return topic
    return "Other"

TOPIC_KEYWORDS = {
    "Earnings & Estimates": [
        "earnings", "eps", "quarterly results", "beats", "misses", "scheduled", "estimate", "guidance"
    ],
    "Price Targets & Analyst Ratings": [
        "price target", "raises pt", "downgrades", "upgrades", "maintains", "initiates coverage",
        "buy", "sell", "hold", "target"
    ],
    "Mergers/Acquisitions/Partnerships": [
        "acquires", "merger", "acquisition", "partnership", "joint venture", "buyout", "deal"
    ],
    "FDA & Regulatory": [
        "fda approval", "fda", "approval", "clinical trial", "phase", "sec filing", "clearance", "warning letter"
    ],
    "Dividends/Splits/Buybacks": [
        "dividend", "split", "buyback", "repurchase", "payout", "special dividend"
    ],
    "Market/Trading Activity": [
        "52 week", "high", "low", "trading", "session", "volume", "moving", "pre market", "mid day", "after hours"
    ],
    "Sales/Revenue/Guidance": [
        "sales", "revenue", "forecast", "guidance", "raises guidance", "outlook", "update"
    ],
    "Company Events/Announcements": [
        "announces", "reports", "scheduled", "conference", "presentation"
    ],
    "Product/Innovation": [
        "launch", "product", "patent", "innovation", "trial", "technology"
    ],
    "Sector/Company Specific": [
        "oil", "mining", "energy", "pharma", "tech", "financial", "retail", "AAPL", "TSLA", "MSFT", "NVDA", "META", "GOOG"
    ],
}

def add_topic_tags(df, headline_col="headline", topic_dict=TOPIC_KEYWORDS):
    """
    Adds a 'topic' column to the DataFrame, assigning a topic to each headline.
    """
    df = df.copy()
    df['topic'] = df[headline_col].apply(lambda x: tag_headline_topics(x, topic_dict))
    return df
