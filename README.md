# analysis-of-financial-news-data
This project basicaly focuses on enhancing financial forecasting using advanced data analytics. It includes sentiment analysis of financial news headlines using NLP techniques and correlation analysis to evaluate how sentiment influences stock price movements.

# Financial News Analysis (CI/CD Setup)

This repo is scaffolded with a working CI/CD pipeline using GitHub Actions.

## How to Use
- Placeholder logic exists in `src/` and is tested via `tests/`
- CI/CD runs on every push via GitHub Actions
- Install dependencies: `pip install -r requirements.txt`
- Run tests locally: `python -m unittest discover tests`
# Analysis of Financial News Data

This project explores the relationship between financial news sentiment and stock price movements. It leverages data analytics, natural language processing, and technical analysis to provide insights for financial forecasting.

---

## Project Structure

```
.
├── data/                # Raw and processed data files
├── notebooks/           # Jupyter notebooks for analysis
│   ├── task_1/          # Data Preparation & Cleaning
│   ├── task_2/          # Sentiment Analysis & Feature Engineering
│   └── task_3/          # Correlation & Predictive Analysis
├── src/                 # Source code modules
├── tests/               # Unit tests
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## Notebooks Overview

### `notebooks/task_1/` — **Data Preparation & Cleaning**
- **Purpose:** Load, merge, and clean financial news and stock price data.
- **Key Steps:**
  - Import raw news and stock data.
  - Normalize date formats and handle missing values.
  - Merge news headlines with corresponding stock prices.
  - Output a cleaned dataset ready for analysis.

### `notebooks/task_2/` — **Sentiment Analysis & Feature Engineering**
- **Purpose:** Extract sentiment from news headlines and engineer features for modeling.
- **Key Steps:**
  - Apply NLP techniques to assign sentiment scores to each headline.
  - Aggregate sentiment scores by date or ticker.
  - Compute technical indicators (e.g., Moving Average, RSI, MACD, ATR, Bollinger Bands).
  - Visualize sentiment and indicator trends.

### `notebooks/task_3/` — **Correlation & Predictive Analysis**
- **Purpose:** Analyze the relationship between news sentiment and stock price movements.
- **Key Steps:**
  - Correlate aggregated sentiment scores with stock returns and volatility.
  - Visualize correlations and trends.
  - (Optional) Build simple predictive models to forecast price movement based on sentiment and technical features.
  - Evaluate model performance.

---

## Main Features

- **Data Loading & Cleaning:** Utilities for importing and preprocessing data (`src/data_loader.py`, `src/data_cleaning.py`).
- **Sentiment Analysis:** NLP-based sentiment scoring for news headlines (`src/sentiment_analysis.py`, `src/text_analysis.py`).
- **Technical Indicators:** Calculation and visualization of Moving Averages, RSI, MACD, ATR, and Bollinger Bands.
- **Correlation Analysis:** Tools to analyze the relationship between sentiment and stock price movements (`src/correlation.py`).
- **Visualization:** Functions for plotting trends, distributions, and technical indicators.
- **Jupyter Notebooks:** Step-by-step workflows for each stage of the analysis.

---

## Setup

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd analysis-of-financial-news-data
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run tests:**
   ```sh
   python -m unittest discover tests
   ```

4. **Explore Notebooks:**
   - Open notebooks in the `notebooks/` directory for guided analysis.

---

## Usage Examples

- **Load and clean news data:**
  ```python
  from src.data_loader import load_news_data
  df = load_news_data('data/news_yfinance_merged.csv')
  ```

- **Run sentiment analysis:**
  ```python
  from src.sentiment_analysis import analyze_sentiment
  df['sentiment'] = df['headline'].apply(analyze_sentiment)
  ```

- **Plot technical indicators:**
  ```python
  from src.plot_ma import plot_ma
  plot_ma(df, 'Close')
  ```

---
