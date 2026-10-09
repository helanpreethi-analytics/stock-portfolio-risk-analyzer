import yfinance as yf
import pandas as pd


def load_data(tickers, start, end):
    """Fetch and clean price data for multiple tickers."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    data = data.dropna()
    return data


def calculate_returns(df):
    return df.pct_change().dropna()
