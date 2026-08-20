"""
data/fetch.py
-------------
Download adjusted close prices from Yahoo Finance.
In a real firm this would hit an internal tick database (Kdb+/Q, Arctic).
"""

import pandas as pd
import yfinance as yf


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers.

    Returns a DataFrame with dates as index and tickers as columns.
    Drops any tickers with >5% missing data.
    Forward-fills small gaps (weekends/holidays already handled by yf).
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    # Drop tickers with too much missing data
    missing_pct = prices.isna().mean()
    good_tickers = missing_pct[missing_pct < 0.05].index.tolist()
    dropped = set(tickers) - set(good_tickers)
    if dropped:
        print(f"      Warning: dropped {dropped} (>5% missing data)")

    prices = prices[good_tickers].ffill().dropna()
    return prices
