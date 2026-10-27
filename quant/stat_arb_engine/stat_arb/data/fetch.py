"""
data/fetch.py
-------------
Download adjusted close prices from Yahoo Finance.
In a real firm this would hit an internal tick database (Kdb+/Q, Arctic).

Downloads are cached as pickles under `cache_dir` (keyed by tickers and
dates), so re-running main.py on a few hundred names doesn't re-hit Yahoo.
Delete the cache directory to force a fresh download.
"""

import hashlib
import os

import pandas as pd
import yfinance as yf


def fetch_prices(
    tickers    : list[str],
    start      : str,
    end        : str,
    max_missing: float = 0.05,
    cache_dir  : str | None = None,
) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers.

    Returns a DataFrame with dates as index and tickers as columns.
    Drops any tickers with >= `max_missing` of their days missing.
    Forward-fills small gaps (weekends/holidays already handled by yf).
    The final dropna() drops any row where a kept ticker has no price yet,
    so on a wide universe keep `max_missing` small or one late listing
    shortens the whole history.
    """
    raw = None
    path = None
    if cache_dir:
        key = hashlib.sha1(f"{sorted(tickers)}|{start}|{end}".encode()).hexdigest()[:16]
        path = os.path.join(cache_dir, f"prices_{key}.pkl")
        if os.path.exists(path):
            raw = pd.read_pickle(path)
    if raw is None:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            raw.to_pickle(path)

    # yfinance returns MultiIndex columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    # Drop tickers with too much missing data
    missing_pct = prices.isna().mean()
    good_tickers = missing_pct[missing_pct < max_missing].index.tolist()
    dropped = set(tickers) - set(good_tickers)
    if dropped:
        print(f"      Warning: dropped {sorted(dropped)} (>={max_missing:.0%} missing data)")

    prices = prices[good_tickers].ffill().dropna()
    return prices
