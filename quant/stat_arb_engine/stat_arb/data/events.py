"""
data/events.py
--------------
Firm news dates from SEC 8-K filings, with the time each became public.

An 8-K lists item codes: 2.02 is "Results of Operations and Financial
Condition", the earnings release. EDGAR stamps each filing with the moment it
was accepted, so this gives point-in-time earnings and news dates for free.
acceptanceDateTime is in UTC (Apple's 16:30 ET release reads 20:30Z).

Amended filings (8-K/A) are left out: they usually repeat old news. ETFs file no
8-Ks, so they have no events. Tickers map to filers through the SEC's current
list, so a filer that has left it would have no events.

The SEC's fair-access policy asks for a contact in the User-Agent, so downloads
read it from the SEC_USER_AGENT environment variable, e.g.
  SEC_USER_AGENT="my-project me@example.com"
It is never written to the repo. Cached runs don't need it.
"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

EARNINGS_ITEM = "2.02"
MARKET_CLOSE_HOUR = 16     # Eastern time
PAUSE = 0.12               # seconds between requests: the SEC allows 10 a second


def _get(url: str, user_agent: str, tries: int = 5) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "identity"})
    for attempt in range(tries):
        try:
            time.sleep(PAUSE)
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as err:
            if err.code not in (429, 500, 502, 503) or attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"no answer from {url}")


def _filings(cik: int, start: str, user_agent: str) -> pd.DataFrame:
    """One filer's 8-Ks accepted from `start` on, following its older pages."""
    base = "https://data.sec.gov/submissions/"
    doc = _get(f"{base}CIK{cik:010d}.json", user_agent)
    pages = [doc["filings"]["recent"]]
    for page in doc["filings"]["files"]:
        if page["filingTo"] >= start:
            pages.append(_get(base + page["name"], user_agent))
    df = pd.concat([pd.DataFrame(p)[["form", "acceptanceDateTime", "items", "filingDate"]] for p in pages])
    df = df[(df["form"] == "8-K") & (df["filingDate"] >= start)]
    return df.rename(columns={"acceptanceDateTime": "accepted"})[["accepted", "items"]]


def fetch_8k_events(tickers: list[str], start: str, as_of: str, cache_dir: str | None = None) -> pd.DataFrame:
    """
    Every 8-K of `tickers` accepted from `start` to the download, as rows of
    ticker, accepted (UTC) and items. Cached by tickers, start and `as_of`.
    Tickers with no filer are listed in .attrs["unmapped"].
    """
    path = None
    if cache_dir:
        key = hashlib.sha1(f"{sorted(tickers)}|{start}|{as_of}".encode()).hexdigest()[:16]
        path = os.path.join(cache_dir, f"events_{key}.pkl")
        if os.path.exists(path):
            return pd.read_pickle(path)

    user_agent = os.environ.get("SEC_USER_AGENT")
    if not user_agent:
        raise RuntimeError('Set SEC_USER_AGENT to a project name and contact email, e.g. "my-project me@example.com".')
    listing = _get("https://www.sec.gov/files/company_tickers.json", user_agent)
    ciks = {row["ticker"]: int(row["cik_str"]) for row in listing.values()}

    frames, by_cik = [], {}
    for ticker in tickers:
        cik = ciks.get(ticker)
        if cik is None:
            continue
        if cik not in by_cik:          # share classes of one filer
            by_cik[cik] = _filings(cik, start, user_agent)
        frames.append(by_cik[cik].assign(ticker=ticker))
    events = pd.concat(frames, ignore_index=True)[["ticker", "accepted", "items"]]
    events["accepted"] = pd.to_datetime(events["accepted"], utc=True)
    events.attrs["unmapped"] = sorted(t for t in tickers if t not in ciks)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        events.to_pickle(path)
    return events


def earnings_only(events: pd.DataFrame) -> pd.DataFrame:
    """The 8-Ks with item 2.02."""
    codes = events["items"].fillna("").str.split(",")
    return events[[EARNINGS_ITEM in c for c in codes]]


def effective_positions(accepted: pd.Series, calendar: pd.DatetimeIndex) -> np.ndarray:
    """
    Calendar position of the first close that could have reacted to each
    filing: its own day if accepted before the 16:00 ET close of a trading day,
    otherwise the next trading day. May equal len(calendar).
    """
    eastern = pd.to_datetime(accepted, utc=True).dt.tz_convert("America/New_York")
    day = eastern.dt.tz_localize(None).dt.normalize().values
    pos = calendar.searchsorted(day)
    on_trading_day = np.zeros(len(pos), dtype=bool)
    inside = pos < len(calendar)
    on_trading_day[inside] = calendar.values[pos[inside]] == day[inside]
    after_close = eastern.dt.hour.values >= MARKET_CLOSE_HOUR
    return pos + (on_trading_day & after_close)


def blocked_days(events: pd.DataFrame, calendar: pd.DatetimeIndex, window: int) -> dict:
    """{ticker: bool array on calendar}, True on the event's effective day and
    the following window - 1 trading days."""
    out = {}
    for ticker, group in events.groupby("ticker"):
        mask = np.zeros(len(calendar), dtype=bool)
        for p in effective_positions(group["accepted"], calendar):
            if p < len(calendar):
                mask[p : p + window] = True
        out[ticker] = mask
    return out


def entry_multipliers(prepared: list, blocked: dict, calendar: pd.DatetimeIndex) -> dict:
    """
    run_portfolio_backtest size_multipliers: 0 on decision dates when either leg
    is blocked, so no entry is taken there, 1 otherwise. Open positions are
    left alone.
    """
    none = np.zeros(len(calendar), dtype=bool)
    out = {}
    for prep in prepared:
        pos = calendar.get_indexer(prep.size.index)
        hit = blocked.get(prep.pp.ticker_a, none)[pos] | blocked.get(prep.pp.ticker_b, none)[pos]
        out[prep.label] = pd.Series(np.where(hit, 0.0, 1.0), index=prep.size.index)
    return out
