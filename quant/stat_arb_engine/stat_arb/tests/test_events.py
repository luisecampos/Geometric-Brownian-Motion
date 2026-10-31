"""
tests/test_events.py
--------------------
Timing checks for the news filter in data/events.py. The one that matters: a
filing can only block an entry once a close could have reacted to it.

Run with `python -m pytest tests` or just `python tests/test_events.py`.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.events import blocked_days, earnings_only, effective_positions, entry_multipliers  # noqa: E402

CAL = pd.bdate_range("2021-03-08", "2021-03-26")   # US clocks changed on Sunday 2021-03-14


def _pos(utc):
    return int(effective_positions(pd.Series([pd.Timestamp(utc, tz="UTC")]), CAL)[0])


def test_before_the_close_counts_that_day():
    assert CAL[_pos("2021-03-10 18:00")] == pd.Timestamp("2021-03-10")   # 13:00 ET


def test_after_the_close_counts_next_day():
    assert CAL[_pos("2021-03-10 21:30")] == pd.Timestamp("2021-03-11")   # 16:30 ET


def test_exactly_the_close_counts_next_day():
    assert CAL[_pos("2021-03-10 21:00")] == pd.Timestamp("2021-03-11")   # 16:00 ET


def test_daylight_saving_shifts_the_close():
    # 20:30 UTC is 15:30 ET before the change and 16:30 ET after it
    assert CAL[_pos("2021-03-12 20:30")] == pd.Timestamp("2021-03-12")
    assert CAL[_pos("2021-03-16 20:30")] == pd.Timestamp("2021-03-17")


def test_late_evening_stays_on_its_eastern_date():
    # 01:00 UTC on the 11th is 20:00 ET on the 10th, after that close
    assert CAL[_pos("2021-03-11 01:00")] == pd.Timestamp("2021-03-11")


def test_weekend_filing_counts_next_trading_day():
    assert CAL[_pos("2021-03-13 15:00")] == pd.Timestamp("2021-03-15")
    assert CAL[_pos("2021-03-13 22:00")] == pd.Timestamp("2021-03-15")   # after hours, no trading day


def test_filing_after_the_last_day_drops_off():
    events = pd.DataFrame({"ticker": ["A"], "accepted": [pd.Timestamp("2021-03-26 21:30", tz="UTC")]})
    assert not blocked_days(events, CAL, 5)["A"].any()


def test_blocked_window_starts_on_effective_day():
    events = pd.DataFrame({"ticker": ["A"], "accepted": [pd.Timestamp("2021-03-10 21:30", tz="UTC")]})
    mask = pd.Series(blocked_days(events, CAL, 3)["A"], index=CAL)
    assert list(mask[mask].index.strftime("%m-%d")) == ["03-11", "03-12", "03-15"]


def test_earnings_only_matches_the_whole_item():
    events = pd.DataFrame({"ticker": list("ABCD"), "accepted": pd.Timestamp("2021-03-10", tz="UTC"),
                           "items": ["2.02,9.01", "8.01", "12.02", None]})
    assert list(earnings_only(events)["ticker"]) == ["A"]


def test_multipliers_block_either_leg_only_on_blocked_days():
    blocked = {"A": np.array([False, True] + [False] * (len(CAL) - 2)),
               "B": np.array([False, False, False, True] + [False] * (len(CAL) - 4))}
    prep = SimpleNamespace(label="x", size=pd.Series(1.0, index=CAL[:6]),
                           pp=SimpleNamespace(ticker_a="A", ticker_b="B"))
    assert list(entry_multipliers([prep], blocked, CAL)["x"]) == [1, 0, 1, 0, 1, 1]
    prep.pp = SimpleNamespace(ticker_a="XLK", ticker_b="XLF")            # no filings at all
    assert list(entry_multipliers([prep], blocked, CAL)["x"]) == [1] * 6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
