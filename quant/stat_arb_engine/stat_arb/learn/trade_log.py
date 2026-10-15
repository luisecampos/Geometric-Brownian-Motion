"""
learn/trade_log.py
------------------
Every candidate signal the primary (z-score) model produces, taken or not,
with its features and its triple-barrier outcome.

Candidate signal:
  The first decision day of each excursion of z into the entry band
  (entry_threshold < |z| <= stop_threshold), plus the first day of each
  active period if z is already in the band. These are exactly the days the
  engine can open a trade on: after a signal exit or a stop, the engine
  can't re-enter until z leaves the band and comes back in. So every engine
  entry is a candidate, and a candidate is "not taken" when the engine is
  already in a position, is blocked after a stop, or the meta-labeler says
  skip.

Triple-barrier label (López de Prado, AFML ch. 3), simulated like engine.py
on $1 gross: share counts frozen at entry, hedge from the decision close,
decisions on z[t-1] filled at the close of t, costs on both legs at entry
and exit.
  upper  "pt"         |z| back inside exit_threshold (the spread reverted
                      to its mean; the engine's own profit-take)
  lower  "sl"         |z| beyond stop_threshold (the spread blew out)
  vertical "time"     max_hold days, about 2x the pair's half-life
           "period_end" end of the active period (the engine force-closes)
  y = 1 if the net return is > 0.

  t1 (the exit date) is when the label becomes known. The meta-labeler only
  trains on candidates with t1 safely before the rebalance date (see
  metalabel.training_mask).
"""

from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd

from backtest.engine import COST_BPS
from learn.features import compute_event_features, selection_window_features


def find_candidate_signals(
    zscores         : pd.Series,
    first           : int,
    entry_threshold : float = 2.0,
    stop_threshold  : float = 3.5,
) -> list[int]:
    """
    Window positions of candidate DECISION days (the engine fills one day
    later). `first` is the engine's first row, i.e. the day before
    period_start. The last row is excluded because the engine never opens a
    trade on the final day.
    """
    z = zscores.values.astype(float)
    n = len(z)
    stop = np.inf if stop_threshold is None else stop_threshold

    def band(v):
        if not np.isfinite(v) or not (entry_threshold < abs(v) <= stop):
            return 0
        return 1 if v > 0 else -1

    out = []
    for d in range(first, n - 2):     # fill d+1 must come before the last row
        cur = band(z[d])
        if cur == 0:
            continue
        if d == first or band(z[d - 1]) != cur:
            out.append(d)
    return out


def triple_barrier(
    series_a       : pd.Series,
    series_b       : pd.Series,
    hedge_ratio    : pd.Series,
    zscores        : pd.Series,
    decision_pos   : int,
    exit_threshold : float = 0.5,
    stop_threshold : float = 3.5,
    max_hold       : int = 40,
    cost_bps       : float = COST_BPS,
) -> dict:
    """
    Outcome of a $1-gross trade decided at window position `decision_pos`
    and filled at the next close. Returns t1, barrier, holding_days, ret, y.
    Uses only data up to t1.
    """
    dates = series_a.index
    a, b = series_a.values.astype(float), series_b.values.astype(float)
    z = zscores.values.astype(float)
    n = len(a)
    stop = np.inf if stop_threshold is None else stop_threshold

    d, i = decision_pos, decision_pos + 1
    position = 1 if z[d] < 0 else -1          # LONG_SPREAD if z < 0
    hedge = float(hedge_ratio.iloc[d])
    notional_a = position / (1.0 + abs(hedge))
    notional_b = -position * hedge / (1.0 + abs(hedge))
    sh_a, sh_b = notional_a / a[i], notional_b / b[i]
    ret = -cost_bps * (abs(notional_a) + abs(notional_b))

    barrier, s = "period_end", n - 1
    for s in range(i + 1, n):
        ret += sh_a * (a[s] - a[s - 1]) + sh_b * (b[s] - b[s - 1])
        zz = z[s - 1]
        if not np.isfinite(zz) or abs(zz) < exit_threshold:
            barrier = "pt"
        elif abs(zz) > stop:
            barrier = "sl"
        elif s - i >= max_hold:
            barrier = "time"
        elif s == n - 1:
            barrier = "period_end"
        else:
            continue
        break
    ret -= cost_bps * (abs(sh_a) * a[s] + abs(sh_b) * b[s])
    return {
        "t1"          : dates[s],
        "barrier"     : barrier,
        "holding_days": s - i,
        "ret"         : float(ret),
        "y"           : int(ret > 0),
    }


def build_candidate_table(
    prepared        : list,
    prices          : pd.DataFrame,
    spy             : pd.Series | None = None,
    sectors         : dict | None = None,
    lookback_window : int = 60,
    entry_threshold : float = 2.0,
    exit_threshold  : float = 0.5,
    stop_threshold  : float = 3.5,
    signal          : str = "rolling",
    hold_multiple   : float = 2.0,
    min_hold        : int = 5,
) -> pd.DataFrame:
    """
    One row per candidate signal across all prepared pair-periods: identity,
    decision/entry dates, features as of the decision date, and the
    triple-barrier outcome (t1, barrier, ret, y).
    """
    calendar = prices.index
    first_qualified: dict = {}
    for prep in sorted(prepared, key=lambda p: p.pp.period_start):
        key = (prep.pp.ticker_a, prep.pp.ticker_b)
        first_qualified.setdefault(key, prep.pp.period_start)

    rows = []
    for prep in prepared:
        pp = prep.pp
        z = prep.zscores(lookback_window, signal)
        cands = find_candidate_signals(z, prep.first, entry_threshold, stop_threshold)
        if not cands:
            continue
        static = selection_window_features(
            prep.series_a.iloc[:prep.train_size], prep.series_b.iloc[:prep.train_size]
        )
        hl = static["half_life"] if np.isfinite(static["half_life"]) else pp.half_life
        max_hold = int(max(min_hold, round(hold_multiple * hl))) if np.isfinite(hl) else 60

        for d in cands:
            decision_date = z.index[d]
            feats = compute_event_features(
                prep, z, decision_date, static, spy,
                first_qualified[(pp.ticker_a, pp.ticker_b)], calendar, sectors,
            )
            out = triple_barrier(
                prep.series_a, prep.series_b, prep.kf.hedge_ratio, z, d,
                exit_threshold, stop_threshold, max_hold,
            )
            rows.append({
                "pair_label"   : prep.label,
                "ticker_a"     : pp.ticker_a,
                "ticker_b"     : pp.ticker_b,
                "period_start" : pp.period_start,
                "decision_date": decision_date,
                "entry_date"   : z.index[d + 1],
                "max_hold"     : max_hold,
                **feats,
                **out,
            })

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["decision_date", "pair_label"]).reset_index(drop=True)
    return df


def mark_taken(candidates: pd.DataFrame, results, column: str) -> pd.DataFrame:
    """
    Add `taken_<column>` (did the engine open a trade on this candidate?) and
    `pnl_<column>` (that trade's realised PnL on book capital) from a
    PortfolioResults run.
    """
    entries = {}
    for (ta, tb, ps), res in results.per_pair.items():
        for t in res.trades:
            entries[(ta, tb, ps, t.entry_date)] = t.pnl
    keys = list(zip(candidates["ticker_a"], candidates["ticker_b"],
                    candidates["period_start"], candidates["entry_date"]))
    candidates[f"taken_{column}"] = [k in entries for k in keys]
    candidates[f"pnl_{column}"] = [entries.get(k, np.nan) for k in keys]
    n_trades = sum(len(r.trades) for r in results.per_pair.values())
    unmatched = n_trades - int(candidates[f"taken_{column}"].sum())
    if unmatched:
        print(f"      Warning: {unmatched} {column} trade(s) didn't match a candidate signal")
    return candidates


class TradeLog:
    """
    Persist the candidate table. `.parquet` needs pyarrow; anything else
    (default `.sqlite`) uses the standard-library sqlite3, table `candidates`.
    """

    def __init__(self, path: str):
        self.path = path

    def save(self, df: pd.DataFrame) -> str:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        if self.path.endswith(".parquet"):
            df.to_parquet(self.path, index=False)
        else:
            out = df.copy()
            for c in out.columns:
                if pd.api.types.is_datetime64_any_dtype(out[c]):
                    out[c] = out[c].dt.strftime("%Y-%m-%d")
                elif out[c].dtype == bool:
                    out[c] = out[c].astype(int)
            with sqlite3.connect(self.path) as con:
                out.to_sql("candidates", con, if_exists="replace", index=False)
        return self.path

    def load(self) -> pd.DataFrame:
        if self.path.endswith(".parquet"):
            return pd.read_parquet(self.path)
        with sqlite3.connect(self.path) as con:
            df = pd.read_sql("SELECT * FROM candidates", con)
        for c in ("period_start", "decision_date", "entry_date", "t1"):
            if c in df:
                df[c] = pd.to_datetime(df[c])
        return df
