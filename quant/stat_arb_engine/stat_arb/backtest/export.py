"""
backtest/export.py
------------------
Write a run's results as flat CSV tables for the results dashboard.

Every table has a "sample" column ("2019-2023" for main.py, "OOS 2024-2026"
for oos_test.py), so the two runs' files can be read side by side. Tables:

  variants.csv     one row per variant: headline stats (+ DSR when given)
  equity.csv       date, variant, cumulative return on book capital
  yearly.csv       variant, year: Sharpe and PnL
  quarters.csv     variant, quarter: PnL and pair-periods starting that quarter
  exits.csv        variant, exit reason: trades, PnL, win rate, holding days
  sectors.csv      variant, sector of the pair: trades and PnL
  grid.csv         static grid point: Sharpe (main.py only)
  calibration.csv  classifier, predicted-p quintile: win rate, mean return
  classifiers.csv  classifier: OOS AUC and Brier (main.py only)
"""

import os

import numpy as np
import pandas as pd

from backtest.report import summary_stats, sharpe_ratio


def _trades(res):
    return [(key, t) for key, r in res.per_pair.items() for t in r.trades]


def variant_tables(sample: str, variants: dict, dsr: dict | None = None, sectors: dict | None = None) -> dict:
    """Tables built from {name: PortfolioResults}."""
    rows, equity, yearly, quarters, exits, by_sector = [], [], [], [], [], []
    for name, res in variants.items():
        st = summary_stats(res)
        rows.append({"sample": sample, "variant": name, **st,
                     "dsr": (dsr or {}).get(name, np.nan)})

        eq = res.equity_curve
        equity.append(pd.DataFrame({"sample": sample, "variant": name,
                                    "date": eq.index.strftime("%Y-%m-%d"), "equity": eq.values}))

        x = res.daily_pnl
        for y in sorted(set(x.index.year)):
            xy = x[x.index.year == y]
            yearly.append({"sample": sample, "variant": name, "year": int(y),
                           "sharpe": sharpe_ratio(xy), "pnl": float(xy.sum())})

        trades = _trades(res)
        active = pd.Series([k[2] for k in res.per_pair]).value_counts()
        q = x.groupby(x.index.to_period("Q")).sum()
        for per, pnl in q.items():
            n = int(active[(active.index >= per.start_time) & (active.index <= per.end_time)].sum())
            quarters.append({"sample": sample, "variant": name, "quarter": str(per),
                             "pnl": float(pnl), "pairs_started": n})

        t = pd.DataFrame([{"reason": tr.exit_reason, "pnl": tr.pnl, "hold": tr.holding_days,
                           "sector": (sectors or {}).get(key[0], "n/a")} for key, tr in trades])
        if t.empty:
            continue
        for reason, g in t.groupby("reason"):
            exits.append({"sample": sample, "variant": name, "exit_reason": reason,
                          "trades": len(g), "pnl": float(g["pnl"].sum()),
                          "win_rate": float((g["pnl"] > 0).mean()),
                          "mean_hold_days": float(g["hold"].mean())})
        for sector, g in t.groupby("sector"):
            by_sector.append({"sample": sample, "variant": name, "sector": sector,
                              "trades": len(g), "pnl": float(g["pnl"].sum()),
                              "win_rate": float((g["pnl"] > 0).mean())})

    return {
        "variants": pd.DataFrame(rows),
        "equity": pd.concat(equity, ignore_index=True),
        "yearly": pd.DataFrame(yearly),
        "quarters": pd.DataFrame(quarters),
        "exits": pd.DataFrame(exits),
        "sectors": pd.DataFrame(by_sector),
    }


def learning_tables(sample: str, learned) -> dict:
    """Tables from learn.pipeline.LearningResults (grid, classifiers, calibration)."""
    grid = [{"sample": sample, "entry": e, "exit": x, "lookback": lb, "sharpe": sharpe_ratio(r.daily_pnl)}
            for (e, x, lb), r in learned.grid_results.items()]
    clf = [{"sample": sample, "model": k, **v} for k, v in learned.clf_stats.items()]
    cal = []
    c = learned.candidates
    for kind in learned.clf_stats:
        d = c[["y", "ret", f"p_{kind}"]].dropna().rename(columns={f"p_{kind}": "p"})
        d["quintile"] = pd.qcut(d["p"], 5, labels=False, duplicates="drop") + 1
        for qn, g in d.groupby("quintile"):
            cal.append({"sample": sample, "model": kind, "quintile": int(qn), "n": len(g),
                        "p_mean": float(g["p"].mean()), "win_rate": float(g["y"].mean()),
                        "mean_ret": float(g["ret"].mean())})
    return {"grid": pd.DataFrame(grid), "classifiers": pd.DataFrame(clf),
            "calibration": pd.DataFrame(cal)}


def write_tables(tables: dict, out_dir: str, prefix: str) -> list[str]:
    """Write each table as <out_dir>/<prefix>_<name>.csv; returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, df in tables.items():
        p = os.path.join(out_dir, f"{prefix}_{name}.csv")
        df.to_csv(p, index=False, float_format="%.6g")
        paths.append(p)
    return paths
