"""
backtest/tearsheet.py
----------------------
Render a 4-panel PNG tearsheet for a portfolio backtest:
  1. Equity curve
  2. Rolling 60-day Sharpe (is performance consistent, or one lucky quarter?)
  3. Drawdown ("underwater") chart
  4. Pair-PnL correlation heatmap (are the pairs actually diversifying?)
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless-safe: never try to open a GUI window
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

TRADING_DAYS_PER_YEAR = 252
ROLLING_SHARPE_WINDOW = 60


def _rolling_sharpe(daily_pnl, window=ROLLING_SHARPE_WINDOW):
    mean = daily_pnl.rolling(window).mean()
    std  = daily_pnl.rolling(window).std()
    return (mean / std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def plot_tearsheet(results, out_dir: str = "outputs") -> str:
    """
    Build and save the tearsheet PNG for a PortfolioResults object.

    Returns the path the file was saved to.
    """
    eq   = results.equity_curve
    dpnl = results.daily_pnl
    # With thousands of pair-periods a full heatmap is unreadable (and slow
    # to build): show the most active ones, by gross PnL.
    max_heat = 60
    heat_title = "Pair PnL Correlation"
    if results.pair_pnl.shape[1] > max_heat:
        top = results.pair_pnl.abs().sum().nlargest(max_heat).index
        corr = results.pair_pnl[top].corr()
        heat_title += f" (top {max_heat} of {results.pair_pnl.shape[1]} by gross PnL)"
    else:
        corr = results.pair_pnl_corr

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Stat-Arb Portfolio Tearsheet", fontsize=14, fontweight="bold")

    # 1. Equity curve
    ax = axes[0, 0]
    ax.plot(eq.index, eq.values, color="#1f77b4", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Cumulative return on capital")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=30)

    # 2. Rolling Sharpe
    ax = axes[0, 1]
    roll_sharpe = _rolling_sharpe(dpnl)
    ax.plot(roll_sharpe.index, roll_sharpe.values, color="#2ca02c", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(f"Rolling {ROLLING_SHARPE_WINDOW}d Sharpe")
    ax.tick_params(axis="x", rotation=30)

    # 3. Drawdown (underwater chart)
    ax = axes[1, 0]
    roll_max = eq.cummax().clip(lower=0.0)   # equity starts at 0
    drawdown = eq - roll_max
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#d62728", alpha=0.6)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (return on capital)")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=30)

    # 4. Pair PnL correlation heatmap
    ax = axes[1, 1]
    if corr.shape[0] > 1:
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        # Keep labels readable even with many pair-periods: only tick a
        # subset if there are a lot of columns.
        n = corr.shape[0]
        step = max(1, n // 12)
        ticks = list(range(0, n, step))
        labels = [corr.columns[i].split(" [")[0] for i in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(heat_title, fontsize=9)
    else:
        ax.text(0.5, 0.5, "Only one pair-period\n(no correlation to show)",
                ha="center", va="center")
        ax.set_title("Pair PnL Correlation")
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"tearsheet_{timestamp}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def plot_comparison(variants: dict, rows: list[dict], importances: dict,
                    calibration: dict, out_dir: str = "outputs") -> str:
    """
    Baseline vs learned tearsheet:
      1. Equity curves of every variant on the same days
      2. Sharpe and deflated Sharpe per variant
      3. Out-of-sample permutation importance per classifier
      4. Out-of-sample calibration (predicted vs realised win rate)
    variants: {name: PortfolioResults}; rows: print_comparison rows;
    importances: {model name: pd.Series}; calibration: {model name: DataFrame}.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Stat-Arb: Baseline vs Self-Learning (out-of-sample)", fontsize=14, fontweight="bold")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax = axes[0, 0]
    for i, (name, res) in enumerate(variants.items()):
        lw, c = (2.0, "black") if i == 0 else (1.1, colors[(i - 1) % len(colors)])
        ax.plot(res.equity_curve.index, res.equity_curve.values, label=name, linewidth=lw, color=c)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_title("Equity curves")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(fontsize=7, loc="best")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[0, 1]
    names = [r["name"] for r in rows]
    y = np.arange(len(rows))
    ax.barh(y - 0.2, [r["sharpe"] for r in rows], height=0.4, label="Sharpe (ann.)", color="#1f77b4")
    ax.barh(y + 0.2, [r["dsr"] for r in rows], height=0.4, label="Deflated Sharpe (prob.)", color="#ff7f0e")
    ax.axvline(0, color="black", linewidth=0.6)
    ax.axvline(0.95, color="#ff7f0e", linewidth=0.6, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Sharpe vs deflated Sharpe (DSR bar at 0.95)")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    if importances:
        imp = pd.DataFrame(importances).fillna(0.0)
        imp = imp.loc[imp.abs().max(axis=1).sort_values().index]
        yy = np.arange(len(imp))
        h = 0.8 / imp.shape[1]
        for j, col in enumerate(imp.columns):
            ax.barh(yy + j * h, imp[col].values, height=h, label=col)
        ax.set_yticks(yy + 0.4 - h / 2)
        ax.set_yticklabels(imp.index, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.legend(fontsize=7)
    ax.set_title("OOS permutation importance (drop in AUC)")

    ax = axes[1, 1]
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8)
    for name, cal in calibration.items():
        if len(cal):
            ax.plot(cal["p_mean"], cal["win_rate"], marker="o", label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(profitable)")
    ax.set_ylabel("Realised win rate")
    ax.set_title("OOS calibration (triple-barrier labels)")
    ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"comparison_{timestamp}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
