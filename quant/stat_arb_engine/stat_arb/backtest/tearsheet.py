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
import matplotlib
matplotlib.use("Agg")   # headless-safe: never try to open a GUI window
import matplotlib.pyplot as plt

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
    corr = results.pair_pnl_corr

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Stat-Arb Portfolio Tearsheet", fontsize=14, fontweight="bold")

    # 1. Equity curve
    ax = axes[0, 0]
    ax.plot(eq.index, eq.values, color="#1f77b4", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Cumulative PnL ($)")
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
    roll_max = eq.cummax()
    drawdown = eq - roll_max
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#d62728", alpha=0.6)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown ($)")
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
        ax.set_title("Pair PnL Correlation")
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
