"""
backtest/report.py
------------------
Compute and print performance metrics from backtest results.

Metrics that actually matter to quant interviewers:
  - Sharpe ratio   : risk-adjusted return (>1 is decent, >2 is good)
  - Max drawdown   : worst peak-to-trough loss (tells you about tail risk)
  - Win rate       : % of trades profitable
  - Avg hold time  : how long positions are open
  - Profit factor  : gross profit / gross loss (>1.5 is reasonable)
"""

import numpy as np
import pandas as pd
from backtest.engine import BacktestResults


TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(daily_pnl: pd.Series) -> float:
    pnl = daily_pnl[daily_pnl != 0]
    if pnl.std() == 0:
        return 0.0
    return float((pnl.mean() / pnl.std()) * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity_curve: pd.Series) -> float:
    roll_max = equity_curve.cummax()
    drawdown = equity_curve - roll_max
    return float(drawdown.min())


def print_report(ticker_a: str, ticker_b: str, results: BacktestResults) -> None:
    trades = results.trades
    eq     = results.equity_curve
    dpnl   = results.daily_pnl

    print("=" * 60)
    print(f"  BACKTEST RESULTS  —  {ticker_a} / {ticker_b}")
    print("=" * 60)

    if not trades:
        print("  No trades were executed.")
        return

    pnls        = [t.pnl for t in trades]
    winners     = [p for p in pnls if p > 0]
    losers      = [p for p in pnls if p <= 0]
    hold_times  = [(t.exit_date - t.entry_date).days for t in trades]

    total_return  = eq.iloc[-1]
    sharpe        = sharpe_ratio(dpnl)
    mdd           = max_drawdown(eq)
    win_rate      = len(winners) / len(pnls) * 100
    avg_win       = np.mean(winners) if winners else 0
    avg_loss      = np.mean(losers)  if losers  else 0
    profit_factor = (sum(winners) / abs(sum(losers))) if losers else float("inf")
    avg_hold      = np.mean(hold_times)

    print(f"\n  {'Total PnL':<28} ${total_return:>10.4f}")
    print(f"  {'Sharpe Ratio':<28} {sharpe:>10.2f}")
    print(f"  {'Max Drawdown':<28} ${mdd:>10.4f}")
    print(f"  {'Total Trades':<28} {len(trades):>10d}")
    print(f"  {'Win Rate':<28} {win_rate:>9.1f}%")
    print(f"  {'Avg Win':<28} ${avg_win:>10.4f}")
    print(f"  {'Avg Loss':<28} ${avg_loss:>10.4f}")
    print(f"  {'Profit Factor':<28} {profit_factor:>10.2f}")
    print(f"  {'Avg Hold Time':<28} {avg_hold:>8.1f} days")

    print(f"\n  {'─' * 50}")
    print(f"  {'Last 5 trades':}")
    print(f"  {'─' * 50}")
    print(f"  {'Entry':<12} {'Exit':<12} {'Dir':<8} {'Entry Z':>8} {'Exit Z':>8} {'PnL':>10}")
    print(f"  {'─' * 50}")

    for t in trades[-5:]:
        direction = "SHORT" if t.direction == -1 else "LONG"
        print(
            f"  {str(t.entry_date.date()):<12}"
            f" {str(t.exit_date.date()):<12}"
            f" {direction:<8}"
            f" {t.entry_z:>8.2f}"
            f" {t.exit_z:>8.2f}"
            f" ${t.pnl:>9.4f}"
        )

    print(f"\n  {'─' * 50}")
    print(f"  Final equity: ${eq.iloc[-1]:.4f}  (start: $0.0000)")
    print("=" * 60)
    print("\n  Tip: Plot eq.curve with matplotlib to visualise the equity curve.")
    print("  Tip: Increase ENTRY_THRESHOLD in main.py to trade less frequently.")
    print()


def print_portfolio_report(results) -> None:
    """
    Print portfolio-level stats for a multi-pair walk-forward backtest.

    `results` is a backtest.portfolio.PortfolioResults — kept untyped here
    (rather than imported) to avoid a circular import between report.py and
    portfolio.py.
    """
    eq   = results.equity_curve
    dpnl = results.daily_pnl
    corr = results.pair_pnl_corr

    n_pair_periods = len(results.per_pair)
    n_distinct_pairs = len({(a, b) for (a, b, _) in results.per_pair.keys()})
    total_trades = sum(len(r.trades) for r in results.per_pair.values())

    # Average pairwise correlation, off-diagonal only
    if corr.shape[0] > 1:
        off_diag = corr.values[~np.eye(corr.shape[0], dtype=bool)]
        avg_corr = float(np.nanmean(off_diag))
    else:
        avg_corr = float("nan")

    print("=" * 60)
    print("  PORTFOLIO BACKTEST RESULTS")
    print("=" * 60)
    print(f"\n  {'Total PnL':<28} ${eq.iloc[-1]:>10.4f}")
    print(f"  {'Sharpe Ratio':<28} {sharpe_ratio(dpnl):>10.2f}")
    print(f"  {'Max Drawdown':<28} ${max_drawdown(eq):>10.4f}")
    print(f"  {'Distinct Pairs Traded':<28} {n_distinct_pairs:>10d}")
    print(f"  {'Pair-Periods (walk-fwd)':<28} {n_pair_periods:>10d}")
    print(f"  {'Total Trades':<28} {total_trades:>10d}")
    print(f"  {'Avg Pairwise PnL Corr':<28} {avg_corr:>10.2f}")
    print(f"\n  {'─' * 50}")
    print(f"  Final equity: ${eq.iloc[-1]:.4f}  (start: $0.0000)")
    print("=" * 60)
    print()
