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
from backtest.engine import BacktestResults, COST_BPS


TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(daily_pnl: pd.Series) -> float:
    """
    Annualised Sharpe over ALL days, flat days included.

    The old version dropped zero-PnL days first. A strategy that's in the
    market a fraction f of the time then reports roughly 1/sqrt(f) times its
    true Sharpe (~1.8x at f=0.32) — idle capital is still capital. This now
    matches the tearsheet's rolling Sharpe, which always used every day.
    """
    std = daily_pnl.std()
    if not np.isfinite(std) or std == 0:
        return 0.0
    return float((daily_pnl.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Worst peak-to-trough drop of the cumulative-return curve (a return, e.g. -0.08)."""
    roll_max = equity_curve.cummax().clip(lower=0.0)   # equity starts at 0
    drawdown = equity_curve - roll_max
    return float(drawdown.min())


def _exit_breakdown(trades) -> str:
    counts = {}
    for t in trades:
        counts[t.exit_reason] = counts.get(t.exit_reason, 0) + 1
    order = ["signal", "stop", "period_end"]
    keys = order + sorted(k for k in counts if k not in order)
    return "  ".join(f"{k}={counts[k]}" for k in keys if k in counts)


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

    print(f"\n  {'Total Return':<28} {total_return:>10.2%}")
    print(f"  {'Sharpe Ratio':<28} {sharpe:>10.2f}")
    print(f"  {'Max Drawdown':<28} {mdd:>10.2%}")
    print(f"  {'Total Trades':<28} {len(trades):>10d}")
    print(f"  {'Exits':<28} {_exit_breakdown(trades)}")
    print(f"  {'Win Rate':<28} {win_rate:>9.1f}%")
    print(f"  {'Avg Win':<28} {avg_win:>10.3%}")
    print(f"  {'Avg Loss':<28} {avg_loss:>10.3%}")
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
            f" {t.pnl:>9.3%}"
        )

    print(f"\n  {'─' * 50}")
    print(f"  Cumulative return on capital: {eq.iloc[-1]:.2%}")
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

    n_pair_periods = len(results.per_pair)
    n_distinct_pairs = len({(a, b) for (a, b, _) in results.per_pair.keys()})
    all_trades = [t for r in results.per_pair.values() for t in r.trades]
    total_trades = len(all_trades)
    win_rate = np.mean([t.pnl > 0 for t in all_trades]) * 100 if all_trades else float("nan")
    ann_ret = dpnl.mean() * TRADING_DAYS_PER_YEAR
    ann_vol = dpnl.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    pct_in_market = (dpnl != 0).mean() * 100

    # Average pairwise correlation, off-diagonal only
    avg_corr = results.avg_pair_corr()

    print("=" * 60)
    print("  PORTFOLIO BACKTEST RESULTS")
    print("=" * 60)
    print(f"\n  {'Total Return':<28} {eq.iloc[-1]:>10.2%}")
    print(f"  {'Annualised Return':<28} {ann_ret:>10.2%}")
    print(f"  {'Annualised Vol':<28} {ann_vol:>10.2%}")
    print(f"  {'Sharpe Ratio (all days)':<28} {sharpe_ratio(dpnl):>10.2f}")
    print(f"  {'Max Drawdown':<28} {max_drawdown(eq):>10.2%}")
    print(f"  {'Days in Market':<28} {pct_in_market:>9.1f}%")
    print(f"  {'Distinct Pairs Traded':<28} {n_distinct_pairs:>10d}")
    print(f"  {'Pair-Periods (walk-fwd)':<28} {n_pair_periods:>10d}")
    print(f"  {'Total Trades':<28} {total_trades:>10d}")
    print(f"  {'Exits':<28} {_exit_breakdown(all_trades)}")
    print(f"  {'Win Rate (net of costs)':<28} {win_rate:>9.1f}%")
    print(f"  {'Avg Pairwise PnL Corr':<28} {avg_corr:>10.2f}")
    print(f"\n  {'─' * 50}")
    print(f"  Returns are on book capital (no compounding), net of {COST_BPS*1e4:.0f}bp/leg costs.")
    print("=" * 60)
    print()


def summary_stats(results) -> dict:
    """Headline numbers for one PortfolioResults, for side-by-side tables."""
    dpnl, eq = results.daily_pnl, results.equity_curve
    trades = [t for r in results.per_pair.values() for t in r.trades]
    return {
        "total_return": float(eq.iloc[-1]),
        "ann_vol"     : float(dpnl.std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "sharpe"      : sharpe_ratio(dpnl),
        "max_dd"      : max_drawdown(eq),
        "trades"      : len(trades),
        "win_rate"    : float(np.mean([t.pnl > 0 for t in trades])) if trades else float("nan"),
    }


def print_comparison(rows: list[dict]) -> None:
    """
    Baseline vs learned variants over the same out-of-sample days. Each row
    is summary_stats() plus "name", "dsr" and optional "note".
    """
    print("=" * 96)
    print("  BASELINE vs SELF-LEARNING  (same pair-periods, same out-of-sample days)")
    print("=" * 96)
    print(f"  {'Variant':<24} {'Sharpe':>7} {'DSR':>6} {'Return':>8} {'Vol':>7} "
          f"{'MaxDD':>8} {'Trades':>7} {'Win%':>6}  Note")
    print(f"  {'─' * 92}")
    for r in rows:
        print(f"  {r['name']:<24} {r['sharpe']:>7.2f} {r['dsr']:>6.2f} {r['total_return']:>8.2%} "
              f"{r['ann_vol']:>7.2%} {r['max_dd']:>8.2%} {r['trades']:>7d} "
              f"{r['win_rate'] * 100:>5.1f}%  {r.get('note', '')}")
    print(f"  {'─' * 92}")
    print("  DSR = deflated Sharpe ratio: P(true Sharpe > 0) after correcting for every")
    print("  configuration tried (grid + variants + offline). > 0.95 would be evidence of skill.")
    print("=" * 96)
    print()
