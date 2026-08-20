# Stat-Arb Pairs Trading Engine

A realistic **portfolio** pairs trading system built the way quant research
desks do it: not one pair picked once, but a walk-forward schedule of pairs
that re-qualifies every quarter, sized by volatility, and combined into a
single book.

## What it does

1. Downloads price data from Yahoo Finance for a ticker universe
2. **Walk-forward**: rolls a cointegration re-test (Engle-Granger) forward in
   quarterly steps — at each step only the trailing `TRAIN_WINDOW` days are
   used to decide which pairs are tradeable for the next `TEST_WINDOW` days,
   so pair selection never peeks at its own trading period
3. For every pair-period: estimates a dynamic hedge ratio via Kalman filter,
   generates z-score entry/exit signals, and sizes the position to a target
   daily volatility instead of a flat $1 notional
4. Backtests every pair-period and combines them into one portfolio equity
   curve, tracking the correlation between pairs' daily PnL
5. Prints a portfolio report (Sharpe, drawdown, distinct pairs, trades,
   average pairwise correlation) and saves a 4-panel tearsheet PNG

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run (no prompts — everything is configured in main.py)
python main.py
```

## Project structure

```
stat_arb/
├── main.py                  # Entry point + config
├── requirements.txt
├── data/
│   └── fetch.py             # Price data download
├── signals/
│   ├── cointegration.py     # Engle-Granger pair finder
│   ├── kalman.py            # Dynamic hedge ratio (Kalman filter)
│   ├── zscore.py            # Rolling z-score signal
│   ├── walkforward.py       # Rolling pair-selection schedule (no look-ahead)
│   └── sizing.py            # Vol-targeted position sizing
├── backtest/
│   ├── engine.py            # Day-by-day simulation for one pair
│   ├── portfolio.py         # Runs every pair-period, combines into one book
│   ├── report.py            # Performance metrics (single-pair + portfolio)
│   └── tearsheet.py         # 4-panel PNG: equity, rolling Sharpe, drawdown, corr heatmap
└── outputs/                 # Generated tearsheets (gitignored)
```

## Key parameters (edit in main.py)

| Parameter          | Default    | Effect                                              |
|---------------------|-----------|------------------------------------------------------|
| `UNIVERSE`          | 15 stocks | Ticker universe walk-forward draws candidate pairs from |
| `TRAIN_WINDOW`      | 252 days  | History used to re-fit cointegration each rebalance   |
| `TEST_WINDOW`       | 63 days   | How long a qualifying pair stays tradeable (~1 quarter) |
| `LOOKBACK_WINDOW`   | 60 days   | Z-score rolling window                                |
| `ENTRY_THRESHOLD`   | 2.0σ      | Higher = fewer, higher-conviction trades              |
| `EXIT_THRESHOLD`    | 0.5σ      | Lower = hold longer, more mean reversion               |
| `PVALUE_CUTOFF`     | 0.05      | Cointegration significance level                       |
| `TARGET_DAILY_VOL`  | 1%        | Per-pair position sizing target (vol-targeting)         |

## What to learn / extend next

- ~~Walk-forward validation~~ — done (`signals/walkforward.py`)
- ~~Position sizing~~ — done, vol-targeted (`signals/sizing.py`); Kelly sizing
  would be a good follow-up variant to compare against
- ~~Portfolio of pairs~~ — done (`backtest/portfolio.py`); note the
  correlation heatmap correlates over each pair-period's full column
  (zero-padded outside its active window), so it reads best as "does this
  pair-period's PnL move with others active around the same time" rather
  than a strict pairwise stat
- **Stop losses** — exit if z-score blows past 3.5σ (pair broke down)
- **Execution layer** — port the signal generation to C++ for live trading
