# Stat-Arb Pairs Trading Engine

A realistic **portfolio** pairs trading system built the way quant research
desks do it: not one pair picked once, but a walk-forward schedule of pairs
that re-qualifies every quarter, sized by volatility, and combined into a
single book.

## What it does

1. Downloads price data from Yahoo Finance for a ticker universe
2. **Walk-forward**: rolls a cointegration re-test (Engle-Granger, on log
   prices) forward in quarterly steps — at each step only the trailing
   `TRAIN_WINDOW` days are used to decide which pairs are tradeable for the
   next `TEST_WINDOW` days, so pair selection never peeks at its own trading
   period. A multiple-testing correction (Benjamini-Hochberg by default) and
   a spread half-life filter decide which pairs survive.
3. For every pair-period: estimates a dynamic **log-space** hedge ratio +
   intercept via Kalman filter (noise fit by maximum likelihood on the
   selection window), generates z-score entry/exit/stop signals, and sizes
   the position as gross notional / capital to hit a target daily volatility
4. Backtests every pair-period — share counts frozen at entry, decisions on
   yesterday's close executed at today's close, costs on entry and exit,
   positions force-closed at period end — and combines them into one
   portfolio of **returns on book capital**
5. Prints a portfolio report (return, vol, all-days Sharpe, drawdown, exit
   reasons, win rate, pairwise correlation) and saves a 4-panel tearsheet PNG

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

# 4. Tests (synthetic-data checks for the Kalman filter and the engine's accounting)
python tests/test_kalman.py
python tests/test_engine.py
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
├── tests/
│   ├── test_kalman.py       # Spread recovers a known synthetic OU spread
│   └── test_engine.py       # PnL accounting, no look-ahead, force-close, stop
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
| `STOP_THRESHOLD`    | 3.5σ      | Stop-loss: exit if \|z\| blows past this (pair broke); no re-entry until \|z\| is back inside the entry band |
| `PVALUE_CUTOFF`     | 0.05      | FDR level under BH, family-wise level under Bonferroni, raw cutoff if no correction |
| `MULTIPLE_TESTING`  | `"bh"`    | `"bh"` (Benjamini-Hochberg), `"bonferroni"`, or `None` (raw p-values) |
| `MIN_HALF_LIFE`     | 5 days    | Drop pairs reverting faster than this (usually noise)  |
| `MAX_HALF_LIFE`     | 60 days   | Drop pairs too slow to revert inside a 63-day window   |
| `SIGNAL`            | `"rolling"` | `"rolling"` z-score of the spread, or `"innovation"` = Kalman e/√S |
| `TARGET_DAILY_VOL`  | 0.25%     | Per-pair daily PnL vol target, as a fraction of book capital (gross leverage capped at 3x). Was 1% when PnL was in arbitrary units; at 1% of real capital, ~6 pairs on at once ran ~48% annual vol |

Kalman settings live in `signals/kalman.py`: noise is estimated per
pair-period by MLE, with `min_state_memory` (default 500 days) capping how
fast the hedge may adapt. See the module docstring for why that cap matters.

## Measurement notes (read before trusting a number)

These were all bugs in earlier versions. Each one inflated or scrambled the
backtest:

- **Kalman spread was one-day noise.** In raw prices with Q=1e-4, R=1e-3 the
  filter's gain on ~$100 stocks was ~1, so beta re-fit itself to every day's
  price and the "spread" was really the daily change: autocorrelation -0.02,
  versus 0.96 for the true spread on a synthetic pair. Now the filter runs in
  log prices with an intercept state, and the spread is the one-step-ahead
  residual. `tests/test_kalman.py` checks recovery (median corr > 0.9).
- **Sizing was always at the cap.** On that noisy spread, vol-target size sat
  at the 3.0 cap on 100% of days. Sizing now targets the vol of the hedged
  *return* and outputs gross notional as a fraction of capital.
- **Look-ahead in PnL.** Daily PnL used `beta[t]`, which had already been
  updated with day-t prices. Share counts are now frozen at entry using the
  prior close's hedge.
- **Entry/exit timing mismatch.** Entries acted on `z[t-1]`, exits on `z[t]`
  at the same close. Both now act on `z[t-1]` and fill at the close of t.
- **Costs and open positions.** The exit cost wasn't in `Trade.pnl`, and
  positions still open at `period_end` vanished from the log. Both fixed.
  Every PnL day now belongs to exactly one trade.
- **Sharpe skipped flat days.** Dropping zero-PnL days inflates Sharpe by
  ~1/√(fraction of days in market). It now uses every day, matching the
  tearsheet's rolling Sharpe.
- **Pair selection was mostly false positives.** 105 pairs tested at p<0.05
  every quarter gives ~5 hits per window from chance alone. On 2018-2024
  data this universe produced 118 raw hits over 20 windows, against ~105
  expected under the null, so most of what the old engine traded was noise.
  With BH, very few pair-periods survive. That's the honest answer for this
  universe, and the next lever is a bigger or better-targeted universe.

Before/after on identical 2018-2024 prices (Sharpe is the all-days version;
returns are on book capital at the 0.25% per-pair vol target):

| Run | Pair-periods | Trades | Sharpe | Total return | Max DD | Exits (signal / stop / period end) |
|-----|-------------:|-------:|-------:|-------------:|-------:|-----------------|
| Old engine, as reported | 107 | 238 | 0.30 | n/a ($ units) | n/a | not logged |
| Old engine, all-days Sharpe | 107 | 238 | 0.17 | | | |
| Fixed engine, raw p<0.05 (no correction) | 118 | 171 | -0.36 | -21.3% | -43.7% | 100 / 27 / 44 |
| Fixed engine, raw p<0.05 + half-life | 113 | 164 | -0.38 | -22.6% | -45.0% | 97 / 27 / 40 |
| **Fixed engine, BH + half-life (default)** | 2 | 3 | 0.59 | +2.3% | -0.8% | 3 / 0 / 0 |

With unbiased accounting, the raw-selection book loses money: a 56% win rate
is outweighed by stop-outs and period-end losses. The default BH run is
positive but rests on 3 trades, so it isn't evidence of anything yet.

## What to learn / extend next

- ~~Walk-forward validation~~ — done (`signals/walkforward.py`)
- ~~Position sizing~~ — done, vol-targeted (`signals/sizing.py`); Kelly sizing
  would be a good follow-up variant to compare against
- ~~Portfolio of pairs~~ — done (`backtest/portfolio.py`); note the
  correlation heatmap correlates over each pair-period's full column
  (zero-padded outside its active window), so it reads best as "does this
  pair-period's PnL move with others active around the same time" rather
  than a strict pairwise stat
- ~~Stop losses~~ — done (`STOP_THRESHOLD`, exits logged as `exit_reason="stop"`)
- **Bigger universe** — with honest multiple-testing control, 15 tickers
  rarely yield a significant pair; sector ETFs + constituents, or a few
  hundred names, would give the selection step something to find
- **Meta-labeling** — `Trade` now records entry z, hedge ratio, gross size,
  holding days and exit reason: the labels/features a learning layer needs
- **Execution layer** — port the signal generation to C++ for live trading
