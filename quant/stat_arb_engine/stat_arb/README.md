# Stat-Arb Pairs Trading Engine

A realistic **portfolio** pairs trading system built the way quant research
desks do it: not one pair picked once, but a walk-forward schedule of pairs
that re-qualifies every quarter, sized by volatility, and combined into a
single book.

## What it does

1. Downloads price data from Yahoo Finance for a ticker universe: by
   default the 11 SPDR sector ETFs plus ~25 large caps per sector (291
   names with full 2018-2023 history, `data/universe.py`), with candidate
   pairs restricted to the same sector
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
6. Runs the **self-learning layer** (meta-labeling, nested walk-forward
   tuning, Thompson allocation) next to the baseline on the same
   out-of-sample days, scored with the deflated Sharpe ratio

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
python oos_test.py              # pre-registered 2024-2026 test (see "Self-learning layer")
# Both write CSV tables for the results dashboard to outputs/dashboard/
python exit_test.py              # pre-registered exit rules on 2019-2023; writes forward_config.json
python forward_test.py           # forward holdout from 2026-11-02; shows no PnL before day 252

# 4. Tests (synthetic-data checks for the Kalman filter and the engine's accounting)
python tests/test_kalman.py
python tests/test_engine.py
python tests/test_learn.py      # no-look-ahead checks for the learning layer (~2-3 min)
```

On the wide universe, `python main.py` takes 15-30 minutes on 8 cores and
needs a few GB of RAM. The ~77,000 Engle-Granger tests (~4.5 min) and
~9,800 Kalman MLE fits (~1-2 min in a 7-process pool) are cached under
`cache/` (~250 MB), so later runs load them in under two minutes. The rest
is the learning layer: 26,000 candidate signals, two classifiers retrained
19 times, and ~17 full-book backtests at ~20 s each.

## Project structure

```
stat_arb/
├── main.py                  # Entry point + config
├── oos_test.py              # Pre-registered 2024-2026 out-of-sample test (x=0)
├── exit_test.py             # Pre-registered exit rules for trades that never revert
├── forward_test.py          # Forward holdout from 2026-11-02, blinded until day 252
├── requirements.txt
├── data/
│   ├── fetch.py             # Price data download (cached under cache/)
│   └── universe.py          # Wide universe: sector ETFs + large caps, sector/industry maps
├── signals/
│   ├── cointegration.py     # Engle-Granger pair finder
│   ├── kalman.py            # Dynamic hedge ratio (Kalman filter)
│   ├── zscore.py            # Rolling z-score signal
│   ├── walkforward.py       # Rolling pair-selection schedule (no look-ahead)
│   ├── sizing.py            # Vol-targeted position sizing
│   └── cache.py             # Disk cache + process pool for coint tests and Kalman fits
├── backtest/
│   ├── engine.py            # Day-by-day simulation for one pair
│   ├── portfolio.py         # Runs every pair-period, combines into one book
│   ├── report.py            # Performance metrics (single-pair + portfolio)
│   ├── export.py            # Results as CSV tables (variants, equity, exits, sectors, ...)
│   └── tearsheet.py         # 4-panel PNG + baseline-vs-learned comparison PNG
├── learn/                   # Self-learning layer (see below)
│   ├── features.py          # Features as of each signal's decision date
│   ├── trade_log.py         # Candidate signals, triple-barrier labels, SQLite/parquet log
│   ├── metalabel.py         # Walk-forward meta-labeler, purge + embargo, Kelly sizing
│   ├── tuning.py            # Nested walk-forward entry/exit/lookback choice
│   ├── allocation.py        # Thompson-sampling capital allocation
│   ├── evaluate.py          # Deflated Sharpe, OOS AUC, permutation importance
│   └── pipeline.py          # Runs every variant next to the baseline
├── tests/
│   ├── test_kalman.py       # Spread recovers a known synthetic OU spread
│   ├── test_engine.py       # PnL accounting, no look-ahead, force-close, stop
│   └── test_learn.py        # Features/labels/training/variants use no future data
├── outputs/                 # Generated tearsheets (gitignored)
└── cache/                   # Prices, coint tests, Kalman fits (gitignored; delete to recompute)
```

## Key parameters (edit in main.py)

| Parameter          | Default    | Effect                                              |
|---------------------|-----------|------------------------------------------------------|
| `UNIVERSE_NAME`     | `"wide"`  | `"wide"` = 11 sector ETFs + ~25 large caps per sector (291 names with data), pairs tested within a sector only; `"narrow"` = the original 15 stocks, every pair tested |
| `TRAIN_WINDOW`      | 252 days  | History used to re-fit cointegration each rebalance   |
| `TEST_WINDOW`       | 63 days   | How long a qualifying pair stays tradeable (~1 quarter) |
| `LOOKBACK_WINDOW`   | 60 days   | Z-score rolling window                                |
| `ENTRY_THRESHOLD`   | 2.0σ      | Higher = fewer, higher-conviction trades              |
| `EXIT_THRESHOLD`    | 0.5σ      | Lower = hold longer, more mean reversion               |
| `STOP_THRESHOLD`    | 3.5σ      | Stop-loss: exit if \|z\| blows past this (pair broke); no re-entry until \|z\| is back inside the entry band |
| `PVALUE_CUTOFF`     | 0.10      | FDR level under BH, family-wise level under Bonferroni, raw cutoff if no correction. Loose on purpose: the meta-labeler is the precision layer (see "Self-learning layer"). Use 0.05 + `"bh"` for the conservative stand-alone engine |
| `MULTIPLE_TESTING`  | `None`    | `"bh"` (Benjamini-Hochberg), `"bonferroni"`, or `None` (raw p-values) |
| `MIN_HALF_LIFE`     | 5 days    | Drop pairs reverting faster than this (usually noise)  |
| `MAX_HALF_LIFE`     | 60 days   | Drop pairs too slow to revert inside a 63-day window   |
| `SIGNAL`            | `"rolling"` | `"rolling"` z-score of the spread, or `"innovation"` = Kalman e/√S |
| `BOOK_DAILY_VOL`    | 0.8% (wide) | Book budget: each period's per-pair target is this / √(pairs active that period). `None` (narrow) uses `TARGET_DAILY_VOL` per pair |
| `N_JOBS`, `CACHE_DIR` | cores−1, `cache/` | Process pool and disk cache for the Engle-Granger tests and Kalman fits |
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
  data the 15-name universe produced 118 raw hits over 20 windows, against
  ~105 expected under the null, so most of what the old engine traded was
  noise. With BH, very few pair-periods survive. The wide universe has more
  to find, but not much more: 3,843 same-sector pairs per window give on
  average 520 raw hits at p<0.10, against 384 expected under the null. After
  BH at 10% and the half-life filter, only 5 of 20 windows have any pair,
  and one quarter holds 217 of the 246.

Before/after on identical 2018-2024 prices for the original 15-name universe (Sharpe is the all-days version;
returns are on book capital at the 0.25% per-pair vol target):

| Run | Pair-periods | Trades | Sharpe | Total return | Max DD | Exits (signal / stop / period end) |
|-----|-------------:|-------:|-------:|-------------:|-------:|-----------------|
| Old engine, as reported | 107 | 238 | 0.30 | n/a ($ units) | n/a | not logged |
| Old engine, all-days Sharpe | 107 | 238 | 0.17 | | | |
| Fixed engine, raw p<0.05 (no correction) | 118 | 171 | -0.36 | -21.3% | -43.7% | 100 / 27 / 44 |
| Fixed engine, raw p<0.05 + half-life | 113 | 164 | -0.38 | -22.6% | -45.0% | 97 / 27 / 40 |
| **Fixed engine, BH + half-life** (default before the learning layer) | 2 | 3 | 0.59 | +2.3% | -0.8% | 3 / 0 / 0 |

With unbiased accounting, the raw-selection book loses money: a 56% win rate
is outweighed by stop-outs and period-end losses. The BH run is positive but
rests on 3 trades, so it isn't evidence of anything yet.

## Self-learning layer

`learn/` adds a layer that learns from the engine's own past signals. It is
re-fit at every walk-forward rebalance and uses only information available
at that rebalance. `python main.py` runs it after the baseline (set
`SELF_LEARNING = False` to skip it). It prints a side-by-side table and
saves `outputs/comparison_*.png` and the trade log `outputs/trade_log.sqlite`.

### Design

**Meta-labeling** (López de Prado, *Advances in Financial Machine Learning*,
ch. 3). The z-score rule stays the *primary* model and picks the side of
every trade. A *secondary* classifier predicts P(trade profitable) and
decides whether to take the signal and how large to make it.

- **Candidate signals** (`learn/trade_log.py`): the first day of every
  excursion of z into the entry band, whether the engine takes it or not.
  These are exactly the days the engine can open a trade on. Every engine
  entry matches a candidate, and the code checks this.
- **Candidate set.** The meta-labeler needs labels. The screen is loose on
  purpose: raw p < 0.10, no multiple-testing correction, half-life filter
  kept. That is the textbook meta-labeling set-up, where a high-recall
  primary model is paired with a precision filter, and the coint p-value is
  a feature, so the classifier can still learn to distrust weak pairs. On
  the original 15 names this gave 213 pair-periods and only 514 labelled
  candidates. The default universe is now ~290 names (sector ETFs + large
  caps, same-sector pairs only), which gives 9,836 pair-periods and 26,095
  labelled candidates for the same 2019-2023 out-of-sample span.
- **Labels: triple barrier**, simulated exactly like the engine. Shares are
  frozen at entry, decisions fill at the next close, and costs are paid on
  both legs. The barriers are: profit-take when |z| < `EXIT_THRESHOLD`;
  stop when |z| > `STOP_THRESHOLD`; a time limit of `HOLD_MULTIPLE` x
  half-life, cut at the pair-period's end. The label is 1 if the net return
  is > 0. It agrees in sign with the engine's realised trade PnL on 96% of
  the trades the baseline takes.
- **Features** (`learn/features.py`), each computed from data up to the
  decision close only:

  | Feature | What it measures |
  |---|---|
  | \|z\| | signal strength at entry |
  | direction | long or short the spread |
  | half-life | OU fit on the selection window |
  | ADF stat, coint p-value | Engle-Granger test on the selection window |
  | Hurst exponent | of the Kalman spread, last 120 days |
  | Kalman beta stability | 20-day std of the hedge ratio |
  | spread vol percentile | current 20-day spread vol vs the past year |
  | leg correlation | 60-day correlation of the legs' returns |
  | SPY 20-day vol, SPY trend | market regime; trend is SPY vs its 50-day average |
  | days since first qualified | how long the pair has been passing the screen |
  | same-sector flag | both legs in the same group of the map passed in. On the wide universe every pair is already same-sector, so this is the finer *industry group* (banks vs insurers, E&P vs refiners; sector ETFs are their own group), 1 for 25% of candidates |
- **Training** (`learn/metalabel.py`): logistic regression (L2, C=0.1) and
  HistGradientBoosting (depth 2), each retrained at every rebalance T on an
  expanding window of past candidates.
  - *Purge*: a candidate is used only if its outcome resolved (t1) before T.
  - *Embargo*: candidates whose outcome resolved in the 5 trading days before
    T are also dropped.
  - *Cold start*: until 40 labels exist with at least 8 of each outcome, the
    model abstains and signals are taken at baseline size.
  - Labels are cut at the pair-period's end, which is always before the next
    rebalance, so the purge rarely binds. The test below shows that a real
    leak, training on the test quarter's own candidates, is caught.
- **Using the probability**: `gate` skips a signal if p < 0.5 or the Kelly
  fraction f\* = p − (1−p)/b ≤ 0, where b is the training set's average win
  over average loss. `kelly` applies the same gate and scales size by
  clip(f\*/0.25, 0, 2).
- **Nested walk-forward tuning** (`learn/tuning.py`): at each rebalance,
  (entry, exit, lookback) is chosen from {1.5, 2, 2.5} x {0, 0.5} x {40, 60}
  (exit 0 = no signal exit; hold to period end or stop).
  The pick is by the Sharpe each setting earned over the previous 252 days of
  its own out-of-sample trading.
- **Thompson allocation** (`learn/allocation.py`): each pair is an arm, and
  its reward is the PnL of its finished pair-periods. A Normal-Normal
  posterior with a skeptical zero-mean prior gives P(pair is best) among the
  active pairs. That probability sets its capital, shrunk halfway to equal
  weight and capped at 3x.

### Leakage tests

`tests/test_learn.py` perturbs every price after a date and asserts that
nothing decided before that date changes:

- features, compared at decision dates throughout a pair-period;
- triple-barrier labels, after their t1;
- the purge/embargo training mask;
- end to end on a synthetic universe: every variant's probabilities, size
  multipliers, tuned parameters and daily PnL before the cut.

I checked that the tests fail when they should. A feature that peeks one day
ahead fails the features test. Training on the test quarter's own candidates
fails the end-to-end test. The end-to-end test takes about 2-3 minutes.

### Results: wide universe (2019-2023 out-of-sample, same 9,836 pair-periods for every row)

Default config: 291 names (11 sector ETFs + large caps, `data/universe.py`),
same-sector pairs only, raw p < 0.10 screen + half-life filter. That gives
9,836 pair-periods across 3,485 distinct pairs, about 500 active per quarter
(195-1,090), and **26,095 labelled candidate signals** (vs 514 on the 15-name
universe), of which the baseline takes 13,852. The triple-barrier win rate
is 53.5%.

The DSR uses N = 76 configurations: the 7 variants and 12 grid points of
this run, plus 57 run before it (the two narrow-universe screens and the
wide BH sensitivity check below, 19 each; see `EXTRA_TRIALS` in `main.py`).
The expected best Sharpe of 76 skill-less trials is SR0 = 1.16
(annualised). It is this high because the grid's Sharpes are spread widely
(-0.27 to +0.99), which raises V[SR].

| Variant | Sharpe | DSR | Return | Vol | Max DD | Trades | Win % |
|---|---:|---:|---:|---:|---:|---:|---:|
| **baseline** | **-0.18** | 0.00 | -35.6% | 40.1% | -118.7% | 13,852 | 57.8% |
| meta-logit gate | -0.50 | 0.00 | -80.2% | 31.9% | -118.3% | 7,825 | 55.9% |
| meta-logit kelly | -0.50 | 0.00 | -104.8% | 42.4% | -137.3% | 7,825 | 55.9% |
| meta-hgb gate | -0.62 | 0.00 | -99.3% | 32.3% | -124.1% | 8,434 | 55.5% |
| meta-hgb kelly | -0.63 | 0.00 | -127.8% | 41.0% | -150.5% | 8,434 | 55.5% |
| tuned (nested WF) | **0.84** | 0.24 | +183.0% | 43.6% | -36.9% | 10,095 | 51.3% |
| thompson alloc | -0.17 | 0.00 | -35.4% | 41.3% | -108.9% | 13,852 | 57.8% |

Returns, vol and drawdown are on book capital, with a 0.8%/day book budget
split as 1/√(active pairs) per period. Realised vol is 40%, not the ~13% the
budget implies, because ~500 concurrent pairs share legs and sector factors.
The 1/√n split assumes independent pairs, and with n that large even small
positive correlations (the report shows an average of 0.00 to two decimals)
add up. Those three columns scale linearly with the budget.
Sharpe, DSR and win rate do not.

Out-of-sample classifier quality, on the 25,192 candidates scored after the
cold start:

| Model | AUC (pooled) | AUC (within quarter) | Brier | Brier of base rate |
|---|---:|---:|---:|---:|
| logit | 0.48 | 0.51 | 0.272 | 0.249 |
| hgb | 0.47 | 0.50 | 0.270 | 0.249 |

Calibration (OOS, quintiles of predicted p, logit; hgb looks the same):

| Mean predicted p | Realised win rate | Mean net return per $1 gross |
|---:|---:|---:|
| 0.44 | 56.9% | +0.63% |
| 0.52 | 54.0% | +0.13% |
| 0.55 | 54.9% | +0.06% |
| 0.60 | 54.8% | -0.14% |
| 0.79 | 49.0% | -0.96% |

The static grid, full-period Sharpe (ex post, for reference only; the
baseline is e=2 x=0.5 lb=60). The engine exits on |z| < exit, so **x = 0
means no signal exit at all**: every trade is held until its period ends or
the stop hits.

| exit \ (entry, lookback) | 1.5, 40 | 1.5, 60 | 2, 40 | 2, 60 | 2.5, 40 | 2.5, 60 |
|---|---:|---:|---:|---:|---:|---:|
| x = 0 | 0.99 | 0.60 | 0.20 | 0.63 | -0.11 | 0.15 |
| x = 0.5 | -0.13 | -0.27 | -0.15 | -0.18 | -0.13 | -0.24 |

Sharpe by year, baseline vs tuned: 2019 -0.98 / 0.47, 2020 -0.70 / 0.24,
2021 0.35 / 0.75, 2022 0.30 / 1.10, 2023 0.42 / 1.82.

**Sensitivity: BH at 10% on the wide universe.** With the false discovery
rate controlled across each window's 3,843 tests, only 246 pair-periods
survive, 217 of them in one quarter (2021-04). That leaves 585 candidates.
Baseline Sharpe is -0.25, every learned variant is between -0.27 and -0.33,
and every DSR is 0.00 (N = 76, SR0 = 0.96). A cleaner primary model doesn't
rescue the layer, but 585 labels concentrated in one quarter can't test the
meta-labeler properly either.

### Results: original 15-name universe (2019-2023, 213 pair-periods)

Kept for comparison. Raw p < 0.10 screen: 514 candidate signals, of which
the baseline takes 293, with a 52% triple-barrier win rate. The DSR here used
N = 21 (7 variants, 12 grid points, 2 offline screens) and SR0 = 0.42.
Set `UNIVERSE_NAME = "narrow"` in `main.py` to re-run it. In the current
environment (scikit-learn 1.9) every row reproduces exactly, except
meta-logit gate, which gives -0.47 / 238 trades. The original commit gives
the same -0.47 in this environment, so the difference comes from the
library version, not the code.

| Variant | Sharpe | DSR | Return | Vol | Max DD | Trades | Win % |
|---|---:|---:|---:|---:|---:|---:|---:|
| **baseline** | **-0.36** | 0.04 | -30.0% | 16.8% | -65.9% | 293 | 56.3% |
| meta-logit gate | -0.45 | 0.02 | -36.4% | 16.1% | -68.5% | 239 | 55.2% |
| meta-logit kelly | -0.44 | 0.02 | -55.9% | 25.7% | -98.5% | 239 | 55.2% |
| meta-hgb gate | -0.33 | 0.04 | -26.1% | 15.6% | -64.1% | 224 | 57.1% |
| meta-hgb kelly | -0.49 | 0.01 | -67.9% | 27.7% | -117.7% | 224 | 57.1% |
| tuned (nested WF) | -0.39 | 0.03 | -36.3% | 18.4% | -78.0% | 229 | 50.2% |
| thompson alloc | -0.38 | 0.03 | -31.9% | 17.0% | -66.4% | 293 | 56.3% |

OOS classifier quality on the 456 candidates scored after the cold start:
logit AUC 0.42 pooled / 0.56 within quarter, hgb 0.43 / 0.57, Brier
0.356 / 0.349 vs 0.250 for the base rate. The p < 0.05 sensitivity check
(113 pair-periods, 276 candidates) gave a baseline Sharpe of -0.38, learned
variants from -0.81 to -0.29, and every DSR ≤ 0.03.

### Takeaways

- **More labels did not help the meta-labeler. It got worse.** With 26,095
  labels instead of 514, both classifiers are below 0.5 pooled AUC and at
  0.50 within a quarter. The "faint ranking signal" the 15-name run showed
  (within-quarter AUC 0.56) was noise and is gone at 50x the sample. Gating
  skips about half of all signals and takes the Sharpe from -0.18 to
  -0.50/-0.62. Kelly sizing adds risk without adding return.
- **It is anti-calibrated, not just weak.** The top quintile by predicted
  p (0.79) has the lowest realised win rate (49%) and loses 0.96% per trade.
  The bottom quintile (p = 0.44) wins 57% of the time and makes +0.63%. The
  model is confidently wrong in exactly the trades it would size up.
- **The cause is the same: the base rate shifts from quarter to quarter.**
  The COVID quarter (2020-01) is still 16% of all candidates (4,139 of
  26,095, at a 47% win rate). The models, trained on 2019's 41-64%, predicted
  0.68-0.74 for it. Later quarters swing between 47% and 65%. No feature
  separates the outcomes enough to overcome that: every feature-label
  correlation is below 0.035 in absolute value, and OOS permutation
  importances are ≤ 0.003 AUC. The sample size was not the binding
  constraint. The features are.
- **Tuning is the one positive result, and it is still not significant.**
  Every grid point with x = 0 beats every point with x = 0.5, and the nested
  walk-forward switches to x = 0 by 2019-Q3 and keeps it in 18 of 20
  quarters. x = 0 is not "exit at the mean": the engine exits on
  |z| < exit, which never fires at 0, so these settings never take profit
  and hold every entry to period end or the stop (53-67 calendar days on
  average, vs 23 for the baseline). With x = 0.5 the profit-taking exits
  make money, but the trades that never revert lose more at period end and
  at the stop. Holding everything turns that around. The tuned Sharpe is
  0.84, positive in every year, against -0.18 for the baseline. But the DSR
  is 0.24: after 76 configurations, a 0.84 Sharpe over 5 years is well
  inside what the best of 76 skill-less trials could show (SR0 = 1.16). It
  also didn't carry over to the 15-name universe, where tuning lost. The
  pre-registered test on 2024-2026 data (below) did not confirm it either.
- **Thompson allocation is a no-op here** (-0.17 vs -0.18). With ~500
  active pairs a quarter, most of them new, there's almost no per-arm history
  to learn from.
- **The wider universe makes the baseline less bad, not good.** Sharpe went
  from -0.36 to -0.18 with a 57.8% win rate, but it still loses. Stops and
  period-end exits (5,089 of 13,852 trades) outweigh the signal exits.

### Out-of-sample test: no profit-taking exit, 2024-01 to 2026-09 (pre-registered)

The one in-sample result worth spending fresh data on was "x = 0 beats
x = 0.5". `oos_test.py` fixes the configurations, data window and verdict
rule in its docstring, and was committed before any 2024-2026 price was
downloaded. The setup is the wide default: same universe, screen and costs,
no learning layer. Pairs are selected on 2023 and trade 2024-01-04 to
2026-09-30: 11 quarters, 5,008 pair-periods, 312-718 pairs per quarter. XLC
has full history in this window, so it is in (292 names, 3,856 pairs per
window). There are 2 candidates, so the DSR uses N = 2.

| Config | Sharpe | DSR (N=2) | Return | Vol | Max DD | Trades | Win % |
|---|---:|---:|---:|---:|---:|---:|---:|
| **primary** e=1.5 x=0 lb=40 (the tuner's last choice) | **-0.44** | 0.22 | -52.8% | 44.5% | -111.6% | 5,631 | 48.4% |
| secondary e=2 x=0 lb=60 (baseline, exit changed only) | -0.32 | 0.27 | -30.5% | 34.6% | -88.4% | 5,029 | 48.4% |
| reference e=2 x=0.5 lb=60 (baseline) | -0.34 | 0.26 | -28.2% | 30.0% | -62.4% | 7,109 | 60.5% |

Sharpe by year (2024 / 2025 / 2026 to September): primary -1.06 / 0.95 /
-1.73, secondary -0.54 / 1.14 / -1.86, reference -1.72 / 0.40 / -0.08.

The exit change on its own (daily PnL of secondary minus reference) has a
Sharpe of -0.03 and t = -0.05 over 687 days. That is no effect at all.

**Verdict, by the pre-registered rule: NOT CONFIRMED.** The primary lost
money and did worse than the baseline. The in-sample 0.84 was selection
plus one regime, which is what its 0.24 DSR said. Every configuration in
this README now loses money out of sample. The 2 trials are added to
`EXTRA_TRIALS`.

### Next steps (not done)

- **Normalise the base rate out of the labels**: predict whether a signal
  beats its own quarter's median, or gate on within-quarter rank instead of
  an absolute p. Sample-uniqueness weights (AFML ch. 4) would also stop one
  crowded quarter from dominating training. Each of these counts as a new
  configuration for the DSR.
- **Better features before more data.** Candidates: the spread's recent
  reversion speed vs its training half-life, sector-ETF residual momentum,
  earnings-date proximity for either leg, and short interest.
- **Point-in-time universe.** Use historical index membership, so the
  backtest also trades the names that were later acquired or delisted.
- **Correlation-aware risk budget.** The 1/√n split assumes independent
  pairs and runs 3x its vol target. Sizing to a covariance estimate of the
  pair PnLs (or capping each leg's total exposure) would fix that.

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
- ~~Bigger universe~~ — done (`data/universe.py`): sector ETFs + ~25 large
  caps per sector, pairs within a sector. It is survivorship-biased (names
  picked with hindsight; 8 that were since acquired or delisted have no
  Yahoo data); a point-in-time constituent list would fix that
- ~~Meta-labeling~~ — done (`learn/`). It does not help, on either universe;
  see "Self-learning layer" for why and what to try next
- **Execution layer** — port the signal generation to C++ for live trading
