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

# 4. Tests (synthetic-data checks for the Kalman filter and the engine's accounting)
python tests/test_kalman.py
python tests/test_engine.py
python tests/test_learn.py      # no-look-ahead checks for the learning layer (~2-3 min)
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
| `PVALUE_CUTOFF`     | 0.10      | FDR level under BH, family-wise level under Bonferroni, raw cutoff if no correction. Loose on purpose: the meta-labeler is the precision layer (see "Self-learning layer"). Use 0.05 + `"bh"` for the conservative stand-alone engine |
| `MULTIPLE_TESTING`  | `None`    | `"bh"` (Benjamini-Hochberg), `"bonferroni"`, or `None` (raw p-values) |
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
- **Candidate set.** The meta-labeler needs labels, and BH at 5% leaves 3
  trades in 2019-2023. The two options were a looser screen on the same
  universe or a wider universe. I loosened the screen: raw p < 0.10, no
  multiple-testing correction, half-life filter kept. That gives 213
  pair-periods and 514 labelled candidates. It keeps the universe and data
  identical to the foundation-fix numbers, so the comparison stays like for
  like. It is also the textbook meta-labeling set-up, where a high-recall
  primary model is paired with a precision filter. The coint p-value is a
  feature, so the classifier can still learn to distrust weak pairs. Widening
  the universe is still the better long-run fix (see below).
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
  | same-sector flag | both legs in the same sector |
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
  (entry, exit, lookback) is chosen from {1.5, 2, 2.5} x {0, 0.5} x {40, 60}.
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

### Results (2019-2023 out-of-sample, same 213 pair-periods for every row)

Default config (raw p < 0.10 screen): 514 candidate signals, of which the
baseline takes 293. The triple-barrier win rate is 52%. The DSR uses N = 21
configurations: the 7 variants, the 12-point grid, and 2 offline candidate
screens. The baseline equals one grid point and is counted twice, which is
slightly conservative. The expected best Sharpe of 21 skill-less trials is
SR0 = 0.42 (annualised).

| Variant | Sharpe | DSR | Return | Vol | Max DD | Trades | Win % |
|---|---:|---:|---:|---:|---:|---:|---:|
| **baseline** | **-0.36** | 0.04 | -30.0% | 16.8% | -65.9% | 293 | 56.3% |
| meta-logit gate | -0.45 | 0.02 | -36.4% | 16.1% | -68.5% | 239 | 55.2% |
| meta-logit kelly | -0.44 | 0.02 | -55.9% | 25.7% | -98.5% | 239 | 55.2% |
| meta-hgb gate | -0.33 | 0.04 | -26.1% | 15.6% | -64.1% | 224 | 57.1% |
| meta-hgb kelly | -0.49 | 0.01 | -67.9% | 27.7% | -117.7% | 224 | 57.1% |
| tuned (nested WF) | -0.39 | 0.03 | -36.3% | 18.4% | -78.0% | 229 | 50.2% |
| thompson alloc | -0.38 | 0.03 | -31.9% | 17.0% | -66.4% | 293 | 56.3% |

Out-of-sample classifier quality, on the 456 candidates scored after the
cold start:

| Model | AUC (pooled) | AUC (within quarter) | Brier | Brier of base rate |
|---|---:|---:|---:|---:|
| logit | 0.42 | 0.56 | 0.356 | 0.250 |
| hgb | 0.43 | 0.57 | 0.349 | 0.250 |

Sensitivity check with the p < 0.05 screen (113 pair-periods, 276
candidates; the baseline matches the measurement table above). The baseline
Sharpe is -0.38. Learned variants range from -0.81 to -0.29: meta-logit gate
-0.34, Thompson -0.29, every other variant worse than the baseline, and every
DSR ≤ 0.03.

### Takeaways

- **The learning layer does not rescue this book.** No variant has a
  positive Sharpe, and no DSR is above 0.05. The best learned rows (meta-hgb
  gate here, Thompson and meta-logit gate in the p<0.05 check) beat the
  baseline by less than the spread between grid points, and the winner
  differs between the two screens. That is noise, not skill.
- **Kelly sizing makes it worse.** The classifier is most confident exactly
  when it is most wrong, so scaling size up with p roughly doubles drawdowns.
- **Why it fails: regime shift in the base rate.** The 2020-01 quarter
  (COVID) holds 145 of the 514 candidates with a 33% win rate. The models had
  been trained on 2019's 66-94% win rates and predicted about 0.89. After
  that, the cluster dominates the training set. Feature-label correlations
  also flip sign between 2019-21 and 2021-23: the coint p-value goes from
  -0.03 to +0.21 and same-sector from 0.00 to -0.28. Pooled AUC is below 0.5
  and Brier is far worse than the base rate.
- **There is a faint ranking signal.** Within a quarter, AUC is about 0.56,
  above 0.5 in 11 of 16 quarters. The level of p is unusable, but the
  ordering isn't pure noise. The features that carry it out of sample are
  coint p-value, leg correlation and days since first qualified.
- **Nested tuning picks a non-default setting in 17 of 20 quarters and
  loses.** Trailing-year Sharpe doesn't persist into the next quarter here.

### Next steps (not done)

- **Widen the universe** (sector ETFs + constituents, a few hundred names),
  so that the primary model has real cointegration to find and the
  meta-labeler gets thousands of labels instead of hundreds.
- **Sample-uniqueness weights** (AFML ch. 4), so that one crowded quarter
  of overlapping, near-duplicate labels doesn't dominate training.
- **Rank-based gating**: skip the bottom fraction of each quarter's
  candidates by p, instead of using an absolute threshold. This uses the
  within-quarter ranking and ignores the unstable level. It counts as a new
  configuration for the DSR.

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
- ~~Meta-labeling~~ — done (`learn/`). It does not help on this universe yet;
  see "Self-learning layer" for why and what to try next
- **Execution layer** — port the signal generation to C++ for live trading
