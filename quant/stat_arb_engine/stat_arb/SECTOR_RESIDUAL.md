# Sector-ETF residual model: design

Status: pre-registered 2026-10-31. Committed before any signal or PnL from this
model ran on real prices. `residual_test.py` carries the rules that bind the run.

## Question

Does a stock's move against its own sector ETF revert strongly enough to clear a
net Sharpe of 1.0 at realistic costs, in every sample, at a size a $25,000 margin
account could hold?

Pairs earn only 5.5bp to 8bp per leg before costs. This changes the signal. It
trades one stock against a cheap, diversified hedge, with no pair to select, so
the selection stage that kept failing goes away.

## Expect it to fail

The paper is Avellaneda and Lee (2010). Net of 5bp per trade, its ETF strategy
had a Sharpe of 1.1 over 1997-2007, with a degradation after 2002. A later student
replication stagnated after about 2012. I found no solid net-of-cost replication
for the 2010s or 2020s. This design also departs from the paper in ways that mostly
hurt, so a failure wouldn't be surprising. It is still the fairest test, because
every rule below is taken from the paper or fixed before any run.

## The model

Taken from the paper unless marked as a change.

**Each day, for each stock, on the last 60 daily returns:**

1. Regress the stock's return on its sector ETF's return: `R = beta0 + beta * R_etf + e`.
2. Build the cumulative residual `X_k = e_1 + ... + e_k`.
3. Fit `X_(n+1) = a + b * X_n + z`.
4. Compute `kappa = -ln(b) * 252`, and reject the stock unless `0 < b` and
   `kappa > 252/30 = 8.4`, meaning reversion within about 1.5 months.
5. Compute `m = a / (1 - b)` and centre it: `m_c = m - mean(m)` across stocks that day.
6. Compute `sigma_eq = sqrt(Var(z) / (1 - b^2))` and the score `s = -m_c / sigma_eq`.

**Rules:**

| Event | Rule |
|---|---|
| Open long | `s < -1.25`: long the stock, short `beta` of the ETF |
| Open short | `s > +1.25`: short the stock, long `beta` of the ETF |
| Close long | `s > -0.50` |
| Close short | `s < +0.75` |
| Close either | `kappa` falls to 8.4 or below |

There is no stop, no time limit and no earnings filter, because the paper has none.

## Changes from the paper

| Change | Why | Likely effect |
|---|---|---|
| Decide on the close of t-1, fill at the close of t | The paper fills at the close the signal uses, which you can't know in time. A one-day lag makes a closing-auction order possible and matches `backtest/engine.py` | Hurts, and may be the largest loss |
| About 230-280 large caps, chosen with hindsight | It's the universe available | Flatters the result; see Limits |
| No more than 20 positions, 5% of capital in each stock | The paper runs 2+2 leverage, which doesn't fit 2x | Reduces diversification |
| Hedge frozen at entry | Matches the engine's "no silent rebalancing" | Small hedge drift |
| No volume-adjusted signal | That would be a second configuration | Gives up the paper's 1.51 variant |

## Sizing

This is the risk-sizing fix, built in from the start.

- **Stock leg:** 5% of capital, so stocks total at most 1x gross with 20 positions.
- **ETF hedge:** `beta` × the stock leg, netted across each sector.
- **Hard cap:** no entry may take total overnight gross above 2.0x. A blocked entry
  is skipped and is not queued.
- **More signals than slots:** take the largest `|s|` first, with ties in ticker order.

On $25,000 a stock leg is $1,250, about 12 shares at the 2023 median price of $100.
Estimated commission on that leg is about 2.8bp on IBKR Pro Tiered and about 0.5bp
at IBKR Lite's on-close rate, which is still unconfirmed. Both are below the 5bp
used here. 5% and 20 are single values set by the account and the 2x limit, not
searched.

## Data

| Sample | Period | Stocks | Sectors |
|---|---|---|---|
| Past | 2010-2017, warm-up from 2009 | 229 | 9: no real estate or communication |
| Recent | 2019-2023, warm-up from 2018 | 281 | 11 |
| Out-of-sample | 2024-01 to 2026-10-30, warm-up from 2023 | About 281, set by the same rule | 11 |

- XLRE starts 2015-10 and XLC 2018-06, so those two sectors can't run through the
  past sample. Both are fetched separately in the recent sample, because the 1%
  missing-data rule would otherwise drop them.
- Sector membership is today's map in `data/universe.py`.
- **Nothing from 2026-11-02 onward is read.** The existing holdout and its
  `forward_test.py` are not touched.

## Costs and trials

| Level | Per leg | Borrow | Use |
|---|---:|---:|---|
| FREE | 0 | 0 | Gross PnL and break-even |
| CURRENT | 2bp | 0 | Comparison with earlier tests |
| REALISTIC | 5bp | 0.50%/yr | **The bar** |
| HARSH | 10bp | 1.00%/yr | Margin for error |

- Stock orders are charged on every order. ETF orders are charged after same-day
  netting within a sector, and the result before netting is also reported.
- Borrow applies to short stocks and short ETF hedges.
- Trials: 1, added to `EXTRA_TRIALS`, so N = 91. One configuration, nothing searched.

## Decision rule

**Stage 1, one run on both samples.** Pass needs REALISTIC Sharpe >= 1.0 in the past
sample **and** in the recent sample.

**Stage 2, only after a Stage 1 pass.** One run on 2024-01 to 2026-10-30. Pass needs
REALISTIC Sharpe >= 1.0. Pairs were tested on this period, but this model hasn't been.

| Outcome | What happens |
|---|---|
| Stage 1 fails | **ADVANCE: NO.** Stage 2 never runs. Daily price-only reversion on these names is set aside |
| Stage 1 passes, Stage 2 fails | **NOT CONFIRMED.** Nothing is frozen |
| Both pass | **CANDIDATE.** It needs a point-in-time universe check, then its own frozen forward test of 252 days |

**No rescue runs.** If this fails, its thresholds, lag and slots stay as they are.
Any other variant would be a new pre-registered trial.

**Reported with no verdict attached:** DSR at N = 91 on the recent sample and N = 1
on the others, t against DISTANCE20 on 2019-2023, Sharpe by year, break-even cost,
turnover, average and peak gross, average positions, days with blocked entries, and
PnL split between long-stock and short-stock trades, and trades open across a
one-day move of 0.25 or more in log price, because Yahoo doesn't adjust every
spin-off.

## Limits

- **Survivorship matters more here than for pairs.** Buying a stock after it falls
  behind its sector works well when the list holds only companies that survived.
  The past sample is the most exposed. The long/short split shows how much of the
  result comes from the long side. Even a full pass isn't believable without a
  point-in-time universe.
- **Self-membership:** large stocks are a big part of their own ETF, so their
  residuals are understated.
- **Sector map:** today's map is applied to the past.
- **Concentration:** 20 positions at 5% is far fewer than the paper's roughly 200,
  and a single gap can move the book by more.
- **Whole shares:** a $25,000 account can't hold exact sizes, and the backtest
  doesn't round.
- **Margin:** the 2x limit is applied to gross overnight exposure, and the broker's
  actual requirement may differ.
- **The bar:** 1.0 is a judgment, set in advance and not derived.

## Build

- `signals/residual.py`: the daily fit and scores
- `backtest/residual_book.py`: the book, costs and gross cap
- `residual_test.py`: the binding rules and the two stages
- `tests/test_residual.py`: synthetic prices only

The tests check the paper's formulas by hand, that a price at t changes no decision
before t+1, the slot and cap rules, ETF netting and borrow, that costs leave the
trades alone, and that a planted reversion passes the bar while noise doesn't.

Stage 1 runs once, after this is committed.

## Sources

- [Avellaneda and Lee, Statistical arbitrage in the US equities market](https://www.math.nyu.edu/faculty/avellane/AvellanedaLeeStatArb20090616.pdf): the formulas, the thresholds, `kappa > 8.4`, the 60-day window, 5bp per trade, and fills at the same close
- [Stanford MS&E 448 student report](https://web.stanford.edu/class/msande448/2017/Final/Reports/gr5.pdf), read only from a search summary
- The fee estimates and the 13.8x peak gross come from desk analysis of DISTANCE20's trades, in a draft that is not part of this pre-registration
