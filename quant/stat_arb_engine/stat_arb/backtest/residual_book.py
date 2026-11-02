"""
backtest/residual_book.py
-------------------------
Daily book for the sector-ETF residual model.

Timing is backtest/engine.py's: every decision uses scores up to the close of
t-1 and executes at the close of t, so a closing-auction order placed the night
before could do the same thing.

Rules (Avellaneda & Lee 2010, except the sizing and the one-day lag):
  open long     s < -1.25   long the stock, short beta of its ETF
  open short    s > +1.25   short the stock, long beta of its ETF
  close long    s > -0.50
  close short   s < +0.75
  close either  the stock is rejected by the fit that day (s is NaN)

Sizing, so the book fits an account limited to 2x gross overnight:
  - each stock leg is 5% of capital, with at most 20 positions
  - with more signals than slots, the largest |s| goes first, ties by ticker
  - no entry may take stock gross plus each sector's net ETF gross above 2.0x
  - an entry that doesn't fit is skipped, not queued

The hedge is frozen at entry, so a position trades only when it opens and closes.
ETF orders are netted within a sector on each close, because that's the order
that would reach the market. Capital is fixed and PnL is a return on it, as in
the engine. Decisions never depend on PnL, so costs can't change the trades.

Trade.pnl covers the stock leg, its share of the hedge's price moves, and the
stock's own costs. ETF costs and ETF borrow depend on the net book, so they are
charged to daily_pnl only.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backtest.engine import COST_BPS, Trade

MAX_POSITIONS = 20
STOCK_LEG     = 0.05    # of capital
MAX_GROSS     = 2.0     # stock gross + net ETF gross, as a multiple of capital
OPEN_S        = 1.25
CLOSE_LONG    = -0.50
CLOSE_SHORT   = 0.75


@dataclass
class _Open:
    stock     : str
    etf       : str
    side      : int      # +1 long the stock, -1 short it
    shares    : float
    etf_shares: float
    entry_t   : int
    entry_s   : float
    beta      : float
    pnl       : float


@dataclass
class ResidualResults:
    equity_curve: pd.Series
    daily_pnl   : pd.Series
    per_pair    : dict = field(default_factory=dict)   # (stock, etf, first entry) -> .trades
    gross       : pd.Series = None      # overnight gross / capital
    short       : pd.Series = None      # overnight short notional / capital
    positions   : pd.Series = None      # open positions at each close
    stock_traded: pd.Series = None      # notional traded / capital
    etf_traded  : pd.Series = None      # after netting when net_etf=True
    blocked     : pd.Series = None      # entries refused by slots or the gross cap


def run_residual_book(
    stocks       : pd.DataFrame,
    etfs         : pd.DataFrame,
    etf_of       : dict[str, str],
    scores       : pd.DataFrame,
    betas        : pd.DataFrame,
    start        : str | pd.Timestamp | None = None,
    cost_bps     : float = COST_BPS,
    borrow_fee   : float = 0.0,
    net_etf      : bool  = True,
    max_positions: int   = MAX_POSITIONS,
    stock_leg    : float = STOCK_LEG,
    max_gross    : float = MAX_GROSS,
) -> ResidualResults:
    """
    Run the book on one sample.

    Args:
        stocks, etfs   : closing prices on the same dates
        etf_of         : stock -> its sector ETF
        scores, betas  : from signals.residual.residual_scores
        start          : first trading date; earlier rows are warm-up only
        cost_bps       : one-way cost per leg, as a fraction of notional
        borrow_fee     : annual fee on short stock and net short ETF notional
        net_etf        : charge ETF orders after same-day netting in a sector
    """
    dates = stocks.index
    names = list(stocks.columns)
    col = {name: i for i, name in enumerate(names)}
    p, e = stocks.to_numpy(), {etf: etfs[etf].to_numpy() for etf in set(etf_of.values())}
    s_all, b_all = scores[names].to_numpy(), betas[names].to_numpy()

    n = len(dates)
    first = max(1, int(dates.searchsorted(pd.Timestamp(start)))) if start is not None else 1
    days = n - first
    pnl, gross, short = np.zeros(days), np.zeros(days), np.zeros(days)
    count, stock_tr, etf_tr, blocked = np.zeros(days), np.zeros(days), np.zeros(days), np.zeros(days)

    book: dict[str, _Open] = {}
    held = {etf: 0.0 for etf in e}          # net ETF shares
    trades: dict[str, list[Trade]] = {}
    first_entry: dict[str, pd.Timestamp] = {}

    for t in range(first, n):
        i = t - first
        last_day = t == n - 1

        # ── Mark what was held over (t-1, t] ─────────────────────────────────
        for pos in book.values():
            a = p[t, col[pos.stock]]
            a0 = p[t - 1, col[pos.stock]]
            day = pos.shares * (a - a0) + pos.etf_shares * (e[pos.etf][t] - e[pos.etf][t - 1])
            day -= borrow_fee / 252 * max(-pos.shares, 0.0) * a0
            pos.pnl += day
            pnl[i] += day
        for etf, sh in held.items():
            if sh < 0:
                pnl[i] -= borrow_fee / 252 * -sh * e[etf][t - 1]

        delta = {etf: 0.0 for etf in e}     # signed ETF shares ordered at this close
        ordered = {etf: 0.0 for etf in e}   # the same, before netting

        # ── Exits, decided on yesterday's close ──────────────────────────────
        closed = set()
        for name in sorted(book):
            pos = book[name]
            s = s_all[t - 1, col[name]]
            reason = None
            if np.isnan(s):
                reason = "model"
            elif (pos.side > 0 and s > CLOSE_LONG) or (pos.side < 0 and s < CLOSE_SHORT):
                reason = "signal"
            elif last_day:
                reason = "period_end"
            if reason is None:
                continue

            value = abs(pos.shares) * p[t, col[name]]
            pos.pnl -= cost_bps * value
            pnl[i] -= cost_bps * value
            stock_tr[i] += value
            delta[pos.etf] -= pos.etf_shares
            ordered[pos.etf] += abs(pos.etf_shares)
            held[pos.etf] -= pos.etf_shares
            trades.setdefault(name, []).append(Trade(
                entry_date   = dates[pos.entry_t],
                exit_date    = dates[t],
                direction    = pos.side,
                entry_z      = pos.entry_s,
                exit_z       = float(s),
                pnl          = pos.pnl,
                exit_reason  = reason,
                hedge_ratio  = pos.beta,
                gross        = stock_leg * (1.0 + abs(pos.beta)),
                holding_days = t - pos.entry_t,
            ))
            del book[name]
            closed.add(name)

        # ── Entries, decided on yesterday's close ────────────────────────────
        if not last_day:
            s_prev = s_all[t - 1]
            hits = [j for j in np.flatnonzero(np.abs(np.nan_to_num(s_prev)) > OPEN_S)
                    if names[j] not in book and names[j] not in closed]
            for j in sorted(hits, key=lambda j: (-abs(s_prev[j]), names[j])):
                name, etf = names[j], etf_of[names[j]]
                beta = b_all[t - 1, j]
                if len(book) >= max_positions or not np.isfinite(beta) or np.isnan(e[etf][t]):
                    blocked[i] += 1
                    continue
                side = -1 if s_prev[j] > 0 else 1
                shares = side * stock_leg / p[t, j]
                etf_shares = -side * beta * stock_leg / e[etf][t]

                after = sum(abs(x.shares) * p[t, col[x.stock]] for x in book.values()) + stock_leg
                after += sum(abs(sh + (etf_shares if k == etf else 0.0)) * e[k][t]
                             for k, sh in held.items() if sh != 0.0 or k == etf)
                if after > max_gross:
                    blocked[i] += 1
                    continue

                pnl[i] -= cost_bps * stock_leg
                stock_tr[i] += stock_leg
                delta[etf] += etf_shares
                ordered[etf] += abs(etf_shares)
                held[etf] += etf_shares
                book[name] = _Open(name, etf, side, shares, etf_shares, t, float(s_prev[j]),
                                   float(beta), -cost_bps * stock_leg)
                first_entry.setdefault(name, dates[t])

        # ── ETF orders and the closing book ──────────────────────────────────
        for etf in e:
            value = (abs(delta[etf]) if net_etf else ordered[etf]) * e[etf][t]
            if value > 0:
                pnl[i] -= cost_bps * value
                etf_tr[i] += value

        for pos in book.values():
            value = abs(pos.shares) * p[t, col[pos.stock]]
            gross[i] += value
            short[i] += value if pos.shares < 0 else 0.0
        for etf, sh in held.items():
            if sh != 0.0:
                gross[i] += abs(sh) * e[etf][t]
                short[i] += -sh * e[etf][t] if sh < 0 else 0.0
        count[i] = len(book)

    index = dates[first:]
    daily = pd.Series(pnl, index=index, name="daily_pnl")
    series = lambda x, name: pd.Series(x, index=index, name=name)
    return ResidualResults(
        equity_curve = daily.cumsum().rename("equity"),
        daily_pnl    = daily,
        per_pair     = {(name, etf_of[name], first_entry[name]): SimpleNamespace(trades=ts)
                        for name, ts in trades.items()},
        gross        = series(gross, "gross"),
        short        = series(short, "short"),
        positions    = series(count, "positions"),
        stock_traded = series(stock_tr, "stock_traded"),
        etf_traded   = series(etf_tr, "etf_traded"),
        blocked      = series(blocked, "blocked"),
    )
