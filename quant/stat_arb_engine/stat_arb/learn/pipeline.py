"""
learn/pipeline.py
-----------------
Run every learned variant next to the baseline on the same prepared
pair-periods and the same out-of-sample days, and score them.

  baseline            the z-score engine as configured in main.py
  meta-<clf> gate     meta-labeler skips signals with p < p_min or Kelly <= 0
  meta-<clf> kelly    same gate, size scaled by capped Kelly from p
  tuned (nested WF)   entry / exit / lookback re-chosen each quarter
  thompson alloc      per-pair capital from Thompson sampling on past OOS PnL

Every learned variant's decisions at rebalance T depend only on data before
T (see the module docstrings and tests/test_learn.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.portfolio import run_portfolio_backtest
from backtest.report import summary_stats
from learn.allocation import thompson_multipliers
from learn.evaluate import (calibration_table, classification_stats, daily_sharpe,
                            deflated_sharpe, oos_permutation_importance)
from learn.features import FEATURES
from learn.metalabel import multiplier_series, size_multiplier, walk_forward_metalabel
from learn.trade_log import build_candidate_table, mark_taken
from learn.tuning import GRID, grid_keys, key_name, nested_walk_forward, run_grid


@dataclass
class LearnConfig:
    lookback_window : int = 60
    entry_threshold : float = 2.0
    exit_threshold  : float = 0.5
    stop_threshold  : float = 3.5
    signal          : str = "rolling"
    classifiers     : tuple = ("logit", "hgb")
    embargo_days    : int = 5
    min_train       : int = 40
    p_min           : float = 0.5
    kelly_unit      : float = 0.25
    max_mult        : float = 2.0
    hold_multiple   : float = 2.0
    sectors         : dict = field(default_factory=dict)
    run_tuning      : bool = True
    run_thompson    : bool = True
    extra_trials    : int = 0      # configurations tried offline, for the DSR


@dataclass
class LearningResults:
    variants    : dict             # name -> PortfolioResults, baseline first
    rows        : list             # comparison rows (summary_stats + dsr)
    candidates  : pd.DataFrame     # the trade log
    clf_stats   : dict             # classifier -> OOS AUC / Brier
    importances : dict             # classifier -> pd.Series
    calibration : dict             # classifier -> DataFrame
    tuned_choice: dict = field(default_factory=dict)
    thompson_log: pd.DataFrame = None


def run_learning_layer(prices, schedule, prepared, baseline, spy=None,
                       cfg: LearnConfig = LearnConfig(), verbose=True) -> LearningResults:
    def say(msg):
        if verbose:
            print(msg)

    bt_kwargs = dict(
        lookback_window=cfg.lookback_window, entry_threshold=cfg.entry_threshold,
        exit_threshold=cfg.exit_threshold, stop_threshold=cfg.stop_threshold,
        signal=cfg.signal, prepared=prepared,
    )
    calendar = prices.index

    # ── Candidate signals + triple-barrier labels ─────────────────────────
    cands = build_candidate_table(
        prepared, prices, spy=spy, sectors=cfg.sectors,
        lookback_window=cfg.lookback_window, entry_threshold=cfg.entry_threshold,
        exit_threshold=cfg.exit_threshold, stop_threshold=cfg.stop_threshold,
        signal=cfg.signal, hold_multiple=cfg.hold_multiple,
    )
    if cands.empty:
        raise ValueError("no candidate signals — nothing for the meta-labeler to learn from")
    cands = mark_taken(cands, baseline, "baseline")
    say(f"      {len(cands)} candidate signals, {int(cands['taken_baseline'].sum())} taken by the baseline; "
        f"triple-barrier win rate {cands['y'].mean():.1%} "
        f"({', '.join(f'{k}={v}' for k, v in cands['barrier'].value_counts().items())})")

    variants = {"baseline": baseline}
    notes = {"baseline": "z-score rule as configured"}
    clf_stats, importances, calibration = {}, {}, {}

    # ── Meta-labeling ─────────────────────────────────────────────────────
    for kind in cfg.classifiers:
        preds, records = walk_forward_metalabel(
            cands, calendar, kind=kind, embargo_days=cfg.embargo_days, min_train=cfg.min_train,
        )
        cands[f"p_{kind}"] = preds["p"]
        st = classification_stats(cands["y"], preds["p"], cands["period_start"])
        clf_stats[kind] = st
        importances[kind] = oos_permutation_importance(records, FEATURES)
        calibration[kind] = calibration_table(cands["y"], preds["p"])
        n_cold = int(preds["p"].isna().sum())
        say(f"      meta-{kind}: {len(records)} retrains, {n_cold} early candidates at baseline size (cold start); "
            f"OOS AUC {st['auc']:.3f} pooled / {st['auc_within']:.3f} within-quarter, "
            f"Brier {st['brier']:.4f} vs {st['brier_base']:.4f} for the base rate")

        for mode in ("gate", "kelly"):
            mult = size_multiplier(preds["p"].values, preds["win_loss"].values, mode=mode,
                                   p_min=cfg.p_min, kelly_unit=cfg.kelly_unit, max_mult=cfg.max_mult)
            name = f"meta-{kind} {mode}"
            col = name.replace(" ", "_").replace("-", "_")
            cands[f"mult_{col}"] = mult
            res = run_portfolio_backtest(prices, schedule,
                                         size_multipliers=multiplier_series(cands, mult, prepared),
                                         **bt_kwargs)
            cands = mark_taken(cands, res, col)
            variants[name] = res
            n_scored = int(np.isfinite(preds["p"]).sum())
            n_skip = int(((mult == 0) & np.isfinite(preds["p"])).sum())
            notes[name] = f"skips {n_skip}/{n_scored} scored signals"

    # ── Nested walk-forward tuning ────────────────────────────────────────
    grid_results, tuned_choice = {}, {}
    if cfg.run_tuning:
        grid_results = run_grid(prices, schedule, prepared, GRID, cfg.stop_threshold, cfg.signal)
        default_key = (cfg.entry_threshold, cfg.exit_threshold, cfg.lookback_window)
        if default_key not in grid_results:
            default_key = grid_keys(GRID)[0]
        tuned, tuned_choice = nested_walk_forward(grid_results, prepared, default_key)
        variants["tuned (nested WF)"] = tuned
        n_changed = sum(k != default_key for k in tuned_choice.values())
        notes["tuned (nested WF)"] = f"non-default in {n_changed}/{len(tuned_choice)} quarters"

    # ── Thompson allocation ───────────────────────────────────────────────
    thompson_log = None
    if cfg.run_thompson:
        mults, thompson_log = thompson_multipliers(baseline, prepared)
        variants["thompson alloc"] = run_portfolio_backtest(prices, schedule, size_multipliers=mults, **bt_kwargs)
        w = thompson_log["mult"]
        notes["thompson alloc"] = f"weights {w.min():.2f}-{w.max():.2f}"

    # ── Score: Sharpe + deflated Sharpe over every configuration tried ────
    trial_series = {k: r.daily_pnl for k, r in variants.items()}
    for key, r in grid_results.items():
        trial_series.setdefault(key_name(key), r.daily_pnl)
    trial_sharpes = [daily_sharpe(s) for s in trial_series.values()]
    n_trials = len(trial_sharpes) + cfg.extra_trials

    rows = []
    for name, res in variants.items():
        row = {"name": name, **summary_stats(res)}
        d = deflated_sharpe(res.daily_pnl, trial_sharpes, n_trials)
        row["dsr"] = d["dsr"]
        row["note"] = notes.get(name, "")
        rows.append(row)
    rows[0]["note"] += f"  [DSR over N={n_trials} trials, SR0={d['sr0_ann']:.2f}]"

    return LearningResults(
        variants=variants, rows=rows, candidates=cands, clf_stats=clf_stats,
        importances=importances, calibration=calibration,
        tuned_choice=tuned_choice, thompson_log=thompson_log,
    )
