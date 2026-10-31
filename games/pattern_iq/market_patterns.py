"""
Spot the Pattern: classify a generated price series into one of six regimes.
Aimed squarely at "market intuition" - the actual skill being trained is
telling a real drift/mean-reversion/regime-shift/cycle apart from noise that
merely looks like one, so difficulty here means shrinking the signal-to-noise
ratio (more randomness relative to the underlying drift/amplitude), not just
bigger numbers like in sequences.py.

Pure-Python (random.gauss), no numpy - keeps this game's only dependency
pygame, same as games/mental_math. Each generator returns a plain list of
floats (a price path starting at 100.0); main.py/ui.py never need to know
the generating process, just the series and its label.
"""

import math
import random
import collections
from dataclasses import dataclass


@dataclass
class MarketQuestion:
    series: list       # price path, starts at 100.0
    label: str          # correct pattern key, e.g. "trend_up"
    choices: list        # 4 pattern keys, one of which equals label, pre-shuffled
    tier: int


_N_POINTS = {1: 70, 2: 90, 3: 110}

# Signal-to-noise gets worse as tier climbs: sigma (noise) scales up while
# drift/amplitude/theta (signal) stays roughly flat or shrinks slightly.
_SIGMA = {1: 0.006, 2: 0.011, 3: 0.017}
_TREND_MU = {1: 0.006, 2: 0.005, 3: 0.0038}
_OU_THETA = {1: 0.14, 2: 0.10, 3: 0.07}
_CYCLE_AMPLITUDE = {1: 6.0, 2: 4.0, 3: 2.6}


def _gbm_path(n, mu, sigma, s0=100.0):
    prices = [s0]
    for _ in range(n - 1):
        shock = random.gauss(0.0, 1.0)
        ret = mu + sigma * shock
        prices.append(max(0.5, prices[-1] * (1.0 + ret)))
    return prices


def generate_trend_up(tier):
    n = _N_POINTS[tier]
    return _gbm_path(n, _TREND_MU[tier], _SIGMA[tier])


def generate_trend_down(tier):
    n = _N_POINTS[tier]
    return _gbm_path(n, -_TREND_MU[tier], _SIGMA[tier])


def generate_random_walk(tier):
    n = _N_POINTS[tier]
    # A hair more noise than the trend classes at the same tier - a pure
    # random walk has no drift to "cut through" the noise with.
    return _gbm_path(n, 0.0, _SIGMA[tier] * 1.15)


def generate_mean_reversion(tier):
    n = _N_POINTS[tier]
    theta = _OU_THETA[tier]
    sigma = _SIGMA[tier] * 100  # OU works in price-point units, not returns
    level = 100.0
    prices = [level + random.uniform(-4, 4)]
    for _ in range(n - 1):
        shock = random.gauss(0.0, 1.0)
        nxt = prices[-1] + theta * (level - prices[-1]) + sigma * shock
        prices.append(max(0.5, nxt))
    return prices


def generate_volatility_breakout(tier):
    n = _N_POINTS[tier]
    calm_sigma = _SIGMA[tier] * 0.4
    wild_sigma = _SIGMA[tier] * (4.0 + tier)
    break_point = random.randint(n // 3, 2 * n // 3)
    prices = [100.0]
    for i in range(1, n):
        sigma = wild_sigma if i >= break_point else calm_sigma
        shock = random.gauss(0.0, 1.0)
        prices.append(max(0.5, prices[-1] * (1.0 + sigma * shock)))
    return prices


def generate_cyclical(tier):
    n = _N_POINTS[tier]
    amplitude = _CYCLE_AMPLITUDE[tier]
    period = random.uniform(14, 24)
    phase = random.uniform(0, 2 * math.pi)
    noise = _SIGMA[tier] * 70
    prices = []
    for i in range(n):
        base = 100.0 + amplitude * math.sin(2 * math.pi * i / period + phase)
        prices.append(max(0.5, base + random.gauss(0.0, noise)))
    return prices


PATTERN_GENERATORS = {
    "trend_up": generate_trend_up,
    "trend_down": generate_trend_down,
    "mean_reversion": generate_mean_reversion,
    "random_walk": generate_random_walk,
    "volatility_breakout": generate_volatility_breakout,
    "cyclical": generate_cyclical,
}

# Even weighting at every tier - unlike sequences.py, difficulty here comes
# entirely from signal-to-noise, not from favoring "harder" categories.
_ALL_PATTERNS = list(PATTERN_GENERATORS.keys())


class MarketEngine:
    """Generates questions while avoiding immediate repeats (by label)."""

    def __init__(self, history_size=3):
        self._recent = collections.deque(maxlen=history_size)

    def next_question(self, tier):
        label = None
        for _ in range(10):
            candidate = random.choice(_ALL_PATTERNS)
            if candidate not in self._recent:
                label = candidate
                break
        if label is None:
            label = random.choice(_ALL_PATTERNS)
        self._recent.append(label)

        series = PATTERN_GENERATORS[label](tier)
        distractors = random.sample([p for p in _ALL_PATTERNS if p != label], 3)
        choices = [label] + distractors
        random.shuffle(choices)
        return MarketQuestion(series, label, choices, tier)


def check_answer(question, choice_value):
    return choice_value == question.label
