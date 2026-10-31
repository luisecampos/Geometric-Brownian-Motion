"""
Probability Quick-Fire (EV/Odds + Card/Dice Probability)
------------------------------------------------------------
Timed quiz combining the two probability skills quant interviews lean on:
computing EV / converting odds to implied probability, and quick card/dice
probability questions. Same tiered-timer shape as mental_math (45s -> 30s ->
15s, 5 correct promotes, 3 incorrect demotes) - this file only supplies the
question generators; games/common/quiz_engine.py runs the actual game.

Run:
    python probability_quickfire.py         (from inside games/)
    python games/probability_quickfire.py   (from the repo root)

Dependencies: pip install pygame-ce   (plain "pygame" also works)
"""

import os
import sys
import math
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import quiz_engine
from common.quiz_engine import Question


# ---------------------------------------------------------------------------
# EV of a two-outcome bet
# ---------------------------------------------------------------------------

def _nice_amount(percent, low, high):
    """Nudges a random dollar amount so percent% of it is a whole number -
    same trick as mental_math's percentage generator."""
    percent = max(1, min(99, percent))
    step = 100 // math.gcd(percent, 100)
    amt = random.randint(low, high)
    return max(step, round(amt / step) * step)


_EV_P_CHOICES = {1: list(range(10, 91, 10)), 2: list(range(10, 91, 10)), 3: list(range(5, 96, 5))}
_EV_AMOUNT_RANGES = {1: ((40, 200), (20, 150)), 2: ((100, 500), (80, 400)), 3: ((150, 900), (100, 700))}


def generate_ev_question(tier):
    p = random.choice(_EV_P_CHOICES[tier])
    q = 100 - p
    (win_low, win_high), (loss_low, loss_high) = _EV_AMOUNT_RANGES[tier]
    win_amt = _nice_amount(p, win_low, win_high)
    loss_amt = _nice_amount(q, loss_low, loss_high)
    win_contribution = p * win_amt // 100
    loss_contribution = q * loss_amt // 100
    ev = win_contribution - loss_contribution
    prompt = f"{p}% to win ${win_amt}, {q}% to lose ${loss_amt}. What's the EV?"
    return Question(prompt, float(ev), "ev", tier)


# ---------------------------------------------------------------------------
# American / decimal odds -> implied probability
# ---------------------------------------------------------------------------

_ODDS_TOLERANCE = 1.0  # percentage points - covers repeating decimals (33.3%, 66.7%, ...)

_AMERICAN_ODDS = {
    1: [-100, 100, -200, 200, -400, 400],
    2: [-150, 150, -250, 250, -125, 125, -350, 350],
    3: [-175, 175, -220, 220, -320, 320, -115, 115, -600, 600],
}


def _implied_prob_from_american(odds):
    if odds > 0:
        return 100.0 / (100 + odds) * 100.0
    return (-odds) / ((-odds) + 100) * 100.0


def generate_american_odds_question(tier):
    odds = random.choice(_AMERICAN_ODDS[tier])
    prob = _implied_prob_from_american(odds)
    sign = "+" if odds > 0 else ""
    prompt = f"American odds: {sign}{odds}. Implied probability (%)?"
    return Question(prompt, prob, "odds_american", tier, tolerance=_ODDS_TOLERANCE)


_DECIMAL_ODDS = {
    1: [2.00, 1.50, 4.00, 1.25, 5.00],
    2: [1.20, 3.00, 1.10, 1.75, 2.50],
    3: [1.40, 2.20, 1.65, 3.50, 1.15, 6.00],
}


def generate_decimal_odds_question(tier):
    odds = random.choice(_DECIMAL_ODDS[tier])
    prob = 1.0 / odds * 100.0
    prompt = f"Decimal odds: {odds:.2f}. Implied probability (%)?"
    return Question(prompt, prob, "odds_decimal", tier, tolerance=_ODDS_TOLERANCE)


# ---------------------------------------------------------------------------
# Dice probability
# ---------------------------------------------------------------------------

_DICE_SUM_COUNTS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
_SINGLE_DIE_CONDITIONS = [
    ("even", 3, "even"),
    ("odd", 3, "odd"),
    (">=4", 3, "4 or higher"),
    (">=5", 2, "5 or higher"),
    ("<=2", 2, "2 or lower"),
]


def generate_dice_question(tier):
    kinds = ["single", "sum"] if tier == 1 else ["single", "sum", "at_least_one"]
    kind = random.choice(kinds)

    if kind == "single":
        _key, count, label = random.choice(_SINGLE_DIE_CONDITIONS)
        prob = count / 6 * 100
        prompt = f"Roll one die. P(result is {label})?"
    elif kind == "sum":
        target = random.randint(2, 12)
        prob = _DICE_SUM_COUNTS[target] / 36 * 100
        prompt = f"Roll two dice. P(sum = {target})?"
    else:  # at_least_one - probability is the same for any specific face by symmetry
        face = random.randint(1, 6)
        prob = (1 - (5 / 6) ** 2) * 100
        prompt = f"Roll two dice. P(at least one {face})?"

    return Question(prompt, prob, "dice", tier, tolerance=_ODDS_TOLERANCE)


# ---------------------------------------------------------------------------
# Card probability (standard 52-card deck)
# ---------------------------------------------------------------------------

def generate_card_question(tier):
    if tier == 1 or random.random() < 0.6:
        kind = random.choice(["suit", "color", "face", "ace"])
        if kind == "suit":
            prob, prompt = 13 / 52 * 100, "Draw 1 card from a full deck. P(it's a heart)?"
        elif kind == "color":
            prob, prompt = 26 / 52 * 100, "Draw 1 card from a full deck. P(it's red)?"
        elif kind == "face":
            prob, prompt = 12 / 52 * 100, "Draw 1 card from a full deck. P(it's a face card - J, Q, or K)?"
        else:
            prob, prompt = 4 / 52 * 100, "Draw 1 card from a full deck. P(it's an ace)?"
    else:
        kind = random.choice(["both_aces", "both_red", "both_hearts"])
        if kind == "both_aces":
            prob, prompt = (4 / 52) * (3 / 51) * 100, "Draw 2 cards (no replacement). P(both are aces)?"
        elif kind == "both_red":
            prob, prompt = (26 / 52) * (25 / 51) * 100, "Draw 2 cards (no replacement). P(both are red)?"
        else:
            prob, prompt = (13 / 52) * (12 / 51) * 100, "Draw 2 cards (no replacement). P(both are hearts)?"

    return Question(prompt, prob, "cards", tier, tolerance=_ODDS_TOLERANCE)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

CATEGORY_GENERATORS = {
    "ev": generate_ev_question,
    "odds_american": generate_american_odds_question,
    "odds_decimal": generate_decimal_odds_question,
    "dice": generate_dice_question,
    "cards": generate_card_question,
}

CATEGORY_WEIGHTS = {
    1: {"ev": 3, "odds_american": 2, "odds_decimal": 2, "dice": 2, "cards": 2},
    2: {"ev": 2, "odds_american": 2, "odds_decimal": 2, "dice": 3, "cards": 3},
    3: {"ev": 2, "odds_american": 3, "odds_decimal": 3, "dice": 3, "cards": 3},
}

CATEGORY_LABELS = {
    "ev": "EV Calculation",
    "odds_american": "American Odds",
    "odds_decimal": "Decimal Odds",
    "dice": "Dice Probability",
    "cards": "Card Probability",
}

STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probability_quickfire_stats.json")


def main():
    quiz_engine.run(
        title="PROBABILITY QUICK-FIRE",
        tagline="CITADEL INTERVIEW PREP",
        category_generators=CATEGORY_GENERATORS,
        category_weights=CATEGORY_WEIGHTS,
        category_labels=CATEGORY_LABELS,
        stats_path=STATS_PATH,
        caption="Probability Quick-Fire — Citadel Prep",
    )


if __name__ == "__main__":
    main()
