"""
Question generation and answer validation.

Each category has its own generator function (generate_addition_question,
generate_multiplication_question, ...) so a new category can be added later
by writing one more function and registering it in CATEGORY_GENERATORS /
CATEGORY_WEIGHTS - nothing else needs to change.

Difficulty here means more than "bigger numbers": ranges, which operands get
picked, and how often each category shows up all shift with the tier (1/2/3),
which is driven externally by difficulty.DifficultyManager.
"""

import math
import random
import collections
from dataclasses import dataclass


@dataclass
class Question:
    prompt: str
    answer: float
    category: str
    tier: int
    tolerance: float = 1e-6  # 0-ish for exact-integer answers, wider for fractions


# ---------------------------------------------------------------------------
# Addition / subtraction (two-digit)
# ---------------------------------------------------------------------------

_ADD_SUB_RANGES = {1: (10, 50), 2: (20, 80), 3: (35, 99)}


def generate_addition_question(tier):
    lo, hi = _ADD_SUB_RANGES[tier]
    a, b = random.randint(lo, hi), random.randint(lo, hi)
    return Question(f"{a} + {b}", float(a + b), "addition", tier)


def generate_subtraction_question(tier):
    lo, hi = _ADD_SUB_RANGES[tier]
    a, b = random.randint(lo, hi), random.randint(lo, hi)
    if b > a:
        a, b = b, a  # keep it positive - the point is speed, not sign-handling
    return Question(f"{a} - {b}", float(a - b), "subtraction", tier)


# ---------------------------------------------------------------------------
# Multiplication - nudged toward "friendly" numbers that reward decomposition
# tricks (25x16 = 25x4x4, 48x25 = 48/4x100, etc.) rather than raw brute force.
# ---------------------------------------------------------------------------

_FRIENDLY_FACTORS = [11, 12, 15, 16, 18, 19, 21, 22, 24, 25, 26]


def generate_multiplication_question(tier):
    if tier == 1:
        a = random.randint(6, 9)           # single digit
        b = random.randint(11, 30)         # small two-digit
    elif tier == 2:
        a = random.choice(_FRIENDLY_FACTORS) if random.random() < 0.5 else random.randint(12, 30)
        b = random.randint(12, 40)
    else:  # tier 3
        a = random.choice(_FRIENDLY_FACTORS) if random.random() < 0.35 else random.randint(12, 40)
        b = random.randint(12, 99)

    if random.random() < 0.5:
        a, b = b, a
    return Question(f"{a} × {b}", float(a * b), "multiplication", tier)


# ---------------------------------------------------------------------------
# Triple-digit addition/subtraction
# ---------------------------------------------------------------------------

_TRIPLE_RANGES = {1: (120, 450), 2: (200, 700), 3: (300, 950)}


def generate_triple_digit_question(tier):
    lo, hi = _TRIPLE_RANGES[tier]
    a, b = random.randint(lo, hi), random.randint(lo, hi)
    op = random.choice(["+", "-"])
    if op == "-" and b > a:
        a, b = b, a
    answer = a + b if op == "+" else a - b
    return Question(f"{a} {op} {b}", float(answer), "triple_digit", tier)


# ---------------------------------------------------------------------------
# Percentages - base is nudged so the answer always lands on a whole number,
# so "18% of 250" always resolves to a clean 45, never a messy decimal.
# ---------------------------------------------------------------------------

_PERCENT_CHOICES = {
    1: [10, 20, 25, 50, 5],
    2: [12, 15, 18, 24, 35, 45, 8],
    3: [17, 23, 32, 37, 42, 63, 68, 3, 7],
}
_PERCENT_BASE_RANGES = {1: (40, 300), 2: (100, 600), 3: (150, 900)}


def _nice_percent_base(percent, low, high):
    step = 100 // math.gcd(percent, 100)
    base = random.randint(low, high)
    return max(step, round(base / step) * step)


def generate_percentage_question(tier):
    percent = random.choice(_PERCENT_CHOICES[tier])
    low, high = _PERCENT_BASE_RANGES[tier]
    base = _nice_percent_base(percent, low, high)
    answer = base * percent // 100
    return Question(f"What is {percent}% of {base}?", float(answer), "percentage", tier)


# ---------------------------------------------------------------------------
# Fractions -> decimals. Non-terminating fractions need a rounding tolerance;
# the UI shows a hint about the 3-decimal convention whenever this category
# comes up (see ui._category_hint / main._category_hint).
# ---------------------------------------------------------------------------

_FRACTION_DENOMS = {1: [2, 4, 5, 10], 2: [3, 6, 8, 10], 3: [7, 9, 11, 12]}
FRACTION_TOLERANCE = 0.0015  # covers rounding to 3 dp either direction, plus slack


def generate_fraction_question(tier):
    denom = random.choice(_FRACTION_DENOMS[tier])
    numer = random.randint(1, denom - 1)
    value = numer / denom
    return Question(f"{numer}/{denom} = ?", value, "fraction", tier, tolerance=FRACTION_TOLERANCE)


# ---------------------------------------------------------------------------
# Squares, 11^2 - 25^2. Higher tiers lean toward numbers where "square around
# a nearby base" (19^2 = 20^2 - 20 - 19, 21^2 = 20^2 + 20 + 21) pays off.
# ---------------------------------------------------------------------------

_SQUARE_RANGES = {1: (11, 16), 2: (14, 21), 3: (18, 25)}


def generate_square_question(tier):
    lo, hi = _SQUARE_RANGES[tier]
    n = random.randint(lo, hi)
    return Question(f"{n}² = ?", float(n * n), "square", tier)


# ---------------------------------------------------------------------------
# Registration - add a new category by adding one entry to each dict below.
# ---------------------------------------------------------------------------

CATEGORY_GENERATORS = {
    "addition": generate_addition_question,
    "subtraction": generate_subtraction_question,
    "multiplication": generate_multiplication_question,
    "triple_digit": generate_triple_digit_question,
    "percentage": generate_percentage_question,
    "fraction": generate_fraction_question,
    "square": generate_square_question,
}

# Relative frequency per category, per tier. Higher tiers lean harder into
# multiplication/percentage/squares, per the "difficulty = frequency too" idea.
CATEGORY_WEIGHTS = {
    1: {"addition": 3, "subtraction": 3, "multiplication": 2, "triple_digit": 1,
        "percentage": 1, "fraction": 1, "square": 1},
    2: {"addition": 2, "subtraction": 2, "multiplication": 3, "triple_digit": 2,
        "percentage": 2, "fraction": 2, "square": 2},
    3: {"addition": 1, "subtraction": 1, "multiplication": 3, "triple_digit": 2,
        "percentage": 3, "fraction": 2, "square": 3},
}


class QuestionEngine:
    """Generates questions while avoiding immediate repeats."""

    def __init__(self, history_size=5):
        self._recent = collections.deque(maxlen=history_size)

    def next_question(self, tier):
        categories = list(CATEGORY_WEIGHTS[tier].keys())
        weights = list(CATEGORY_WEIGHTS[tier].values())
        question = None
        for _ in range(10):  # a handful of retries is plenty to dodge a repeat
            category = random.choices(categories, weights=weights, k=1)[0]
            question = CATEGORY_GENERATORS[category](tier)
            if question.prompt not in self._recent:
                break
        self._recent.append(question.prompt)
        return question


# ---------------------------------------------------------------------------
# Answer validation
# ---------------------------------------------------------------------------

def _parse_number(raw):
    text = raw.strip().replace(",", "").replace(" ", "")
    if not text:
        raise ValueError("empty input")
    if text.startswith("."):
        text = "0" + text
    elif text.startswith("-."):
        text = "-0" + text[1:]
    return float(text)


def check_answer(question, raw_text):
    """True if raw_text is close enough to question.answer given its tolerance."""
    try:
        value = _parse_number(raw_text)
    except ValueError:
        return False
    return abs(value - question.answer) <= question.tolerance
