"""
Number Sequences: "what comes next" IQ-test-style puzzles, multiple choice.

Each category has its own generator function (generate_arithmetic_question,
generate_geometric_question, ...) that returns a SequenceQuestion, so a new
category can be added later by writing one more function and registering it
in CATEGORY_GENERATORS / CATEGORY_WEIGHTS - nothing else needs to change.
Mirrors the shape of games/mental_math/questions.py.

Difficulty (tier 1/2/3) widens number ranges, lengthens the shown sequence,
and shifts which categories show up more often - driven externally by
difficulty.DifficultyManager.
"""

import random
import collections
from dataclasses import dataclass


@dataclass
class SequenceQuestion:
    terms: list       # sequence terms shown, e.g. [2, 4, 8, 16]
    answer: float
    choices: list      # 4 numbers, one of which equals answer, pre-shuffled
    category: str
    tier: int
    rule_hint: str     # plain-English rule, shown in feedback either way


def _make_choices(answer, wrongs):
    """Dedupe wrongs against the answer and each other, top up with small
    jitter if there weren't enough distinct distractors, then shuffle."""
    pool = []
    seen = {answer}
    for w in wrongs:
        if w not in seen:
            pool.append(w)
            seen.add(w)
    jitter = 1
    while len(pool) < 3:
        candidate = answer + jitter if jitter % 2 else answer - jitter
        jitter += 1
        if candidate not in seen:
            pool.append(candidate)
            seen.add(candidate)
    choices = [answer] + pool[:3]
    random.shuffle(choices)
    return choices


# ---------------------------------------------------------------------------
# Arithmetic: a, a+d, a+2d, ...
# ---------------------------------------------------------------------------

_ARITH_D_RANGE = {1: (1, 6), 2: (2, 14), 3: (3, 30)}
_ARITH_START_RANGE = {1: (-10, 30), 2: (-40, 60), 3: (-100, 150)}
_ARITH_N_TERMS = {1: 4, 2: 5, 3: 5}


def generate_arithmetic_question(tier):
    d_lo, d_hi = _ARITH_D_RANGE[tier]
    d = random.randint(d_lo, d_hi) * random.choice([1, -1])
    start = random.randint(*_ARITH_START_RANGE[tier])
    n = _ARITH_N_TERMS[tier]
    terms = [start + i * d for i in range(n)]
    answer = start + n * d
    wrongs = [answer + d, answer - d, answer + 2 * d, terms[-1]]
    sign = "+" if d >= 0 else "-"
    hint = f"Add {sign}{abs(d)} each step."
    return SequenceQuestion(terms, answer, _make_choices(answer, wrongs), "arithmetic", tier, hint)


# ---------------------------------------------------------------------------
# Geometric: a, a*r, a*r^2, ... - start chosen so every term stays an integer.
# ---------------------------------------------------------------------------

_GEOM_RATIOS = {1: [2, 3, -2], 2: [2, 3, 4, -2, -3], 3: [2, 3, 4, 5, -2, -3, -4]}
_GEOM_N_TERMS = {1: 4, 2: 4, 3: 5}


def generate_geometric_question(tier):
    ratio = random.choice(_GEOM_RATIOS[tier])
    start = random.randint(1, 6) * random.choice([1, -1]) if random.random() < 0.3 else random.randint(1, 6)
    if start == 0:
        start = 1
    n = _GEOM_N_TERMS[tier]
    terms = [start * ratio ** i for i in range(n)]
    answer = start * ratio ** n
    last = terms[-1]
    wrongs = [answer + last, answer - last, last * (ratio + 1), last * (ratio - 1) if ratio != 1 else answer + 1]
    hint = f"Multiply by {ratio} each step."
    return SequenceQuestion(terms, answer, _make_choices(answer, wrongs), "geometric", tier, hint)


# ---------------------------------------------------------------------------
# Fibonacci-style: each term is the sum of the previous two.
# ---------------------------------------------------------------------------

_FIB_START_RANGE = {1: (1, 5), 2: (1, 12), 3: (1, 30)}
_FIB_N_TERMS = {1: 5, 2: 6, 3: 6}


def generate_fibonacci_question(tier):
    a, b = random.randint(*_FIB_START_RANGE[tier]), random.randint(*_FIB_START_RANGE[tier])
    terms = [a, b]
    n = _FIB_N_TERMS[tier]
    while len(terms) < n:
        terms.append(terms[-1] + terms[-2])
    answer = terms[-1] + terms[-2]
    wrongs = [answer + terms[-1], answer - terms[-1], terms[-1] + terms[-3] if len(terms) >= 3 else answer + 1, answer + 2]
    hint = "Each term is the sum of the two before it."
    return SequenceQuestion(terms, answer, _make_choices(answer, wrongs), "fibonacci", tier, hint)


# ---------------------------------------------------------------------------
# Quadratic: the *differences between terms* form their own arithmetic
# sequence (constant second difference) - e.g. 2, 3, 5, 8, 12 (diffs 1,2,3,4).
# ---------------------------------------------------------------------------

_QUAD_START_RANGE = {1: (0, 10), 2: (-10, 20), 3: (-20, 40)}
_QUAD_D0_RANGE = {1: (1, 3), 2: (1, 5), 3: (2, 8)}
_QUAD_DD_RANGE = {1: (1, 2), 2: (1, 4), 3: (2, 6)}
_QUAD_N_TERMS = {1: 5, 2: 5, 3: 6}


def generate_quadratic_question(tier):
    start = random.randint(*_QUAD_START_RANGE[tier])
    d = random.randint(*_QUAD_D0_RANGE[tier])
    dd = random.randint(*_QUAD_DD_RANGE[tier]) * random.choice([1, -1]) or 1
    n = _QUAD_N_TERMS[tier]
    terms = [start]
    step = d
    for _ in range(n - 1):
        terms.append(terms[-1] + step)
        step += dd
    answer = terms[-1] + step
    wrongs = [terms[-1] + step - dd, terms[-1] + step + dd, answer + 1, answer - 1]
    hint = f"The gap between terms itself grows by {dd:+d} each step."
    return SequenceQuestion(terms, answer, _make_choices(answer, wrongs), "quadratic", tier, hint)


# ---------------------------------------------------------------------------
# Interleaved: two independent arithmetic sequences woven together
# (e.g. 1, 10, 2, 9, 3, 8 -> next is 4, continuing the odd-position sequence).
# ---------------------------------------------------------------------------

_ALT_D_RANGE = {1: (1, 4), 2: (1, 8), 3: (2, 15)}
_ALT_START_RANGE = {1: (0, 15), 2: (-10, 30), 3: (-30, 60)}
_ALT_PAIRS = {1: 3, 2: 3, 3: 4}  # pairs of terms shown (n = 2*pairs)


def generate_alternating_question(tier):
    d_lo, d_hi = _ALT_D_RANGE[tier]
    da = random.randint(d_lo, d_hi) * random.choice([1, -1])
    db = random.randint(d_lo, d_hi) * random.choice([1, -1])
    a0 = random.randint(*_ALT_START_RANGE[tier])
    b0 = random.randint(*_ALT_START_RANGE[tier])
    pairs = _ALT_PAIRS[tier]
    seq_a = [a0 + i * da for i in range(pairs + 1)]
    seq_b = [b0 + i * db for i in range(pairs)]
    terms = []
    for i in range(pairs):
        terms.append(seq_a[i])
        terms.append(seq_b[i])
    answer = seq_a[pairs]  # next term continues the odd-position (A) sequence
    off_track = seq_b[-1] + db  # common mistake: continuing the other subsequence instead
    wrongs = [off_track, answer + da, answer - da]
    hint = "Two interleaved sequences - track the one at the same position, ignoring the other."
    return SequenceQuestion(terms, answer, _make_choices(answer, wrongs), "alternating", tier, hint)


# ---------------------------------------------------------------------------
# Primes
# ---------------------------------------------------------------------------

def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _next_prime(n):
    candidate = n + 1
    while not _is_prime(candidate):
        candidate += 1
    return candidate


def _primes_from(start, count):
    out = []
    n = start
    while len(out) < count:
        if _is_prime(n):
            out.append(n)
        n += 1
    return out


_PRIME_START_RANGE = {1: (2, 15), 2: (10, 40), 3: (30, 90)}
_PRIME_N_TERMS = {1: 4, 2: 5, 3: 5}


def generate_primes_question(tier):
    start = random.randint(*_PRIME_START_RANGE[tier])
    n = _PRIME_N_TERMS[tier]
    terms = _primes_from(start, n)
    answer = _next_prime(terms[-1])
    skip_one = _next_prime(answer)               # skipped a prime
    near_composite = answer + 1 if not _is_prime(answer + 1) else answer - 1  # nearby non-prime
    wrongs = [skip_one, near_composite, answer + 2 if _is_prime(answer + 2) is False else answer - 2]
    hint = "The sequence of prime numbers."
    return SequenceQuestion(terms, answer, _make_choices(answer, wrongs), "primes", tier, hint)


# ---------------------------------------------------------------------------
# Powers: squares or cubes of consecutive integers.
# ---------------------------------------------------------------------------

_POWER_START_RANGE = {1: (1, 5), 2: (2, 8), 3: (3, 12)}
_POWER_N_TERMS = {1: 4, 2: 4, 3: 5}


def generate_powers_question(tier):
    power = 2 if tier == 1 or random.random() < 0.6 else 3
    start = random.randint(*_POWER_START_RANGE[tier])
    n = _POWER_N_TERMS[tier]
    terms = [(start + i) ** power for i in range(n)]
    base_next = start + n
    answer = base_next ** power
    wrongs = [(base_next - 1) ** power + base_next, answer + base_next, answer - base_next, (base_next + 1) ** power]
    label = "squares" if power == 2 else "cubes"
    hint = f"Consecutive integers raised to the power of {power} ({label} of {start}, {start + 1}, ...)."
    return SequenceQuestion(terms, answer, _make_choices(answer, wrongs), "powers", tier, hint)


# ---------------------------------------------------------------------------
# Doubling + constant: a(n+1) = a(n)*2 + c
# ---------------------------------------------------------------------------

_DBL_START_RANGE = {1: (1, 8), 2: (1, 15), 3: (1, 25)}
_DBL_C_RANGE = {1: (0, 3), 2: (-5, 5), 3: (-10, 10)}
_DBL_N_TERMS = {1: 4, 2: 5, 3: 5}


def generate_doubling_question(tier):
    a0 = random.randint(*_DBL_START_RANGE[tier])
    c = random.randint(*_DBL_C_RANGE[tier])
    n = _DBL_N_TERMS[tier]
    terms = [a0]
    for _ in range(n - 1):
        terms.append(terms[-1] * 2 + c)
    answer = terms[-1] * 2 + c
    wrongs = [answer + c, answer - c, terms[-1] * 3 + c, terms[-1] * 2]
    sign = "+" if c >= 0 else "-"
    hint = f"Double the previous term, then {sign} {abs(c)}."
    return SequenceQuestion(terms, answer, _make_choices(answer, wrongs), "doubling", tier, hint)


# ---------------------------------------------------------------------------
# Registration - add a new category by adding one entry to each dict below.
# ---------------------------------------------------------------------------

CATEGORY_GENERATORS = {
    "arithmetic": generate_arithmetic_question,
    "geometric": generate_geometric_question,
    "fibonacci": generate_fibonacci_question,
    "quadratic": generate_quadratic_question,
    "alternating": generate_alternating_question,
    "primes": generate_primes_question,
    "powers": generate_powers_question,
    "doubling": generate_doubling_question,
}

# Relative frequency per category, per tier - higher tiers lean into the
# categories that need more working-memory/pattern-hunting under time pressure.
CATEGORY_WEIGHTS = {
    1: {"arithmetic": 4, "geometric": 3, "fibonacci": 2, "quadratic": 1,
        "alternating": 1, "primes": 1, "powers": 2, "doubling": 1},
    2: {"arithmetic": 2, "geometric": 3, "fibonacci": 2, "quadratic": 2,
        "alternating": 2, "primes": 2, "powers": 2, "doubling": 2},
    3: {"arithmetic": 1, "geometric": 2, "fibonacci": 2, "quadratic": 3,
        "alternating": 3, "primes": 3, "powers": 2, "doubling": 2},
}


class SequenceEngine:
    """Generates questions while avoiding immediate repeats (by category)."""

    def __init__(self, history_size=4):
        self._recent = collections.deque(maxlen=history_size)

    def next_question(self, tier):
        categories = list(CATEGORY_WEIGHTS[tier].keys())
        weights = list(CATEGORY_WEIGHTS[tier].values())
        question = None
        for _ in range(10):
            category = random.choices(categories, weights=weights, k=1)[0]
            if category not in self._recent or len(self._recent) < len(categories):
                question = CATEGORY_GENERATORS[category](tier)
                break
        if question is None:
            category = random.choices(categories, weights=weights, k=1)[0]
            question = CATEGORY_GENERATORS[category](tier)
        self._recent.append(question.category)
        return question


def check_answer(question, choice_value):
    return choice_value == question.answer
