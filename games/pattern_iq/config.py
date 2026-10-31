"""Shared constants for Pattern IQ. No pygame dependency here so other
modules (sequences, market_patterns, difficulty, stats) can import it
without needing pygame to already be initialized.
"""

import os

WIN_WIDTH = 760
WIN_HEIGHT = 800
FPS = 60

FONT_NAME = "Segoe UI"

# ---- Colors (same family as games/mental_math, for a consistent "house
# style" across the repo's games) ----
BG_COLOR = (15, 15, 20)
TEXT_COLOR = (255, 255, 255)
MUTED_TEXT_COLOR = (159, 160, 161)
ACCENT_COLOR = (20, 74, 165)
ACCENT_TEXT_COLOR = (255, 255, 255)
CORRECT_COLOR = (110, 232, 233)
INCORRECT_COLOR = (232, 93, 93)
CHOICE_BG_COLOR = (26, 26, 34)
CHOICE_BORDER_COLOR = (60, 64, 74)
CHOICE_HOVER_BORDER = (125, 173, 255)
BUTTON_IDLE_COLOR = (30, 34, 44)
TIMER_TRACK_COLOR = (30, 34, 44)
TIMER_SAFE_COLOR = (110, 232, 233)
TIMER_WARN_COLOR = (222, 168, 74)
TIMER_DANGER_COLOR = (232, 93, 93)
CHART_LINE_COLOR = (125, 173, 255)
CHART_FILL_COLOR = (30, 45, 74)
CHART_AXIS_COLOR = (50, 54, 64)

# ---- Modes ----
MODE_SEQUENCE = "sequence"
MODE_MARKET = "market"
MODE_LABELS = {MODE_SEQUENCE: "Number Sequences", MODE_MARKET: "Spot the Pattern"}

# ---- Difficulty tiers ----
# Three tiers, same shape as mental_math: pick a starting tier on the menu
# ("adjustable difficulty"), then streak-based promote/demote nudges it up
# or down mid-session automatically.
MIN_TIER = 1
MAX_TIER = 3
TIER_LABELS = {1: "Easy", 2: "Medium", 3: "Hard"}
PROMOTE_STREAK = 5   # consecutive correct answers needed to move up a tier
DEMOTE_STREAK = 3    # consecutive incorrect answers needed to move down a tier

# Sequences are read-and-solve, so they get a slightly longer clock than a
# single arithmetic fact; market charts need a beat longer still to actually
# look at the shape of the line before it's read as a decision.
SEQUENCE_TIME_LIMITS = {1: 20.0, 2: 14.0, 3: 9.0}
MARKET_TIME_LIMITS = {1: 16.0, 2: 11.0, 3: 7.0}

# ---- Session length options shown on the menu ----
SESSION_LENGTHS = [
    ("1 min", 60),
    ("3 min", 180),
    ("5 min", 300),
    ("Endless", None),
]
DEFAULT_SESSION_INDEX = 3  # Endless

FEEDBACK_DISPLAY_SECONDS = 2.2  # feedback carries a rule/pattern explanation, so give it a beat longer than mental_math's

# ---- Corner badge image (same convention as mental_math - drop a file in
# assets/ and it's picked up automatically) ----
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
LOGO_FILENAMES = ["logo.png", "logo.jpg", "logo.jpeg"]

SEQUENCE_CATEGORY_LABELS = {
    "arithmetic": "Arithmetic",
    "geometric": "Geometric",
    "fibonacci": "Fibonacci-style",
    "quadratic": "Quadratic (2nd diff.)",
    "alternating": "Interleaved",
    "primes": "Primes",
    "powers": "Powers",
    "doubling": "Double + Constant",
}

MARKET_PATTERN_LABELS = {
    "trend_up": "Uptrend",
    "trend_down": "Downtrend",
    "mean_reversion": "Mean-Reverting",
    "random_walk": "Random Walk",
    "volatility_breakout": "Volatility Breakout",
    "cyclical": "Cyclical",
}

# One-liner shown in the feedback panel after each market question - the
# actual "market intuition" payload, independent of whether the answer was right.
MARKET_PATTERN_TIPS = {
    "trend_up": "Higher highs & higher lows despite the noise - drift dominates over time.",
    "trend_down": "Lower highs & lower lows - a persistent downward drift, not a single bad bar.",
    "mean_reversion": "Keeps getting pulled back toward a central level - fades away from extremes.",
    "random_walk": "No persistent drift or pull - the up/down moves are directionless noise.",
    "volatility_breakout": "A tight, calm range suddenly explodes into much bigger swings - a regime shift, not a new trend.",
    "cyclical": "Rhythmic up-down swings around a roughly fixed period - look for the repeating wave.",
}
