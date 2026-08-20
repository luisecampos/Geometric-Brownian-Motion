"""Shared constants for the Mental Math game. No pygame dependency here so
other modules (questions, difficulty, stats) can import it without needing
pygame to already be initialized.
"""

import os

WIN_WIDTH = 640
WIN_HEIGHT = 760
FPS = 60

FONT_NAME = "Segoe UI"

# ---- Colors ----
# Pulled from citadelsecurities.com's own CSS custom properties (--blue-primary
# #144aa5, --blue-dark-sec/--gray-dark-alt for the dark surfaces, --green
# #6ee8e9 which is actually their cyan CTA-button color despite the name) -
# navy blue + cyan, no gold. Kept the dark background for a "terminal" feel;
# their actual site is white/light if you'd rather match that more literally.
BG_COLOR = (15, 15, 20)                # their --gray-dark-alt (#0f0f14)
TEXT_COLOR = (255, 255, 255)
MUTED_TEXT_COLOR = (159, 160, 161)     # their --gray-dark-sec (#9fa0a1)
ACCENT_COLOR = (20, 74, 165)           # their --blue-primary (#144aa5)
ACCENT_TEXT_COLOR = (255, 255, 255)    # white text on blue, same as their real buttons
CORRECT_COLOR = (110, 232, 233)        # their --green / CTA-button cyan (#6ee8e9)
INCORRECT_COLOR = (232, 93, 93)        # not a brand color - just needs to read as "wrong"
INPUT_BG_COLOR = (26, 26, 34)
INPUT_BORDER_COLOR = (60, 64, 74)
INPUT_BORDER_ACTIVE = (125, 173, 255)  # their --blue-light-accent (#7dadff), visible on dark bg
BUTTON_IDLE_COLOR = (30, 34, 44)
TIMER_TRACK_COLOR = (30, 34, 44)
TIMER_SAFE_COLOR = (110, 232, 233)
TIMER_WARN_COLOR = (222, 168, 74)      # not brand-derived - functional "caution" color
TIMER_DANGER_COLOR = (232, 93, 93)

# ---- Difficulty tiers ----
# Easy to retune later: just edit these numbers, nothing else has to change.
TIME_LIMITS = {1: 45.0, 2: 30.0, 3: 15.0}
MIN_TIER = 1
MAX_TIER = 3
PROMOTE_STREAK = 5   # consecutive correct answers needed to move up a tier
DEMOTE_STREAK = 3    # consecutive incorrect answers needed to move down a tier

# ---- Session length options shown on the menu ----
# (label, duration_in_seconds); None duration = endless.
SESSION_LENGTHS = [
    ("1 min", 60),
    ("3 min", 180),
    ("5 min", 300),
    ("Endless", None),
]
DEFAULT_SESSION_INDEX = 3  # Endless

FEEDBACK_DISPLAY_SECONDS = 0.9  # how long correct/incorrect feedback shows before advancing

# ---- Corner badge image ----
# Drop a logo file at games/mental_math/assets/logo.(png|jpg|jpeg) and the
# corner badge will use it automatically; otherwise it falls back to a drawn
# monogram (see ui._draw_badge).
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
LOGO_FILENAMES = ["logo.png", "logo.jpg", "logo.jpeg"]

CATEGORY_LABELS = {
    "addition": "Addition",
    "subtraction": "Subtraction",
    "multiplication": "Multiplication",
    "triple_digit": "Triple-Digit +/-",
    "percentage": "Percentages",
    "fraction": "Fractions",
    "square": "Squares",
}
