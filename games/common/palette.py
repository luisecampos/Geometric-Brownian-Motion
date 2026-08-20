"""
Shared color palette + font name for the brain-training game family.

Pulled from citadelsecurities.com's own CSS custom properties
(--blue-primary #144aa5, --gray-dark-alt for the dark background, --green
which is actually their cyan CTA-button color #6ee8e9) - navy blue + cyan,
no gold. See games/mental_math/config.py for the original lookup; this is
the same palette, just shared so the newer games don't redefine it.
"""

FONT_NAME = "Segoe UI"

BG_COLOR = (15, 15, 20)
TEXT_COLOR = (255, 255, 255)
MUTED_TEXT_COLOR = (159, 160, 161)
ACCENT_COLOR = (20, 74, 165)
ACCENT_TEXT_COLOR = (255, 255, 255)
CORRECT_COLOR = (110, 232, 233)
INCORRECT_COLOR = (232, 93, 93)
INPUT_BG_COLOR = (26, 26, 34)
INPUT_BORDER_COLOR = (60, 64, 74)
INPUT_BORDER_ACTIVE = (125, 173, 255)
BUTTON_IDLE_COLOR = (30, 34, 44)
TIMER_TRACK_COLOR = (30, 34, 44)
TIMER_SAFE_COLOR = (110, 232, 233)
TIMER_WARN_COLOR = (222, 168, 74)
TIMER_DANGER_COLOR = (232, 93, 93)
