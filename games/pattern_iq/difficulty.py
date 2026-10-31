"""
Streak-based tier progression, generalized from games/mental_math/difficulty.py
to take its time-limit table as a parameter, since Number Sequences and Spot
the Pattern run on different clocks (config.SEQUENCE_TIME_LIMITS /
config.MARKET_TIME_LIMITS) but should promote/demote the same way.

"Adjustable difficulty" here is two things at once: you pick a starting tier
on the menu (Easy/Medium/Hard), and then a streak of rights/wrongs nudges the
tier up or down mid-session automatically - same promote/demote thresholds as
mental_math, just retunable in config.py without touching this file.
"""

from . import config


class DifficultyManager:
    def __init__(self, time_limits, start_tier=config.MIN_TIER):
        self.time_limits = time_limits
        self.tier = start_tier
        self._correct_streak = 0
        self._incorrect_streak = 0

    @property
    def time_limit(self):
        return self.time_limits[self.tier]

    def record_result(self, correct):
        """Update streaks and move a tier up/down if thresholds are hit."""
        if correct:
            self._correct_streak += 1
            self._incorrect_streak = 0
            if self._correct_streak >= config.PROMOTE_STREAK and self.tier < config.MAX_TIER:
                self.tier += 1
                self._correct_streak = 0
        else:
            self._incorrect_streak += 1
            self._correct_streak = 0
            if self._incorrect_streak >= config.DEMOTE_STREAK and self.tier > config.MIN_TIER:
                self.tier -= 1
                self._incorrect_streak = 0
