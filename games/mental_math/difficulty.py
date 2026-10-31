"""
Streak-based tier progression.

Deliberately dumb and easy to retune: promotion/demotion is just consecutive
correct/incorrect counters compared against two constants in config.py.
Swap those numbers (or the comparison logic below) without touching anything
else in the game.
"""

from . import config


class DifficultyManager:
    def __init__(self, start_tier=config.MIN_TIER):
        self.tier = start_tier
        self._correct_streak = 0
        self._incorrect_streak = 0

    @property
    def time_limit(self):
        return config.TIME_LIMITS[self.tier]

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
