"""
Statistics tracking: per-session (SessionStats) and all-time (PersistentStats,
backed by a small local JSON file). Same shape as games/mental_math/stats.py,
generalized so one PersistentStats instance covers both modes (sequence
categories and market pattern labels are just different "category" strings)
and keeps their all-time records separate in the JSON file.
"""

import json
import os

STATS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pattern_iq_stats.json")


class SessionStats:
    def __init__(self):
        self.correct = 0
        self.incorrect = 0
        self.timeouts = 0
        self.current_streak = 0
        self.best_streak = 0
        self.response_times = []
        self.category_totals = {}  # category -> [correct, incorrect]

    def record(self, category, correct, response_time, timed_out=False):
        counts = self.category_totals.setdefault(category, [0, 0])
        if correct:
            self.correct += 1
            counts[0] += 1
            self.current_streak += 1
            self.best_streak = max(self.best_streak, self.current_streak)
        else:
            self.incorrect += 1
            counts[1] += 1
            self.current_streak = 0
            if timed_out:
                self.timeouts += 1
        self.response_times.append(response_time)

    @property
    def total(self):
        return self.correct + self.incorrect

    @property
    def accuracy(self):
        return (self.correct / self.total * 100) if self.total else 0.0

    @property
    def avg_time(self):
        return (sum(self.response_times) / len(self.response_times)) if self.response_times else 0.0

    def strongest_and_weakest(self):
        """Returns ((category, pct), (category, pct)) or (None, None) if not
        enough data yet. Categories need >= 2 answers to count."""
        results = {}
        for category, (correct, incorrect) in self.category_totals.items():
            total = correct + incorrect
            if total >= 2:
                results[category] = correct / total * 100
        if not results:
            return None, None
        strongest = max(results, key=results.get)
        weakest = min(results, key=results.get)
        return (strongest, results[strongest]), (weakest, results[weakest])


class PersistentStats:
    """All-time bests per mode, persisted to a local JSON file across sessions."""

    def __init__(self, path=STATS_FILE):
        self.path = path
        self.data = self._load()

    def _empty_mode(self):
        return {
            "best_accuracy": 0.0,
            "best_streak": 0,
            "fastest_avg_time": None,
            "total_questions": 0,
            "category_history": {},  # category -> [correct, incorrect], all-time
        }

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for mode in ("sequence", "market"):
                        data.setdefault(mode, self._empty_mode())
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return {"sequence": self._empty_mode(), "market": self._empty_mode()}

    def mode_data(self, mode):
        return self.data[mode]

    def update_from_session(self, mode, session):
        if session.total == 0:
            return  # nothing played - don't let a 0/0 session touch the records
        record = self.data[mode]
        record["best_accuracy"] = max(record["best_accuracy"], session.accuracy)
        record["best_streak"] = max(record["best_streak"], session.best_streak)
        if session.response_times:
            fastest = record["fastest_avg_time"]
            record["fastest_avg_time"] = session.avg_time if fastest is None else min(fastest, session.avg_time)
        record["total_questions"] += session.total
        history = record["category_history"]
        for category, (correct, incorrect) in session.category_totals.items():
            entry = history.setdefault(category, [0, 0])
            entry[0] += correct
            entry[1] += incorrect
        self._save()

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
