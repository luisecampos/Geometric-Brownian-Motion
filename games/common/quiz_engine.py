"""
Generic timed multiple-category "type the answer, hit Enter" quiz runner -
the shared engine behind mental_math-style games. Handles the menu/playing/
results state machine, timer-tier difficulty progression (45s -> 30s -> 15s,
same shape as mental_math), session stats, and persistent local JSON stats.
Callers just supply question-generator functions per category and call run().

Not used by games/mental_math itself (that package predates this engine and
stays fully self-contained) - but any *new* timed-quiz game in this family
should build on this instead of rebuilding the loop from scratch. See
games/probability_quickfire.py for the reference usage.
"""

import os
import json
import random
import collections
from dataclasses import dataclass

import pygame

from . import palette, widgets, badge

# ---- Difficulty tiers - same shape as mental_math, so the whole family
# progresses the same way and stats are easy to compare across games. ----
TIME_LIMITS = {1: 45.0, 2: 30.0, 3: 15.0}
MIN_TIER, MAX_TIER = 1, 3
PROMOTE_STREAK = 5
DEMOTE_STREAK = 3

FEEDBACK_DISPLAY_SECONDS = 0.9
SESSION_LENGTHS = [("1 min", 60), ("3 min", 180), ("5 min", 300), ("Endless", None)]
DEFAULT_SESSION_INDEX = 3

WIN_WIDTH = 640
WIN_HEIGHT = 760


@dataclass
class Question:
    prompt: str
    answer: float
    category: str
    tier: int
    tolerance: float = 1e-6


def check_answer(question, raw_text):
    try:
        text = raw_text.strip().replace(",", "").replace(" ", "")
        if not text:
            raise ValueError("empty input")
        if text.startswith("."):
            text = "0" + text
        elif text.startswith("-."):
            text = "-0" + text[1:]
        value = float(text)
    except ValueError:
        return False
    return abs(value - question.answer) <= question.tolerance


class DifficultyManager:
    def __init__(self):
        self.tier = MIN_TIER
        self._correct_streak = 0
        self._incorrect_streak = 0

    @property
    def time_limit(self):
        return TIME_LIMITS[self.tier]

    def record_result(self, correct):
        if correct:
            self._correct_streak += 1
            self._incorrect_streak = 0
            if self._correct_streak >= PROMOTE_STREAK and self.tier < MAX_TIER:
                self.tier += 1
                self._correct_streak = 0
        else:
            self._incorrect_streak += 1
            self._correct_streak = 0
            if self._incorrect_streak >= DEMOTE_STREAK and self.tier > MIN_TIER:
                self.tier -= 1
                self._incorrect_streak = 0


class QuestionEngine:
    """Generates questions from the registered categories while avoiding
    immediate repeats."""

    def __init__(self, category_generators, category_weights, history_size=5):
        self._generators = category_generators
        self._weights = category_weights
        self._recent = collections.deque(maxlen=history_size)

    def next_question(self, tier):
        categories = list(self._weights[tier].keys())
        weights = list(self._weights[tier].values())
        question = None
        for _ in range(10):
            category = random.choices(categories, weights=weights, k=1)[0]
            question = self._generators[category](tier)
            if question.prompt not in self._recent:
                break
        self._recent.append(question.prompt)
        return question


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
    """All-time bests, persisted to a local JSON file across sessions."""

    def __init__(self, path):
        self.path = path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "best_accuracy": 0.0,
            "best_streak": 0,
            "fastest_avg_time": None,
            "total_questions": 0,
            "category_history": {},
        }

    def update_from_session(self, session):
        if session.total == 0:
            return
        self.data["best_accuracy"] = max(self.data["best_accuracy"], session.accuracy)
        self.data["best_streak"] = max(self.data["best_streak"], session.best_streak)
        if session.response_times:
            fastest = self.data["fastest_avg_time"]
            self.data["fastest_avg_time"] = session.avg_time if fastest is None else min(fastest, session.avg_time)
        self.data["total_questions"] += session.total
        history = self.data["category_history"]
        for category, (correct, incorrect) in session.category_totals.items():
            entry = history.setdefault(category, [0, 0])
            entry[0] += correct
            entry[1] += incorrect
        self._save()

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

# Question prompts range from short ("47 x 26"-style, in other games built on
# this engine) to full sentences (this game's "40% to win $500, 60% to lose
# $80. What's the EV?"). Rather than fix a single font size and let long ones
# run off the edge of the window, _fit_question_lines() below tries these
# sizes largest-first and wraps to the first one that fits in <= 2 lines.
_QUESTION_FONT_SIZES = [40, 34, 30, 26, 22, 19]


def _load_fonts():
    return {
        "title": pygame.font.SysFont(palette.FONT_NAME, 40, bold=True),
        "question_sizes": [pygame.font.SysFont(palette.FONT_NAME, size, bold=True) for size in _QUESTION_FONT_SIZES],
        "label": pygame.font.SysFont(palette.FONT_NAME, 18),
        "stat": pygame.font.SysFont(palette.FONT_NAME, 20, bold=True),
        "button": pygame.font.SysFont(palette.FONT_NAME, 22, bold=True),
        "timer": pygame.font.SysFont(palette.FONT_NAME, 24, bold=True),
        "small": pygame.font.SysFont(palette.FONT_NAME, 16),
        "input": pygame.font.SysFont(palette.FONT_NAME, 32),
        "badge": pygame.font.SysFont(palette.FONT_NAME, 15, bold=True),
    }


def _center_text(win, font, text, y, color=palette.TEXT_COLOR):
    surf = font.render(text, True, color)
    win.blit(surf, surf.get_rect(center=(WIN_WIDTH // 2, y)))


def _wrap_text(font, text, max_width):
    """Greedy word-wrap: fills each line up to max_width before breaking."""
    words = text.split(" ")
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if font.size(candidate)[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _fit_question_lines(fonts, text, max_width, max_lines=2):
    """Picks the largest font (from fonts["question_sizes"]) that wraps the
    question to at most max_lines within max_width. Falls back to the
    smallest size, wrapped as far as it goes, if nothing fits cleanly."""
    for font in fonts["question_sizes"]:
        lines = _wrap_text(font, text, max_width)
        if len(lines) <= max_lines and all(font.size(line)[0] <= max_width for line in lines):
            return font, lines
    font = fonts["question_sizes"][-1]
    return font, _wrap_text(font, text, max_width)


def _draw_question(win, fonts, text, center_y, max_width):
    font, lines = _fit_question_lines(fonts, text, max_width)
    line_height = font.get_height()
    total_height = line_height * len(lines)
    start_y = center_y - total_height // 2 + line_height // 2
    for i, line in enumerate(lines):
        _center_text(win, font, line, start_y + i * line_height)
    return start_y + (len(lines) - 1) * line_height + line_height // 2  # bottom edge, for layout below it


def _draw_menu(win, fonts, title, tagline, category_rows, session_buttons, selected_index, start_button, logo):
    win.fill(palette.BG_COLOR)
    _center_text(win, fonts["title"], title, 80)
    _center_text(win, fonts["small"], tagline, 112, palette.ACCENT_COLOR)
    _center_text(win, fonts["small"], "45 seconds  →  30 seconds  →  15 seconds", 148, palette.MUTED_TEXT_COLOR)

    y = 180
    for row in category_rows:
        _center_text(win, fonts["small"], row, y, palette.MUTED_TEXT_COLOR)
        y += 22

    _center_text(win, fonts["label"], "Session length", 268, palette.MUTED_TEXT_COLOR)
    for i, button in enumerate(session_buttons):
        selected = i == selected_index
        color = palette.ACCENT_COLOR if selected else palette.BUTTON_IDLE_COLOR
        text_color = palette.ACCENT_TEXT_COLOR if selected else palette.TEXT_COLOR
        button.draw(win, fonts["label"], color=color, text_color=text_color)

    start_button.draw(win, fonts["button"])
    badge.draw(win, WIN_WIDTH, fonts, logo)
    pygame.display.update()


def _draw_game(win, fonts, question, input_box, submit_button, timer_remaining, timer_limit,
               stats, tier, feedback, category_hint, logo):
    win.fill(palette.BG_COLOR)

    stat_rows = [
        (f"Correct: {stats.correct}", f"Accuracy: {stats.accuracy:.0f}%"),
        (f"Streak: {stats.current_streak}", f"Best: {stats.best_streak}"),
        (f"Avg Time: {stats.avg_time:.1f}s", f"Tier: {TIME_LIMITS[tier]:.0f}s"),
    ]
    y = 44
    for left, right in stat_rows:
        left_surf = fonts["stat"].render(left, True, palette.MUTED_TEXT_COLOR)
        right_surf = fonts["stat"].render(right, True, palette.MUTED_TEXT_COLOR)
        win.blit(left_surf, (40, y))
        win.blit(right_surf, (WIN_WIDTH - 40 - right_surf.get_width(), y))
        y += 28

    question_bottom = _draw_question(win, fonts, question.prompt, 260, WIN_WIDTH - 80)
    if category_hint:
        _center_text(win, fonts["small"], category_hint, question_bottom + 24, palette.MUTED_TEXT_COLOR)

    input_box.draw(win, fonts["input"])
    submit_button.draw(win, fonts["button"])

    if feedback is not None:
        color = palette.CORRECT_COLOR if feedback["correct"] else palette.INCORRECT_COLOR
        message = "Correct!" if feedback["correct"] else f"Answer: {feedback['correct_answer']}"
        pygame.draw.rect(win, color, (0, WIN_HEIGHT - 150, WIN_WIDTH, 6))
        _center_text(win, fonts["stat"], message, WIN_HEIGHT - 118, color)

    ratio = 0.0 if timer_limit <= 0 else max(0.0, timer_remaining / timer_limit)
    if ratio > 0.5:
        timer_color = palette.TIMER_SAFE_COLOR
    elif ratio > 0.2:
        timer_color = palette.TIMER_WARN_COLOR
    else:
        timer_color = palette.TIMER_DANGER_COLOR

    bar_rect = pygame.Rect(60, WIN_HEIGHT - 70, WIN_WIDTH - 120, 14)
    pygame.draw.rect(win, palette.TIMER_TRACK_COLOR, bar_rect, border_radius=7)
    fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, int(bar_rect.width * ratio), bar_rect.height)
    if fill_rect.width > 0:
        pygame.draw.rect(win, timer_color, fill_rect, border_radius=7)
    _center_text(win, fonts["timer"], f"Time: {timer_remaining:.1f}s", WIN_HEIGHT - 40, timer_color)

    badge.draw(win, WIN_WIDTH, fonts, logo)
    pygame.display.update()


def _draw_results(win, fonts, category_labels, session, persistent, play_again_button, logo):
    win.fill(palette.BG_COLOR)
    _center_text(win, fonts["title"], "SESSION COMPLETE", 70)

    lines = [
        f"Correct:        {session.correct}",
        f"Incorrect:      {session.incorrect}",
        f"Accuracy:       {session.accuracy:.1f}%",
        f"Best Streak:    {session.best_streak}",
        f"Avg. Time:      {session.avg_time:.1f}s",
    ]
    y = 150
    for line in lines:
        _center_text(win, fonts["stat"], line, y)
        y += 32

    strongest, weakest = session.strongest_and_weakest()
    y += 14
    if strongest:
        category, pct = strongest
        label = category_labels.get(category, category)
        _center_text(win, fonts["label"], f"Strongest Category: {label} — {pct:.0f}%", y, palette.CORRECT_COLOR)
        y += 26
    if weakest:
        category, pct = weakest
        label = category_labels.get(category, category)
        _center_text(win, fonts["label"], f"Weakest Category: {label} — {pct:.0f}%", y, palette.INCORRECT_COLOR)
        y += 26

    y += 20
    _center_text(win, fonts["small"], "All-Time Bests", y, palette.MUTED_TEXT_COLOR)
    y += 22
    fastest = persistent.data["fastest_avg_time"]
    fastest_part = f"Fastest Avg: {fastest:.1f}s" if fastest is not None else "Fastest Avg: —"
    best_line = (
        f"Best Accuracy: {persistent.data['best_accuracy']:.0f}%   "
        f"Best Streak: {persistent.data['best_streak']}   "
        f"{fastest_part}"
    )
    _center_text(win, fonts["small"], best_line, y, palette.MUTED_TEXT_COLOR)

    play_again_button.draw(win, fonts["button"])
    badge.draw(win, WIN_WIDTH, fonts, logo)
    pygame.display.update()


# ---------------------------------------------------------------------------
# Game-flow helpers (kept as free functions, not methods, so they're easy to
# unit test directly the same way games/mental_math/main.py's are)
# ---------------------------------------------------------------------------

def format_answer(question):
    if float(question.answer).is_integer():
        return f"{question.answer:.0f}"
    return f"{question.answer:.2f}"


def start_session(engine, duration_seconds, input_rect, submit_rect):
    difficulty = DifficultyManager()
    game = {
        "engine": engine,
        "difficulty": difficulty,
        "stats": SessionStats(),
        "q_timer": None,  # set below via a fresh QuestionTimer import to avoid a circular top-level import
        "s_timer": None,
        "input_box": widgets.TextInput(input_rect),
        "submit_button": widgets.Button(submit_rect, "SUBMIT"),
        "feedback": None,
        "feedback_until": 0.0,
        "session_over": False,
    }
    from .timer import QuestionTimer, SessionTimer  # local import: keeps timer.py dependency-free of engine
    game["q_timer"] = QuestionTimer()
    game["s_timer"] = SessionTimer(duration_seconds)
    game["question"] = engine.next_question(difficulty.tier)
    game["q_timer"].start(difficulty.time_limit)
    return game


def record_result(game, correct, response_time, timed_out, category_hint_fn=None):
    question = game["question"]
    game["stats"].record(question.category, correct, response_time, timed_out=timed_out)
    game["difficulty"].record_result(correct)
    game["feedback"] = {"correct": correct, "correct_answer": format_answer(question)}
    game["feedback_until"] = pygame.time.get_ticks() / 1000.0 + FEEDBACK_DISPLAY_SECONDS
    game["input_box"].clear()


def submit_answer(game):
    question = game["question"]
    raw = game["input_box"].text
    response_time = game["q_timer"].elapsed()
    correct = bool(raw.strip()) and check_answer(question, raw)
    record_result(game, correct, response_time, timed_out=False)


def advance_question(game):
    game["input_box"].clear()
    if game["s_timer"].is_over():
        game["session_over"] = True
        game["feedback"] = None
        return
    game["question"] = game["engine"].next_question(game["difficulty"].tier)
    game["q_timer"].start(game["difficulty"].time_limit)
    game["feedback"] = None


def update_playing(game):
    if game["session_over"]:
        return
    if game["feedback"] is not None:
        if pygame.time.get_ticks() / 1000.0 >= game["feedback_until"]:
            advance_question(game)
        return
    if game["s_timer"].is_over():
        game["session_over"] = True
        return
    if game["q_timer"].expired():
        record_result(game, correct=False, response_time=game["difficulty"].time_limit, timed_out=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(title, tagline, category_generators, category_weights, category_labels,
        stats_path, category_hint_fn=None, caption=None):
    """Runs a full menu -> playing -> results game loop. Blocks until the
    window is closed. See games/probability_quickfire.py for usage."""
    pygame.init()
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption(caption or title)
    fonts = _load_fonts()
    logo = badge.load_logo()
    clock = pygame.time.Clock()

    persistent = PersistentStats(stats_path)
    engine = QuestionEngine(category_generators, category_weights)

    category_rows = _wrap_categories(list(category_labels.values()))

    session_buttons = [
        widgets.Button((45 + i * 138, 300, 122, 40), label)
        for i, (label, _seconds) in enumerate(SESSION_LENGTHS)
    ]
    start_button = widgets.Button((WIN_WIDTH // 2 - 90, 380, 180, 50), "START GAME")
    play_again_button = widgets.Button((WIN_WIDTH // 2 - 100, 620, 200, 50), "PLAY AGAIN")

    input_rect = (WIN_WIDTH // 2 - 110, 340, 220, 56)
    submit_rect = (WIN_WIDTH // 2 - 80, 410, 160, 44)

    state = "menu"
    session_index = DEFAULT_SESSION_INDEX
    game = None

    running = True
    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif state == "menu":
                for i, button in enumerate(session_buttons):
                    if button.is_clicked(event):
                        session_index = i
                if start_button.is_clicked(event):
                    duration = SESSION_LENGTHS[session_index][1]
                    game = start_session(engine, duration, input_rect, submit_rect)
                    state = "playing"

            elif state == "playing":
                submitted = game["input_box"].handle_event(event) == "submit"
                if game["submit_button"].is_clicked(event):
                    submitted = True
                if submitted and game["feedback"] is None:
                    submit_answer(game)

            elif state == "results":
                if play_again_button.is_clicked(event):
                    state = "menu"

        if state == "playing":
            update_playing(game)
            if game["session_over"]:
                persistent.update_from_session(game["stats"])
                state = "results"

        if state == "menu":
            _draw_menu(win, fonts, title, tagline, category_rows, session_buttons, session_index, start_button, logo)
        elif state == "playing":
            hint = category_hint_fn(game["question"]) if category_hint_fn else None
            _draw_game(win, fonts, game["question"], game["input_box"], game["submit_button"],
                       game["q_timer"].remaining(), game["difficulty"].time_limit,
                       game["stats"], game["difficulty"].tier, game["feedback"], hint, logo)
        elif state == "results":
            _draw_results(win, fonts, category_labels, game["stats"], persistent, play_again_button, logo)

    pygame.quit()


def _wrap_categories(labels, per_row=4):
    rows = []
    for i in range(0, len(labels), per_row):
        rows.append(" · ".join(labels[i:i + per_row]))
    return rows
