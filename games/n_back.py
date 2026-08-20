"""
N-Back Trainer
------------------------------------------------------------
Classic working-memory drill: a letter appears roughly every ~2 seconds.
Press SPACE if it's the same letter that appeared N steps back in the
sequence; do nothing if it isn't. Consistent accuracy over a block of
trials raises N; a rough block drops it back down - same promote/demote
shape as mental_math's time tiers, just driven by N instead of seconds.

Run:
    python n_back.py            (from inside games/)
    python games/n_back.py      (from the repo root)

Dependencies: pip install pygame-ce   (plain "pygame" also works)
"""

import os
import sys
import json
import random

import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import palette, widgets, badge
from common.timer import QuestionTimer, SessionTimer

# ---- Config ----
WIN_WIDTH = 640
WIN_HEIGHT = 640
FPS = 60

STIMULUS_LETTERS = ["C", "H", "K", "L", "Q", "R", "S", "T"]  # no vowels - avoids accidental word cues
STIMULUS_DISPLAY_SECONDS = 0.6   # how long the letter itself is visible
STIMULUS_INTERVAL_SECONDS = 2.2  # time from one stimulus to the next
MATCH_PROBABILITY = 0.3          # rough fraction of trials nudged to be deliberate matches

MIN_N, MAX_N = 1, 5
TRIALS_PER_BLOCK = 20
PROMOTE_ACCURACY = 90.0   # block accuracy % needed to move N up
DEMOTE_ACCURACY = 60.0    # block accuracy % that moves N down

SESSION_LENGTHS = [("1 min", 60), ("3 min", 180), ("5 min", 300), ("Endless", None)]
DEFAULT_SESSION_INDEX = 3

STATS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "n_back_stats.json")


# ---------------------------------------------------------------------------
# Persistent stats (local JSON, same pattern as the rest of the family)
# ---------------------------------------------------------------------------

def load_persistent():
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"best_n": MIN_N, "best_accuracy": 0.0, "total_trials": 0, "sessions_played": 0}


def save_persistent(data):
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Game state / flow
# ---------------------------------------------------------------------------

def next_letter(sequence, n):
    """Picks the next stimulus, nudged toward MATCH_PROBABILITY so n-back
    matches show up at a reasonable rate instead of only by chance."""
    if len(sequence) >= n and random.random() < MATCH_PROBABILITY:
        return sequence[-n]
    return random.choice(STIMULUS_LETTERS)


def new_session(duration_seconds):
    return {
        "n": MIN_N,
        "sequence": [],
        "trial_index": 0,
        "responded": False,
        "stim_timer": QuestionTimer(),
        "s_timer": SessionTimer(duration_seconds),
        "hits": 0, "misses": 0, "false_alarms": 0, "correct_rejections": 0,
        "block_correct": 0, "block_scored": 0,
        "best_n": MIN_N,
        "last_feedback": None,
        "feedback_until": 0.0,
        "session_over": False,
    }


def start_trial(game):
    letter = next_letter(game["sequence"], game["n"])
    game["sequence"].append(letter)
    game["trial_index"] += 1
    game["responded"] = False
    game["stim_timer"].start(STIMULUS_INTERVAL_SECONDS)


def register_response(game):
    if game["trial_index"] > game["n"]:  # only meaningful once there's enough history to judge
        game["responded"] = True


def finalize_trial(game):
    n = game["n"]
    idx = game["trial_index"]
    if idx > n:
        is_match = game["sequence"][-1] == game["sequence"][-1 - n]
        responded = game["responded"]
        if is_match and responded:
            game["hits"] += 1
            game["block_correct"] += 1
            game["last_feedback"] = "hit"
        elif is_match and not responded:
            game["misses"] += 1
            game["last_feedback"] = "miss"
        elif not is_match and responded:
            game["false_alarms"] += 1
            game["last_feedback"] = "false_alarm"
        else:
            game["correct_rejections"] += 1
            game["block_correct"] += 1
            game["last_feedback"] = "correct_rejection"
        game["block_scored"] += 1
        game["feedback_until"] = pygame.time.get_ticks() / 1000.0 + 0.5
    else:
        game["last_feedback"] = None

    if game["block_scored"] >= TRIALS_PER_BLOCK:
        block_accuracy = game["block_correct"] / game["block_scored"] * 100
        if block_accuracy >= PROMOTE_ACCURACY and game["n"] < MAX_N:
            game["n"] += 1
        elif block_accuracy <= DEMOTE_ACCURACY and game["n"] > MIN_N:
            game["n"] -= 1
        game["best_n"] = max(game["best_n"], game["n"])
        game["block_correct"] = 0
        game["block_scored"] = 0


def update(game):
    if game["session_over"]:
        return
    if game["s_timer"].is_over():
        game["session_over"] = True
        return
    if game["stim_timer"].expired():
        finalize_trial(game)
        start_trial(game)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def load_fonts():
    return {
        "title": pygame.font.SysFont(palette.FONT_NAME, 40, bold=True),
        "stimulus": pygame.font.SysFont(palette.FONT_NAME, 110, bold=True),
        "label": pygame.font.SysFont(palette.FONT_NAME, 18),
        "stat": pygame.font.SysFont(palette.FONT_NAME, 18, bold=True),
        "button": pygame.font.SysFont(palette.FONT_NAME, 22, bold=True),
        "small": pygame.font.SysFont(palette.FONT_NAME, 15),
        "n_level": pygame.font.SysFont(palette.FONT_NAME, 26, bold=True),
        "badge": pygame.font.SysFont(palette.FONT_NAME, 15, bold=True),
    }


def _center_text(win, font, text, y, color=palette.TEXT_COLOR):
    surf = font.render(text, True, color)
    win.blit(surf, surf.get_rect(center=(WIN_WIDTH // 2, y)))


def draw_menu(win, fonts, session_buttons, selected_index, start_button, logo):
    win.fill(palette.BG_COLOR)
    _center_text(win, fonts["title"], "N-BACK TRAINER", 70)
    _center_text(win, fonts["small"], "CITADEL INTERVIEW PREP", 102, palette.ACCENT_COLOR)
    _center_text(win, fonts["label"], "A letter appears every ~2 seconds.", 138, palette.MUTED_TEXT_COLOR)
    _center_text(win, fonts["label"], "Press SPACE if it matches N steps back.", 162, palette.MUTED_TEXT_COLOR)
    _center_text(win, fonts["small"], "Consistent accuracy raises N; a rough block lowers it.", 190, palette.MUTED_TEXT_COLOR)

    _center_text(win, fonts["label"], "Session length", 280, palette.MUTED_TEXT_COLOR)
    for i, button in enumerate(session_buttons):
        selected = i == selected_index
        color = palette.ACCENT_COLOR if selected else palette.BUTTON_IDLE_COLOR
        text_color = palette.ACCENT_TEXT_COLOR if selected else palette.TEXT_COLOR
        button.draw(win, fonts["label"], color=color, text_color=text_color)

    start_button.draw(win, fonts["button"])
    badge.draw(win, WIN_WIDTH, fonts, logo)
    pygame.display.update()


_FEEDBACK_LABELS = {
    "hit": ("Hit", palette.CORRECT_COLOR),
    "correct_rejection": ("Correct", palette.CORRECT_COLOR),
    "miss": ("Miss", palette.INCORRECT_COLOR),
    "false_alarm": ("False alarm", palette.INCORRECT_COLOR),
}


def draw_game(win, fonts, game, logo):
    win.fill(palette.BG_COLOR)

    _center_text(win, fonts["n_level"], f"N = {game['n']}", 40, palette.ACCENT_COLOR)

    total_scored = game["hits"] + game["misses"] + game["false_alarms"] + game["correct_rejections"]
    accuracy = ((game["hits"] + game["correct_rejections"]) / total_scored * 100) if total_scored else 0.0
    stat_rows = [
        (f"Hits: {game['hits']}", f"Accuracy: {accuracy:.0f}%"),
        (f"Misses: {game['misses']}", f"False Alarms: {game['false_alarms']}"),
    ]
    y = 78
    for left, right in stat_rows:
        left_surf = fonts["stat"].render(left, True, palette.MUTED_TEXT_COLOR)
        right_surf = fonts["stat"].render(right, True, palette.MUTED_TEXT_COLOR)
        win.blit(left_surf, (40, y))
        win.blit(right_surf, (WIN_WIDTH - 40 - right_surf.get_width(), y))
        y += 26

    _center_text(win, fonts["small"], f"Block: {game['block_scored']}/{TRIALS_PER_BLOCK} trials", y + 8, palette.MUTED_TEXT_COLOR)

    elapsed = game["stim_timer"].elapsed()
    if game["sequence"] and elapsed < STIMULUS_DISPLAY_SECONDS:
        _center_text(win, fonts["stimulus"], game["sequence"][-1], WIN_HEIGHT // 2)
    else:
        _center_text(win, fonts["stimulus"], "•", WIN_HEIGHT // 2, palette.MUTED_TEXT_COLOR)

    ratio = max(0.0, game["stim_timer"].remaining() / STIMULUS_INTERVAL_SECONDS)
    bar_rect = pygame.Rect(60, WIN_HEIGHT - 110, WIN_WIDTH - 120, 10)
    pygame.draw.rect(win, palette.TIMER_TRACK_COLOR, bar_rect, border_radius=5)
    fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, int(bar_rect.width * ratio), bar_rect.height)
    if fill_rect.width > 0:
        pygame.draw.rect(win, palette.ACCENT_COLOR, fill_rect, border_radius=5)

    if game["last_feedback"] and pygame.time.get_ticks() / 1000.0 < game["feedback_until"]:
        label, color = _FEEDBACK_LABELS[game["last_feedback"]]
        _center_text(win, fonts["label"], label, WIN_HEIGHT - 70, color)

    _center_text(win, fonts["small"], "SPACE = matches N-back", WIN_HEIGHT - 40, palette.MUTED_TEXT_COLOR)
    badge.draw(win, WIN_WIDTH, fonts, logo)
    pygame.display.update()


def draw_results(win, fonts, game, persistent, play_again_button, logo):
    win.fill(palette.BG_COLOR)
    _center_text(win, fonts["title"], "SESSION COMPLETE", 60)

    total_scored = game["hits"] + game["misses"] + game["false_alarms"] + game["correct_rejections"]
    accuracy = ((game["hits"] + game["correct_rejections"]) / total_scored * 100) if total_scored else 0.0

    lines = [
        f"Peak N reached:      {game['best_n']}",
        f"Accuracy:            {accuracy:.1f}%",
        f"Hits:                {game['hits']}",
        f"Misses:              {game['misses']}",
        f"False Alarms:        {game['false_alarms']}",
        f"Correct Rejections:  {game['correct_rejections']}",
    ]
    y = 140
    for line in lines:
        _center_text(win, fonts["stat"], line, y)
        y += 30

    y += 20
    _center_text(win, fonts["small"], "All-Time Bests", y, palette.MUTED_TEXT_COLOR)
    y += 22
    _center_text(win, fonts["small"],
                 f"Best N: {persistent['best_n']}   Best Accuracy: {persistent['best_accuracy']:.0f}%",
                 y, palette.MUTED_TEXT_COLOR)

    play_again_button.draw(win, fonts["button"])
    badge.draw(win, WIN_WIDTH, fonts, logo)
    pygame.display.update()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("N-Back Trainer — Citadel Prep")
    fonts = load_fonts()
    logo = badge.load_logo()
    clock = pygame.time.Clock()

    persistent = load_persistent()

    session_buttons = [
        widgets.Button((45 + i * 138, 300, 122, 40), label)
        for i, (label, _s) in enumerate(SESSION_LENGTHS)
    ]
    start_button = widgets.Button((WIN_WIDTH // 2 - 90, 380, 180, 50), "START GAME")
    play_again_button = widgets.Button((WIN_WIDTH // 2 - 100, 460, 200, 50), "PLAY AGAIN")

    state = "menu"
    session_index = DEFAULT_SESSION_INDEX
    game = None

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif state == "menu":
                for i, button in enumerate(session_buttons):
                    if button.is_clicked(event):
                        session_index = i
                if start_button.is_clicked(event):
                    duration = SESSION_LENGTHS[session_index][1]
                    game = new_session(duration)
                    start_trial(game)
                    state = "playing"

            elif state == "playing":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    register_response(game)

            elif state == "results":
                if play_again_button.is_clicked(event):
                    state = "menu"

        if state == "playing":
            update(game)
            if game["session_over"]:
                persistent["best_n"] = max(persistent["best_n"], game["best_n"])
                total_scored = game["hits"] + game["misses"] + game["false_alarms"] + game["correct_rejections"]
                if total_scored:
                    accuracy = (game["hits"] + game["correct_rejections"]) / total_scored * 100
                    persistent["best_accuracy"] = max(persistent["best_accuracy"], accuracy)
                    persistent["total_trials"] += total_scored
                persistent["sessions_played"] += 1
                save_persistent(persistent)
                state = "results"

        if state == "menu":
            draw_menu(win, fonts, session_buttons, session_index, start_button, logo)
        elif state == "playing":
            draw_game(win, fonts, game, logo)
        elif state == "results":
            draw_results(win, fonts, game, persistent, play_again_button, logo)

    pygame.quit()


if __name__ == "__main__":
    main()
