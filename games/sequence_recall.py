"""
Sequence Recall (Digit Span / Simon-style)
------------------------------------------------------------
Working-memory drill in the classic "digit span" family: a sequence of
numbered tiles lights up one at a time, then you repeat it back - by
clicking the tiles or pressing the matching number keys, in order. Get it
right and the sequence grows by one; get it wrong and it shrinks back down.
A simplified single-trial-per-length staircase (the clinical digit-span test
uses two trials per length) - close enough for training, much simpler to
build.

Run:
    python sequence_recall.py           (from inside games/)
    python games/sequence_recall.py     (from the repo root)

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
WIN_WIDTH = 560
WIN_HEIGHT = 680
FPS = 60

TILE_COUNT = 9
TILE_ON_SECONDS = 0.55
TILE_GAP_SECONDS = 0.15
FEEDBACK_SECONDS = 1.0

START_SPAN = 3
MIN_SPAN = 3
MAX_SPAN = 12

SESSION_LENGTHS = [("1 min", 60), ("3 min", 180), ("5 min", 300), ("Endless", None)]
DEFAULT_SESSION_INDEX = 3

STATS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sequence_recall_stats.json")

TILE_KEYS = {
    pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3, pygame.K_4: 4, pygame.K_5: 5,
    pygame.K_6: 6, pygame.K_7: 7, pygame.K_8: 8, pygame.K_9: 9,
    pygame.K_KP1: 1, pygame.K_KP2: 2, pygame.K_KP3: 3, pygame.K_KP4: 4, pygame.K_KP5: 5,
    pygame.K_KP6: 6, pygame.K_KP7: 7, pygame.K_KP8: 8, pygame.K_KP9: 9,
}


def load_persistent():
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"best_span": 0, "rounds_correct": 0, "rounds_total": 0, "sessions_played": 0}


def save_persistent(data):
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def tile_rects():
    """3x3 grid of tile rects, numbered 1-9 in reading order."""
    size, gap = 120, 16
    grid_w = size * 3 + gap * 2
    left = (WIN_WIDTH - grid_w) // 2
    top = 220
    rects = {}
    for i in range(9):
        row, col = divmod(i, 3)
        x = left + col * (size + gap)
        y = top + row * (size + gap)
        rects[i + 1] = pygame.Rect(x, y, size, size)
    return rects


def new_sequence(length):
    return [random.randint(1, TILE_COUNT) for _ in range(length)]


def new_session(duration_seconds):
    game = {
        "span": START_SPAN,
        "sequence": new_sequence(START_SPAN),
        "phase": "show",
        "show_index": 0,
        "show_timer": QuestionTimer(),
        "showing_tile": None,
        "input_index": 0,
        "feedback_timer": QuestionTimer(),
        "last_result": None,
        "correct_rounds": 0,
        "total_rounds": 0,
        "best_span": 0,
        "s_timer": SessionTimer(duration_seconds),
        "session_over": False,
    }
    start_show_step(game)
    return game


def start_show_step(game):
    game["show_timer"].start(TILE_ON_SECONDS)
    game["showing_tile"] = game["sequence"][game["show_index"]]


def begin_round(game, span):
    game["span"] = span
    game["sequence"] = new_sequence(span)
    game["phase"] = "show"
    game["show_index"] = 0
    game["input_index"] = 0
    game["last_result"] = None
    start_show_step(game)


def advance_show(game):
    """Called once the current tile's on/gap timing elapses."""
    if game["showing_tile"] is not None:
        game["showing_tile"] = None
        game["show_timer"].start(TILE_GAP_SECONDS)
        return
    game["show_index"] += 1
    if game["show_index"] >= len(game["sequence"]):
        game["phase"] = "input"
    else:
        start_show_step(game)


def submit_tile(game, tile_number):
    if game["phase"] != "input":
        return
    expected = game["sequence"][game["input_index"]]
    if tile_number != expected:
        finish_round(game, correct=False)
        return
    game["input_index"] += 1
    if game["input_index"] >= len(game["sequence"]):
        finish_round(game, correct=True)


def finish_round(game, correct):
    attempted_span = len(game["sequence"])
    game["total_rounds"] += 1
    if correct:
        game["correct_rounds"] += 1
        game["last_result"] = "correct"
        game["best_span"] = max(game["best_span"], attempted_span)
        game["span"] = min(MAX_SPAN, attempted_span + 1)
    else:
        game["last_result"] = "wrong"
        game["span"] = max(MIN_SPAN, attempted_span - 1)
    game["phase"] = "feedback"
    game["feedback_timer"].start(FEEDBACK_SECONDS)


def update(game):
    if game["session_over"]:
        return
    if game["s_timer"].is_over() and game["phase"] != "show":
        # let an in-progress flash finish rather than cutting it off mid-sequence
        game["session_over"] = True
        return
    if game["phase"] == "show" and game["show_timer"].expired():
        advance_show(game)
    elif game["phase"] == "feedback" and game["feedback_timer"].expired():
        if game["s_timer"].is_over():
            game["session_over"] = True
        else:
            begin_round(game, game["span"])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def load_fonts():
    return {
        "title": pygame.font.SysFont(palette.FONT_NAME, 36, bold=True),
        "label": pygame.font.SysFont(palette.FONT_NAME, 18),
        "small": pygame.font.SysFont(palette.FONT_NAME, 15),
        "stat": pygame.font.SysFont(palette.FONT_NAME, 18, bold=True),
        "button": pygame.font.SysFont(palette.FONT_NAME, 22, bold=True),
        "tile": pygame.font.SysFont(palette.FONT_NAME, 40, bold=True),
        "badge": pygame.font.SysFont(palette.FONT_NAME, 15, bold=True),
    }


def _center_text(win, font, text, y, color=palette.TEXT_COLOR):
    surf = font.render(text, True, color)
    win.blit(surf, surf.get_rect(center=(WIN_WIDTH // 2, y)))


def draw_menu(win, fonts, session_buttons, selected_index, start_button, logo):
    win.fill(palette.BG_COLOR)
    _center_text(win, fonts["title"], "SEQUENCE RECALL", 60)
    _center_text(win, fonts["small"], "CITADEL INTERVIEW PREP", 92, palette.ACCENT_COLOR)
    _center_text(win, fonts["label"], "Watch the tiles light up, then repeat the", 128, palette.MUTED_TEXT_COLOR)
    _center_text(win, fonts["label"], "sequence - click tiles or press 1-9.", 150, palette.MUTED_TEXT_COLOR)
    _center_text(win, fonts["small"], "Right = sequence grows. Wrong = it shrinks.", 178, palette.MUTED_TEXT_COLOR)

    _center_text(win, fonts["label"], "Session length", 260, palette.MUTED_TEXT_COLOR)
    for i, button in enumerate(session_buttons):
        selected = i == selected_index
        color = palette.ACCENT_COLOR if selected else palette.BUTTON_IDLE_COLOR
        text_color = palette.ACCENT_TEXT_COLOR if selected else palette.TEXT_COLOR
        button.draw(win, fonts["label"], color=color, text_color=text_color)

    start_button.draw(win, fonts["button"])
    badge.draw(win, WIN_WIDTH, fonts, logo)
    pygame.display.update()


def draw_game(win, fonts, game, rects, logo):
    win.fill(palette.BG_COLOR)
    _center_text(win, fonts["title"], "SEQUENCE RECALL", 40)
    _center_text(win, fonts["small"], f"Span: {game['span']}   Best: {game['best_span']}", 72, palette.MUTED_TEXT_COLOR)

    total = game["total_rounds"]
    accuracy = (game["correct_rounds"] / total * 100) if total else 0.0
    _center_text(win, fonts["small"], f"Rounds: {total}   Accuracy: {accuracy:.0f}%", 94, palette.MUTED_TEXT_COLOR)

    if game["phase"] == "show":
        phase_label, color = "Watch the sequence...", palette.TEXT_COLOR
    elif game["phase"] == "input":
        phase_label = f"Repeat it back ({game['input_index']}/{len(game['sequence'])})"
        color = palette.TEXT_COLOR
    else:
        correct = game["last_result"] == "correct"
        phase_label = "Correct!" if correct else "Wrong"
        color = palette.CORRECT_COLOR if correct else palette.INCORRECT_COLOR
    _center_text(win, fonts["label"], phase_label, 136, color)

    if game["phase"] == "feedback" and game["last_result"] == "wrong":
        seq_text = " - ".join(str(n) for n in game["sequence"])
        _center_text(win, fonts["small"], f"Sequence was: {seq_text}", 162, palette.MUTED_TEXT_COLOR)

    for number, rect in rects.items():
        fill = palette.BUTTON_IDLE_COLOR
        if game["phase"] == "show" and game["showing_tile"] == number:
            fill = palette.ACCENT_COLOR
        pygame.draw.rect(win, fill, rect, border_radius=10)
        label_color = palette.ACCENT_TEXT_COLOR if fill == palette.ACCENT_COLOR else palette.TEXT_COLOR
        surf = fonts["tile"].render(str(number), True, label_color)
        win.blit(surf, surf.get_rect(center=rect.center))

    _center_text(win, fonts["small"], "Click tiles or press 1-9", WIN_HEIGHT - 30, palette.MUTED_TEXT_COLOR)
    badge.draw(win, WIN_WIDTH, fonts, logo)
    pygame.display.update()


def draw_results(win, fonts, game, persistent, play_again_button, logo):
    win.fill(palette.BG_COLOR)
    _center_text(win, fonts["title"], "SESSION COMPLETE", 60)

    total = game["total_rounds"]
    accuracy = (game["correct_rounds"] / total * 100) if total else 0.0
    lines = [
        f"Best span this session: {game['best_span']}",
        f"Rounds played:          {total}",
        f"Rounds correct:         {game['correct_rounds']}",
        f"Accuracy:               {accuracy:.1f}%",
    ]
    y = 150
    for line in lines:
        _center_text(win, fonts["stat"], line, y)
        y += 32

    y += 20
    _center_text(win, fonts["small"], "All-Time Bests", y, palette.MUTED_TEXT_COLOR)
    y += 22
    _center_text(win, fonts["small"], f"Best Span Ever: {persistent['best_span']}", y, palette.MUTED_TEXT_COLOR)

    play_again_button.draw(win, fonts["button"])
    badge.draw(win, WIN_WIDTH, fonts, logo)
    pygame.display.update()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Sequence Recall — Citadel Prep")
    fonts = load_fonts()
    logo = badge.load_logo()
    clock = pygame.time.Clock()
    rects = tile_rects()

    persistent = load_persistent()

    session_buttons = [
        widgets.Button((30 + i * 128, 290, 112, 40), label)
        for i, (label, _s) in enumerate(SESSION_LENGTHS)
    ]
    start_button = widgets.Button((WIN_WIDTH // 2 - 90, 360, 180, 50), "START GAME")
    play_again_button = widgets.Button((WIN_WIDTH // 2 - 100, 480, 200, 50), "PLAY AGAIN")

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
                    state = "playing"

            elif state == "playing":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for number, rect in rects.items():
                        if rect.collidepoint(event.pos):
                            submit_tile(game, number)
                            break
                elif event.type == pygame.KEYDOWN and event.key in TILE_KEYS:
                    submit_tile(game, TILE_KEYS[event.key])

            elif state == "results":
                if play_again_button.is_clicked(event):
                    state = "menu"

        if state == "playing":
            update(game)
            if game["session_over"]:
                persistent["best_span"] = max(persistent["best_span"], game["best_span"])
                persistent["rounds_correct"] += game["correct_rounds"]
                persistent["rounds_total"] += game["total_rounds"]
                persistent["sessions_played"] += 1
                save_persistent(persistent)
                state = "results"

        if state == "menu":
            draw_menu(win, fonts, session_buttons, session_index, start_button, logo)
        elif state == "playing":
            draw_game(win, fonts, game, rects, logo)
        elif state == "results":
            draw_results(win, fonts, game, persistent, play_again_button, logo)

    pygame.quit()


if __name__ == "__main__":
    main()
