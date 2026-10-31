"""
Mental Math - Speed Training Game
------------------------------------------------------------
Adaptive mental-arithmetic reflex trainer. One problem at a time, type the
answer, hit Enter. Consistent accuracy speeds the clock up (45s -> 30s ->
15s); a rough patch slows it back down. See package docstring (__init__.py)
for how the modules split up.

Run:
    python main.py          (from inside games/mental_math/)
    python games/mental_math/main.py   (from the repo root)

Dependencies: pip install pygame-ce   (plain "pygame" also works - same API)
"""

import os
import sys

import pygame

# Let this file run directly (python main.py) while its sibling modules
# still use clean relative imports internally - put the *games* folder on
# sys.path so "mental_math" resolves as a normal package from outside it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mental_math import config, ui
from mental_math.questions import QuestionEngine, check_answer
from mental_math.difficulty import DifficultyManager
from mental_math.stats import SessionStats, PersistentStats
from mental_math.timer import QuestionTimer, SessionTimer


def _format_answer(question):
    if question.category == "fraction":
        return f"{question.answer:.3f}"
    return f"{question.answer:.0f}"


def _category_hint(question):
    if question.category == "fraction":
        return "Round to 3 decimal places — small tolerance allowed"
    return None


def _start_session(duration_seconds):
    difficulty = DifficultyManager()
    engine = QuestionEngine()
    game = {
        "engine": engine,
        "difficulty": difficulty,
        "stats": SessionStats(),
        "q_timer": QuestionTimer(),
        "s_timer": SessionTimer(duration_seconds),
        "input_box": ui.TextInput((config.WIN_WIDTH // 2 - 110, 340, 220, 56)),
        "submit_button": ui.Button((config.WIN_WIDTH // 2 - 80, 410, 160, 44), "SUBMIT"),
        "feedback": None,
        "feedback_until": 0.0,
        "session_over": False,
    }
    game["question"] = engine.next_question(difficulty.tier)
    game["q_timer"].start(difficulty.time_limit)
    return game


def _record_result(game, correct, response_time, timed_out):
    question = game["question"]
    game["stats"].record(question.category, correct, response_time, timed_out=timed_out)
    game["difficulty"].record_result(correct)
    game["feedback"] = {"correct": correct, "correct_answer": _format_answer(question)}
    game["feedback_until"] = pygame.time.get_ticks() / 1000.0 + config.FEEDBACK_DISPLAY_SECONDS
    game["input_box"].clear()


def _submit_answer(game):
    question = game["question"]
    raw = game["input_box"].text
    response_time = game["q_timer"].elapsed()
    correct = bool(raw.strip()) and check_answer(question, raw)
    _record_result(game, correct, response_time, timed_out=False)


def _advance_question(game):
    game["input_box"].clear()
    if game["s_timer"].is_over():
        game["session_over"] = True
        game["feedback"] = None
        return
    game["question"] = game["engine"].next_question(game["difficulty"].tier)
    game["q_timer"].start(game["difficulty"].time_limit)
    game["feedback"] = None


def _update_playing(game):
    if game["session_over"]:
        return

    if game["feedback"] is not None:
        if pygame.time.get_ticks() / 1000.0 >= game["feedback_until"]:
            _advance_question(game)
        return

    if game["s_timer"].is_over():
        game["session_over"] = True
        return

    if game["q_timer"].expired():
        _record_result(game, correct=False, response_time=game["difficulty"].time_limit, timed_out=True)


def main():
    pygame.init()
    win = pygame.display.set_mode((config.WIN_WIDTH, config.WIN_HEIGHT))
    pygame.display.set_caption("Mental Math — Citadel Prep")
    fonts = ui.load_fonts()
    logo = ui.load_logo()  # None until you drop a file in games/mental_math/assets/
    clock = pygame.time.Clock()

    persistent = PersistentStats()

    session_buttons = [
        ui.Button((45 + i * 138, 300, 122, 40), label)
        for i, (label, _seconds) in enumerate(config.SESSION_LENGTHS)
    ]
    start_button = ui.Button((config.WIN_WIDTH // 2 - 90, 380, 180, 50), "START GAME")
    play_again_button = ui.Button((config.WIN_WIDTH // 2 - 100, 620, 200, 50), "PLAY AGAIN")

    state = "menu"
    session_index = config.DEFAULT_SESSION_INDEX
    game = None

    running = True
    while running:
        clock.tick(config.FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif state == "menu":
                for i, button in enumerate(session_buttons):
                    if button.is_clicked(event):
                        session_index = i
                if start_button.is_clicked(event):
                    duration = config.SESSION_LENGTHS[session_index][1]
                    game = _start_session(duration)
                    state = "playing"

            elif state == "playing":
                submitted = game["input_box"].handle_event(event) == "submit"
                if game["submit_button"].is_clicked(event):
                    submitted = True
                if submitted and game["feedback"] is None:
                    _submit_answer(game)

            elif state == "results":
                if play_again_button.is_clicked(event):
                    state = "menu"

        if state == "playing":
            _update_playing(game)
            if game["session_over"]:
                persistent.update_from_session(game["stats"])
                state = "results"

        if state == "menu":
            ui.draw_menu(win, fonts, session_buttons, session_index, start_button, logo=logo)
        elif state == "playing":
            ui.draw_game(
                win, fonts, game["question"], game["input_box"], game["submit_button"],
                game["q_timer"].remaining(), game["difficulty"].time_limit,
                game["stats"], game["difficulty"].tier, game["feedback"],
                category_hint=_category_hint(game["question"]), logo=logo,
            )
        elif state == "results":
            ui.draw_results(win, fonts, game["stats"], persistent, play_again_button, logo=logo)

    pygame.quit()


if __name__ == "__main__":
    main()
