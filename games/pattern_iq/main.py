"""
Pattern IQ - Speed Pattern Recognition
------------------------------------------------------------
Two modes off one shell (pick on the menu):

    Number Sequences   IQ-test "what comes next" puzzles, multiple choice.
    Spot the Pattern   name the regime behind a generated price chart
                       (uptrend / downtrend / mean-reversion / random walk /
                       volatility breakout / cyclical) - built for the "is
                       this signal or noise" instinct that's useful for
                       market intuition.

Both modes share one adaptive-difficulty engine: pick a starting tier
(Easy/Medium/Hard) on the menu, then a streak of rights/wrongs nudges the
tier - and the clock - up or down automatically, same as games/mental_math.

Run:
    python main.py                       (from inside games/pattern_iq/)
    python games/pattern_iq/main.py       (from the repo root)

Dependencies: pip install pygame-ce   (plain "pygame" also works - same API)
"""

import os
import sys

import pygame

# Let this file run directly (python main.py) while its sibling modules
# still use clean relative imports internally - put the *games* folder on
# sys.path so "pattern_iq" resolves as a normal package from outside it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_iq import config, ui
from pattern_iq.sequences import SequenceEngine, check_answer as check_sequence_answer
from pattern_iq.market_patterns import MarketEngine, check_answer as check_market_answer
from pattern_iq.difficulty import DifficultyManager
from pattern_iq.stats import SessionStats, PersistentStats
from pattern_iq.timer import QuestionTimer, SessionTimer


_SEQ_CHOICE_RECTS = [
    (50, 340, 320, 64), (390, 340, 320, 64),
    (50, 414, 320, 64), (390, 414, 320, 64),
]
_MARKET_CHOICE_RECTS = [
    (50, 410, 320, 64), (390, 410, 320, 64),
    (50, 478, 320, 64), (390, 478, 320, 64),
]


def _start_session(mode, start_tier):
    if mode == config.MODE_SEQUENCE:
        engine = SequenceEngine()
        difficulty = DifficultyManager(config.SEQUENCE_TIME_LIMITS, start_tier=start_tier)
        question = engine.next_question(difficulty.tier)
        choice_rects = _SEQ_CHOICE_RECTS
        labels = [f"{c:,}" for c in question.choices]
    else:
        engine = MarketEngine()
        difficulty = DifficultyManager(config.MARKET_TIME_LIMITS, start_tier=start_tier)
        question = engine.next_question(difficulty.tier)
        choice_rects = _MARKET_CHOICE_RECTS
        labels = [config.MARKET_PATTERN_LABELS[c] for c in question.choices]

    game = {
        "mode": mode,
        "engine": engine,
        "difficulty": difficulty,
        "stats": SessionStats(),
        "q_timer": QuestionTimer(),
        "question": question,
        "choice_buttons": ui.make_choice_grid(choice_rects, question.choices, labels),
        "feedback": None,
        "feedback_until": 0.0,
        "session_over": False,
    }
    game["q_timer"].start(difficulty.time_limit)
    return game


def _next_question(game):
    question = game["engine"].next_question(game["difficulty"].tier)
    if game["mode"] == config.MODE_SEQUENCE:
        rects = _SEQ_CHOICE_RECTS
        labels = [f"{c:,}" for c in question.choices]
        category = question.category
    else:
        rects = _MARKET_CHOICE_RECTS
        labels = [config.MARKET_PATTERN_LABELS[c] for c in question.choices]
        category = question.label
    game["question"] = question
    game["choice_buttons"] = ui.make_choice_grid(rects, question.choices, labels)
    return category


def _record_result(game, correct, response_time, picked_value, timed_out):
    question = game["question"]
    category = question.category if game["mode"] == config.MODE_SEQUENCE else question.label
    game["stats"].record(category, correct, response_time, timed_out=timed_out)
    game["difficulty"].record_result(correct)
    game["feedback"] = {"correct": correct, "picked": picked_value}
    game["feedback_until"] = pygame.time.get_ticks() / 1000.0 + config.FEEDBACK_DISPLAY_SECONDS


def _submit_choice(game, choice_value):
    question = game["question"]
    response_time = game["q_timer"].elapsed()
    if game["mode"] == config.MODE_SEQUENCE:
        correct = check_sequence_answer(question, choice_value)
    else:
        correct = check_market_answer(question, choice_value)
    _record_result(game, correct, response_time, choice_value, timed_out=False)


def _advance_question(game, session_timer):
    if session_timer.is_over():
        game["session_over"] = True
        game["feedback"] = None
        return
    _next_question(game)
    game["q_timer"].start(game["difficulty"].time_limit)
    game["feedback"] = None


def _update_playing(game, session_timer):
    if game["session_over"]:
        return

    if game["feedback"] is not None:
        if pygame.time.get_ticks() / 1000.0 >= game["feedback_until"]:
            _advance_question(game, session_timer)
        return

    if session_timer.is_over():
        game["session_over"] = True
        return

    if game["q_timer"].expired():
        _record_result(game, correct=False, response_time=game["difficulty"].time_limit,
                        picked_value=None, timed_out=True)


def main():
    pygame.init()
    win = pygame.display.set_mode((config.WIN_WIDTH, config.WIN_HEIGHT))
    pygame.display.set_caption("Pattern IQ — Speed Pattern Recognition")
    fonts = ui.load_fonts()
    logo = ui.load_logo()
    clock = pygame.time.Clock()

    persistent = PersistentStats()

    mode_buttons = [
        ui.Button((80, 150, 290, 44), config.MODE_LABELS[config.MODE_SEQUENCE], config.MODE_SEQUENCE),
        ui.Button((390, 150, 290, 44), config.MODE_LABELS[config.MODE_MARKET], config.MODE_MARKET),
    ]
    tier_buttons = [
        ui.Button((80 + i * 210, 266, 190, 40), config.TIER_LABELS[tier], tier)
        for i, tier in enumerate((1, 2, 3))
    ]
    session_buttons = [
        ui.Button((50 + i * 168, 384, 152, 40), label, seconds)
        for i, (label, seconds) in enumerate(config.SESSION_LENGTHS)
    ]
    start_button = ui.Button((config.WIN_WIDTH // 2 - 90, 440, 180, 50), "START GAME")
    play_again_button = ui.Button((config.WIN_WIDTH // 2 - 100, 660, 200, 50), "PLAY AGAIN")

    state = "menu"
    selected_mode = config.MODE_SEQUENCE
    selected_tier = config.MIN_TIER
    session_index = config.DEFAULT_SESSION_INDEX
    game = None
    session_timer = None

    running = True
    while running:
        clock.tick(config.FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif state == "menu":
                for button in mode_buttons:
                    if button.is_clicked(event):
                        selected_mode = button.value
                for button in tier_buttons:
                    if button.is_clicked(event):
                        selected_tier = button.value
                for i, button in enumerate(session_buttons):
                    if button.is_clicked(event):
                        session_index = i
                if start_button.is_clicked(event):
                    game = _start_session(selected_mode, selected_tier)
                    duration = config.SESSION_LENGTHS[session_index][1]
                    session_timer = SessionTimer(duration)
                    state = "playing"

            elif state == "playing":
                if game["feedback"] is not None:
                    continue
                picked = None
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for button in game["choice_buttons"]:
                        if button.is_clicked(event):
                            picked = button.value
                elif event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4
                ):
                    index = event.key - pygame.K_1
                    if index < len(game["choice_buttons"]):
                        picked = game["choice_buttons"][index].value
                if picked is not None:
                    _submit_choice(game, picked)

            elif state == "results":
                if play_again_button.is_clicked(event):
                    state = "menu"

        if state == "playing":
            _update_playing(game, session_timer)
            if game["session_over"]:
                persistent.update_from_session(game["mode"], game["stats"])
                state = "results"

        if state == "menu":
            ui.draw_menu(win, fonts, mode_buttons, selected_mode, tier_buttons, selected_tier,
                         session_buttons, config.SESSION_LENGTHS[session_index][1], start_button, logo=logo)
        elif state == "playing":
            draw_fn = ui.draw_game_sequence if game["mode"] == config.MODE_SEQUENCE else ui.draw_game_market
            draw_fn(
                win, fonts, game["question"], game["choice_buttons"],
                game["q_timer"].remaining(), game["difficulty"].time_limit,
                game["stats"], game["difficulty"].tier, game["feedback"], logo=logo,
            )
        elif state == "results":
            ui.draw_results(win, fonts, game["mode"], game["stats"], persistent, play_again_button, logo=logo)

    pygame.quit()


if __name__ == "__main__":
    main()
