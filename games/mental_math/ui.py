"""
Pygame rendering: two small reusable widgets (TextInput, Button) plus one
draw function per screen (menu / playing / results) - mirrors flappy_base.py's
draw_window()-style functions: pure "given this state, draw it" calls.
"""

import os

import pygame

from . import config


def load_fonts():
    return {
        "title": pygame.font.SysFont(config.FONT_NAME, 40, bold=True),
        "question": pygame.font.SysFont(config.FONT_NAME, 46, bold=True),
        "label": pygame.font.SysFont(config.FONT_NAME, 18),
        "stat": pygame.font.SysFont(config.FONT_NAME, 20, bold=True),
        "button": pygame.font.SysFont(config.FONT_NAME, 22, bold=True),
        "timer": pygame.font.SysFont(config.FONT_NAME, 24, bold=True),
        "small": pygame.font.SysFont(config.FONT_NAME, 16),
        "input": pygame.font.SysFont(config.FONT_NAME, 32),
        "badge": pygame.font.SysFont(config.FONT_NAME, 15, bold=True),
    }


def _center_text(win, font, text, y, color=config.TEXT_COLOR):
    surf = font.render(text, True, color)
    win.blit(surf, surf.get_rect(center=(config.WIN_WIDTH // 2, y)))


_BADGE_CENTER = (config.WIN_WIDTH - 26, 26)
_BADGE_RADIUS = 15


def load_logo():
    """Look for games/mental_math/assets/logo.(png|jpg|jpeg) and scale it to
    fit the corner badge, preserving aspect ratio. Returns None if nothing's
    there yet, in which case _draw_badge() falls back to a drawn monogram.
    Call this once after pygame.display.set_mode(), same as load_fonts().
    """
    for name in config.LOGO_FILENAMES:
        path = os.path.join(config.ASSETS_DIR, name)
        if not os.path.exists(path):
            continue
        try:
            image = pygame.image.load(path).convert_alpha()
        except pygame.error:
            continue
        max_size = _BADGE_RADIUS * 2
        w, h = image.get_size()
        scale = max_size / max(w, h)
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        return pygame.transform.smoothscale(image, new_size)
    return None


def _draw_badge(win, fonts, logo=None):
    """Corner emblem - uses a dropped-in logo image if load_logo() found one,
    otherwise falls back to a small original drawn monogram (not a
    reproduction of any real company's actual logo).
    """
    cx, cy = _BADGE_CENTER
    if logo is not None:
        win.blit(logo, logo.get_rect(center=(cx, cy)))
        return
    r = _BADGE_RADIUS
    diamond = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    pygame.draw.polygon(win, config.ACCENT_COLOR, diamond, width=2)
    letter = fonts["badge"].render("C", True, config.ACCENT_COLOR)
    win.blit(letter, letter.get_rect(center=(cx, cy)))


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class TextInput:
    """Minimal numeric entry box. Always considered 'active' since it's the
    only focusable control during gameplay - that's what lets the player
    play almost entirely from the keyboard with no click-to-focus step.
    """

    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        self.text = ""
        self.active = True

    def clear(self):
        self.text = ""

    def handle_event(self, event):
        """Returns "submit" on Enter, otherwise None."""
        if event.type != pygame.KEYDOWN:
            return None
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            return "submit"
        if event.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
        elif event.unicode and (event.unicode.isdigit() or event.unicode in ".-") and len(self.text) < 12:
            self.text += event.unicode
        return None

    def draw(self, win, font):
        border = config.INPUT_BORDER_ACTIVE if self.active else config.INPUT_BORDER_COLOR
        pygame.draw.rect(win, config.INPUT_BG_COLOR, self.rect, border_radius=8)
        pygame.draw.rect(win, border, self.rect, width=2, border_radius=8)

        surf = font.render(self.text, True, config.TEXT_COLOR)
        text_rect = surf.get_rect(midleft=(self.rect.x + 16, self.rect.centery))
        win.blit(surf, text_rect)

        if self.active and (pygame.time.get_ticks() // 500) % 2 == 0:
            caret_x = text_rect.right + 3
            pygame.draw.line(win, config.TEXT_COLOR, (caret_x, self.rect.y + 12), (caret_x, self.rect.bottom - 12), 2)


class Button:
    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect)
        self.label = label

    def is_hovered(self):
        return self.rect.collidepoint(pygame.mouse.get_pos())

    def is_clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.rect.collidepoint(event.pos))

    def draw(self, win, font, color=config.ACCENT_COLOR, text_color=config.ACCENT_TEXT_COLOR):
        fill = tuple(min(255, c + 15) for c in color) if self.is_hovered() else color
        pygame.draw.rect(win, fill, self.rect, border_radius=8)
        surf = font.render(self.label, True, text_color)
        win.blit(surf, surf.get_rect(center=self.rect.center))


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------

_MENU_CATEGORY_ROWS = [
    "Addition · Subtraction · Multiplication · Triple-Digit +/-",
    "Percentages · Fractions · Squares",
]


def draw_menu(win, fonts, session_buttons, selected_index, start_button, logo=None):
    win.fill(config.BG_COLOR)
    _center_text(win, fonts["title"], "MENTAL MATH", 80)
    _center_text(win, fonts["small"], "CITADEL INTERVIEW PREP", 112, config.ACCENT_COLOR)
    _center_text(win, fonts["label"], "Test your calculation speed.", 140, config.MUTED_TEXT_COLOR)
    _center_text(win, fonts["small"], "45 seconds  →  30 seconds  →  15 seconds", 166, config.MUTED_TEXT_COLOR)

    y = 198
    for row in _MENU_CATEGORY_ROWS:
        _center_text(win, fonts["small"], row, y, config.MUTED_TEXT_COLOR)
        y += 22

    _center_text(win, fonts["label"], "Session length", 268, config.MUTED_TEXT_COLOR)
    for i, button in enumerate(session_buttons):
        selected = i == selected_index
        color = config.ACCENT_COLOR if selected else config.BUTTON_IDLE_COLOR
        text_color = config.ACCENT_TEXT_COLOR if selected else config.TEXT_COLOR
        button.draw(win, fonts["label"], color=color, text_color=text_color)

    start_button.draw(win, fonts["button"])
    _draw_badge(win, fonts, logo)
    pygame.display.update()


def draw_game(win, fonts, question, input_box, submit_button, timer_remaining, timer_limit,
              stats, tier, feedback, category_hint=None, logo=None):
    win.fill(config.BG_COLOR)

    stat_rows = [
        (f"Correct: {stats.correct}", f"Accuracy: {stats.accuracy:.0f}%"),
        (f"Streak: {stats.current_streak}", f"Best: {stats.best_streak}"),
        (f"Avg Time: {stats.avg_time:.1f}s", f"Tier: {config.TIME_LIMITS[tier]:.0f}s"),
    ]
    y = 44  # leaves clearance above for the corner badge
    for left, right in stat_rows:
        left_surf = fonts["stat"].render(left, True, config.MUTED_TEXT_COLOR)
        right_surf = fonts["stat"].render(right, True, config.MUTED_TEXT_COLOR)
        win.blit(left_surf, (40, y))
        win.blit(right_surf, (config.WIN_WIDTH - 40 - right_surf.get_width(), y))
        y += 28

    _center_text(win, fonts["question"], question.prompt, 260)
    if category_hint:
        _center_text(win, fonts["small"], category_hint, 300, config.MUTED_TEXT_COLOR)

    input_box.draw(win, fonts["input"])
    submit_button.draw(win, fonts["button"])

    if feedback is not None:
        color = config.CORRECT_COLOR if feedback["correct"] else config.INCORRECT_COLOR
        message = "Correct!" if feedback["correct"] else f"Answer: {feedback['correct_answer']}"
        pygame.draw.rect(win, color, (0, config.WIN_HEIGHT - 150, config.WIN_WIDTH, 6))
        _center_text(win, fonts["stat"], message, config.WIN_HEIGHT - 118, color)

    ratio = 0.0 if timer_limit <= 0 else max(0.0, timer_remaining / timer_limit)
    if ratio > 0.5:
        timer_color = config.TIMER_SAFE_COLOR
    elif ratio > 0.2:
        timer_color = config.TIMER_WARN_COLOR
    else:
        timer_color = config.TIMER_DANGER_COLOR

    bar_rect = pygame.Rect(60, config.WIN_HEIGHT - 70, config.WIN_WIDTH - 120, 14)
    pygame.draw.rect(win, config.TIMER_TRACK_COLOR, bar_rect, border_radius=7)
    fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, int(bar_rect.width * ratio), bar_rect.height)
    if fill_rect.width > 0:
        pygame.draw.rect(win, timer_color, fill_rect, border_radius=7)
    _center_text(win, fonts["timer"], f"Time: {timer_remaining:.1f}s", config.WIN_HEIGHT - 40, timer_color)

    _draw_badge(win, fonts, logo)
    pygame.display.update()


def draw_results(win, fonts, session, persistent, play_again_button, logo=None):
    win.fill(config.BG_COLOR)
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
        label = config.CATEGORY_LABELS.get(category, category)
        _center_text(win, fonts["label"], f"Strongest Category: {label} — {pct:.0f}%", y, config.CORRECT_COLOR)
        y += 26
    if weakest:
        category, pct = weakest
        label = config.CATEGORY_LABELS.get(category, category)
        _center_text(win, fonts["label"], f"Weakest Category: {label} — {pct:.0f}%", y, config.INCORRECT_COLOR)
        y += 26

    y += 20
    _center_text(win, fonts["small"], "All-Time Bests", y, config.MUTED_TEXT_COLOR)
    y += 22
    fastest = persistent.data["fastest_avg_time"]
    fastest_part = f"Fastest Avg: {fastest:.1f}s" if fastest is not None else "Fastest Avg: —"
    best_line = (
        f"Best Accuracy: {persistent.data['best_accuracy']:.0f}%   "
        f"Best Streak: {persistent.data['best_streak']}   "
        f"{fastest_part}"
    )
    _center_text(win, fonts["small"], best_line, y, config.MUTED_TEXT_COLOR)

    play_again_button.draw(win, fonts["button"])
    _draw_badge(win, fonts, logo)
    pygame.display.update()
