"""
Pygame rendering: Button widget, a tiny line-chart renderer for Spot the
Pattern, and one draw function per screen (menu / sequence / market /
results) - mirrors games/mental_math/ui.py's "given this state, draw it" style.
"""

import os

import pygame

from . import config


def load_fonts():
    return {
        "title": pygame.font.SysFont(config.FONT_NAME, 38, bold=True),
        "sequence": pygame.font.SysFont(config.FONT_NAME, 40, bold=True),
        "label": pygame.font.SysFont(config.FONT_NAME, 18),
        "stat": pygame.font.SysFont(config.FONT_NAME, 20, bold=True),
        "button": pygame.font.SysFont(config.FONT_NAME, 21, bold=True),
        "choice": pygame.font.SysFont(config.FONT_NAME, 24, bold=True),
        "choice_small": pygame.font.SysFont(config.FONT_NAME, 19, bold=True),
        "timer": pygame.font.SysFont(config.FONT_NAME, 24, bold=True),
        "small": pygame.font.SysFont(config.FONT_NAME, 16),
        "tip": pygame.font.SysFont(config.FONT_NAME, 17),
        "badge": pygame.font.SysFont(config.FONT_NAME, 15, bold=True),
    }


def _center_text(win, font, text, y, color=config.TEXT_COLOR):
    surf = font.render(text, True, color)
    win.blit(surf, surf.get_rect(center=(config.WIN_WIDTH // 2, y)))


def _wrap_center_text(win, font, text, y, color, max_width, line_height=20):
    words = text.split(" ")
    lines, current = [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if font.size(trial)[0] > max_width and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    for i, line in enumerate(lines):
        _center_text(win, font, line, y + i * line_height, color)
    return y + len(lines) * line_height


_BADGE_CENTER = (config.WIN_WIDTH - 26, 26)
_BADGE_RADIUS = 15


def load_logo():
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
    cx, cy = _BADGE_CENTER
    if logo is not None:
        win.blit(logo, logo.get_rect(center=(cx, cy)))
        return
    r = _BADGE_RADIUS
    diamond = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    pygame.draw.polygon(win, config.ACCENT_COLOR, diamond, width=2)
    letter = fonts["badge"].render("P", True, config.ACCENT_COLOR)
    win.blit(letter, letter.get_rect(center=(cx, cy)))


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class Button:
    def __init__(self, rect, label, value=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.value = value  # arbitrary payload (choice value, mode key, tier int, ...)

    def is_hovered(self):
        return self.rect.collidepoint(pygame.mouse.get_pos())

    def is_clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.rect.collidepoint(event.pos))

    def draw(self, win, font, color=config.ACCENT_COLOR, text_color=config.ACCENT_TEXT_COLOR, border=None):
        fill = tuple(min(255, c + 15) for c in color) if self.is_hovered() else color
        pygame.draw.rect(win, fill, self.rect, border_radius=8)
        if border is not None:
            pygame.draw.rect(win, border, self.rect, width=2, border_radius=8)
        surf = font.render(self.label, True, text_color)
        win.blit(surf, surf.get_rect(center=self.rect.center))


def make_choice_grid(rects, values, labels):
    """Builds/updates a fixed list of choice Buttons in place - rects stay
    put across questions, only label/value change, so the click targets
    never jump around mid-session."""
    return [Button(rect, label, value) for rect, value, label in zip(rects, values, labels)]


def draw_choice_buttons(win, font, buttons, feedback, correct_value):
    for button in buttons:
        if feedback is None:
            button.draw(win, font, color=config.CHOICE_BG_COLOR, text_color=config.TEXT_COLOR,
                        border=config.CHOICE_HOVER_BORDER if button.is_hovered() else config.CHOICE_BORDER_COLOR)
        else:
            if button.value == correct_value:
                button.draw(win, font, color=config.CHOICE_BG_COLOR, text_color=config.CORRECT_COLOR,
                            border=config.CORRECT_COLOR)
            elif feedback.get("picked") == button.value:
                button.draw(win, font, color=config.CHOICE_BG_COLOR, text_color=config.INCORRECT_COLOR,
                            border=config.INCORRECT_COLOR)
            else:
                button.draw(win, font, color=config.CHOICE_BG_COLOR, text_color=config.MUTED_TEXT_COLOR,
                            border=config.CHOICE_BORDER_COLOR)


def draw_chart(win, rect, series):
    """Minimal auto-scaled line chart, no axes/labels - the point is to read
    the shape at a glance, same as glancing at a ticker."""
    x, y, w, h = rect
    pygame.draw.rect(win, config.CHOICE_BG_COLOR, rect, border_radius=6)
    pygame.draw.rect(win, config.CHART_AXIS_COLOR, rect, width=1, border_radius=6)

    lo, hi = min(series), max(series)
    span = (hi - lo) or 1.0
    pad = 10
    plot_w, plot_h = w - 2 * pad, h - 2 * pad
    n = len(series)

    points = []
    for i, value in enumerate(series):
        px = x + pad + (i / (n - 1)) * plot_w
        py = y + pad + (1 - (value - lo) / span) * plot_h
        points.append((px, py))

    fill_points = [(x + pad, y + h - pad)] + points + [(x + w - pad, y + h - pad)]
    pygame.draw.polygon(win, config.CHART_FILL_COLOR, fill_points)
    pygame.draw.lines(win, config.CHART_LINE_COLOR, False, points, width=3)


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------

def draw_menu(win, fonts, mode_buttons, selected_mode, tier_buttons, selected_tier,
              session_buttons, selected_session, start_button, logo=None):
    win.fill(config.BG_COLOR)
    _center_text(win, fonts["title"], "PATTERN IQ", 62)
    _center_text(win, fonts["small"], "SPEED PATTERN RECOGNITION", 92, config.ACCENT_COLOR)

    _center_text(win, fonts["label"], "Mode", 128, config.MUTED_TEXT_COLOR)
    for button in mode_buttons:
        selected = button.value == selected_mode
        color = config.ACCENT_COLOR if selected else config.BUTTON_IDLE_COLOR
        text_color = config.ACCENT_TEXT_COLOR if selected else config.TEXT_COLOR
        button.draw(win, fonts["button"], color=color, text_color=text_color)

    description = (
        "Number sequences, IQ-test style - see the rule, pick what's next."
        if selected_mode == config.MODE_SEQUENCE else
        "A generated price chart - name the pattern before the clock runs out."
    )
    _center_text(win, fonts["small"], description, 208, config.MUTED_TEXT_COLOR)

    _center_text(win, fonts["label"], "Difficulty", 244, config.MUTED_TEXT_COLOR)
    for button in tier_buttons:
        selected = button.value == selected_tier
        color = config.ACCENT_COLOR if selected else config.BUTTON_IDLE_COLOR
        text_color = config.ACCENT_TEXT_COLOR if selected else config.TEXT_COLOR
        button.draw(win, fonts["label"], color=color, text_color=text_color)
    _center_text(win, fonts["small"], "Streaks adjust the tier further as you play.", 322, config.MUTED_TEXT_COLOR)

    _center_text(win, fonts["label"], "Session length", 360, config.MUTED_TEXT_COLOR)
    for button in session_buttons:
        selected = button.value == selected_session
        color = config.ACCENT_COLOR if selected else config.BUTTON_IDLE_COLOR
        text_color = config.ACCENT_TEXT_COLOR if selected else config.TEXT_COLOR
        button.draw(win, fonts["label"], color=color, text_color=text_color)

    start_button.draw(win, fonts["button"])
    _draw_badge(win, fonts, logo)
    pygame.display.update()


def _draw_stat_rows(win, fonts, stats, tier, time_limit):
    stat_rows = [
        (f"Correct: {stats.correct}", f"Accuracy: {stats.accuracy:.0f}%"),
        (f"Streak: {stats.current_streak}", f"Best: {stats.best_streak}"),
        (f"Avg Time: {stats.avg_time:.1f}s", f"Tier: {config.TIER_LABELS[tier]} ({time_limit:.0f}s)"),
    ]
    y = 44
    for left, right in stat_rows:
        left_surf = fonts["stat"].render(left, True, config.MUTED_TEXT_COLOR)
        right_surf = fonts["stat"].render(right, True, config.MUTED_TEXT_COLOR)
        win.blit(left_surf, (40, y))
        win.blit(right_surf, (config.WIN_WIDTH - 40 - right_surf.get_width(), y))
        y += 28
    return y


def _draw_timer_bar(win, fonts, timer_remaining, timer_limit):
    ratio = 0.0 if timer_limit <= 0 else max(0.0, timer_remaining / timer_limit)
    if ratio > 0.5:
        timer_color = config.TIMER_SAFE_COLOR
    elif ratio > 0.2:
        timer_color = config.TIMER_WARN_COLOR
    else:
        timer_color = config.TIMER_DANGER_COLOR

    bar_rect = pygame.Rect(60, config.WIN_HEIGHT - 56, config.WIN_WIDTH - 120, 14)
    pygame.draw.rect(win, config.TIMER_TRACK_COLOR, bar_rect, border_radius=7)
    fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, int(bar_rect.width * ratio), bar_rect.height)
    if fill_rect.width > 0:
        pygame.draw.rect(win, timer_color, fill_rect, border_radius=7)
    _center_text(win, fonts["timer"], f"Time: {timer_remaining:.1f}s", config.WIN_HEIGHT - 26, timer_color)


def draw_game_sequence(win, fonts, question, choice_buttons, timer_remaining, timer_limit,
                        stats, tier, feedback, logo=None):
    win.fill(config.BG_COLOR)
    _draw_stat_rows(win, fonts, stats, tier, timer_limit)

    _center_text(win, fonts["label"], config.SEQUENCE_CATEGORY_LABELS.get(question.category, question.category), 158, config.MUTED_TEXT_COLOR)
    sequence_text = ", ".join(f"{t:,}" for t in question.terms) + ",  ?"
    _center_text(win, fonts["sequence"], sequence_text, 200)

    draw_choice_buttons(win, fonts["choice"], choice_buttons, feedback, question.answer)

    if feedback is not None:
        color = config.CORRECT_COLOR if feedback["correct"] else config.INCORRECT_COLOR
        message = "Correct!" if feedback["correct"] else f"Answer: {question.answer:,}"
        _center_text(win, fonts["stat"], message, 490, color)
        _wrap_center_text(win, fonts["tip"], question.rule_hint, 518, config.MUTED_TEXT_COLOR, config.WIN_WIDTH - 140)

    _draw_timer_bar(win, fonts, timer_remaining, timer_limit)
    _draw_badge(win, fonts, logo)
    pygame.display.update()


_CHART_RECT = (60, 172, config.WIN_WIDTH - 120, 220)


def draw_game_market(win, fonts, question, choice_buttons, timer_remaining, timer_limit,
                      stats, tier, feedback, logo=None):
    win.fill(config.BG_COLOR)
    _draw_stat_rows(win, fonts, stats, tier, timer_limit)

    _center_text(win, fonts["label"], "What pattern is this?", 152, config.MUTED_TEXT_COLOR)
    draw_chart(win, _CHART_RECT, question.series)

    draw_choice_buttons(win, fonts["choice_small"], choice_buttons, feedback, question.label)

    if feedback is not None:
        color = config.CORRECT_COLOR if feedback["correct"] else config.INCORRECT_COLOR
        correct_label = config.MARKET_PATTERN_LABELS[question.label]
        message = "Correct!" if feedback["correct"] else f"Answer: {correct_label}"
        _center_text(win, fonts["stat"], message, 620, color)
        tip = config.MARKET_PATTERN_TIPS[question.label]
        _wrap_center_text(win, fonts["tip"], tip, 646, config.MUTED_TEXT_COLOR, config.WIN_WIDTH - 140)

    _draw_timer_bar(win, fonts, timer_remaining, timer_limit)
    _draw_badge(win, fonts, logo)
    pygame.display.update()


def draw_results(win, fonts, mode, session, persistent, play_again_button, logo=None):
    win.fill(config.BG_COLOR)
    _center_text(win, fonts["title"], "SESSION COMPLETE", 62)
    _center_text(win, fonts["small"], config.MODE_LABELS[mode], 92, config.ACCENT_COLOR)

    lines = [
        f"Correct:        {session.correct}",
        f"Incorrect:      {session.incorrect}",
        f"Accuracy:       {session.accuracy:.1f}%",
        f"Best Streak:    {session.best_streak}",
        f"Avg. Time:      {session.avg_time:.1f}s",
    ]
    y = 140
    for line in lines:
        _center_text(win, fonts["stat"], line, y)
        y += 30

    labels = config.SEQUENCE_CATEGORY_LABELS if mode == config.MODE_SEQUENCE else config.MARKET_PATTERN_LABELS
    strongest, weakest = session.strongest_and_weakest()
    y += 12
    if strongest:
        category, pct = strongest
        _center_text(win, fonts["label"], f"Strongest: {labels.get(category, category)} — {pct:.0f}%", y, config.CORRECT_COLOR)
        y += 26
    if weakest:
        category, pct = weakest
        _center_text(win, fonts["label"], f"Weakest: {labels.get(category, category)} — {pct:.0f}%", y, config.INCORRECT_COLOR)
        y += 26

    y += 18
    _center_text(win, fonts["small"], "All-Time Bests", y, config.MUTED_TEXT_COLOR)
    y += 22
    record = persistent.mode_data(mode)
    fastest = record["fastest_avg_time"]
    fastest_part = f"Fastest Avg: {fastest:.1f}s" if fastest is not None else "Fastest Avg: —"
    best_line = (
        f"Best Accuracy: {record['best_accuracy']:.0f}%   "
        f"Best Streak: {record['best_streak']}   "
        f"{fastest_part}"
    )
    _center_text(win, fonts["small"], best_line, y, config.MUTED_TEXT_COLOR)

    play_again_button.draw(win, fonts["button"])
    _draw_badge(win, fonts, logo)
    pygame.display.update()
