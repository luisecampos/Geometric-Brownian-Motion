"""
Shared corner badge - the "why am I doing this" reminder, drawn on every
screen of every game in the family. Looks for a logo image at
games/common/assets/logo.(png|jpg|jpeg); falls back to a small original
drawn monogram if nothing's there yet (never reproduces any real company's
actual logo artwork - see games/mental_math/ui.py for the original design
and reasoning).

Shared here (rather than per-game, like mental_math still does) so you only
need to drop the logo file in once and every new game picks it up.
"""

import os

import pygame

from . import palette

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
LOGO_FILENAMES = ["logo.png", "logo.jpg", "logo.jpeg"]

RADIUS = 15


def load_logo():
    """Returns a scaled, aspect-preserved Surface, or None if no logo file
    has been dropped into games/common/assets/ yet. Call once after
    pygame.display.set_mode(), same as loading fonts."""
    for name in LOGO_FILENAMES:
        path = os.path.join(ASSETS_DIR, name)
        if not os.path.exists(path):
            continue
        try:
            image = pygame.image.load(path).convert_alpha()
        except pygame.error:
            continue
        max_size = RADIUS * 2
        w, h = image.get_size()
        scale = max_size / max(w, h)
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        return pygame.transform.smoothscale(image, new_size)
    return None


def draw(win, width, fonts, logo=None, margin=26):
    cx, cy = width - margin, margin
    if logo is not None:
        win.blit(logo, logo.get_rect(center=(cx, cy)))
        return
    diamond = [(cx, cy - RADIUS), (cx + RADIUS, cy), (cx, cy + RADIUS), (cx - RADIUS, cy)]
    pygame.draw.polygon(win, palette.ACCENT_COLOR, diamond, width=2)
    letter = fonts["badge"].render("C", True, palette.ACCENT_COLOR)
    win.blit(letter, letter.get_rect(center=(cx, cy)))
