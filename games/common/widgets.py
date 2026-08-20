"""
Shared pygame widgets: a numeric TextInput box and a clickable Button.
Ported from games/mental_math/ui.py so newer games don't reimplement the
same ~40 lines each.
"""

import pygame

from . import palette


class TextInput:
    """Minimal numeric entry box, always 'active' so keyboard input works
    with no click-to-focus step."""

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
        border = palette.INPUT_BORDER_ACTIVE if self.active else palette.INPUT_BORDER_COLOR
        pygame.draw.rect(win, palette.INPUT_BG_COLOR, self.rect, border_radius=8)
        pygame.draw.rect(win, border, self.rect, width=2, border_radius=8)

        surf = font.render(self.text, True, palette.TEXT_COLOR)
        text_rect = surf.get_rect(midleft=(self.rect.x + 16, self.rect.centery))
        win.blit(surf, text_rect)

        if self.active and (pygame.time.get_ticks() // 500) % 2 == 0:
            caret_x = text_rect.right + 3
            pygame.draw.line(win, palette.TEXT_COLOR, (caret_x, self.rect.y + 12), (caret_x, self.rect.bottom - 12), 2)


class Button:
    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect)
        self.label = label

    def is_hovered(self):
        return self.rect.collidepoint(pygame.mouse.get_pos())

    def is_clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.rect.collidepoint(event.pos))

    def draw(self, win, font, color=palette.ACCENT_COLOR, text_color=palette.ACCENT_TEXT_COLOR):
        fill = tuple(min(255, c + 15) for c in color) if self.is_hovered() else color
        pygame.draw.rect(win, fill, self.rect, border_radius=8)
        surf = font.render(self.label, True, text_color)
        win.blit(surf, surf.get_rect(center=self.rect.center))
