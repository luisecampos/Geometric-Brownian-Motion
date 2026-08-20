"""
Flappy Bird - Base Game (Step 1)
---------------------------------
Playable version, no AI yet. Press SPACE to flap.
This is the foundation the NEAT AI will plug into later:
the Bird, Pipe, and collision logic here stay almost
identical once we add AI control.
"""

import pygame
import random

pygame.init()

# ---- Config ----
WIN_WIDTH = 500
WIN_HEIGHT = 700
FLOOR_Y = 630

GRAVITY = 0.6
FLAP_STRENGTH = -9
MAX_FALL_SPEED = 15

PIPE_GAP = 170          # vertical gap the bird flies through
PIPE_SPEED = 4
PIPE_SPACING = 280      # horizontal distance between pipes

BIRD_X = 100
BIRD_SIZE = 30

BG_COLOR = (135, 206, 235)   # sky blue
PIPE_COLOR = (34, 139, 34)
BIRD_COLOR = (255, 200, 0)
FLOOR_COLOR = (222, 184, 135)
TEXT_COLOR = (20, 20, 20)

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird - Base Game")
FONT = pygame.font.SysFont("Segoe UI", 28)
CLOCK = pygame.time.Clock()


class Bird:
    def __init__(self):
        self.x = BIRD_X
        self.y = WIN_HEIGHT // 2
        self.velocity = 0

    def flap(self):
        self.velocity = FLAP_STRENGTH

    def move(self):
        self.velocity += GRAVITY
        self.velocity = min(self.velocity, MAX_FALL_SPEED)
        self.y += self.velocity

    def get_rect(self):
        return pygame.Rect(self.x, int(self.y), BIRD_SIZE, BIRD_SIZE)

    def draw(self, win):
        pygame.draw.circle(win, BIRD_COLOR, (self.x + BIRD_SIZE // 2, int(self.y) + BIRD_SIZE // 2), BIRD_SIZE // 2)


class Pipe:
    def __init__(self, x):
        self.x = x
        self.gap_y = random.randint(150, FLOOR_Y - 150)  # center of the gap
        self.passed = False  # has the bird flown past this pipe (for scoring)

    def move(self):
        self.x -= PIPE_SPEED

    def top_rect(self):
        top_height = self.gap_y - PIPE_GAP // 2
        return pygame.Rect(self.x, 0, 60, top_height)

    def bottom_rect(self):
        bottom_y = self.gap_y + PIPE_GAP // 2
        return pygame.Rect(self.x, bottom_y, 60, FLOOR_Y - bottom_y)

    def collides_with(self, bird_rect):
        return bird_rect.colliderect(self.top_rect()) or bird_rect.colliderect(self.bottom_rect())

    def draw(self, win):
        pygame.draw.rect(win, PIPE_COLOR, self.top_rect())
        pygame.draw.rect(win, PIPE_COLOR, self.bottom_rect())


def draw_window(win, bird, pipes, score):
    win.fill(BG_COLOR)
    for pipe in pipes:
        pipe.draw(win)
    pygame.draw.rect(win, FLOOR_COLOR, (0, FLOOR_Y, WIN_WIDTH, WIN_HEIGHT - FLOOR_Y))
    bird.draw(win)

    score_surf = FONT.render(f"Score: {score}", True, TEXT_COLOR)
    win.blit(score_surf, (10, 10))
    pygame.display.update()


def show_message(win, text):
    surf = FONT.render(text, True, TEXT_COLOR)
    rect = surf.get_rect(center=(WIN_WIDTH // 2, WIN_HEIGHT // 2))
    win.blit(surf, rect)
    pygame.display.update()


def main():
    bird = Bird()
    pipes = [Pipe(WIN_WIDTH + 100)]
    score = 0
    game_over = False

    running = True
    while running:
        CLOCK.tick(60)  # 60 frames per second

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not game_over:
                    bird.flap()
                if event.key == pygame.K_r and game_over:
                    bird, pipes, score, game_over = Bird(), [Pipe(WIN_WIDTH + 100)], 0, False

        if not game_over:
            bird.move()

            # Move pipes, add new ones, remove off-screen ones
            for pipe in pipes:
                pipe.move()

            if pipes[-1].x < WIN_WIDTH - PIPE_SPACING:
                pipes.append(Pipe(WIN_WIDTH + 20))

            if pipes[0].x + 60 < 0:
                pipes.pop(0)

            # Scoring: bird has passed a pipe's x position
            for pipe in pipes:
                if not pipe.passed and pipe.x + 60 < bird.x:
                    pipe.passed = True
                    score += 1

            # Collision checks
            bird_rect = bird.get_rect()
            if bird.y + BIRD_SIZE >= FLOOR_Y or bird.y <= 0:
                game_over = True
            for pipe in pipes:
                if pipe.collides_with(bird_rect):
                    game_over = True

        draw_window(WIN, bird, pipes, score)
        if game_over:
            show_message(WIN, "Game Over - Press R to Restart")

    pygame.quit()


if __name__ == "__main__":
    main()