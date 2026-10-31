"""
Watch a trained DQN checkpoint play Snake, or play it yourself for
comparison (the agent's actions and the human's are driven through the
exact same SnakeEnv.step() - neither mode gets a different game).

Run (from inside games/snake_dqn/):
    python play.py                       watch the best checkpoint play
    python play.py --checkpoint latest   watch the latest checkpoint instead
    python play.py --human               play it yourself with arrow keys
    python play.py --fps 30              change playback speed

Run (from the repo root):
    python -m games.snake_dqn.play [same flags]

Dependencies: pip install -r requirements.txt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame

from snake_dqn import config as cfg
from snake_dqn import snake_env
from snake_dqn.agent import DQNAgent
from snake_dqn.snake_env import SnakeEnv

_CLOCKWISE = [snake_env.UP, snake_env.RIGHT, snake_env.DOWN, snake_env.LEFT]

_KEY_TO_DIRECTION = {
    pygame.K_UP: snake_env.UP,
    pygame.K_DOWN: snake_env.DOWN,
    pygame.K_LEFT: snake_env.LEFT,
    pygame.K_RIGHT: snake_env.RIGHT,
}


def _absolute_to_action(current_dir, desired_dir):
    """Human input arrives as an absolute compass direction; the env wants
    a relative action. A 180-degree reversal (the one illegal Snake move)
    silently falls back to "keep going" instead of being rejected."""
    if desired_dir is None or desired_dir == current_dir:
        return cfg.ACTION_STRAIGHT
    idx = _CLOCKWISE.index(current_dir)
    if _CLOCKWISE[(idx + 1) % 4] == desired_dir:
        return cfg.ACTION_RIGHT
    if _CLOCKWISE[(idx - 1) % 4] == desired_dir:
        return cfg.ACTION_LEFT
    return cfg.ACTION_STRAIGHT


def _draw(win, font, env):
    win.fill(cfg.BG_COLOR)

    board_top = cfg.SCORE_BAR_HEIGHT
    for x, y in [(x, y) for x in range(env.width) for y in range(env.height)]:
        rect = pygame.Rect(x * cfg.CELL_SIZE, board_top + y * cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE)
        pygame.draw.rect(win, cfg.GRID_LINE_COLOR, rect, 1)

    fx, fy = env.food
    pygame.draw.rect(win, cfg.FOOD_COLOR, (
        fx * cfg.CELL_SIZE + 2, board_top + fy * cfg.CELL_SIZE + 2,
        cfg.CELL_SIZE - 4, cfg.CELL_SIZE - 4,
    ))

    for i, (sx, sy) in enumerate(env.snake):
        color = cfg.SNAKE_HEAD_COLOR if i == 0 else cfg.SNAKE_BODY_COLOR
        pygame.draw.rect(win, color, (
            sx * cfg.CELL_SIZE + 1, board_top + sy * cfg.CELL_SIZE + 1,
            cfg.CELL_SIZE - 2, cfg.CELL_SIZE - 2,
        ))

    score_surf = font.render(f"Score: {env.score}", True, cfg.TEXT_COLOR)
    win.blit(score_surf, (10, 8))
    pygame.display.update()


def _show_message(win, font, text):
    surf = font.render(text, True, cfg.TEXT_COLOR)
    rect = surf.get_rect(center=(cfg.WIN_WIDTH // 2, cfg.WIN_HEIGHT // 2))
    win.blit(surf, rect)
    pygame.display.update()


def run(human=False, checkpoint_path=None, fps=None, max_episodes=None):
    pygame.init()
    win = pygame.display.set_mode((cfg.WIN_WIDTH, cfg.WIN_HEIGHT))
    pygame.display.set_caption("Snake - " + ("Human" if human else "DQN Agent"))
    font = pygame.font.SysFont(cfg.FONT_NAME, 24)
    clock = pygame.time.Clock()

    agent = None
    if not human:
        agent = DQNAgent()
        path = checkpoint_path or cfg.BEST_CHECKPOINT_PATH
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}. Run train.py first, or pass --human to play yourself.")
            pygame.quit()
            return
        agent.load(path)
        print(f"Loaded checkpoint from {path} ({agent.train_steps} train steps).")

    fps = fps or (cfg.FPS_HUMAN if human else cfg.FPS_WATCH)
    env = SnakeEnv()
    episodes_played = 0
    running = True

    while running:
        state = env.reset()
        done = False
        desired_dir = env.direction

        while not done and running:
            clock.tick(fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif human and event.type == pygame.KEYDOWN and event.key in _KEY_TO_DIRECTION:
                    desired_dir = _KEY_TO_DIRECTION[event.key]

            if not running:
                break

            if human:
                action = _absolute_to_action(env.direction, desired_dir)
            else:
                action = agent.act(state, epsilon=0.0)  # greedy - no exploration when watching

            state, _reward, done, info = env.step(action)
            _draw(win, font, env)

        if running:
            episodes_played += 1
            _show_message(win, font, f"Score: {info['score']}  -  press any key to play again")
            waiting = True
            while waiting and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    elif event.type == pygame.KEYDOWN:
                        waiting = False
            if max_episodes and episodes_played >= max_episodes:
                running = False

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Play or watch Snake + DQN.")
    parser.add_argument("--human", action="store_true", help="play yourself with arrow keys")
    parser.add_argument("--checkpoint", default=None,
                         help="'best' (default), 'latest', or a path to a .pt checkpoint")
    parser.add_argument("--fps", type=int, default=None, help="override playback speed")
    parser.add_argument("--episodes", type=int, default=None, help="auto-stop after N episodes")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path == "best":
        checkpoint_path = cfg.BEST_CHECKPOINT_PATH
    elif checkpoint_path == "latest":
        checkpoint_path = cfg.LATEST_CHECKPOINT_PATH

    run(human=args.human, checkpoint_path=checkpoint_path, fps=args.fps, max_episodes=args.episodes)


if __name__ == "__main__":
    main()
