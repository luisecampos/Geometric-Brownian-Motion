"""
The Snake game itself, with no rendering and no AI in it - just state and
rules, exposed through a small Gym-like API (reset / step). play.py drives
it with pygame + keyboard for a human; train.py and play.py's "watch mode"
drive it with a DQNAgent instead. Neither caller needs to know this file
exists beyond that one interface.
"""

import random

from . import config as cfg

# Clockwise order of absolute headings. "Turn right" / "turn left" (relative
# to the snake's own facing) just means "next" / "previous" in this list -
# this is what lets the agent's action space be direction-free.
UP, RIGHT, DOWN, LEFT = (0, -1), (1, 0), (0, 1), (-1, 0)
_CLOCKWISE = [UP, RIGHT, DOWN, LEFT]


class SnakeEnv:
    """Grid-based Snake. Coordinates are (col, row) cells, origin top-left."""

    def __init__(self, width=cfg.GRID_WIDTH, height=cfg.GRID_HEIGHT):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        cx, cy = self.width // 2, self.height // 2
        self.direction = RIGHT
        # Head first, tail last - matches draw order and collision checks below.
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.score = 0
        self.steps_without_food = 0
        self.food = self._spawn_food()
        return self.get_state()

    def _spawn_food(self):
        occupied = set(self.snake)
        empty = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in occupied
        ]
        return random.choice(empty) if empty else None

    def _turn(self, action):
        idx = _CLOCKWISE.index(self.direction)
        if action == cfg.ACTION_RIGHT:
            return _CLOCKWISE[(idx + 1) % 4]
        if action == cfg.ACTION_LEFT:
            return _CLOCKWISE[(idx - 1) % 4]
        return self.direction  # ACTION_STRAIGHT

    def _is_collision(self, point):
        x, y = point
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return point in self.snake

    def step(self, action):
        """action is one of cfg.ACTION_STRAIGHT / ACTION_RIGHT / ACTION_LEFT."""
        prev_head = self.snake[0]
        prev_dist = abs(prev_head[0] - self.food[0]) + abs(prev_head[1] - self.food[1])

        self.direction = self._turn(action)
        new_head = (prev_head[0] + self.direction[0], prev_head[1] + self.direction[1])

        self.steps_without_food += 1
        done = False
        reward = cfg.REWARD_STEP

        if self._is_collision(new_head) or self.steps_without_food > cfg.MAX_STEPS_WITHOUT_FOOD:
            done = True
            reward = cfg.REWARD_DEATH
            return self.get_state(), reward, done, {"score": self.score}

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.steps_without_food = 0
            reward = cfg.REWARD_FOOD
            self.food = self._spawn_food()
            if self.food is None:  # filled the whole board - about as done as it gets
                done = True
        else:
            self.snake.pop()  # move forward without growing
            new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            reward += cfg.REWARD_CLOSER if new_dist < prev_dist else cfg.REWARD_FARTHER

        return self.get_state(), reward, done, {"score": self.score}

    def get_state(self):
        """11-element feature vector: 3 relative danger flags, 4 absolute
        heading one-hot flags, 4 relative food-direction flags. Small and
        hand-built rather than raw pixels, so the MLP in model.py can learn
        from a few hundred episodes on CPU instead of needing a CNN."""
        head = self.snake[0]
        idx = _CLOCKWISE.index(self.direction)
        straight = self.direction
        right = _CLOCKWISE[(idx + 1) % 4]
        left = _CLOCKWISE[(idx - 1) % 4]

        def danger(d):
            return float(self._is_collision((head[0] + d[0], head[1] + d[1])))

        return [
            danger(straight),
            danger(right),
            danger(left),
            float(self.direction == UP),
            float(self.direction == DOWN),
            float(self.direction == LEFT),
            float(self.direction == RIGHT),
            float(self.food[1] < head[1]),  # food up
            float(self.food[1] > head[1]),  # food down
            float(self.food[0] < head[0]),  # food left
            float(self.food[0] > head[0]),  # food right
        ]
