"""Shared constants and hyperparameters for Snake + DQN.

No pygame or torch imports here, so snake_env.py (and anything testing
it) can import this without either dependency already set up.
"""

import os

# ---- Grid ----
GRID_WIDTH = 20
GRID_HEIGHT = 20
CELL_SIZE = 24

# ---- Rendering (play.py only) ----
SCORE_BAR_HEIGHT = 40
WIN_WIDTH = GRID_WIDTH * CELL_SIZE
WIN_HEIGHT = GRID_HEIGHT * CELL_SIZE + SCORE_BAR_HEIGHT
FPS_HUMAN = 10
FPS_WATCH = 12
FONT_NAME = "Segoe UI"

BG_COLOR = (15, 15, 20)
GRID_LINE_COLOR = (26, 26, 34)
SNAKE_HEAD_COLOR = (110, 232, 233)
SNAKE_BODY_COLOR = (20, 74, 165)
FOOD_COLOR = (232, 93, 93)
TEXT_COLOR = (255, 255, 255)
MUTED_TEXT_COLOR = (159, 160, 161)

# ---- Actions (relative to current heading - the agent never has to
# learn the absolute compass, just "keep going / turn") ----
ACTION_STRAIGHT, ACTION_RIGHT, ACTION_LEFT = 0, 1, 2
ACTION_SIZE = 3

# ---- State vector (see snake_env.SnakeEnv.get_state) ----
STATE_SIZE = 11

# ---- Reward shaping ----
REWARD_FOOD = 10.0
REWARD_DEATH = -10.0
REWARD_STEP = -0.01     # tiny time penalty so the agent prefers direct paths
REWARD_CLOSER = 0.0     # optional dense shaping; 0 keeps rewards sparse by default
REWARD_FARTHER = 0.0    # bump both of these (e.g. +-0.1) if training stalls

MAX_STEPS_WITHOUT_FOOD = GRID_WIDTH * GRID_HEIGHT * 4  # truncate episodes that go nowhere

# ---- DQN hyperparameters ----
HIDDEN_SIZE = 256
LEARNING_RATE = 5e-4
GAMMA = 0.95

REPLAY_CAPACITY = 100_000
BATCH_SIZE = 1024
MIN_REPLAY_BEFORE_TRAIN = 1_000

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_EPISODES = 300   # linear decay from start to end over this many episodes

TARGET_SYNC_EVERY_STEPS = 1_000

NUM_EPISODES = 1_000
CHECKPOINT_EVERY_EPISODES = 50
LOG_EVERY_EPISODES = 10

# ---- Paths ----
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PACKAGE_DIR, "checkpoints")
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best.pt")
LATEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest.pt")
TRAINING_LOG_PATH = os.path.join(PACKAGE_DIR, "training_log.json")
