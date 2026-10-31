# Snake + DQN

A Deep Q-Network agent that learns to play Snake from scratch - no
hand-coded strategy, just reward, trial, and error. This is the repo's
second "AI learning machine" (alongside the planned NEAT agent for
[`games/flappy_base.py`](../flappy_base.py)), and a deliberate contrast
in approach:

| | NEAT (flappy_base, planned) | DQN (this project) |
|---|---|---|
| Learns | a population of whole policies | one Q-function |
| Update rule | evolution (selection + mutation) | gradient descent (Bellman backup) |
| On/off-policy | on-policy, gradient-free | off-policy, gradient-based |
| Needs a replay buffer? | no | yes |

## How it works

- **`snake_env.py`** - the game, with no rendering and no AI: a small
  Gym-like `reset()` / `step(action)` API. Actions are *relative* to the
  snake's current heading (`straight` / `turn right` / `turn left`), so
  the agent never has to learn the absolute compass.
- **State representation** - rather than raw pixels, the agent sees an
  11-element hand-built feature vector: 3 "is there danger if I go
  straight/right/left" flags, 4 one-hot current-heading flags, and 4
  relative food-direction flags. This is the standard compact encoding
  for this problem - it lets a small MLP converge in a few hundred
  episodes on a CPU, instead of needing a CNN over the raw grid.
- **`model.py`** - the Q-network (a 3-layer MLP, 11 -> 256 -> 256 -> 3)
  and a uniform-random experience replay buffer.
- **`agent.py`** - epsilon-greedy action selection, the DQN training
  step (Huber loss against a target network's TD target), and a hard
  target-network sync every `TARGET_SYNC_EVERY_STEPS` steps. Both the
  target network and replay buffer exist for the same reason: vanilla
  online Q-learning on correlated, non-stationary targets doesn't
  converge reliably.
- **`train.py`** - headless training loop. Logs to the console and to
  `training_log.json`, checkpoints the best-scoring and the latest
  policy under `checkpoints/`, and plots a learning curve on exit.
- **`play.py`** - watch a checkpoint play (greedy, `epsilon=0`), or play
  it yourself with arrow keys (`--human`) on the exact same `SnakeEnv`.

## Run it

```bash
pip install -r requirements.txt
# or, CPU-only torch explicitly:
pip install torch --index-url https://download.pytorch.org/whl/cpu

python train.py              # train for config.NUM_EPISODES episodes (headless, ~a few minutes)
python train.py 2000         # or pass an episode count directly

python play.py               # watch the best checkpoint play
python play.py --checkpoint latest --fps 30
python play.py --human       # play it yourself
```

## Tuning

Every hyperparameter (learning rate, gamma, epsilon schedule, replay
size, reward shaping, network size) lives in `config.py`. A few notes
if training looks stuck:

- **Score plateaus early (~5-10):** the snake is likely trapping itself
  with its own body. Try increasing `HIDDEN_SIZE` or `EPSILON_DECAY_EPISODES`
  (more exploration before committing to a policy).
- **Score is noisy / loss spikes:** lower `LEARNING_RATE`, or increase
  `TARGET_SYNC_EVERY_STEPS` (a target that updates too often reintroduces
  the instability it's meant to prevent).
- **Learning is slow to start:** the agent does nothing but explore
  randomly until `MIN_REPLAY_BEFORE_TRAIN` transitions have accumulated -
  that's expected, not a bug.
- **Dense shaping:** `REWARD_CLOSER` / `REWARD_FARTHER` are 0 by default
  (sparse reward: only food and death matter). Setting them to e.g.
  `+-0.1` rewards moving toward the food every step, which usually speeds
  up early learning at some risk of the agent over-fitting to "chase the
  food" over "don't trap yourself."

## Possible extensions

- Swap the hand-built state vector for the raw grid + a small CNN, to
  see how much of the 11-feature encoding's head start that costs.
- Double DQN (decouple action selection from evaluation in the TD
  target) or Dueling DQN (separate value/advantage heads) - both are
  small changes to `agent.py` / `model.py` and a natural next step once
  vanilla DQN plateaus.
- Once `flappy_base.py`'s NEAT agent exists, compare learning curves:
  generations-to-competence (NEAT) vs. episodes-to-competence (DQN) on
  two different games isn't an apples-to-apples benchmark, but the
  shape of the curves (smooth vs. punctuated) is still an interesting
  contrast between the two paradigms.
