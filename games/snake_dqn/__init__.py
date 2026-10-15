"""
Snake + DQN - a value-based reinforcement-learning agent that learns to
play Snake from scratch via trial and error (no hand-coded strategy).

This is the repo's second "AI learning machine" data point, alongside
games/flappy_base.py's planned NEAT agent: NEAT evolves a population of
whole policies (on-policy, gradient-free), while this trains a single
network's Q-function with experience replay (off-policy, gradient-based).
Same underlying goal - a game-playing agent - two very different ways of
getting there.

Modules are split by concern:
    config.py       grid/rendering constants + all DQN hyperparameters
    snake_env.py    the game itself: a small Gym-like reset()/step() API,
                    deliberately rendering-agnostic
    model.py        the Q-network (small MLP) and the replay buffer
    agent.py        DQNAgent: epsilon-greedy action selection, learning
                    step, target-network sync, checkpoint save/load
    train.py        the training loop entry point (python train.py)
    play.py         watch a trained checkpoint play, or play by hand
                    for comparison (python play.py [--human] [--checkpoint PATH])
"""
