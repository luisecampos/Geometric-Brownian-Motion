"""
Train the DQN agent on Snake. Headless (no pygame) - nothing needs to
render every step, so this runs far faster than real-time play.

Logs score/reward per episode to the console and to training_log.json,
saves the best-scoring and the latest checkpoint under checkpoints/, and
plots a learning curve (raw score + moving average) when training ends.

Run:
    python train.py                      (from inside games/snake_dqn/)
    python -m games.snake_dqn.train      (from the repo root)

Dependencies: pip install -r requirements.txt
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snake_dqn import config as cfg
from snake_dqn.agent import DQNAgent
from snake_dqn.snake_env import SnakeEnv


def moving_average(values, window=50):
    if len(values) < window:
        return values[:]
    out = []
    running = sum(values[:window])
    out.extend([None] * (window - 1))
    out.append(running / window)
    for i in range(window, len(values)):
        running += values[i] - values[i - window]
        out.append(running / window)
    return out


def train(num_episodes=cfg.NUM_EPISODES):
    env = SnakeEnv()
    agent = DQNAgent()

    scores, rewards_log, losses_log = [], [], []
    best_score = -1
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        epsilon = agent.epsilon_for_episode(episode)
        total_reward = 0.0
        episode_losses = []
        done = False

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            loss = agent.learn_step()
            if loss is not None:
                episode_losses.append(loss)

        score = info["score"]
        scores.append(score)
        rewards_log.append(total_reward)
        losses_log.append(sum(episode_losses) / len(episode_losses) if episode_losses else None)

        if score > best_score:
            best_score = score
            agent.save(cfg.BEST_CHECKPOINT_PATH)

        if episode % cfg.CHECKPOINT_EVERY_EPISODES == 0:
            agent.save(cfg.LATEST_CHECKPOINT_PATH)

        if episode % cfg.LOG_EVERY_EPISODES == 0:
            recent = scores[-cfg.LOG_EVERY_EPISODES:]
            avg_score = sum(recent) / len(recent)
            elapsed = time.time() - start_time
            print(
                f"episode {episode:5d}/{num_episodes}  "
                f"score {score:3d}  avg({cfg.LOG_EVERY_EPISODES}) {avg_score:5.1f}  "
                f"best {best_score:3d}  eps {epsilon:.3f}  "
                f"buffer {len(agent.replay):6d}  elapsed {elapsed:6.1f}s"
            )

    agent.save(cfg.LATEST_CHECKPOINT_PATH)
    _write_log(scores, rewards_log, losses_log)
    _plot_learning_curve(scores)
    print(f"\nTraining done. Best score: {best_score}. "
          f"Checkpoints saved to {cfg.CHECKPOINT_DIR}")
    return scores


def _write_log(scores, rewards_log, losses_log):
    with open(cfg.TRAINING_LOG_PATH, "w") as f:
        json.dump({"scores": scores, "rewards": rewards_log, "losses": losses_log}, f)


def _plot_learning_curve(scores):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping learning-curve plot.")
        return

    avg = moving_average(scores, window=min(50, max(1, len(scores) // 10)))
    plt.figure(figsize=(10, 5))
    plt.plot(scores, color="#4a4a4a", alpha=0.35, label="score per episode")
    plt.plot(avg, color="#1a4ca5", linewidth=2, label="moving average")
    plt.xlabel("episode")
    plt.ylabel("score (food eaten)")
    plt.title("Snake DQN - learning curve")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(cfg.PACKAGE_DIR, "learning_curve.png")
    plt.savefig(out_path, dpi=120)
    print(f"Learning curve saved to {out_path}")


if __name__ == "__main__":
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else cfg.NUM_EPISODES
    train(num_episodes=episodes)
