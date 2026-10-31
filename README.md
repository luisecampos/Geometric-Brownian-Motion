# Code

Personal projects, organized by area.

```
quant/               Quant/trading tools
  Black-Scholes.py
  EDGARX.py                SEC EDGAR 10-K scraper + dashboard (see EDGARX_README.md)
  EDGARX_README.md
  GBM.py
  WTI_OU.py                Ornstein-Uhlenbeck model for WTI crude
  dcf_ep_model.py
  Market_Widget.py         Always-on-top desktop widget: DOW/NASDAQ/S&P/10Y/BTC
  vol_surface.py           Black-Scholes pricer + implied vol surface
  outputs/                 Generated charts (tornado sensitivity charts, etc.)
  simulator_visuals/       Saved GBM/Pi/WTI simulation chart images
  stat_arb_engine/         Statistical arbitrage backtesting engine (own subproject)

agents/               Voice / tool-using AI agents
  Aurelis.py                Push-to-talk voice assistant with tool use + memory
  aurelis_memory.json       Aurelis's durable facts store (git-ignored, created at runtime)

games/                Games (some doubling as AI/ML sandboxes)
  flappy_base.py            Flappy Bird base game, staged for a NEAT AI to plug into
  snake_dqn/                Deep Q-Network agent that learns Snake from scratch (own subproject, see its README)
    snake_env.py              Gym-like game engine (reset/step), no rendering, no AI
    model.py                  Q-network (MLP) + experience replay buffer
    agent.py                  Epsilon-greedy DQN agent: learning step, target-network sync, checkpoints
    train.py                  Headless training loop - logs, checkpoints, learning-curve plot
    play.py                   Watch a trained checkpoint play, or play yourself (`--human`) for comparison
  mental_math/              Adaptive mental-math speed trainer (pygame)
    main.py                   Entry point - run this (`python main.py`)
    config.py                 Colors/fonts/window size, tier + session-length constants
    questions.py               Per-category question generators + answer validation
    difficulty.py              Streak-based tier progression (45s -> 30s -> 15s)
    timer.py                   Framework-agnostic countdown timers
    stats.py                   Session stats + persistent local-JSON all-time stats
    ui.py                      pygame widgets (TextInput, Button) + screen rendering
    mental_math_stats.json    All-time bests, git-ignored, created at first run
  pattern_iq/               Speed pattern-recognition trainer (pygame), two modes
    main.py                   Entry point - run this (`python main.py`)
    config.py                 Colors/fonts/window size, tier + session-length constants
    sequences.py                Number Sequences mode: "what comes next" generators + distractors
    market_patterns.py          Spot the Pattern mode: synthetic price-series generators (trend/mean-reversion/
                                 random walk/volatility breakout/cyclical) for market-intuition training
    difficulty.py              Streak-based tier progression, parametrized per mode's clock
    timer.py                   Framework-agnostic countdown timers
    stats.py                   Session stats + persistent local-JSON all-time stats, per mode
    ui.py                      pygame widgets (Button, chart renderer) + screen rendering
    pattern_iq_stats.json     All-time bests, git-ignored, created at first run
  common/                   Shared code for *newer* brain-training games (mental_math and pattern_iq predate
                             this and stay fully self-contained - see Notes below)
    palette.py                 Citadel-derived colors + font name
    widgets.py                  TextInput, Button
    timer.py                    QuestionTimer, SessionTimer
    badge.py                    Shared corner badge - drop a logo in common/assets/ once, used by every
                                 game below
    quiz_engine.py               Generic timed/tiered category-quiz runner (menu/playing/results loop,
                                 difficulty, stats) - callers just supply question generators
  probability_quickfire.py  EV calc + American/decimal odds + card/dice probability (uses quiz_engine)
  n_back.py                 Working-memory drill: press SPACE on an N-back letter match, standalone
                             (continuous-stream mechanic doesn't fit quiz_engine's shape)
  sequence_recall.py        Digit-span/Simon-style recall: watch a tile sequence, repeat it back,
                             adaptive staircase, standalone (show-then-recall mechanic, not a quiz)

misc/                 Learning material and one-off experiments
  nbody_simulator.py        N-body gravitational simulator
  Python Learning Guide.pdf
  C++ Learning Guide.pdf
  Harvard CS50 Class.py
```

## Setup

Each script lists its own dependencies in its docstring. Shared virtualenv is `.venv/`.
Aurelis needs a `.env` file in the repo root with `ANTHROPIC_API_KEY=sk-ant-...` (see `agents/Aurelis.py`).

## Notes

- `quant/stat_arb_engine/stat_arb/` has a stray literal folder named `{data,signals,backtest,utils}`
  (leftover from a shell brace-expansion that didn't run as intended on Windows) — worth cleaning up separately.
- `quant/EDGARX_README.md` documents `edgar_scraper.py` / `analyze_10k.py` as two separate scripts, but the
  current code lives combined in `EDGARX.py` — the README is stale and could use an update.
- Two architectures coexist in `games/`: `mental_math/` and `pattern_iq/` are each fully self-contained
  packages (their own difficulty/timer/stats/ui, own `assets/logo.*`); `n_back.py`, `sequence_recall.py`,
  and `probability_quickfire.py` share `games/common/` instead. Not unified on purpose yet — ask if you want
  everything retrofit onto one shared base. Practical effect: there are now **three** separate places a logo
  file can go (`mental_math/assets/`, `pattern_iq/assets/`, `common/assets/`) if you want the badge everywhere.
