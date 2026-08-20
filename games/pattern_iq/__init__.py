"""
Pattern IQ - speed pattern-recognition trainer, two modes in one shell.

    Number Sequences   "what comes next" IQ-test style number puzzles.
    Spot the Pattern   classify a generated price series (uptrend, mean-
                       reversion, random walk, volatility breakout, ...) -
                       aimed at building the "is this signal or noise"
                       instinct that's useful for market intuition.

See main.py for the entry point. Modules split by concern, mirroring the
sibling games/mental_math package:
    config.py          shared constants (window/colors/fonts, tiers, session lengths)
    sequences.py        Number Sequences: per-category generators + multiple-choice distractors
    market_patterns.py  Spot the Pattern: synthetic price-series generators + pattern labels
    difficulty.py        streak-based tier progression, parametrized per mode's time limits
    timer.py             framework-agnostic countdown timers
    stats.py              in-session stats + persistent (local JSON) all-time stats, per mode
    ui.py                  pygame drawing + Button/ChoiceGrid widgets + chart renderer
    main.py                 wires it all together, runs the game loop
"""
