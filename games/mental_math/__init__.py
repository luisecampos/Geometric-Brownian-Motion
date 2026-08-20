"""
Mental Math - adaptive mental-arithmetic speed trainer.

See main.py for the entry point. Modules are split by concern:
    config.py      shared constants (window/colors/fonts, tiers, session lengths)
    questions.py   per-category question generators + answer validation
    difficulty.py  streak-based tier progression (45s -> 30s -> 15s)
    timer.py       framework-agnostic countdown timers
    stats.py       in-session stats + persistent (local JSON) all-time stats
    ui.py          pygame drawing + TextInput/Button widgets
    main.py        wires it all together, runs the game loop
"""
