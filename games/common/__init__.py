"""
Shared code for the brain-training game family (mental_math, n_back,
sequence_recall, probability_quickfire, ...): color palette, pygame widgets,
countdown timers, the corner "why am I doing this" badge, and a reusable
timed-quiz engine for anything shaped like mental_math (categories + tiers +
type-the-answer).

mental_math predates this package and stays fully self-contained - these
modules exist so every *new* game in the family doesn't reimplement the same
~500 lines of menu/timer/stats/UI plumbing from scratch.
"""
