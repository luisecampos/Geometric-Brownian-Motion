# To-do

## Quant
- [x] Finish/continue `stat_arb_engine` — now walk-forward, multi-pair, vol-targeted, with a portfolio tearsheet
- [x] Fix the stray `{data,signals,backtest,utils}` literal folder in `quant/stat_arb_engine/stat_arb/`
- [ ] `stat_arb_engine`: Kelly-sizing comparison (stop-losses at |z|>3.5 done)
- [ ] Update `quant/EDGARX_README.md` — it documents `edgar_scraper.py`/`analyze_10k.py` but the code now lives combined in `EDGARX.py`
- [ ] Options strategy backtester using the `vol_surface.py` BS pricer
- [ ] Monte Carlo exotic-option pricer (revive/extend `GBM.py` for Asian/barrier options)
- [ ] Vol surface arbitrage scanner (calendar/butterfly violations on live chains)
- [ ] `Market_Widget.py` polish: sparkline history + price-cross alerts

## Agents
- [ ] Wire Aurelis's `web_search`/market tools together — ask it "how's the Nasdaq doing" out loud
- [ ] Optional `forget_fact` tool to complement `remember_fact`
- [ ] Local-only voice pipeline (whisper.cpp + local LLM) for offline use

## Games
- [ ] Finish the NEAT AI for `flappy_base.py` (it's literally staged for this)
- [ ] Market maker game — quote bid/ask on a simulated random-walk price (reuse GBM/OU code), get run over by informed flow, PnL tracked live
- [x] Snake + DQN agent, as a second NEAT-vs-RL comparison point (`games/snake_dqn/`)
- [ ] `snake_dqn`: try Double DQN / Dueling DQN once vanilla DQN plateaus (see its README's "Possible extensions")
- [ ] Connect 4 with minimax/alpha-beta
- [ ] 2048 + expectimax solver
- [ ] Wordle clone + entropy-based solver
- [ ] Blackjack simulator w/ card counting overlay

## Repo maintenance
- [x] Decide fate of `GBM.py` vs `GBM_citadel.py` — keeping `GBM.py`, deleted `GBM_citadel.py`
- [x] Commit the folder reorg (on a branch, given `main` has diverged from `origin`)
- [ ] Decide whether/how to reconcile `main` with `origin` (unrelated histories — `origin/main` was never merged into the current `main`)
