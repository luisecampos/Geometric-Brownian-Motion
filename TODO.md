# To-do

## Quant
- [x] Finish/continue `stat_arb_engine` — now walk-forward, multi-pair, vol-targeted, with a portfolio tearsheet
- [x] Fix the stray `{data,signals,backtest,utils}` literal folder in `quant/stat_arb_engine/stat_arb/`
- [ ] `stat_arb_engine`: add stop-losses (exit if z-score blows past 3.5σ) and Kelly-sizing comparison
- [ ] Update `quant/EDGARX_README.md` — it documents `edgar_scraper.py`/`analyze_10k.py` but the code now lives combined in `EDGARX.py`
- [ ] Options strategy backtester using the `vol_surface.py` BS pricer
- [ ] Monte Carlo exotic-option pricer (revive/extend `GBM_citadel.py` for Asian/barrier options)
- [ ] Vol surface arbitrage scanner (calendar/butterfly violations on live chains)
- [ ] `Market_Widget.py` polish: sparkline history + price-cross alerts

## Agents
- [ ] Wire Aurelis's `web_search`/market tools together — ask it "how's the Nasdaq doing" out loud
- [ ] Optional `forget_fact` tool to complement `remember_fact`
- [ ] Local-only voice pipeline (whisper.cpp + local LLM) for offline use

## Games
- [ ] Finish the NEAT AI for `flappy_base.py` (it's literally staged for this)
- [ ] Market maker game — quote bid/ask on a simulated random-walk price (reuse GBM/OU code), get run over by informed flow, PnL tracked live
- [ ] Snake + DQN agent, as a second NEAT-vs-RL comparison point
- [ ] Connect 4 with minimax/alpha-beta
- [ ] 2048 + expectimax solver
- [ ] Wordle clone + entropy-based solver
- [ ] Blackjack simulator w/ card counting overlay

## Repo maintenance
- [ ] Decide fate of `GBM.py` vs `GBM_citadel.py` (git still shows `GBM.py` as deleted/uncommitted from before the reorg)
- [ ] Commit the folder reorg (on a branch, given `main` has diverged from `origin`)
