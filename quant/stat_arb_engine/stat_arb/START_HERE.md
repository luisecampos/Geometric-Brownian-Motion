# Start here: this project in plain English

No coding or finance background needed. Words in **bold** are explained in the
glossary at the end. For the technical version, see the [README](README.md).

## The short version

This project asks one question: **could a computer have made money from a
well-known trading idea, after paying the real costs of trading?**

So far the answer is no. The best version made money before costs, and
realistic costs took most or all of that away. It runs on historical prices only.
It doesn't trade real money, and it isn't investment advice.

The project's real strength is how it tests. It's built to avoid the usual ways
people fool themselves with trading results, which is why it can report "this
doesn't work" with some confidence.

## The trading idea: pairs trading

Some companies are so alike that their share prices tend to move together.
Think of two big banks, or two oil producers.

Sometimes one pulls ahead of the other for no lasting reason. Pairs trading bets
that this gap will close:

1. **Buy** the one that fell behind.
2. **Short** the one that pulled ahead, which means betting its price will fall.
3. When the gap closes, exit both and keep the difference.

It's like a person walking a dog on a stretchy lead. The two can drift apart for
a while, but the lead tends to pull them back together.

The appeal is that you're not guessing whether the whole market goes up or down.
Because you hold one of each, a market-wide move mostly cancels out. All you need
is for the gap to close.

The risk is that sometimes the gap has a real cause, such as bad news at one
company. Then it doesn't close, and the trade loses money.

## How it works

Every three months, the program does the same steps:

1. **Gets prices.** It downloads daily prices for about 290 large US companies
   and sector funds.
2. **Picks pairs.** It looks at the past year and finds pairs of similar
   companies whose prices have stayed close.
3. **Waits for gaps.** Over the next three months, it watches those pairs for an
   unusually wide gap.
4. **Places pretend trades.** It simulates the trade, charging fees along the way.
5. **Adds it all up.** It combines every trade into a single result.

Testing an idea on past prices like this is called a **backtest**.

## Why this is harder than it sounds

A backtest is easy to fool, even by accident. Try 100 ideas and one will look
brilliant by luck alone. If 100 people each flip a coin ten times, someone will
get a streak that looks like skill.

The project has four rules against this:

- **Write the test down first.** Before each test runs, its rules and its pass
  mark are saved in the project's history, so the goalposts can't move afterwards.
  This is called **pre-registration**.
- **Count every attempt.** The project has tried 91 configurations so far, and
  results are marked down for that. More attempts means a good result needs to be
  better to count. This is the **deflated Sharpe ratio**.
- **No second chances.** A failed idea stays failed. Adjusting it and trying
  again counts as a brand-new attempt.
- **No peeking at the future.** Each simulated decision uses only prices that
  were available on that day. Automated checks confirm this.

## How to read the results

Results are scored with the **Sharpe ratio**, which measures how much a strategy
earned for the ups and downs it went through.

| Sharpe ratio | Rough meaning |
|---|---|
| Below 0 | Lost money |
| 0 to 0.5 | Earned little for the risk taken |
| 0.5 to 1 | Respectable, but could still be luck |
| 1 or more | Good. The project's last test used this as its pass mark |

Costs are measured in **basis points**, where 1 basis point is 0.01%, or one
cent on every $100 traded. The tests use four cost levels:

| Level | Cost per trade | Fee for borrowing shares to short |
|---|---:|---:|
| Free | None | None |
| Current | 2 basis points | None |
| Realistic | 5 basis points | 0.50% a year |
| Harsh | 10 basis points | 1.00% a year |

These are assumptions. The real costs haven't been measured yet.

## What's been tried

| # | Test | What happened | Sharpe ratio |
|---:|---|---|---:|
| 1 | The basic version, 2019-2023 | Won 58% of its trades, but still lost money overall | -0.18 |
| 2 | Machine learning to choose trades | Made things worse. Its most confident picks were its worst | -0.50 to -0.63 |
| 3 | Automatically tuned settings | Looked good at 0.84, then lost money on fresh 2024-2026 prices | -0.44 |
| 4 | New rules for leaving losing trades | Barely any change. Most of those trades lose however they're closed | 0.08 |
| 5 | **Pickier choice of pairs (DISTANCE20)** | The best so far: only the 20 closest pairs in each period | 0.70 |
| 6 | Skipping trades around company news | Too small a change to tell apart from luck | 0.80 |
| 7 | DISTANCE20 at realistic and harsh costs | Most of the profit went to costs | 0.27 and -0.40 |
| 8 | DISTANCE20 on unused years, 2010-2017 | Made money before costs, and about zero after them | 0.83, then -0.04 |
| 9 | A different idea: each stock against its sector fund | Failed its pass mark of 1.0 on both periods | -0.24 and -0.59 |

Unless the row says otherwise, these use the Current cost level. Row 9 and the
second number in row 8 use Realistic. Row 4's 0.08 is the combined rules, on the
basic version. Row 6's 0.80 is DISTANCE20 skipping earnings news.

## Where it stands now

**Nothing has passed yet.** The best version, DISTANCE20, makes a small profit
before costs. That isn't proof of a real edge, and it may not survive costs.

**Costs decide it.** Each trade costs very little, but the strategy trades its
full account value about 134 times a year. It earns about 10.7% a year before
costs, and every extra basis point takes away about 1.3% a year. Around 7 or 8
basis points the profit is gone. On 2010-2017, that point is lower, at about 5.

**The results are probably flattered.** The company list is today's large
companies, so firms that collapsed or got bought aren't in it. That's
**survivorship bias**, and it makes old results look better than they really were.

**The strategy also borrows more than a normal account allows.** It holds about
3.8 times its capital in positions, and a standard margin account allows about 2.

**The final exam starts on 2 November 2026.** DISTANCE20 is locked in and will
run on new prices that nobody has seen, for 252 trading days. The results stay
hidden until the end and are judged once, around November 2027. This is the
**forward holdout**, and no amount of hindsight can affect it. DISTANCE20 didn't
actually pass its earlier test, so this was a declared exception, recorded as one.

**More daily-price ideas are unlikely to help.** Two different approaches have
now failed at realistic costs. Any new research would need different information,
and real costs would need measuring.

## What's in this folder

You don't need to run anything to follow the project. If you're curious:

| Item | What it is |
|---|---|
| `README.md` | The full technical write-up, with every result |
| `SECTOR_RESIDUAL.md` | The plan for test 9, written before it ran |
| `main.py` | The main program |
| Files ending `_test.py` | One file for each test above |
| `data/` | Downloads prices and holds the company list |
| `signals/` | Decides when to trade |
| `backtest/` | Simulates the trades and scores them |
| `learn/` | The machine-learning experiments |
| `tests/` | Automated checks that the program is working correctly |

## Glossary

| Term | Meaning |
|---|---|
| **Backtest** | Simulating a strategy on past prices to see how it would have done |
| **Basis point** | 0.01%. Five basis points is five cents on every $100 |
| **Deflated Sharpe ratio** | A Sharpe ratio marked down for how many attempts it took to find it |
| **DISTANCE20** | This project's best version: the 20 same-sector pairs whose prices stayed closest |
| **Forward holdout** | A test on future prices, with the strategy locked in before they exist |
| **Pre-registration** | Saving a test's rules and pass mark before running it |
| **Sector fund** | A single investment holding many companies from one industry, also called a sector ETF |
| **Sharpe ratio** | Return earned for the ups and downs taken along the way. Higher is better |
| **Short** | Borrowing shares and selling them, hoping to buy them back cheaper later |
| **Survivorship bias** | Testing only on companies that survived, which makes the past look better |
