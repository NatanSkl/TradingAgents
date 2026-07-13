# monthly_run — cheap monthly review

Deep-*ish* look at every holding without the 8-min full pipeline. One Opus call
per stock over a free yfinance fact sheet, then one Opus call to conclude.

## Flow (`review.py`)
1. **Tier 0 — free data** per stock: momentum (3m/6m, RSI, SMA trend), fundamentals
   (`.info`: P/E, PEG, growth, margin, analyst view + target, 52wk position),
   latest headlines, insider net (120d). All yfinance, seconds.
2. **Tier 1 — deep-ish, every stock:** 1 Opus call → ~150-word analysis ending in
   `RATING` + `CONVICTION`.
3. **Tier 2 — conclusion:** 1 Opus call over all verdicts → allocate the budget +
   name the 1-2 stocks worth escalating to the full pipeline (`PROMOTE:` line).
4. **Tier 3 — optional/manual:** run `analyze_portfolio.py` on those names.

~10 model calls for the whole portfolio instead of ~100+.

## Run
```bash
.venv/bin/python3 -u monthly_run/review.py                 # defaults: 9 holdings, Opus
.venv/bin/python3 -u monthly_run/review.py --budget 600 --tickers PLTR SMH
.venv/bin/python3 -u monthly_run/review.py --model claude-sonnet-4-6   # cheaper
```
Output → `~/reports/monthly_review_<stamp>.md`. Knobs: `--tickers --date --budget --top --model`.

## Caveat
Tier 0 is a *snapshot* (ratios, headlines, insider) — not the full financial-statement
+ macro dig the pipeline does. That deeper read is what Tier 3 / `monthly_deep` is for.
