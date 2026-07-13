# monthly_deep — full pipeline in parallel + Opus conclusion

Runs the complete multi-agent pipeline on several tickers **at once**, then
summarizes each report and asks Opus for a final allocation. This is the
expensive/thorough tier (vs. `monthly_run`'s cheap snapshot).

## Flow (`run.py`)
- **Stage A — parallel pipelines:** launches `analyze_one.py` as one process per
  ticker → each writes a clean `<TICKER>.md` (final decision, trader plan, all
  analyst reports) into a fresh run dir `~/reports/deep_run_<stamp>/`.
- **Stage B — summarize + conclude:** Haiku summarizes each `.md` (keeping the
  *reasoning* behind each call), then Opus reads all summaries + the budget and
  writes `CONCLUSION.md` (allocation table + biggest risk).

`analyze_one.py` is the per-ticker worker; you normally just run `run.py`.

## Run
```bash
# full run (Stage A + B): 6 default tickers in parallel
.venv/bin/python3 -u monthly_deep/run.py --budget 600

# Stage B only, against an existing run dir (fast/cheap — iterate on the final step)
.venv/bin/python3 -u monthly_deep/run.py --run-dir ~/reports/deep_run_<stamp>
```
Knobs: `--tickers --date --budget --run-dir --sum-model --final-model`.

## Notes
- Parallel = same wall-clock (~8-9 min) as one run; token cost is unchanged.
- Many concurrent pipelines can hit API 429s — batch 3-4 at a time if so.
- Reddit 403s in the logs are harmless (social data rate-limited, handled).
