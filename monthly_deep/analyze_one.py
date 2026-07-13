"""
Worker: run the full multi-agent pipeline on ONE ticker and write a clean
markdown report into a run dir. Meant to be launched in parallel (one process
per ticker) by run.py.

    .venv/bin/python3 -u monthly_deep/analyze_one.py --ticker PLTR --date 2026-07-13 --out ~/reports/deep_run_x/PLTR.md
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

parser = argparse.ArgumentParser()
parser.add_argument("--ticker", required=True)
parser.add_argument("--date", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

ticker = args.ticker.upper()
out_path = os.path.expanduser(args.out)

config = DEFAULT_CONFIG.copy()
config["max_recur_limit"] = 500
config["checkpoint_enabled"] = True  # resume from last node if interrupted

print(f"[{ticker}] starting…", flush=True)
t0 = time.time()
ta = TradingAgentsGraph(debug=True, config=config)
final_state, signal = ta.propagate(ticker, args.date)
elapsed = (time.time() - t0) / 60

g = lambda k: final_state.get(k, "") or "_(none)_"
md = f"""# {ticker} — Deep Analysis ({args.date})

**Signal:** {signal}  |  **Runtime:** {elapsed:.1f} min

## Final Trade Decision
{g("final_trade_decision")}

## Trader Plan
{g("trader_investment_plan")}

## Research Manager — Investment Plan
{g("investment_plan")}

## Analyst Reports

### Market / Technicals
{g("market_report")}

### Fundamentals
{g("fundamentals_report")}

### News
{g("news_report")}

### Sentiment / Social
{g("sentiment_report")}
"""

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(md)

print(f"[{ticker}] done → {signal}  ({elapsed:.1f} min) → {out_path}", flush=True)
