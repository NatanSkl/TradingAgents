"""
Portfolio Monthly Investment Advisor
=====================================
Runs the full TradingAgents pipeline on every holding, streams all agent
thinking live, then asks Claude for a final allocation recommendation.

Usage:
    .venv/bin/python3 -u analyze_portfolio.py
    .venv/bin/python3 -u analyze_portfolio.py --budget 1000
    .venv/bin/python3 -u analyze_portfolio.py --date 2026-05-14
    .venv/bin/python3 -u analyze_portfolio.py --tickers VOO TSLA

Notes:
  • Checkpointing is ON — if the run is interrupted, restarting picks up
    from the last completed node for each ticker.
  • All output is tee'd to ~/reports/portfolio_<timestamp>.md in real time.
  • "======= Human Message =======  Continue" between analysts is NORMAL —
    it is an Anthropic-compatibility placeholder, not a hang.
"""

import argparse
import os
import sys
import threading
import time
from datetime import datetime, date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients import create_llm_client

# ── Portfolio ────────────────────────────────────────────────────────────────
PORTFOLIO = ["VOO", "TSLA", "SMH", "QTUM", "QQQM", "GOP", "ETHA", "BITB"]
DEFAULT_BUDGET = 680.0

# ── CLI args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--budget",     type=float,  default=DEFAULT_BUDGET)
parser.add_argument("--date",       type=str,    default=date.today().isoformat())
parser.add_argument("--tickers",    nargs="+",   default=PORTFOLIO)
parser.add_argument("--output-dir", type=str,    default="~/reports")
args = parser.parse_args()

tickers    = [t.upper() for t in args.tickers]
trade_date = args.date
budget     = args.budget

# ── Output file (tee stdout → file) ──────────────────────────────────────────
class Tee:
    def __init__(self, f):
        self._f   = f
        self._out = sys.__stdout__
    def write(self, data):
        self._out.write(data);  self._out.flush()
        self._f.write(data);    self._f.flush()
    def flush(self):
        self._out.flush(); self._f.flush()

out_dir   = Path(args.output_dir).expanduser()
out_dir.mkdir(parents=True, exist_ok=True)
stamp     = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_file  = out_dir / f"portfolio_{stamp}.md"
log       = open(out_file, "w", encoding="utf-8")
sys.stdout = Tee(log)

# ── Header ────────────────────────────────────────────────────────────────────
print(f"""# Portfolio Analysis — {datetime.now().strftime("%B %d, %Y %H:%M")}
**Budget:** ${budget:,.2f}  |  **Date:** {trade_date}  |  **Tickers:** {", ".join(tickers)}
**Models:** {os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM","?")} (deep) / {os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM","?")} (quick)

> 'Continue' messages between analysts are normal Anthropic placeholders — not a hang.

{"=" * 70}
""")

# ── TradingAgentsGraph setup ──────────────────────────────────────────────────
config = DEFAULT_CONFIG.copy()
config["max_recur_limit"]     = 500   # Claude is verbose; 100 default is too low
config["checkpoint_enabled"]  = True  # resume from last node if run is interrupted

ta = TradingAgentsGraph(debug=True, config=config)

# ── Per-ticker analysis ───────────────────────────────────────────────────────
results = []  # list of {"ticker", "signal", "decision"}

for ticker in tickers:
    print(f"\n{'=' * 70}")
    print(f"  ▶  ANALYSING {ticker}   ({trade_date})")
    print(f"{'=' * 70}\n")

    t0 = time.time()
    try:
        final_state, signal = ta.propagate(ticker, trade_date)
        elapsed = time.time() - t0
        decision = final_state.get("final_trade_decision", "")
        results.append({"ticker": ticker, "signal": signal, "decision": decision})
        print(f"\n{'─' * 70}")
        print(f"  ✓  {ticker}  →  {signal}   ({elapsed/60:.1f} min)")
        print(f"{'─' * 70}\n")
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n  ✗  {ticker}  ERROR after {elapsed/60:.1f} min: {exc}\n")
        results.append({"ticker": ticker, "signal": "ERROR", "decision": str(exc)})

# ── Final allocation call ─────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("  FINAL ALLOCATION — asking Claude to recommend how to deploy the budget")
print(f"{'=' * 70}\n")

# Build summary of all decisions for the allocation prompt
decisions_text = "\n\n".join(
    f"### {r['ticker']}  (Signal: {r['signal']})\n{r['decision']}"
    for r in results
)

allocation_prompt = f"""You are a portfolio advisor. A client has completed a full multi-agent analysis of their portfolio for {trade_date}.
They want to invest ${budget:,.2f} today across some or all of these positions.

Here are the full Portfolio Manager decisions for each holding:

{decisions_text}

---

Based on these analyses, please:
1. Recommend exactly how to allocate the ${budget:,.2f} across the tickers (can be $0 for any ticker).
2. Give a one-sentence rationale per ticker explaining your allocation choice.
3. End with a concise summary table:

| Ticker | Allocation | Rationale |
|--------|-----------|-----------|

Be specific with dollar amounts. The total must equal exactly ${budget:,.2f}.
Do not add any tickers outside the list above.
"""

try:
    provider  = os.environ.get("TRADINGAGENTS_LLM_PROVIDER", config["llm_provider"])
    model     = os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM", config["deep_think_llm"])
    client    = create_llm_client(provider=provider, model=model)
    llm       = client.get_llm()
    response  = llm.invoke(allocation_prompt)
    allocation_text = response.content if hasattr(response, "content") else str(response)
except Exception as exc:
    allocation_text = f"(Allocation call failed: {exc})"

print(allocation_text)

# ── Quick signal summary ──────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("  SIGNAL SUMMARY")
print(f"{'=' * 70}")
for r in results:
    print(f"  {r['ticker']:<6}  {r['signal']}")
print(f"{'=' * 70}")
print(f"\nReport saved → {out_file}\n")

log.close()
