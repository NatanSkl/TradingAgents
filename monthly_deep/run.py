"""
Monthly Deep Run  —  full pipeline on several tickers in parallel, then a
summarize-each + Opus-conclusion final stage.

Stage A (parallel): launch analyze_one.py per ticker → one <TICKER>.md each in
        a fresh run dir  ~/reports/deep_run_<stamp>/ .
Stage B (final): summarize every .md (cheap model, keeping the WHY behind each
        decision) → feed all summaries to Opus for a final investment
        conclusion + allocation. Written to <run dir>/CONCLUSION.md .

Stage B is dynamic — point it at an existing run dir to re-run just the final
stage (fast, cheap) without redoing the expensive pipeline:

    # full run:
    .venv/bin/python3 -u monthly_deep/run.py
    # only the final stage against an existing run dir:
    .venv/bin/python3 -u monthly_deep/run.py --run-dir ~/reports/deep_run_2026-07-13_12-00-00
"""

import argparse
import os
import subprocess
import sys
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients import create_llm_client

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PY = os.path.join(REPO, ".venv", "bin", "python3")
TICKERS = ["PLTR", "QQQM", "QTUM", "SMH", "TSLA", "VOO"]

parser = argparse.ArgumentParser()
parser.add_argument("--tickers", nargs="+", default=TICKERS)
parser.add_argument("--date", default=date.today().isoformat())
parser.add_argument("--budget", type=float, default=600.0)
parser.add_argument("--run-dir", default=None, help="existing run dir → skip analysis, run only the final stage")
parser.add_argument("--sum-model", default="claude-haiku-4-5", help="model that summarizes each report")
parser.add_argument("--final-model", default="claude-opus-4-8", help="model for the final conclusion")
args = parser.parse_args()

tickers = [t.upper() for t in args.tickers]
provider = os.environ.get("TRADINGAGENTS_LLM_PROVIDER", DEFAULT_CONFIG["llm_provider"])


# ── Stage A: parallel full pipeline (skipped when --run-dir is given) ──────────
if args.run_dir:
    run_dir = os.path.expanduser(args.run_dir)
    print(f"Final stage only, against existing run dir: {run_dir}")
else:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.expanduser(f"~/reports/deep_run_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}\nLaunching {len(tickers)} pipelines in parallel: {', '.join(tickers)}\n")

    procs = []
    for tk in tickers:
        out = os.path.join(run_dir, f"{tk}.md")
        log = open(os.path.join(run_dir, f"{tk}.log"), "w")
        p = subprocess.Popen(
            [PY, "-u", os.path.join(HERE, "analyze_one.py"), "--ticker", tk, "--date", args.date, "--out", out],
            stdout=log, stderr=subprocess.STDOUT, cwd=REPO,
        )
        procs.append((tk, p, log))
        print(f"  launched {tk}  (pid {p.pid}) → {os.path.join(run_dir, tk + '.log')}")

    print("\nWaiting for all pipelines to finish…")
    for tk, p, log in procs:
        p.wait()
        log.close()
        status = "ok" if p.returncode == 0 else f"FAILED (exit {p.returncode})"
        print(f"  {tk}: {status}")


# ── Stage B: summarize each report, then Opus conclusion ──────────────────────
mds = sorted(f for f in os.listdir(run_dir) if f.endswith(".md") and f != "CONCLUSION.md")
if not mds:
    sys.exit(f"No per-ticker .md reports found in {run_dir}")

print(f"\nSummarizing {len(mds)} reports with {args.sum_model}…")
sum_llm = create_llm_client(provider=provider, model=args.sum_model).get_llm()
final_llm = create_llm_client(provider=provider, model=args.final_model).get_llm()


def ask(llm, prompt):
    r = llm.invoke(prompt)
    return r.content if hasattr(r, "content") else str(r)


summaries = []
for fn in mds:
    tk = fn[:-3]
    text = open(os.path.join(run_dir, fn), encoding="utf-8").read()
    prompt = f"""Summarize this deep multi-agent analysis of {tk} in ~180 words.
KEEP the final recommendation and, most importantly, the REASONING behind it — the specific
bull/bear points, the trader's plan, position sizing, stop loss, and the key risks. Drop boilerplate.

{text}"""
    s = ask(sum_llm, prompt)
    summaries.append(f"### {tk}\n{s}")
    print(f"  summarized {tk}")

print(f"\nFinal conclusion with {args.final_model}…")
final_prompt = f"""You are the client's portfolio manager. Today is {args.date}. Below are condensed deep-analysis
summaries for each holding, each retaining the reasoning behind its recommendation. The client will deploy
${args.budget:,.2f} of new cash this month.

{chr(10).join(summaries)}

---

Give a decisive final investment conclusion:
1. Allocate ${args.budget:,.2f} across the tickers (can be $0 for any; total must equal ${args.budget:,.2f}).
2. One-sentence rationale per non-zero position, grounded in the analyses above.
3. Call out the single biggest risk to this month's plan.
4. End with a summary table: | Ticker | Allocation | Rationale |"""
conclusion = ask(final_llm, final_prompt)

out = os.path.join(run_dir, "CONCLUSION.md")
with open(out, "w", encoding="utf-8") as f:
    f.write(f"# Final Investment Conclusion — {args.date}\n")
    f.write(f"**Budget:** ${args.budget:,.2f}  |  **Summaries:** {args.sum_model}  |  **Conclusion:** {args.final_model}\n\n")
    f.write("## Per-holding summaries\n\n" + "\n\n".join(summaries) + "\n\n---\n\n")
    f.write("## Conclusion\n\n" + conclusion + "\n")

print("\n" + "=" * 72)
print(conclusion)
print("=" * 72)
print(f"\nSaved → {out}")
