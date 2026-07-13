"""
Monthly Portfolio Review  —  cheap depth on every holding, then a conclusion.
============================================================================
Two Opus tiers instead of the full 8-min multi-agent pipeline per stock:

  Tier 1 (deep-ish, every stock): one Opus call per holding over a free
          fact sheet (momentum + fundamentals + news + insider) → analysis
          & verdict. ~1 call each instead of the pipeline's ~15.
  Tier 2 (conclusion, one call):  Opus reads all Tier-1 verdicts and the
          budget, allocates, and names the 1-2 holdings that most deserve
          the expensive full pipeline this month.

  Tier 3 (optional, manual):      run analyze_portfolio.py on those names.

All Tier-0 data is free yfinance (seconds). Only the Opus calls cost.

Usage:
    .venv/bin/python3 -u monthly_run/review.py
    .venv/bin/python3 -u monthly_run/review.py --budget 600 --tickers PLTR SMH TSLA
    .venv/bin/python3 -u monthly_run/review.py --model claude-sonnet-4-6   # cheaper
"""

import argparse
import os
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients import create_llm_client

PORTFOLIO = ["BITB", "ETHA", "GOP", "PLTR", "QQQM", "QTUM", "SMH", "TSLA", "VOO"]

parser = argparse.ArgumentParser()
parser.add_argument("--tickers", nargs="+", default=PORTFOLIO)
parser.add_argument("--date", default=date.today().isoformat())
parser.add_argument("--budget", type=float, default=680.0)
parser.add_argument("--top", type=int, default=2, help="how many names to promote to Tier 3")
parser.add_argument("--model", default="claude-opus-4-8", help="model for both tiers")
args = parser.parse_args()

tickers = [t.upper() for t in args.tickers]
as_of = datetime.strptime(args.date, "%Y-%m-%d")

# ── Output: print live AND tee to a report file ───────────────────────────────
out_dir = os.path.expanduser("~/reports")
os.makedirs(out_dir, exist_ok=True)
stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
report = open(os.path.join(out_dir, f"monthly_review_{stamp}.md"), "w", encoding="utf-8")


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    report.write(line + "\n"); report.flush()


# ── Tier 0: free fact-sheet gatherers (yfinance) ──────────────────────────────
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-9)  # ponytail: avoid div0; RSI→100 when no losses
    return float((100 - 100 / (1 + rs)).iloc[-1])


def momentum_line(sym):
    start = (as_of - timedelta(days=430)).strftime("%Y-%m-%d")
    end = (as_of + timedelta(days=1)).strftime("%Y-%m-%d")
    px = yf.Ticker(sym).history(start=start, end=end)["Close"].dropna()
    if len(px) < 60:
        return f"insufficient price history ({len(px)} bars)"
    last = float(px.iloc[-1])
    sma50 = float(px.rolling(50).mean().iloc[-1]) if len(px) >= 50 else last
    sma200 = float(px.rolling(200).mean().iloc[-1]) if len(px) >= 200 else last
    r3m = last / float(px.iloc[-63]) - 1 if len(px) > 63 else 0.0
    r6m = last / float(px.iloc[-126]) - 1 if len(px) > 126 else 0.0
    return (f"price={last:.2f}, 3mo={r3m*100:+.1f}%, 6mo={r6m*100:+.1f}%, RSI={rsi(px):.0f}, "
            f"{'above' if last > sma50 else 'below'} 50SMA, {'above' if last > sma200 else 'below'} 200SMA")


def fundamentals_line(sym):
    try:
        info = yf.Ticker(sym).info
    except Exception:
        return "fundamentals unavailable"
    pct = lambda v: f"{v*100:+.0f}%" if isinstance(v, (int, float)) else "n/a"
    num = lambda v: f"{v:.1f}" if isinstance(v, (int, float)) else "n/a"
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    tgt = info.get("targetMeanPrice")
    upside = f"{(tgt/price-1)*100:+.0f}%" if isinstance(tgt, (int, float)) and isinstance(price, (int, float)) else "n/a"
    lo, hi = info.get("fiftyTwoWeekLow"), info.get("fiftyTwoWeekHigh")
    pos = f"{(price-lo)/(hi-lo)*100:.0f}% of 52wk range" if all(isinstance(x, (int, float)) for x in (price, lo, hi)) and hi != lo else "n/a"
    return (f"type={info.get('quoteType','?')}, PE(fwd)={num(info.get('forwardPE'))}, PEG={num(info.get('pegRatio'))}, "
            f"revGrowth={pct(info.get('revenueGrowth'))}, earnGrowth={pct(info.get('earningsGrowth'))}, "
            f"margin={pct(info.get('profitMargins'))}, analystView={info.get('recommendationKey') or 'n/a'}"
            f"({info.get('numberOfAnalystOpinions') or 0} analysts, target {upside}), {pos}")


def news_insider_line(sym):
    headlines, insider = [], "no recent insider filings"
    try:
        for a in (yf.Ticker(sym).news or [])[:4]:
            title = a.get("title") or a.get("content", {}).get("title")
            if title:
                headlines.append(title)
    except Exception:
        pass
    try:
        df = yf.Ticker(sym).insider_transactions
        if df is not None and len(df):
            recent = df[pd.to_datetime(df["Start Date"], errors="coerce") >= as_of - timedelta(days=120)]
            buys = recent[recent["Text"].str.contains("Purchase", case=False, na=False)]["Value"].sum()
            sells = recent[recent["Text"].str.contains("Sale", case=False, na=False)]["Value"].sum()
            if buys or sells:
                insider = f"insider last 120d: ${buys:,.0f} bought / ${sells:,.0f} sold"
    except Exception:
        pass
    return f"headlines: {' | '.join(headlines) if headlines else 'none'}\ninsider: {insider}"


def fact_sheet(sym):
    return (f"MOMENTUM: {momentum_line(sym)}\n"
            f"FUNDAMENTALS: {fundamentals_line(sym)}\n"
            f"{news_insider_line(sym)}")


# ── LLM ───────────────────────────────────────────────────────────────────────
provider = os.environ.get("TRADINGAGENTS_LLM_PROVIDER", DEFAULT_CONFIG["llm_provider"])
llm = create_llm_client(provider=provider, model=args.model).get_llm()


def ask(prompt):
    resp = llm.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)


# ── Run ────────────────────────────────────────────────────────────────────────
say(f"# Monthly Portfolio Review — {args.date}")
say(f"**Budget:** ${args.budget:,.2f}  |  **Model:** {args.model}  |  **Holdings:** {', '.join(tickers)}\n")

# Tier 1 — deep-ish analysis on every holding
say("=" * 72)
say("  TIER 1 — deep-ish analysis on every holding")
say("=" * 72)
verdicts = []
for tk in tickers:
    say(f"\n{'─'*72}\n## {tk}\n{'─'*72}")
    sheet = fact_sheet(tk)
    say(f"```\n{sheet}\n```\n")
    prompt = f"""You are a rigorous equity analyst. Today is {args.date}. Analyze this holding from the free data below.

TICKER: {tk}
{sheet}

Write a focused analysis (~150 words): what the momentum, fundamentals, news and insider activity together imply.
For an ETF, fundamentals are mostly N/A — judge it on trend, momentum and its sector/theme.
End with EXACTLY these two lines:
RATING: <Strong Buy | Buy | Hold | Reduce | Sell>
CONVICTION: <1-5, where 5 = high confidence, and note if this is a HIGH-STAKES / low-conviction call worth a deep dive>"""
    analysis = ask(prompt)
    say(analysis)
    verdicts.append(f"### {tk}\n{sheet}\n\n{analysis}")

# Tier 2 — cross-portfolio conclusion
say(f"\n{'='*72}")
say("  TIER 2 — cross-portfolio conclusion")
say("=" * 72 + "\n")
synthesis_prompt = f"""You are a portfolio manager. Today is {args.date}. Below are per-holding analyses for a client who
will deploy ${args.budget:,.2f} of new cash this month across some or all of these holdings.

{chr(10).join(verdicts)}

---

Do THREE things:

1. ALLOCATION — recommend how to split ${args.budget:,.2f} across the holdings (can be $0 for any; total must equal
   ${args.budget:,.2f}). Give a one-sentence rationale per non-zero position and a summary table:
   | Ticker | Allocation | Rationale |

2. PROMOTE — name the {args.top} holdings that would benefit MOST from an expensive full multi-agent deep dive this
   month (highest stakes and/or lowest conviction — where deeper analysis is most likely to change the decision).

3. Finish with a single final line, exactly:
   PROMOTE: <TICKER1>, <TICKER2>"""
conclusion = ask(synthesis_prompt)
say(conclusion)

# Extract the promoted names for the optional Tier-3 command
promoted = []
for line in conclusion.splitlines():
    if line.strip().upper().startswith("PROMOTE:"):
        promoted = [t.strip().upper() for t in line.split(":", 1)[1].replace(",", " ").split() if t.strip().upper() in tickers]
if not promoted:
    promoted = tickers[: args.top]

say(f"\n{'='*72}")
say(f"  TIER 3 (optional) — full pipeline on the promoted names:")
say(f"  .venv/bin/python3 -u analyze_portfolio.py --tickers {' '.join(promoted[:args.top])}")
say("=" * 72)
say(f"\nReport saved → {report.name}")
report.close()
