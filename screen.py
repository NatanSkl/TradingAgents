"""
Portfolio Momentum Screen  (Stage 1 of the two-stage funnel)
============================================================
Cheap, fast ranking of a watchlist so you only spend the full multi-agent
pipeline (analyze_portfolio.py, ~8 min/ticker) on the top 1-2 names.

Stage 1a — quant score (no LLM, ~seconds): trend + momentum + RSI from yfinance.
Stage 1b — optional single LLM call (--llm) that reads the table and picks the
           best 1-2 to promote, using the QUICK model (Haiku by default).

Usage:
    .venv/bin/python3 screen.py
    .venv/bin/python3 screen.py --llm
    .venv/bin/python3 screen.py --tickers PLTR TSLA SMH --top 3
    .venv/bin/python3 screen.py --date 2026-07-13

Then run the full pipeline only on the winners it prints, e.g.:
    .venv/bin/python3 -u analyze_portfolio.py --tickers PLTR
"""

import argparse
import os
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

PORTFOLIO = ["BITB", "ETHA", "GOP", "PLTR", "QQQM", "QTUM", "SMH", "TSLA", "VOO"]

parser = argparse.ArgumentParser()
parser.add_argument("--tickers", nargs="+", default=PORTFOLIO)
parser.add_argument("--date", default=date.today().isoformat())
parser.add_argument("--top", type=int, default=2, help="how many to promote")
parser.add_argument("--llm", action="store_true", help="add an LLM pick on top of the quant rank")
parser.add_argument("--quick", action="store_true", help="use the QUICK model (Haiku) for the pick; default is DEEP (Sonnet)")
args = parser.parse_args()

tickers = [t.upper() for t in args.tickers]
as_of = datetime.strptime(args.date, "%Y-%m-%d")


def fundamentals_snapshot(sym: str) -> str:
    """One free yfinance .info call → compact fundamentals line. ETFs return mostly
    N/A, which is itself the signal: judge them on momentum, not fundamentals."""
    try:
        info = yf.Ticker(sym).info
    except Exception:
        return "fundamentals unavailable"

    def pct(v):
        return f"{v * 100:+.0f}%" if isinstance(v, (int, float)) else "n/a"

    def num(v, f="{:.1f}"):
        return f.format(v) if isinstance(v, (int, float)) else "n/a"

    kind = info.get("quoteType", "?")
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    tgt = info.get("targetMeanPrice")
    upside = f"{(tgt / price - 1) * 100:+.0f}%" if isinstance(tgt, (int, float)) and isinstance(price, (int, float)) else "n/a"
    lo, hi = info.get("fiftyTwoWeekLow"), info.get("fiftyTwoWeekHigh")
    pos52 = f"{(price - lo) / (hi - lo) * 100:.0f}% of 52wk range" if all(isinstance(x, (int, float)) for x in (price, lo, hi)) and hi != lo else "n/a"

    return (
        f"type={kind}, PE(fwd)={num(info.get('forwardPE'))}, PEG={num(info.get('pegRatio'))}, "
        f"revGrowth={pct(info.get('revenueGrowth'))}, earnGrowth={pct(info.get('earningsGrowth'))}, "
        f"margin={pct(info.get('profitMargins'))}, analystView={info.get('recommendationKey') or 'n/a'}"
        f"({info.get('numberOfAnalystOpinions') or 0} analysts, target {upside}), {pos52}"
    )


def news_insider_snapshot(sym: str) -> str:
    """Free: latest few headlines + recent insider net. ETFs typically have neither."""
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
            cutoff = as_of - timedelta(days=120)
            recent = df[pd.to_datetime(df["Start Date"], errors="coerce") >= cutoff]
            buys = recent[recent["Text"].str.contains("Purchase", case=False, na=False)]["Value"].sum()
            sells = recent[recent["Text"].str.contains("Sale", case=False, na=False)]["Value"].sum()
            if buys or sells:
                insider = f"insider last 120d: ${buys:,.0f} bought / ${sells:,.0f} sold"
    except Exception:
        pass
    news = " | ".join(headlines) if headlines else "no recent headlines"
    return f"headlines: {news}\n        {insider}"


def rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-9)  # ponytail: avoid div0; RSI→100 when no losses
    return float((100 - 100 / (1 + rs)).iloc[-1])


def score_ticker(sym: str) -> dict:
    # ~14 months of daily bars so 200-SMA and 6-mo momentum both have room
    start = (as_of - timedelta(days=430)).strftime("%Y-%m-%d")
    end = (as_of + timedelta(days=1)).strftime("%Y-%m-%d")
    px = yf.Ticker(sym).history(start=start, end=end)["Close"].dropna()
    if len(px) < 200:
        return {"ticker": sym, "score": float("nan"), "note": f"only {len(px)} bars"}

    last = float(px.iloc[-1])
    sma50 = float(px.rolling(50).mean().iloc[-1])
    sma200 = float(px.rolling(200).mean().iloc[-1])
    r3m = last / float(px.iloc[-63]) - 1 if len(px) > 63 else 0.0
    r6m = last / float(px.iloc[-126]) - 1 if len(px) > 126 else 0.0
    r = rsi(px)

    # Composite: momentum-weighted, trend as confirmation, RSI as an overbought brake.
    score = (
        0.45 * r3m * 100          # 3-mo momentum, biggest weight
        + 0.25 * r6m * 100        # 6-mo momentum
        + 8.0 * (last > sma50)    # above medium-term trend
        + 8.0 * (last > sma200)   # above long-term trend
        - 6.0 * (r > 70)          # penalize overbought
        + 4.0 * (r < 30)          # small reward if oversold (bounce setup)
    )
    return {
        "ticker": sym, "score": score, "last": last,
        "r3m": r3m * 100, "r6m": r6m * 100, "rsi": r,
        "above50": last > sma50, "above200": last > sma200, "note": "",
    }


rows = [score_ticker(t) for t in tickers]
ranked = sorted(rows, key=lambda x: (x["score"] != x["score"], -(x["score"] if x["score"] == x["score"] else 0)))

print(f"\n  MOMENTUM SCREEN — {args.date}   (higher score = stronger trend/momentum)\n")
print(f"  {'#':>2}  {'TICKER':<6} {'SCORE':>7} {'3M%':>7} {'6M%':>7} {'RSI':>5}  TREND")
print("  " + "-" * 56)
for i, x in enumerate(ranked, 1):
    if x["score"] != x["score"]:  # NaN
        print(f"  {i:>2}  {x['ticker']:<6} {'  n/a':>7}                        {x['note']}")
        continue
    trend = ("↑50" if x["above50"] else "·50") + " " + ("↑200" if x["above200"] else "·200")
    print(f"  {i:>2}  {x['ticker']:<6} {x['score']:>7.1f} {x['r3m']:>6.1f}% {x['r6m']:>6.1f}% {x['rsi']:>5.0f}  {trend}")

valid = [x for x in ranked if x["score"] == x["score"]]
promoted = [x["ticker"] for x in valid[: args.top]]

print("\n  " + "-" * 56)
print(f"  Quant pick (top {args.top}): {', '.join(promoted)}")

# ── Stage 1b: optional single cheap LLM call ──────────────────────────────────
if args.llm and valid:
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.llm_clients import create_llm_client

    print("\n  Pulling fundamentals + news + insider snapshot for the LLM pick...")
    table = "\n".join(
        f"{x['ticker']}: score={x['score']:.1f}, 3mo={x['r3m']:+.1f}%, 6mo={x['r6m']:+.1f}%, "
        f"RSI={x['rsi']:.0f}, above50SMA={x['above50']}, above200SMA={x['above200']}\n"
        f"        fundamentals: {fundamentals_snapshot(x['ticker'])}\n"
        f"        {news_insider_snapshot(x['ticker'])}"
        for x in valid
    )
    prompt = f"""You are a portfolio screener choosing which names deserve an expensive deep-dive this month.
As of {args.date}, here is a free quant + fundamentals scan of the watchlist:

{table}

Pick the {args.top} best candidates to promote to a full multi-agent analysis.
Weigh momentum (durable uptrends beat overbought spikes, RSI>75 is a caution) together with fundamentals
(growth, valuation, analyst view / target upside). For ETFs most fundamentals are N/A — judge those on trend
and momentum. Prefer names where the deep-dive is most likely to change the decision, not just the strongest chart.
Reply with the tickers on the first line (comma-separated), then one short sentence each explaining why."""

    provider = os.environ.get("TRADINGAGENTS_LLM_PROVIDER", DEFAULT_CONFIG["llm_provider"])
    env_key = "TRADINGAGENTS_QUICK_THINK_LLM" if args.quick else "TRADINGAGENTS_DEEP_THINK_LLM"
    cfg_key = "quick_think_llm" if args.quick else "deep_think_llm"
    model = os.environ.get(env_key, DEFAULT_CONFIG[cfg_key])
    llm = create_llm_client(provider=provider, model=model).get_llm()
    resp = llm.invoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)
    print(f"\n  LLM pick ({model}):\n")
    for line in text.splitlines():
        print(f"    {line}")

    # First line is the comma-separated tickers; keep only ones we actually scanned.
    known = {x["ticker"] for x in valid}
    llm_pick = [t.strip().upper() for t in text.splitlines()[0].replace(",", " ").split() if t.strip().upper() in known]
    if llm_pick:
        promoted = llm_pick[: args.top]

print(f"\n  Next: .venv/bin/python3 -u analyze_portfolio.py --tickers {' '.join(promoted)}\n")
