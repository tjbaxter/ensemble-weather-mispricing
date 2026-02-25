"""analyze_alpha.py

Phase 2: Financial Alpha Analysis.

Answers the ONLY question that matters for trading:
"When our model disagrees with the Polymarket crowd, who is right more often?"

A model with 35% bucket accuracy but 65% disagreement alpha is MORE VALUABLE
than a 50%-accurate model with 50% disagreement alpha. This is the trading signal.

Classifies each market as:
  AGREE_BOTH_RIGHT    — model == crowd == actual    (no edge, boring)
  AGREE_BOTH_WRONG    — model == crowd != actual    (no edge, everyone wrong)
  DISAGREE_MODEL_RIGHT — model != crowd, model right (THIS IS ALPHA)
  DISAGREE_CROWD_RIGHT — model != crowd, crowd right (model wrong)

Simulates two strategies:
  B: Value betting — buy the model's predicted bucket when it disagrees with crowd
  C: Tail hunting  — buy cheap buckets (≤15¢) the model likes

Usage:
    python -m backtest.analyze_alpha
    python -m backtest.analyze_alpha --city seoul
    python -m backtest.analyze_alpha --skip-prices   # use model-median fallback
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "backtest" / "data"
PRED_JSON   = DATA_DIR / "model_predictions.json"
PRICES_JSON = DATA_DIR / "market_prices.json"
OUT_JSON    = DATA_DIR / "alpha_results.json"

MODELS = [
    "ncep_aigfs025",
    "gfs_graphcast025",
    "kma_gdps",
    "ecmwf_ifs025",
    "gem_global",
    "gfs_seamless",
    "icon_seamless",
]

ENSEMBLES = {
    "ensemble_avg":    MODELS,
    "ensemble_elite3": ["ncep_aigfs025", "gfs_graphcast025", "kma_gdps"],
    "ensemble_ai2":    ["ncep_aigfs025", "gfs_graphcast025"],
    "ensemble_trad3":  ["ecmwf_ifs025", "gem_global", "icon_seamless"],
}

TRADE_SIZE   = 5.0    # $ per trade
TAIL_MAX_PRICE = 0.15  # Strategy C: only buy buckets priced ≤15¢
OVERROUND_CAP  = 1.15  # markets above this are untradeable for YES
MIN_DISAGREE_FOR_SIGNAL = 10  # need at least this many disagreements to claim alpha


# ── Bucket mapping ────────────────────────────────────────────────────────────

def _parse_canon(label: str) -> tuple[float | None, float | None, str]:
    """Return (lo, hi, kind) for a canonical bucket label.

    kind: 'below', 'above', 'single', 'range'
    lo/hi: inclusive bounds (inf for unbounded)
    """
    import math
    label = str(label).strip()
    if label.endswith("+"):
        lo = float(label[:-1])
        return lo, math.inf, "above"
    if label.endswith("-"):
        hi = float(label[:-1])
        return -math.inf, hi, "below"
    if "-" in label and not label.lstrip("-").startswith("-"):
        # range like "82-83"
        parts = label.split("-", 1)
        try:
            lo, hi = float(parts[0]), float(parts[1])
            return lo, hi, "range"
        except ValueError:
            pass
    try:
        v = float(label)
        return v - 0.5, v + 0.5, "single"
    except ValueError:
        return None, None, "unknown"


def _temp_in_bucket(temp: float, canon: str) -> bool:
    """Does temp fall in the bucket with this canonical label?"""
    lo, hi, kind = _parse_canon(canon)
    if lo is None:
        return False
    if kind == "above":
        return temp >= lo
    if kind == "below":
        return temp <= hi
    if kind == "single":
        return round(temp) == round(float(canon))
    if kind == "range":
        return lo <= temp < hi
    return False


def _temp_to_bucket(temp: float, canons: list[str]) -> str | None:
    """Map temp to the matching canonical bucket, given the full bucket list."""
    for c in canons:
        if _temp_in_bucket(temp, c):
            return c
    # Fallback: nearest
    try:
        return str(int(round(temp)))
    except Exception:
        return None


def _buckets_equal(a: str, b: str) -> bool:
    """Are two canonical bucket labels the same outcome?"""
    return str(a).strip() == str(b).strip()


# ── Model ensemble helpers ────────────────────────────────────────────────────

def _model_temp(row: dict, model: str) -> float | None:
    return row.get(f"pred_{model}")


def _ensemble_temp(row: dict, models: list[str]) -> float | None:
    vals = [_model_temp(row, m) for m in models if _model_temp(row, m) is not None]
    return sum(vals) / len(vals) if vals else None


def _median_temp(row: dict) -> float | None:
    """Median of all available model predictions — used as crowd proxy."""
    vals = sorted(
        v for m in MODELS
        for v in [_model_temp(row, m)]
        if v is not None
    )
    if not vals:
        return None
    mid = len(vals) // 2
    return (vals[mid - 1] + vals[mid]) / 2 if len(vals) % 2 == 0 else vals[mid]


# ── Per-engine alpha computation ──────────────────────────────────────────────

OUTCOME_AGREE_RIGHT  = "agree_right"
OUTCOME_AGREE_WRONG  = "agree_wrong"
OUTCOME_MODEL_RIGHT  = "disagree_model"  # alpha signal
OUTCOME_CROWD_RIGHT  = "disagree_crowd"

def _classify(
    model_canon: str | None,
    crowd_canon: str | None,
    actual_canon: str,
) -> str:
    if model_canon is None or crowd_canon is None:
        return "skip"
    model_right = _buckets_equal(model_canon, actual_canon)
    crowd_right = _buckets_equal(crowd_canon, actual_canon)
    agree       = _buckets_equal(model_canon, crowd_canon)

    if agree and model_right:
        return OUTCOME_AGREE_RIGHT
    if agree and not model_right:
        return OUTCOME_AGREE_WRONG
    if not agree and model_right:
        return OUTCOME_MODEL_RIGHT
    return OUTCOME_CROWD_RIGHT


def _simulate_pnl_B(
    outcome: str,
    buy_price: float | None,
) -> float:
    """Strategy B: buy the model's bucket on every disagreement.
    Returns PnL for this trade ($TRADE_SIZE position).
    """
    if outcome not in (OUTCOME_MODEL_RIGHT, OUTCOME_CROWD_RIGHT):
        return 0.0
    if buy_price is None:
        buy_price = 0.25  # default when no price data
    buy_price = max(0.01, min(0.99, buy_price))
    shares = TRADE_SIZE / buy_price
    if outcome == OUTCOME_MODEL_RIGHT:
        return shares * (1.0 - buy_price)
    return -TRADE_SIZE


def _simulate_pnl_C(
    outcome: str,
    buy_price: float | None,
    model_canon: str | None,
    crowd_canon: str | None,
) -> float:
    """Strategy C: tail hunting — only trade when crowd prices model's bucket ≤15¢.
    Returns PnL for this trade or 0 if we don't take it.
    """
    if model_canon is None or crowd_canon is None:
        return 0.0
    if _buckets_equal(model_canon, crowd_canon):
        return 0.0  # no disagreement — skip
    if buy_price is None or buy_price > TAIL_MAX_PRICE:
        return 0.0  # not cheap enough
    shares = TRADE_SIZE / buy_price
    if outcome == OUTCOME_MODEL_RIGHT:
        return shares * (1.0 - buy_price)
    return -TRADE_SIZE


def _compute_engine_stats(
    rows: list[dict],
    prices: dict[str, dict],
    engine_name: str,
    engine_models: list[str] | None,  # None for single model
    single_model: str | None = None,
    skip_prices: bool = False,
) -> dict:
    """Compute financial alpha stats for one model or ensemble."""
    counts = defaultdict(int)
    pnl_B = 0.0
    pnl_C = 0.0
    pnl_B_trades = 0
    pnl_C_trades = 0
    overround_blocked = 0
    buy_prices_seen: list[float] = []   # for break-even calculation
    highlights: list[dict] = []  # DISAGREE_MODEL_RIGHT cases

    for row in rows:
        eid          = row["event_id"]
        actual_canon = row.get("bucket_label")
        if actual_canon is None:
            continue

        # Model/ensemble temperature
        if single_model:
            model_t = _model_temp(row, single_model)
        else:
            model_t = _ensemble_temp(row, engine_models or MODELS)
        if model_t is None:
            continue

        # All bucket labels for this market (from prices data)
        price_info = prices.get(eid, {}) if not skip_prices else {}
        bucket_prices = price_info.get("bucket_prices", {})
        overround = sum(bucket_prices.values()) if bucket_prices else None
        canons = list(bucket_prices.keys()) if bucket_prices else [actual_canon]

        # Overround filter
        if overround and overround > OVERROUND_CAP:
            overround_blocked += 1

        # Model's bucket
        model_canon = _temp_to_bucket(model_t, canons or [actual_canon])

        # Crowd's bucket — ONLY use real CLOB market data, never proxy.
        # Proxy (model-median) contaminates the alpha signal by forcing the bot
        # to fight a supercomputer consensus instead of real market inefficiencies.
        if skip_prices:
            # Explicit proxy mode requested by caller
            med = _median_temp(row)
            if med is None:
                continue
            crowd_canon = _temp_to_bucket(med, canons or [actual_canon])
            crowd_source = "model_median"
        elif price_info.get("crowd_bucket") and bucket_prices:
            crowd_canon = price_info["crowd_bucket"]
            crowd_source = "market"
        else:
            # No real price data — skip this market entirely.
            # Including it with a proxy crowd would produce meaningless results.
            continue

        # Classify
        outcome = _classify(model_canon, crowd_canon, actual_canon)
        if outcome == "skip":
            continue
        counts[outcome] += 1
        counts["total"] += 1
        if crowd_source == "market":
            counts["market_priced"] += 1

        # Buy price for model's bucket
        buy_price = bucket_prices.get(model_canon) if bucket_prices else None

        # Track real buy prices for break-even calculation
        eff_buy = buy_price if buy_price else 0.25
        if outcome in (OUTCOME_MODEL_RIGHT, OUTCOME_CROWD_RIGHT):
            buy_prices_seen.append(eff_buy)

        # PnL simulations
        pnl_this_B = _simulate_pnl_B(outcome, buy_price)
        if outcome in (OUTCOME_MODEL_RIGHT, OUTCOME_CROWD_RIGHT):
            pnl_B += pnl_this_B
            pnl_B_trades += 1

        pnl_this_C = _simulate_pnl_C(outcome, buy_price, model_canon, crowd_canon)
        if pnl_this_C != 0.0:
            pnl_C += pnl_this_C
            pnl_C_trades += 1

        # Highlight reel
        if outcome == OUTCOME_MODEL_RIGHT:
            highlights.append({
                "date":         row["target_date"],
                "city":         row["city"],
                "model_pred":   round(model_t, 1),
                "model_bucket": model_canon,
                "crowd_bucket": crowd_canon,
                "actual":       actual_canon,
                "buy_price":    round(buy_price, 3) if buy_price else None,
                "pnl_B":        round(pnl_this_B, 2),
            })

    total    = counts["total"]
    disagree = counts[OUTCOME_MODEL_RIGHT] + counts[OUTCOME_CROWD_RIGHT]
    alpha    = counts[OUTCOME_MODEL_RIGHT] / disagree if disagree else None
    avg_buy  = sum(buy_prices_seen) / len(buy_prices_seen) if buy_prices_seen else None

    return {
        "engine":             engine_name,
        "total":              total,
        "agree_right":        counts[OUTCOME_AGREE_RIGHT],
        "agree_wrong":        counts[OUTCOME_AGREE_WRONG],
        "disagree_model":     counts[OUTCOME_MODEL_RIGHT],
        "disagree_crowd":     counts[OUTCOME_CROWD_RIGHT],
        "n_disagree":         disagree,
        "disagree_alpha":     alpha,
        "avg_buy_price":      avg_buy,
        "break_even_rate":    avg_buy,   # break-even win rate = avg buy price
        "pnl_B":              round(pnl_B, 2),
        "pnl_B_trades":       pnl_B_trades,
        "pnl_C":              round(pnl_C, 2),
        "pnl_C_trades":       pnl_C_trades,
        "roi_B":              round(pnl_B / (pnl_B_trades * TRADE_SIZE), 3) if pnl_B_trades else None,
        "overround_blocked":  overround_blocked,
        "market_priced":      counts["market_priced"],
        "highlights":         highlights[:20],
        "crowd_source":       "market+fallback",
    }


# ── Printing ──────────────────────────────────────────────────────────────────

def _alpha_flag(alpha: float | None, n: int, pnl_b: float = 0.0,
                n_trades: int = 0, avg_buy_price: float | None = None) -> str:
    """Evaluate edge accounting for actual buy price (break-even ≠ 50%).

    At 30¢ buy price the break-even win rate is only 30%, not 50%.
    Use PnL-B as the primary truth and alpha as secondary context.
    """
    if alpha is None or n < MIN_DISAGREE_FOR_SIGNAL:
        return "⚠ n<10"

    # Break-even win rate = avg buy price (at buy_price p, EV=0 when alpha=p)
    be = avg_buy_price if avg_buy_price and 0.05 < avg_buy_price < 0.95 else 0.50

    # Primary: simulated PnL is the ground truth
    roi = pnl_b / (n_trades * TRADE_SIZE) if n_trades > 0 else 0.0
    profitable = pnl_b > 0

    if profitable and alpha >= be + 0.10:
        return "✅✅ STRONG"
    if profitable and alpha >= be + 0.05:
        return "✅ EDGE"
    if profitable and alpha >= be:
        return "🟡 MARGINAL+"
    if profitable:
        # PnL positive even though alpha < break-even — favourable buy prices
        roi_pct = roi * 100
        return f"💰 PnL+{roi_pct:.0f}% ROI"
    if alpha >= 0.50:
        return "🟡 MARGINAL (high cost)"
    return "❌ NO EDGE"


def _print_alpha_table(stats_list: list[dict], label: str = "") -> None:
    if label:
        print(f"\n  {label}")
    w = 22
    hdr = f"  {'Engine':<{w}} {'Agree%':>7} {'#Dis':>5} {'Win%':>6} {'BE%':>5} {'ROI':>6} {'PnL-B':>7} {'PnL-C':>7}  Assessment"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for s in stats_list:
        total = s["total"]
        agree = s["agree_right"] + s["agree_wrong"]
        agree_pct = f"{agree/total*100:.0f}%" if total else "—"
        nd   = s["n_disagree"]
        alp  = s["disagree_alpha"]
        alp_str = f"{alp*100:.0f}%" if alp is not None else "—"
        be   = s.get("avg_buy_price")
        be_str = f"{be*100:.0f}%" if be else "~30%"
        roi  = s.get("roi_B")
        roi_str = f"{roi*100:+.0f}%" if roi is not None else "—"
        pb   = f"${s['pnl_B']:+.0f}"
        pc   = f"${s['pnl_C']:+.0f}"
        flag = _alpha_flag(alp, nd, s["pnl_B"], s["pnl_B_trades"], be)
        print(f"  {s['engine']:<{w}} {agree_pct:>7} {nd:>5} {alp_str:>6} {be_str:>5} {roi_str:>6} {pb:>7} {pc:>7}  {flag}")


def _print_highlights(highlights: list[dict], engine: str) -> None:
    if not highlights:
        return
    print(f"\n  HIGHLIGHT REEL — {engine} (DISAGREE_MODEL_RIGHT events):")
    print(f"  {'Date':<12} {'City':<12} {'Pred°':>6} {'ModelB':>7} {'CrowdB':>7} {'Actual':>7} {'BuyPx':>6} {'PnL':>6}")
    print("  " + "─" * 68)
    for h in highlights:
        buy = f"{h['buy_price']:.0%}" if h["buy_price"] else "?"
        pnl = f"${h['pnl_B']:+.2f}"
        print(f"  {h['date']:<12} {h['city']:<12} {h['model_pred']:>6.1f} {h['model_bucket']:>7} {h['crowd_bucket']:>7} {h['actual']:>7} {buy:>6} {pnl:>6}")


def _auto_decision(all_stats: list[dict]) -> str:
    eligible = [s for s in all_stats if s["n_disagree"] >= MIN_DISAGREE_FOR_SIGNAL]
    if not eligible:
        return "⚠️  INSUFFICIENT DATA — need ≥10 disagreements per engine. Collect more data."

    # Rank by ROI (accounts for buy price), then by PnL absolute
    profitable = [s for s in eligible if s["pnl_B"] > 0]
    profitable.sort(key=lambda s: s.get("roi_B") or 0, reverse=True)

    if not profitable:
        best_alpha = max(eligible, key=lambda s: s["disagree_alpha"] or 0)
        alpha = best_alpha["disagree_alpha"] or 0
        return (f"❌ NO FINANCIAL ALPHA — best engine {best_alpha['engine']} loses money on disagreements "
                f"(alpha={alpha*100:.1f}%, PnL-B={best_alpha['pnl_B']:+.0f}).\n"
                f"   Pivot to latency arbitrage or collect more data (NYC/London fetches pending).")

    best = profitable[0]
    alpha = best["disagree_alpha"] or 0
    n = best["n_disagree"]
    roi = best.get("roi_B") or 0
    be = best.get("avg_buy_price")
    be_str = f"{be*100:.0f}%" if be else "~30%"
    engine = best["engine"]
    pnl = best["pnl_B"]
    n_trades = best["pnl_B_trades"]

    # Note: when buying at ~30¢, break-even alpha is ~30%, not 50%
    note = (f"(Break-even at avg buy {be_str} is only {be_str} win rate, not 50% — "
            f"positive PnL confirms real edge despite alpha<50%)")

    if roi >= 0.40 and n >= 15:
        return (f"✅ SIGNAL CONFIRMED — {engine}: {n} disagreements, {alpha*100:.0f}% win rate, "
                f"{roi*100:+.0f}% ROI, simulated PnL ${pnl:+.0f} on {n_trades} trades.\n"
                f"   {note}\n"
                f"   Deploy with $300 bankroll, $3-5 per trade. Prioritise NYC/London for liquidity.")
    if roi >= 0.20 and n >= 10:
        return (f"🟡 SIGNAL PROMISING — {engine}: {roi*100:+.0f}% ROI on {n_trades} trades "
                f"(alpha={alpha*100:.0f}%, n={n} disagreements).\n"
                f"   {note}\n"
                f"   Continue collecting data. Paper trade NYC/London. Fund when n≥30.")
    if pnl > 0:
        return (f"💰 EDGE PRESENT but small sample — {engine} shows ${pnl:+.0f} PnL on {n_trades} trades.\n"
                f"   {note}\n"
                f"   Need NYC + London data to confirm. Do NOT fund yet.")
    return ("❌ NO FINANCIAL ALPHA — the crowd beats your model on disagreements.\n"
            "   Pivot to latency arbitrage or wait for NYC/London data.")


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_alpha_analysis(
    city_filter: list[str] | None = None,
    skip_prices: bool = False,
) -> dict:
    if not PRED_JSON.exists():
        print(f"ERROR: {PRED_JSON} not found. Run Phase 1 first.")
        sys.exit(1)

    rows = json.loads(PRED_JSON.read_text())
    rows = [r for r in rows if any(r.get(f"pred_{m}") is not None for m in MODELS)]

    prices: dict[str, dict] = {}
    if not skip_prices and PRICES_JSON.exists():
        try:
            prices = json.loads(PRICES_JSON.read_text())
        except (ValueError, json.JSONDecodeError):
            pass

    if city_filter:
        rows = [r for r in rows if r["city_slug"] in city_filter or r["city"].lower() in city_filter]

    print("\n" + "╔" + "═" * 63 + "╗")
    print("║" + "          PHASE 2: FINANCIAL ALPHA REPORT".center(63) + "║")
    print("╚" + "═" * 63 + "╝")

    cities = sorted(set(r["city_slug"] for r in rows))
    priced  = sum(1 for r in rows if prices.get(r["event_id"], {}).get("crowd_bucket"))
    tradeable = sum(
        1 for r in rows
        if sum(prices.get(r["event_id"], {}).get("bucket_prices", {}).values()) <= OVERROUND_CAP
    ) if prices else "?"
    print(f"\n  Total markets with model predictions : {len(rows)}")
    print(f"  Markets with CLOB price data         : {priced}")
    print(f"  Tradeable (overround ≤115%)           : {tradeable}")
    crowd_note = "CLOB price history (proxy markets excluded)" if priced else "model-median proxy (--skip-prices mode)"
    print(f"  Crowd source                          : {crowd_note}")

    # Build engines
    def run_all_engines(subset: list[dict], label: str) -> list[dict]:
        results = []
        for m in MODELS:
            s = _compute_engine_stats(subset, prices, m, None, single_model=m, skip_prices=skip_prices)
            results.append(s)
        for ename, emodels in ENSEMBLES.items():
            s = _compute_engine_stats(subset, prices, ename, emodels, skip_prices=skip_prices)
            results.append(s)
        results.sort(key=lambda s: -(s["disagree_alpha"] or 0))
        _print_alpha_table(results, label)
        return results

    # === OVERALL ===
    print("\n" + "=" * 65)
    print("  OVERALL — ALL CITIES COMBINED")
    print("=" * 65)
    all_stats = run_all_engines(rows, "")

    # Best highlights
    best_overall = all_stats[0] if all_stats else None
    if best_overall:
        _print_highlights(best_overall["highlights"], best_overall["engine"])

    # === PER CITY ===
    city_best: dict[str, dict] = {}
    for city in cities:
        city_rows = [r for r in rows if r["city_slug"] == city]
        if len(city_rows) < 5:
            continue
        print(f"\n{'─'*65}")
        print(f"  CITY: {city.upper()}  (n={len(city_rows)})")
        city_stats = run_all_engines(city_rows, "")
        city_best[city] = city_stats[0] if city_stats else {}

    # === RECOMMENDED ENGINES ===
    print("\n" + "=" * 65)
    print("  RECOMMENDED ENGINE PER CITY")
    print("=" * 65)
    for city, best in city_best.items():
        if not best:
            continue
        alpha  = best.get("disagree_alpha")
        n      = best.get("n_disagree", 0)
        roi    = best.get("roi_B")
        be     = best.get("avg_buy_price")
        alpha_str = f"{alpha*100:.1f}%" if alpha is not None else "—"
        roi_str   = f"{roi*100:+.0f}% ROI" if roi is not None else ""
        flag = _alpha_flag(alpha, n, best.get("pnl_B", 0), best.get("pnl_B_trades", 0), be)
        print(f"  {city:<12} {best['engine']:<22} alpha={alpha_str}  n={n}  {roi_str}  {flag}")

    # === DECISION ===
    print("\n" + "=" * 65)
    print("  DECISION")
    print("=" * 65)
    decision = _auto_decision(all_stats)
    for line in decision.split("\n"):
        print(f"  {line}")
    print("=" * 65)

    # Save
    output = {
        "overall": all_stats,
        "by_city": {c: s for c, s in city_best.items()},
        "decision": decision,
        "n_markets": len(rows),
        "n_priced":  priced,
    }
    with OUT_JSON.open("w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved → {OUT_JSON}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city",        nargs="*", default=None)
    parser.add_argument("--skip-prices", action="store_true",
                        help="Use model-median as crowd proxy (no CLOB price data needed)")
    args = parser.parse_args()

    run_alpha_analysis(
        city_filter=args.city,
        skip_prices=args.skip_prices,
    )


if __name__ == "__main__":
    main()
