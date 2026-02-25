"""simulate_pnl.py

The honest PnL simulation. Four parts:

  1. EXACT equity curve — day-by-day PnL on the ~85 real CLOB-priced markets.
     This directly answers "what would my bankroll look like right now?"

  2. Monte Carlo — 10,000 paths using the real trades (same buy prices)
     but randomised win/loss outcomes to show the RANGE of possible results.

  3. CLOB-only vs proxy breakdown with BINOMIAL p-values.
     Separates reliable signal from contaminated data.

  4. Random buy-price simulation — uses real model predictions + real outcomes
     across all 824 markets, random buy prices [20¢–45¢] per trade.
     Shows model accuracy → money translation with price uncertainty.

Usage:
    python -m backtest.simulate_pnl
    python -m backtest.simulate_pnl --city seoul
    python -m backtest.simulate_pnl --no-montecarlo   # skip slow MC if needed
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR   = ROOT / "backtest" / "data"
PRED_JSON  = DATA_DIR / "model_predictions.json"
PRICE_JSON = DATA_DIR / "market_prices.json"
OUT_JSON   = DATA_DIR / "simulation_results.json"

MODELS = [
    "ncep_aigfs025",
    "gfs_graphcast025",
    "kma_gdps",
    "ecmwf_ifs025",
    "gem_global",
    "gfs_seamless",
    "icon_seamless",
]

TRADE_SIZE    = 5.0
START_BANKROLL = 100.0
OVERROUND_CAP  = 1.15
MIN_BUY_PRICE  = 0.05
MAX_BUY_PRICE  = 0.50    # don't buy if our bucket is already ≥50¢ (crowd agrees)
ENTRY_PRICE_FLOOR = 0.20  # random-buy-price sim range low
ENTRY_PRICE_CEIL  = 0.45  # random-buy-price sim range high
MC_SIMS = 10_000


# ── Bucket helpers ─────────────────────────────────────────────────────────────

def _temp_in_bucket(temp: float, canon: str) -> bool:
    if canon.endswith("+"):
        try:
            return temp >= float(canon[:-1])
        except ValueError:
            return False
    if canon.endswith("-"):
        try:
            return temp <= float(canon[:-1])
        except ValueError:
            return False
    if re_range := _parse_range(canon):
        lo, hi = re_range
        return lo <= temp < hi
    try:
        return round(temp) == int(canon)
    except ValueError:
        return False


def _parse_range(canon: str):
    """Return (lo, hi) for range buckets like '82-83', else None."""
    import re
    m = re.match(r"^(-?\d+)-(\d+)$", canon)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def temp_to_bucket(temp: float, canons: list[str]) -> str | None:
    for c in canons:
        if _temp_in_bucket(temp, c):
            return c
    return str(int(round(temp)))


def buckets_equal(a: str | None, b: str | None) -> bool:
    return (a is not None) and (b is not None) and str(a).strip() == str(b).strip()


# ── Ensemble helpers ───────────────────────────────────────────────────────────

def ensemble_avg_temp(row: dict, models: list[str] = MODELS) -> float | None:
    vals = [row.get(f"pred_{m}") for m in models if row.get(f"pred_{m}") is not None]
    return sum(vals) / len(vals) if vals else None


# ── Statistics ─────────────────────────────────────────────────────────────────

def binomial_pvalue(wins: int, n: int, p_null: float) -> float:
    """One-sided binomial p-value: P(X >= wins | n, p_null).

    Tests whether observed win rate significantly exceeds break-even.
    """
    if n == 0:
        return 1.0
    p_null = max(0.001, min(0.999, p_null))
    p_val = 0.0
    for k in range(wins, n + 1):
        p_val += math.comb(n, k) * (p_null ** k) * ((1 - p_null) ** (n - k))
    return min(1.0, p_val)


def significance_label(p: float) -> str:
    if p < 0.05:
        return "✅ SIGNIFICANT (p<0.05)"
    if p < 0.10:
        return "🟡 PROMISING   (p<0.10)"
    if p < 0.20:
        return "⚠  WEAK        (p<0.20)"
    return "❌ NOT SIG     (p≥0.20)"


def min_trades_for_significance(observed_win_rate: float, break_even: float,
                                 target_p: float = 0.05) -> int:
    """Estimate how many trades needed so observed_win_rate yields target_p."""
    if observed_win_rate <= break_even:
        return 9999
    for n in range(5, 500):
        wins = int(observed_win_rate * n)
        p = binomial_pvalue(wins, n, break_even)
        if p < target_p:
            return n
    return 9999


# ── PnL simulation for one trade ──────────────────────────────────────────────

def sim_trade(outcome_right: bool, buy_price: float) -> float:
    """Return PnL for a $TRADE_SIZE position."""
    buy_price = max(0.01, min(0.99, buy_price))
    if outcome_right:
        shares = TRADE_SIZE / buy_price
        return shares * (1.0 - buy_price)
    return -TRADE_SIZE


# ── Data loading and trade extraction ─────────────────────────────────────────

def load_trades(city_filter: list[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Return (clob_trades, all_prediction_rows).

    clob_trades: records that have real CLOB crowd pricing, model disagrees,
                 overround OK, buy price in sensible range — ready to simulate.
    all_prediction_rows: all rows with at least one model prediction (for Part 4).
    """
    rows = json.loads(PRED_JSON.read_text())
    rows = [r for r in rows if any(r.get(f"pred_{m}") is not None for m in MODELS)]
    rows = [r for r in rows if r.get("bucket_label")]

    prices: dict[str, dict] = {}
    if PRICE_JSON.exists():
        try:
            prices = json.loads(PRICE_JSON.read_text())
        except (ValueError, json.JSONDecodeError):
            pass

    if city_filter:
        city_filter_lower = [c.lower() for c in city_filter]
        rows = [r for r in rows
                if r.get("city_slug", "").lower() in city_filter_lower
                or r.get("city", "").lower() in city_filter_lower]

    clob_trades: list[dict] = []

    for row in rows:
        eid         = str(row["event_id"])
        actual      = row["bucket_label"]
        model_t     = ensemble_avg_temp(row)
        if model_t is None:
            continue

        price_info  = prices.get(eid, {})
        bucket_ps   = price_info.get("bucket_prices", {})
        crowd_b     = price_info.get("crowd_bucket")
        method      = price_info.get("method", "")

        is_real_clob = (
            bool(bucket_ps)
            and crowd_b
            and method != "failed"
            and "proxy" not in method.lower()
        )

        if not is_real_clob:
            continue

        overround = sum(bucket_ps.values())
        if overround > OVERROUND_CAP:
            continue

        canons     = list(bucket_ps.keys())
        model_b    = temp_to_bucket(model_t, canons)
        buy_price  = bucket_ps.get(model_b)

        if buy_price is None or buy_price < MIN_BUY_PRICE or buy_price >= MAX_BUY_PRICE:
            continue

        # Only trade when model disagrees with crowd
        if buckets_equal(model_b, crowd_b):
            continue

        model_right = buckets_equal(model_b, actual)

        clob_trades.append({
            "date":        row["target_date"],
            "city":        row.get("city", ""),
            "city_slug":   row.get("city_slug", ""),
            "model_temp":  round(model_t, 1),
            "model_bucket": model_b,
            "crowd_bucket": crowd_b,
            "actual_bucket": actual,
            "buy_price":   buy_price,
            "overround":   round(overround, 3),
            "model_right": model_right,
            "pnl":         sim_trade(model_right, buy_price),
            "eid":         eid,
        })

    clob_trades.sort(key=lambda t: t["date"])
    return clob_trades, rows


# ── PART 1: Exact equity curve ─────────────────────────────────────────────────

def part1_equity_curve(trades: list[dict]) -> dict:
    print("\n" + "═" * 68)
    print("  PART 1 — EXACT HISTORICAL PnL (CLOB-priced markets only)")
    print("═" * 68)

    if not trades:
        print("  ⚠  No trades found with real CLOB data + model disagreement.")
        return {}

    bankroll = START_BANKROLL
    wins = losses = 0
    peak = START_BANKROLL
    max_dd = 0.0
    streak = cur_streak = 0

    print(f"\n  Starting bankroll: ${START_BANKROLL:.2f}\n")
    hdr = f"  {'Date':<12} {'City':<8} {'Model°':>7} {'MB':>5} {'Crowd':>6} {'Actual':>6} {'BuyPx':>6} {'Result':>7} {'PnL':>8} {'Bankroll':>9}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    daily_pnl: list[dict] = []

    for t in trades:
        result_str = f"{'WIN' if t['model_right'] else 'LOSS':>7}"
        pnl_str    = f"${t['pnl']:+.2f}"
        bankroll  += t["pnl"]

        if t["model_right"]:
            wins += 1
            cur_streak = max(0, cur_streak) + 1
        else:
            losses += 1
            cur_streak = min(0, cur_streak) - 1
        streak = max(streak, -cur_streak)

        peak   = max(peak, bankroll)
        dd     = peak - bankroll
        max_dd = max(max_dd, dd)

        print(f"  {t['date']:<12} {t['city']:<8} {t['model_temp']:>7.1f} "
              f"{t['model_bucket']:>5} {t['crowd_bucket']:>6} {t['actual_bucket']:>6} "
              f"{t['buy_price']:>5.0%} {result_str} {pnl_str:>8} ${bankroll:>8.2f}")

        daily_pnl.append({"date": t["date"], "city": t["city"], "bankroll": round(bankroll, 2),
                           "pnl": round(t["pnl"], 2), "win": t["model_right"]})

    n       = wins + losses
    roi     = (bankroll - START_BANKROLL) / START_BANKROLL * 100
    win_rate = wins / n if n else 0
    avg_buy = sum(t["buy_price"] for t in trades) / n if n else 0
    be_rate = avg_buy

    print(f"\n  {'─'*62}")
    print(f"  Trades : {n}  (Wins: {wins}  Losses: {losses})")
    print(f"  Win rate: {win_rate:.1%}  |  Break-even: {be_rate:.1%}  |  vs. {'+' if win_rate > be_rate else ''}{(win_rate-be_rate)*100:.1f}pp")
    print(f"  Total PnL   : ${bankroll - START_BANKROLL:+.2f}")
    print(f"  Final bank  : ${bankroll:.2f}  ({roi:+.1f}% ROI)")
    print(f"  Max drawdown: ${max_dd:.2f}")
    print(f"  Max loss streak: {streak} trades")

    p = binomial_pvalue(wins, n, be_rate)
    print(f"\n  Binomial test (H₀: win rate = break-even {be_rate:.0%})")
    print(f"  p-value = {p:.3f}  →  {significance_label(p)}")

    if win_rate > be_rate:
        n_needed = min_trades_for_significance(win_rate, be_rate)
        print(f"  Trades needed for significance (p<0.05): ~{n_needed}")
    print("═" * 68)

    return {
        "n": n, "wins": wins, "losses": losses,
        "win_rate": win_rate, "break_even": be_rate,
        "final_bankroll": round(bankroll, 2),
        "roi_pct": round(roi, 2),
        "max_drawdown": round(max_dd, 2),
        "longest_loss_streak": streak,
        "p_value": round(p, 4),
        "trades": daily_pnl,
    }


# ── PART 2: Monte Carlo ────────────────────────────────────────────────────────

def part2_montecarlo(trades: list[dict], run_mc: bool = True) -> dict:
    print("\n" + "═" * 68)
    print("  PART 2 — MONTE CARLO SENSITIVITY ANALYSIS")
    print("═" * 68)

    if not trades:
        print("  ⚠  No trades to simulate.")
        return {}

    n         = len(trades)
    obs_wins  = sum(1 for t in trades if t["model_right"])
    obs_rate  = obs_wins / n
    be_rate   = sum(t["buy_price"] for t in trades) / n

    if not run_mc:
        print(f"  (Skipped — pass without --no-montecarlo to run {MC_SIMS:,} simulations)")
        return {}

    print(f"\n  Running {MC_SIMS:,} simulations on {n} real trades …")

    finals: list[float] = []
    random.seed(42)

    for _ in range(MC_SIMS):
        bankroll = START_BANKROLL
        for t in trades:
            win = random.random() < obs_rate
            bankroll += sim_trade(win, t["buy_price"])
        finals.append(bankroll)

    finals.sort()
    pct = lambda p: finals[int(p * MC_SIMS / 100)]

    p5, p25, p50, p75, p95 = pct(5), pct(25), pct(50), pct(75), pct(95)
    prob_loss    = sum(1 for f in finals if f < START_BANKROLL) / MC_SIMS
    prob_gain20  = sum(1 for f in finals if f >= START_BANKROLL * 1.20) / MC_SIMS
    prob_gain50  = sum(1 for f in finals if f >= START_BANKROLL * 1.50) / MC_SIMS

    print(f"\n  Observed win rate: {obs_rate:.1%}  |  Break-even: {be_rate:.1%}")
    print(f"  Simulating: {n} trades at ${TRADE_SIZE}/trade from ${START_BANKROLL} bankroll\n")
    print(f"  {'Percentile':<20} {'Bankroll':>10}  {'ROI':>8}")
    print(f"  {'─'*42}")
    for label, val in [("5th (bad case)", p5), ("25th", p25),
                        ("Median (50th)", p50), ("75th", p75),
                        ("95th (good case)", p95)]:
        roi = (val - START_BANKROLL) / START_BANKROLL * 100
        print(f"  {label:<20} ${val:>9.2f}  {roi:>+7.1f}%")

    print(f"\n  P(final < ${START_BANKROLL:.0f}):  {prob_loss:.1%}  — probability of net loss")
    print(f"  P(+20% gain) :  {prob_gain20:.1%}")
    print(f"  P(+50% gain) :  {prob_gain50:.1%}")
    print("═" * 68)

    return {
        "n_simulations": MC_SIMS,
        "observed_win_rate": round(obs_rate, 4),
        "break_even": round(be_rate, 4),
        "p5": round(p5, 2), "p25": round(p25, 2), "p50": round(p50, 2),
        "p75": round(p75, 2), "p95": round(p95, 2),
        "prob_loss": round(prob_loss, 4),
        "prob_gain20": round(prob_gain20, 4),
        "prob_gain50": round(prob_gain50, 4),
    }


# ── PART 3: CLOB-only leaderboard + binomial tests ────────────────────────────

def part3_clob_leaderboard(all_rows: list[dict], prices: dict,
                            city_filter: list[str] | None = None) -> dict:
    print("\n" + "═" * 68)
    print("  PART 3 — SIGNAL QUALITY: CLOB-ONLY vs PROXY (with p-values)")
    print("═" * 68)

    cities = sorted(set(r.get("city_slug", "") for r in all_rows if r.get("city_slug")))
    results_out: dict = {}

    engines = {
        "ensemble_avg": MODELS,
        "ncep_aigfs025": None,
        "gfs_graphcast025": None,
        "ensemble_ai2": ["ncep_aigfs025", "gfs_graphcast025"],
        "ensemble_elite3": ["ncep_aigfs025", "gfs_graphcast025", "kma_gdps"],
        "icon_seamless": None,
        "ecmwf_ifs025": None,
        "gem_global": None,
        "kma_gdps": None,
    }

    def get_temp(row, engine_name, engine_models):
        if engine_models is None:
            return row.get(f"pred_{engine_name}")
        vals = [row.get(f"pred_{m}") for m in engine_models if row.get(f"pred_{m}") is not None]
        return sum(vals) / len(vals) if vals else None

    def run_engine_on_rows(subset, engine_name, engine_models, clob_only=True):
        wins = losses = 0
        buy_prices = []
        for row in subset:
            eid = str(row["event_id"])
            actual = row.get("bucket_label")
            if not actual:
                continue
            model_t = get_temp(row, engine_name, engine_models)
            if model_t is None:
                continue
            price_info = prices.get(eid, {})
            bucket_ps  = price_info.get("bucket_prices", {})
            crowd_b    = price_info.get("crowd_bucket")
            method     = price_info.get("method", "")
            is_clob    = bool(bucket_ps) and crowd_b and method != "failed" and "proxy" not in method.lower()
            if clob_only and not is_clob:
                continue
            if not clob_only and is_clob:
                continue
            overround = sum(bucket_ps.values()) if bucket_ps else None
            if overround and overround > OVERROUND_CAP:
                continue
            canons   = list(bucket_ps.keys()) if bucket_ps else [actual]
            model_b  = temp_to_bucket(model_t, canons)
            buy_p    = bucket_ps.get(model_b) if bucket_ps else None
            if buy_p is not None and (buy_p < MIN_BUY_PRICE or buy_p >= MAX_BUY_PRICE):
                continue
            # Need crowd to disagree
            if not crowd_b:
                # proxy mode: use model-median
                med_vals = [row.get(f"pred_{m}") for m in MODELS if row.get(f"pred_{m}") is not None]
                if not med_vals:
                    continue
                sorted_vals = sorted(med_vals)
                mid = len(sorted_vals) // 2
                med = (sorted_vals[mid-1] + sorted_vals[mid]) / 2 if len(sorted_vals) % 2 == 0 else sorted_vals[mid]
                crowd_b = temp_to_bucket(med, canons)
            if buckets_equal(model_b, crowd_b):
                continue
            eff_buy = buy_p if buy_p else 0.25
            buy_prices.append(eff_buy)
            if buckets_equal(model_b, actual):
                wins += 1
            else:
                losses += 1
        return wins, losses, buy_prices

    for label, clob_only in [("CLOB-ONLY (reliable)", True), ("PROXY-ONLY ⚠ UNRELIABLE", False)]:
        print(f"\n  ── {label} ──")
        any_found = False
        for city_slug in (city_filter or cities):
            city_rows = [r for r in all_rows if r.get("city_slug") == city_slug]
            if not city_rows:
                continue
            hdr_printed = False
            city_results = []
            for eng_name, eng_models in engines.items():
                w, l, bps = run_engine_on_rows(city_rows, eng_name, eng_models, clob_only)
                n_dis = w + l
                if n_dis == 0:
                    continue
                any_found = True
                win_rate = w / n_dis
                avg_buy  = sum(bps) / len(bps) if bps else 0.25
                be       = avg_buy
                pnl      = sim_trade(True, avg_buy) * w + (-TRADE_SIZE) * l
                roi      = pnl / (n_dis * TRADE_SIZE)
                p        = binomial_pvalue(w, n_dis, be)
                city_results.append((eng_name, w, n_dis, win_rate, be, roi, pnl, p))

            if city_results:
                if not hdr_printed:
                    print(f"\n  {city_slug.upper()}:")
                    print(f"  {'Engine':<18} {'W/N':>7} {'Win%':>6} {'BE%':>5} {'ROI':>7} {'PnL':>7} {'p-val':>7}  Status")
                    print("  " + "─" * 72)
                    hdr_printed = True
                city_results.sort(key=lambda x: x[7])  # sort by p-value
                for eng_name, w, n_dis, win_rate, be, roi, pnl, p in city_results:
                    sig = significance_label(p)
                    print(f"  {eng_name:<18} {w:>3}/{n_dis:<3} {win_rate:>5.0%} {be:>4.0%} "
                          f"{roi:>+6.0%} ${pnl:>+5.0f} {p:>7.3f}  {sig}")

        if not any_found:
            print(f"  (no {label.split()[0]} markets found)")

    print("═" * 68)
    return {}


# ── PART 4: Random buy-price simulation ───────────────────────────────────────

def part4_random_buy_sim(all_rows: list[dict], city_filter: list[str] | None = None,
                          n_sims: int = MC_SIMS) -> dict:
    print("\n" + "═" * 68)
    print("  PART 4 — RANDOM BUY-PRICE SIMULATION")
    print("  (Real predictions + real outcomes, random entry price [20¢–45¢])")
    print("═" * 68)

    # Use ensemble_avg accuracy against actual bucket labels
    random.seed(123)

    def sim_city(rows):
        correct  = sum(1 for r in rows if buckets_equal(
            temp_to_bucket(ensemble_avg_temp(r), [r.get("bucket_label", "")]),
            r.get("bucket_label")))
        n        = len(rows)
        acc_rate = correct / n if n else 0

        finals = []
        for _ in range(n_sims):
            bankroll = START_BANKROLL
            for r in rows:
                model_t = ensemble_avg_temp(r)
                if model_t is None:
                    continue
                actual  = r.get("bucket_label")
                if not actual:
                    continue
                m_bucket = temp_to_bucket(model_t, [actual])
                is_right = buckets_equal(m_bucket, actual)
                buy_p    = random.uniform(ENTRY_PRICE_FLOOR, ENTRY_PRICE_CEIL)
                bankroll += sim_trade(is_right, buy_p)
            finals.append(bankroll)

        finals.sort()
        p5  = finals[int(0.05 * n_sims)]
        p50 = finals[int(0.50 * n_sims)]
        p95 = finals[int(0.95 * n_sims)]
        prob_loss = sum(1 for f in finals if f < START_BANKROLL) / n_sims
        return n, acc_rate, correct, p5, p50, p95, prob_loss

    cities = city_filter or sorted(set(r.get("city_slug", "") for r in all_rows))
    print(f"\n  (Trading EVERY market regardless of crowd disagreement — pure accuracy test)")
    print(f"  {'City':<10} {'n':>5} {'Correct':>8} {'Acc%':>6} {'P5':>9} {'Median':>9} {'P95':>9} {'P(loss)':>8}")
    print("  " + "─" * 68)

    all_results = {}
    all_city_rows = []
    for city_slug in cities:
        city_rows = [r for r in all_rows if r.get("city_slug") == city_slug]
        if not city_rows:
            continue
        all_city_rows.extend(city_rows)
        n, acc, correct, p5, p50, p95, prob_loss = sim_city(city_rows)
        roi50 = (p50 - START_BANKROLL) / START_BANKROLL * 100
        print(f"  {city_slug:<10} {n:>5} {correct:>8} {acc:>5.0%} "
              f"${p5:>8.2f} ${p50:>8.2f} ${p95:>8.2f} {prob_loss:>7.1%}")
        all_results[city_slug] = {"n": n, "accuracy": round(acc, 4), "p50": round(p50, 2),
                                   "prob_loss": round(prob_loss, 4)}

    if len(cities) > 1 and all_city_rows:
        n, acc, correct, p5, p50, p95, prob_loss = sim_city(all_city_rows)
        print(f"  {'ALL':<10} {n:>5} {correct:>8} {acc:>5.0%} "
              f"${p5:>8.2f} ${p50:>8.2f} ${p95:>8.2f} {prob_loss:>7.1%}")
        all_results["all"] = {"n": n, "accuracy": round(acc, 4), "p50": round(p50, 2),
                               "prob_loss": round(prob_loss, 4)}

    print(f"\n  Note: This trades EVERY market (no disagreement filter).")
    print(f"        Real alpha comes from the CLOB-only disagreement trades in Part 1.")
    print("═" * 68)
    return all_results


# ── Final verdict ─────────────────────────────────────────────────────────────

def print_verdict(equity: dict, mc: dict) -> None:
    print("\n" + "╔" + "═" * 66 + "╗")
    print("║" + "  FINAL VERDICT".center(66) + "║")
    print("╚" + "═" * 66 + "╝")

    if not equity:
        print("\n  ⚠  No CLOB data found to evaluate.\n")
        return

    p   = equity.get("p_value", 1.0)
    n   = equity.get("n", 0)
    roi = equity.get("roi_pct", 0)
    wr  = equity.get("win_rate", 0)
    be  = equity.get("break_even", 0.30)

    print(f"\n  Exact equity (CLOB markets):  ${equity['final_bankroll']:.2f}  ({roi:+.1f}% ROI)")
    print(f"  Win rate: {wr:.1%}  vs break-even {be:.1%}  ({n} trades)")
    print(f"  Binomial p-value: {p:.3f}  →  {significance_label(p)}")
    if mc:
        print(f"  Monte Carlo median outcome: ${mc['p50']:.2f}  |  P(loss): {mc['prob_loss']:.1%}")

    print()
    if p < 0.05 and roi > 0:
        print("  ✅ STATISTICALLY SIGNIFICANT EDGE DETECTED.")
        print("     Safe to deploy small capital ($50–$100). $3–5/trade.")
        print("     Continue tracking. Scale when n≥100 live trades confirms.")
    elif p < 0.15 and roi > 0:
        n_more = equity.get("n_needed", "~50")
        print(f"  🟡 PROMISING BUT NOT PROVEN (p={p:.3f}).")
        print(f"     Deploy minimum capital ($50). $3/trade.")
        print(f"     Need ~{n_more} more disagreement trades to reach significance.")
        print(f"     Run binomial test after every 10 live trades.")
    elif roi > 0:
        print(f"  ⚠  MARGINAL POSITIVE PnL BUT NOT SIGNIFICANT (p={p:.3f}).")
        print(f"     Do NOT deploy significant capital.")
        print(f"     Collect more data. Re-run when n≥50 CLOB-priced markets.")
    else:
        print(f"  ❌ INSUFFICIENT EVIDENCE — backtest shows negative PnL on real data.")
        print(f"     Do NOT deploy. The crowd beats the model on these markets.")
        print(f"     Analyse what types of days the model gets right vs wrong.")

    print("\n  TRACKING PROTOCOL (once live):")
    print("  ├─ After 30 live trades: if p < 0.15 → continue. If p > 0.30 → stop.")
    print("  ├─ After 50 live trades: if p < 0.05 → scale to $300 bankroll.")
    print("  └─ After 100 live trades: institutional confidence if p < 0.01.\n")
    print("═" * 68)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Honest PnL simulation with statistics")
    parser.add_argument("--city",          nargs="*", default=None)
    parser.add_argument("--no-montecarlo", action="store_true",
                        help="Skip the Monte Carlo (faster)")
    args = parser.parse_args()

    if not PRED_JSON.exists():
        print(f"ERROR: {PRED_JSON} not found. Run Phase 1 first.")
        sys.exit(1)

    city_filter = [c.lower() for c in args.city] if args.city else None

    # Load
    clob_trades, all_rows = load_trades(city_filter)

    # Load prices for Part 3
    prices: dict[str, dict] = {}
    if PRICE_JSON.exists():
        try:
            prices = json.loads(PRICE_JSON.read_text())
        except (ValueError, json.JSONDecodeError):
            pass

    print(f"\n  Loaded {len(all_rows)} markets with predictions")
    print(f"  CLOB-priced disagreement trades: {len(clob_trades)}")

    # Run all parts
    equity   = part1_equity_curve(clob_trades)
    mc       = part2_montecarlo(clob_trades, run_mc=not args.no_montecarlo)
    _        = part3_clob_leaderboard(all_rows, prices, city_filter)
    rand_sim = part4_random_buy_sim(all_rows, city_filter)

    if equity:
        n_needed = None
        if equity["win_rate"] > equity["break_even"]:
            n_needed = min_trades_for_significance(equity["win_rate"], equity["break_even"])
            equity["n_needed"] = n_needed

    print_verdict(equity, mc)

    # Save
    output = {"equity": equity, "montecarlo": mc, "random_sim": rand_sim}
    with OUT_JSON.open("w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results → {OUT_JSON}\n")


if __name__ == "__main__":
    main()
