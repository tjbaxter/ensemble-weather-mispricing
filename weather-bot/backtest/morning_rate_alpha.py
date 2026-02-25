#!/usr/bin/env python3
"""
Morning Temperature Rise Rate Alpha Analysis — London (EGLC)
============================================================
Hypothesis: the rate at which temperature rises in the morning hours
(06:00–12:00 local time) on the event day correlates with the actual high
overshooting the D+1 model prediction from the previous evening.

If true, this gives an intraday signal: you can update your position at
noon with better information than any numerical model provides at 12Z.

Method
------
For each resolved London date:
  1. Fetch WU EGLC hourly observations for that date.
  2. Extract temps in the 06:00–12:00 window (local London time).
  3. Compute:
       morning_rise   = max(window) - min(window)   [°C]
       rise_rate      = morning_rise / hours         [°C/hr]
       temp_at_6am    = earliest reading >= 06:00
       temp_at_noon   = latest reading <= 12:00
  4. Compare to:
       mf_pred        = meteofrance_arome_france_d1  (primary signal)
       h1_pred        = h1_5model_d1                 (ensemble)
       actual         = res_int                      (Polymarket resolution)
       overshoot_mf   = actual - round(mf_pred)      (positive = actual higher)
       overshoot_h1   = actual - round(h1_pred)
  5. Pearson correlation & scatter.
  6. Bucket analysis: do high-rise-rate mornings predict upside surprises?
"""

import json
import time
import math
import sys
from datetime import datetime, timezone, date
from pathlib import Path
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
CACHE_PATH   = ROOT / "weather-bot" / "data" / "accuracy_rows_cache.json"
OBS_CACHE    = ROOT / "weather-bot" / "data" / "morning_obs_cache.json"

WU_OBS_KEY   = "e1f10a1e78da46f5b10a1e78da96f525"
WU_OBS_URL   = "https://api.weather.com/v1/location/{station}/observations/historical.json"
STATION      = "EGLC:9:GB"
UNITS        = "m"  # metric

# Morning window in local London time (24h)
WINDOW_START = 6   # 06:00
WINDOW_END   = 14  # up to but not including 14:00 (includes 13:xx readings)

# ---------------------------------------------------------------------------
# Load accuracy cache
# ---------------------------------------------------------------------------
def load_london_rows() -> list[dict]:
    with open(CACHE_PATH) as f:
        d = json.load(f)
    rows = d.get("London", [])
    # Only rows with MF AROME d1 prediction and resolved integer temp
    return [r for r in rows if r.get("meteofrance_arome_france_d1") and r.get("res_int") is not None]


# ---------------------------------------------------------------------------
# Fetch / cache WU intraday observations
# ---------------------------------------------------------------------------
def load_obs_cache() -> dict:
    if OBS_CACHE.exists():
        try:
            return json.loads(OBS_CACHE.read_text())
        except Exception:
            pass
    return {}


def save_obs_cache(cache: dict) -> None:
    OBS_CACHE.write_text(json.dumps(cache, indent=2))


def fetch_wu_obs_for_date(date_str: str, cache: dict) -> list[dict] | None:
    """Return list of hourly observation dicts for date_str, cached to disk."""
    if date_str in cache:
        return cache[date_str]

    date_compact = date_str.replace("-", "")
    try:
        r = requests.get(
            WU_OBS_URL.format(station=STATION),
            params={
                "apiKey": WU_OBS_KEY,
                "units":  UNITS,
                "startDate": date_compact,
                "endDate":   date_compact,
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        if r.status_code == 204:
            cache[date_str] = []
            return []
        r.raise_for_status()
        obs = r.json().get("observations", [])
        cache[date_str] = obs
        return obs
    except Exception as e:
        print(f"    WU error for {date_str}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Extract morning rise from obs list
# ---------------------------------------------------------------------------
def extract_morning_features(obs: list[dict]) -> dict | None:
    """
    From a list of WU observation dicts, extract morning window features.
    WU timestamps are valid_time_gmt (Unix). Convert to London local time.
    """
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("Europe/London")

    window_obs = []
    for o in obs:
        gmt = o.get("valid_time_gmt")
        temp = o.get("temp")
        if gmt is None or temp is None:
            continue
        local_dt = datetime.fromtimestamp(gmt, tz=tz)
        hour = local_dt.hour
        if WINDOW_START <= hour < WINDOW_END:
            window_obs.append((local_dt, float(temp)))

    if len(window_obs) < 2:
        return None

    window_obs.sort(key=lambda x: x[0])
    temps       = [t for _, t in window_obs]
    hours_span  = (window_obs[-1][0] - window_obs[0][0]).total_seconds() / 3600
    if hours_span < 0.5:
        return None

    morning_rise = max(temps) - temps[0]   # rise from first reading in window
    rise_rate    = morning_rise / hours_span if hours_span > 0 else 0.0
    peak_morning = max(temps)
    start_temp   = temps[0]

    return {
        "start_temp":    round(start_temp, 1),
        "peak_morning":  round(peak_morning, 1),
        "morning_rise":  round(morning_rise, 2),
        "rise_rate":     round(rise_rate, 3),
        "n_readings":    len(window_obs),
        "hours_span":    round(hours_span, 2),
    }


# ---------------------------------------------------------------------------
# Correlation & stats helpers
# ---------------------------------------------------------------------------
def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num   = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / denom if denom else float("nan")


def buckets(xs: list[float], ys: list[float], n_buckets: int = 4) -> list[dict]:
    """Split xs into n equal-frequency buckets, report mean y per bucket."""
    paired = sorted(zip(xs, ys))
    size = len(paired) // n_buckets
    result = []
    for i in range(n_buckets):
        chunk = paired[i * size: (i + 1) * size] if i < n_buckets - 1 else paired[i * size:]
        bxs = [p[0] for p in chunk]
        bys = [p[1] for p in chunk]
        result.append({
            "bucket":        i + 1,
            "rise_rate_min": round(min(bxs), 3),
            "rise_rate_max": round(max(bxs), 3),
            "mean_rise_rate": round(sum(bxs) / len(bxs), 3),
            "mean_overshoot": round(sum(bys) / len(bys), 2),
            "n":             len(chunk),
            "pct_positive":  round(100 * sum(1 for y in bys if y > 0) / len(bys), 1),
        })
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("London Morning Rise Rate Alpha Analysis")
    print("=" * 60)

    rows = load_london_rows()
    print(f"\nLoaded {len(rows)} resolved London days with MF AROME D1 prediction.")

    obs_cache = load_obs_cache()
    records: list[dict] = []

    print("\nFetching WU intraday observations...")
    for i, row in enumerate(sorted(rows, key=lambda r: r["date"])):
        date_str = row["date"]
        # Skip future or today
        if date_str >= date.today().isoformat():
            continue

        print(f"  [{i+1:02d}/{len(rows)}] {date_str}", end=" ", flush=True)
        cached_hit = date_str in obs_cache

        obs = fetch_wu_obs_for_date(date_str, obs_cache)
        if obs is None:
            print("API error — skipping")
            continue
        if not obs:
            print("no data")
            continue

        feats = extract_morning_features(obs)
        if feats is None:
            print(f"insufficient morning readings ({len(obs)} total obs)")
            continue

        mf_pred  = row.get("meteofrance_arome_france_d1")
        h1_pred  = row.get("h1_5model_d1")
        actual   = row["res_int"]
        is_plus  = row.get("is_plus", False)

        if mf_pred is None:
            print("no MF pred — skip")
            continue

        # Overshoot: positive = actual was higher than model predicted
        # For "+" bucket days, use res_int as floor (actual >= res_int), so
        # overshoot is conservative (0 if actual == res_int).
        overshoot_mf = actual - round(mf_pred)
        overshoot_h1 = (actual - round(h1_pred)) if h1_pred else None

        rec = {
            "date":        date_str,
            "actual":      actual,
            "is_plus":     is_plus,
            "mf_pred":     round(mf_pred, 1),
            "h1_pred":     round(h1_pred, 1) if h1_pred else None,
            "overshoot_mf": overshoot_mf,
            "overshoot_h1": overshoot_h1,
            **feats,
        }
        records.append(rec)

        status = f"rise={feats['morning_rise']:+.1f}°C  rate={feats['rise_rate']:.3f}°C/hr  overshoot={overshoot_mf:+d}°C"
        print(("✓ " if not cached_hit else "📦 ") + status)

        if not cached_hit:
            time.sleep(0.4)   # rate limit courtesy

    save_obs_cache(obs_cache)
    print(f"\nTotal usable records: {len(records)}")

    if len(records) < 10:
        print("Not enough data for meaningful analysis. Exiting.")
        return

    # ---------------------------------------------------------------------------
    # Correlation analysis
    # ---------------------------------------------------------------------------
    xs_rate  = [r["rise_rate"]    for r in records]
    xs_rise  = [r["morning_rise"] for r in records]
    ys_mf    = [r["overshoot_mf"] for r in records]
    ys_h1    = [r["overshoot_h1"] for r in records if r["overshoot_h1"] is not None]
    xs_h1    = [r["rise_rate"]    for r in records if r["overshoot_h1"] is not None]

    print("\n" + "=" * 60)
    print("PEARSON CORRELATIONS")
    print("=" * 60)
    print(f"  rise_rate  vs MF overshoot  : r = {pearson(xs_rate, ys_mf):+.3f}  (n={len(xs_rate)})")
    print(f"  morning_rise vs MF overshoot: r = {pearson(xs_rise, ys_mf):+.3f}  (n={len(xs_rise)})")
    print(f"  rise_rate  vs H1 overshoot  : r = {pearson(xs_h1, ys_h1):+.3f}  (n={len(xs_h1)})")

    print("\n" + "=" * 60)
    print("BUCKET ANALYSIS — rise_rate quartiles vs MF overshoot")
    print("=" * 60)
    bkts = buckets(xs_rate, ys_mf, 4)
    print(f"  {'Bucket':<8} {'Rate range (°C/hr)':<25} {'Mean overshoot':<18} {'% positive':<12} n")
    print("  " + "-" * 72)
    for b in bkts:
        print(f"  Q{b['bucket']:<7} {b['rise_rate_min']:.3f} – {b['rise_rate_max']:.3f}{'':>12} "
              f"{b['mean_overshoot']:+.2f}°C{'':>9} {b['pct_positive']:>5.1f}%{'':>6} {b['n']}")

    print("\n" + "=" * 60)
    print("RAW DATA (sorted by date)")
    print("=" * 60)
    print(f"  {'Date':<12} {'Actual':>7} {'MF pred':>8} {'Overshoot':>10} {'Rise°C':>8} {'Rate/hr':>8} {'6am°C':>7}")
    print("  " + "-" * 65)
    for r in sorted(records, key=lambda x: x["date"]):
        flag = "🔺" if r["overshoot_mf"] > 0 else ("🔻" if r["overshoot_mf"] < 0 else "  ")
        print(f"  {r['date']:<12} {r['actual']:>5}°C  {r['mf_pred']:>6.1f}°C  "
              f"{r['overshoot_mf']:>+7d}°C {flag}  {r['morning_rise']:>5.1f}°C  "
              f"{r['rise_rate']:>7.3f}  {r['start_temp']:>5.1f}°C")

    # ---------------------------------------------------------------------------
    # Actionable summary
    # ---------------------------------------------------------------------------
    high_rise = [r for r in records if r["rise_rate"] >= sorted(xs_rate)[len(xs_rate) * 3 // 4]]
    low_rise  = [r for r in records if r["rise_rate"] <= sorted(xs_rate)[len(xs_rate) * 1 // 4]]

    high_overshoot = sum(r["overshoot_mf"] for r in high_rise) / len(high_rise) if high_rise else 0
    low_overshoot  = sum(r["overshoot_mf"] for r in low_rise)  / len(low_rise)  if low_rise  else 0

    print("\n" + "=" * 60)
    print("ACTIONABLE SUMMARY")
    print("=" * 60)
    print(f"  Top quartile rise rate ({sorted(xs_rate)[len(xs_rate) * 3 // 4]:.3f}+ °C/hr):")
    print(f"    → mean overshoot vs MF AROME = {high_overshoot:+.2f}°C  ({len(high_rise)} days)")
    print(f"  Bottom quartile rise rate ({sorted(xs_rate)[len(xs_rate) * 1 // 4]:.3f}– °C/hr):")
    print(f"    → mean overshoot vs MF AROME = {low_overshoot:+.2f}°C  ({len(low_rise)} days)")
    print()
    if abs(high_overshoot - low_overshoot) >= 1.0:
        print("  ✅ Spread ≥ 1°C between Q4 and Q1 — POTENTIALLY EXPLOITABLE")
        print("     Morning rate may be a useful intraday adjustment signal.")
    else:
        print("  ❌ Spread < 1°C — weak signal, not yet actionable.")
    print()


if __name__ == "__main__":
    main()
