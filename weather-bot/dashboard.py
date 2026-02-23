from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:  # pragma: no cover
    st_autorefresh = None

VM_NAME = "weather-bot"
VM_ZONE = "us-east1-b"
VM_PROJECT = "weather-488111"
VM_REMOTE_USER = "tombaxter"
VM_WORKDIR = f"/home/{VM_REMOTE_USER}/weather-bot"

# Mirror of settings.py MODEL_RUN_TRIGGER_TIMES_UTC — kept in sync manually
_TRIGGER_SCHEDULE: list[tuple[int, int, str]] = [
    (6, 10, "GFS_00Z+ECMWF_00Z"),
    (10, 0, "GFS_06Z"),
    (13, 0, "MIDDAY_DISCOVERY"),
    (18, 30, "GFS_12Z+ECMWF_12Z"),
    (23, 0, "GFS_18Z"),
]


ROOT = Path(__file__).resolve().parent
TRADES_CSV = ROOT / "logs" / "trades.csv"
SIGNALS_CSV = ROOT / "logs" / "signals.csv"
POSITIONS_JSON = ROOT / "data" / "positions.json"
DEFAULT_ENV = ROOT / ".env"
VM_ENV = Path("/etc/weather-bot.env")

BG = "#0E1117"
GREEN = "#00FF88"
RED = "#FF4444"
BLUE = "#4DA6FF"
GRAY = "#888888"
PANEL = "#141A22"
TEXT = "#E6EDF3"


def _empty_trades_df() -> pd.DataFrame:
    cols = [
        "timestamp",
        "city",
        "date",
        "bucket",
        "side",
        "forecast_prob",
        "market_prob",
        "edge",
        "size_usd",
        "fill_price",
        "outcome",
        "pnl",
    ]
    return pd.DataFrame(columns=cols)


def _empty_signals_df() -> pd.DataFrame:
    cols = ["timestamp", "city", "date", "bucket", "forecast_prob", "market_prob", "edge", "action_taken"]
    return pd.DataFrame(columns=cols)


@st.cache_data(ttl=15)
def load_trades_df() -> pd.DataFrame:
    try:
        df = pd.read_csv(TRADES_CSV)
    except Exception:
        return _empty_trades_df()
    for col in ("timestamp", "date", "city", "bucket", "side", "outcome"):
        if col not in df.columns:
            df[col] = ""
    for col in ("pnl", "fill_price", "size_usd", "edge", "market_prob", "forecast_prob"):
        if col not in df.columns:
            df[col] = 0.0
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    return df


@st.cache_data(ttl=15)
def load_signals_df() -> pd.DataFrame:
    try:
        df = pd.read_csv(SIGNALS_CSV)
    except Exception:
        return _empty_signals_df()
    for col in ("timestamp", "city", "date", "bucket", "action_taken"):
        if col not in df.columns:
            df[col] = ""
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df


def load_positions() -> list[dict]:
    try:
        payload = json.loads(POSITIONS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        return [p for p in payload.values() if isinstance(p, dict)]
    return []


def load_mode_from_env(path: Path) -> tuple[bool, bool]:
    paper = True
    live = False
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return paper, live
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        val = value.strip().lower()
        truthy = val in {"1", "true", "yes", "on"}
        if key == "PAPER_TRADING":
            paper = truthy
        elif key == "LIVE_TRADING":
            live = truthy
    return paper, live


def write_mode_to_env(path: Path, target_live: bool) -> tuple[bool, str]:
    paper_val = "false" if target_live else "true"
    live_val = "true" if target_live else "false"
    lines: list[str] = []
    try:
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        lines = []

    found_paper = False
    found_live = False
    out: list[str] = []
    for raw in lines:
        if raw.startswith("PAPER_TRADING="):
            out.append(f"PAPER_TRADING={paper_val}")
            found_paper = True
        elif raw.startswith("LIVE_TRADING="):
            out.append(f"LIVE_TRADING={live_val}")
            found_live = True
        else:
            out.append(raw)
    if not found_paper:
        out.append(f"PAPER_TRADING={paper_val}")
    if not found_live:
        out.append(f"LIVE_TRADING={live_val}")

    try:
        path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
        return True, f"Updated mode flags in {path}"
    except Exception as exc:
        return False, f"Could not write {path}: {exc}"


def realized_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df.copy()
    mask = trades_df["outcome"].astype(str).str.lower().isin({"won", "lost"})
    out = trades_df.loc[mask].copy()
    out = out.sort_values("timestamp", ascending=True)
    out["pnl"] = pd.to_numeric(out["pnl"], errors="coerce").fillna(0.0)
    out["cum_pnl"] = out["pnl"].cumsum()
    return out


def kpis(trades_df: pd.DataFrame, positions: list[dict]) -> dict[str, float]:
    resolved = realized_trades(trades_df)
    wins = 0
    losses = 0
    if not resolved.empty:
        outcomes = resolved["outcome"].astype(str).str.lower()
        wins = int((outcomes == "won").sum())
        losses = int((outcomes == "lost").sum())

    exposure = sum(float(p.get("cost", 0.0) or 0.0) for p in positions)
    open_count = len(positions)

    now_utc = datetime.now(UTC).date()
    today = now_utc.isoformat()
    tomorrow = (now_utc + timedelta(days=1)).isoformat()
    resolving_today = sum(1 for p in positions if str(p.get("date", "")) == today)
    resolving_tomorrow = sum(1 for p in positions if str(p.get("date", "")) == tomorrow)

    total_pnl = float(resolved["pnl"].sum()) if not resolved.empty else 0.0
    total_resolved = wins + losses
    win_rate = (wins / total_resolved) if total_resolved else 0.0

    return {
        "realized_pnl": total_pnl,
        "win_rate": win_rate,
        "open_positions": float(open_count),
        "open_exposure": exposure,
        "resolving_today": float(resolving_today),
        "resolving_tomorrow": float(resolving_tomorrow),
        "wins": float(wins),
        "losses": float(losses),
    }


def apply_style() -> None:
    st.markdown(
        f"""
<style>
.stApp {{
    background-color: {BG};
    color: {TEXT};
}}
.block-container {{
    padding-top: 1rem;
    max-width: 1200px;
}}
.panel {{
    background: {PANEL};
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 10px;
}}
.kpi-card {{
    background: {PANEL};
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 10px 12px;
    text-align: center;
}}
.kpi-label {{
    color: {GRAY};
    font-size: 0.78rem;
    margin-top: 4px;
}}
.kpi-value {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 1.35rem;
    font-weight: 700;
}}
.banner-title {{
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}}
.muted {{
    color: {GRAY};
    font-size: 0.82rem;
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def next_model_run_trigger(now_utc: datetime) -> tuple[datetime, str]:
    candidates: list[tuple[datetime, str]] = []
    for delta_days in (0, 1):
        base = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=delta_days)
        for hour, minute, label in _TRIGGER_SCHEDULE:
            candidate = base.replace(hour=hour, minute=minute)
            if candidate > now_utc:
                candidates.append((candidate, label))
    candidates.sort(key=lambda x: x[0])
    return candidates[0]


def sync_from_vm() -> tuple[bool, str]:
    files = [
        ("logs/trades.csv", f"{VM_WORKDIR}/logs/trades.csv"),
        ("logs/signals.csv", f"{VM_WORKDIR}/logs/signals.csv"),
        ("data/positions.json", f"{VM_WORKDIR}/data/positions.json"),
        ("logs/calibration.json", f"{VM_WORKDIR}/logs/calibration.json"),
    ]
    messages: list[str] = []
    for local_rel, remote_path in files:
        local_abs = ROOT / local_rel
        local_abs.parent.mkdir(parents=True, exist_ok=True)
        src = f"{VM_NAME}:{remote_path}"
        result = subprocess.run(
            [
                "gcloud", "compute", "scp", src, str(local_abs),
                "--zone", VM_ZONE, "--project", VM_PROJECT,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            messages.append(f"✓ {local_rel}")
        else:
            messages.append(f"— {local_rel} (not found or error)")
    return True, "\n".join(messages)


# ---------------------------------------------------------------------------
# Model Accuracy — data layer
# ---------------------------------------------------------------------------

ACCURACY_CITIES: dict[str, dict] = {
    "Seoul": {
        "lat": 37.4492, "lon": 126.451,
        "timezone": "Asia/Seoul",
        "models": {
            "ncep_aigfs025":               ("NCEP AI-GFS",       "🤖"),
            "gfs_graphcast025":            ("GFS GraphCast",     "🌐"),
            "ecmwf_ifs025":                ("ECMWF IFS",         "🌍"),
            "ncep_hgefs025_ensemble_mean": ("NCEP HGEFS Ens",    "📊"),
            "kma_gdps":                    ("KMA GDPS",          "🇰🇷"),
        },
        "best_ensemble": {
            "short":      "AVG(NCEP+GC)",
            "label":      "AVG(NCEP AI-GFS + GraphCast)",
            "model_keys": ["ncep_aigfs025", "gfs_graphcast025"],
        },
        "top_model_key":   "ncep_aigfs025",
        "top_model_label": "NCEP AI-GFS D1",
        "chart_models":    ["ncep_aigfs025", "gfs_graphcast025", "ecmwf_ifs025"],
        "notes": (
            "**Best signal:** AVG(NCEP AI-GFS + GFS GraphCast) D1 — exhaustive search confirmed this "
            "as the accuracy ceiling for Seoul.\n\n"
            "**Coverage:** `ncep_aigfs025` data starts ~Jan 7 2026. HGEFS Ensemble starts ~Jan 22 2026."
        ),
        "polymarket": {
            "2026-01-07": ("5+",   5, True),  "2026-01-08": ("-2",  -2, False),
            "2026-01-09": ("7",    7, False),  "2026-01-10": ("8",   8, False),
            "2026-01-11": ("-3",  -3, False),  "2026-01-12": ("4",   4, False),
            "2026-01-13": ("1+",   1, True),   "2026-01-14": ("2+",  2, True),
            "2026-01-15": ("7+",   7, True),   "2026-01-16": ("6",   6, False),
            "2026-01-17": ("2",    2, False),  "2026-01-18": ("2",   2, False),
            "2026-01-19": ("0",    0, False),  "2026-01-20": ("-6", -6, False),
            "2026-01-21": ("-6",  -6, False),  "2026-01-22": ("-5", -5, False),
            "2026-01-23": ("-1",  -1, False),  "2026-01-24": ("-2", -2, False),
            "2026-01-25": ("-3",  -3, False),  "2026-01-26": ("-1", -1, False),
            "2026-01-27": ("-1",  -1, False),  "2026-01-28": ("-2", -2, False),
            "2026-01-29": ("-1",  -1, False),  "2026-01-30": ("-3", -3, False),
            "2026-01-31": ("-1",  -1, False),
            "2026-02-01": ("0",    0, False),  "2026-02-02": ("0",   0, False),
            "2026-02-03": ("3",    3, False),  "2026-02-04": ("7+",  7, True),
            "2026-02-05": ("8+",   8, True),   "2026-02-06": ("-4", -4, False),
            "2026-02-07": ("-5",  -5, False),  "2026-02-08": ("-4", -4, False),
            "2026-02-09": ("2",    2, False),  "2026-02-10": ("3",   3, False),
            "2026-02-11": ("4",    4, False),  "2026-02-12": ("7+",  7, True),
            "2026-02-13": ("9+",   9, True),   "2026-02-14": ("11+",11, True),
            "2026-02-15": ("8",    8, False),  "2026-02-16": ("5",   5, False),
            "2026-02-17": ("4",    4, False),  "2026-02-18": ("6",   6, False),
            "2026-02-19": ("5",    5, False),  "2026-02-20": ("11+",11, True),
            "2026-02-21": ("14+", 14, True),   "2026-02-22": ("11", 11, False),
        },
    },
    "London": {
        "lat": 51.5053, "lon": 0.0553,
        "timezone": "Europe/London",
        "models": {
            "meteofrance_arome_france":    ("MF AROME France",  "🇫🇷"),
            "meteofrance_seamless":        ("MF Seamless",      "🇫🇷"),
            "meteofrance_arome_france_hd": ("MF AROME HD",      "🇫🇷"),
            "icon_seamless":               ("ICON Seamless",    "🇩🇪"),
            "ecmwf_ifs025":                ("ECMWF IFS",        "🌍"),
            "kma_seamless":                ("KMA Seamless",     "🇰🇷"),
            "knmi_seamless":               ("KNMI Seamless",    "🌊"),
            "dmi_seamless":                ("DMI Seamless",     "🇩🇰"),
        },
        "best_ensemble": {
            "short":      "AVG(MF+Seamless)",
            "label":      "AVG(MF AROME France + MF Seamless)",
            "model_keys": ["meteofrance_arome_france", "meteofrance_seamless"],
        },
        "top_model_key":   "meteofrance_arome_france",
        "top_model_label": "MF AROME France D1",
        "chart_models": [
            "meteofrance_arome_france",
            "meteofrance_seamless",
            "meteofrance_arome_france_hd",
            "ecmwf_ifs025",
        ],
        "notes": (
            "**Best signal:** AVG(MF AROME France + MF Seamless) D1 — brute-force exhaustive search "
            "over all 4,095 model combinations confirmed **78.8%** as the absolute accuracy ceiling "
            "for London Jan–Feb 2026.\n\n"
            "**Coverage:** MF AROME France and MF AROME HD are high-resolution regional models; "
            "D2 (T-48h) archive data is typically unavailable for these — use D1 (T-24h) only."
        ),
        "polymarket": {
            "2026-01-01": ("6",   6, False), "2026-01-02": ("4",   4, False),
            "2026-01-03": ("3",   3, False), "2026-01-04": ("3",   3, False),
            "2026-01-05": ("2",   2, False), "2026-01-06": ("4+",  4, True),
            "2026-01-07": ("6",   6, False), "2026-01-08": ("8+",  8, True),
            "2026-01-09": ("6+",  6, True),  "2026-01-10": ("3",   3, False),
            "2026-01-11": ("8+",  8, True),  "2026-01-12": ("11", 11, False),
            "2026-01-13": ("11", 11, False), "2026-01-14": ("9+",  9, True),
            "2026-01-15": ("10", 10, False), "2026-01-16": ("10", 10, False),
            "2026-01-17": ("11", 11, False), "2026-01-18": ("11+",11, True),
            "2026-01-19": ("11", 11, False), "2026-01-20": ("10", 10, False),
            "2026-01-21": ("10", 10, False), "2026-01-22": ("9",   9, False),
            "2026-01-23": ("9",   9, False), "2026-01-24": ("10", 10, False),
            "2026-01-25": ("8",   8, False), "2026-01-26": ("6",   6, False),
            "2026-01-27": ("9+",  9, True),  "2026-01-28": ("10", 10, False),
            "2026-01-29": ("7",   7, False), "2026-01-30": ("11", 11, False),
            "2026-01-31": ("11+",11, True),
            "2026-02-01": ("9",   9, False), "2026-02-02": ("8",   8, False),
            "2026-02-03": ("7",   7, False), "2026-02-04": ("10", 10, False),
            "2026-02-05": ("8",   8, False), "2026-02-06": ("12", 12, False),
            "2026-02-07": ("11", 11, False), "2026-02-08": ("11", 11, False),
            "2026-02-09": ("9",   9, False), "2026-02-10": ("12", 12, False),
            "2026-02-11": ("12", 12, False), "2026-02-12": ("11", 11, False),
            "2026-02-13": ("8",   8, False), "2026-02-14": ("6",   6, False),
            "2026-02-15": ("9",   9, False), "2026-02-16": ("10", 10, False),
            "2026-02-17": ("7",   7, False), "2026-02-18": ("6",   6, False),
            "2026-02-19": ("7",   7, False), "2026-02-20": ("12", 12, False),
            "2026-02-21": ("14", 14, False),
        },
    },
}

_OM_PREV_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


def _wins(pred: float, res_int: int, is_plus: bool) -> bool:
    return round(pred) >= res_int if is_plus else round(pred) == res_int


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_accuracy_data(city: str) -> dict:
    """Fetch previous_day1 + previous_day2 for all city-specific models, Jan–present."""
    cfg = ACCURACY_CITIES[city]
    now = datetime.now(UTC)
    end = (now - timedelta(days=1)).strftime("%Y-%m-%d")

    raw: dict[str, tuple[dict, dict]] = {}
    for model_key in cfg["models"]:
        params = {
            "latitude": cfg["lat"], "longitude": cfg["lon"],
            "hourly": "temperature_2m_previous_day1,temperature_2m_previous_day2",
            "models": model_key,
            "timezone": cfg["timezone"],
            "start_date": "2026-01-01",
            "end_date": end,
        }
        try:
            r = requests.get(_OM_PREV_URL, params=params, timeout=30)
            d = r.json()
            if "error" in d:
                raw[model_key] = ({}, {})
                continue
            times = d["hourly"]["time"]
            v1 = d["hourly"].get("temperature_2m_previous_day1", [])
            v2 = d["hourly"].get("temperature_2m_previous_day2", [])
            daily1: dict[str, list] = defaultdict(list)
            daily2: dict[str, list] = defaultdict(list)
            for t, a, b in zip(times, v1, v2):
                dt = t[:10]
                if a is not None: daily1[dt].append(a)
                if b is not None: daily2[dt].append(b)
            raw[model_key] = (dict(daily1), dict(daily2))
        except Exception:
            raw[model_key] = ({}, {})

    polymarket = cfg["polymarket"]
    ens_keys = cfg["best_ensemble"]["model_keys"]
    rows = []
    for date in sorted(polymarket.keys()):
        lbl, res_int, is_plus = polymarket[date]
        row: dict = {"date": date, "resolved": lbl, "res_int": res_int, "is_plus": is_plus}

        for mk in cfg["models"]:
            d1_map, d2_map = raw.get(mk, ({}, {}))
            p1 = round(max(d1_map[date]), 1) if d1_map.get(date) else None
            p2 = round(max(d2_map[date]), 1) if d2_map.get(date) else None
            row[f"{mk}_d1"] = p1
            row[f"{mk}_d2"] = p2
            row[f"{mk}_d1_win"] = _wins(p1, res_int, is_plus) if p1 is not None else None
            row[f"{mk}_d2_win"] = _wins(p2, res_int, is_plus) if p2 is not None else None

        # Best ensemble — D1
        ens_d1 = [row[f"{k}_d1"] for k in ens_keys if row.get(f"{k}_d1") is not None]
        best_ens_d1 = round(sum(ens_d1) / len(ens_d1), 1) if len(ens_d1) == len(ens_keys) else None
        row["best_ens_d1"] = best_ens_d1
        row["best_ens_d1_win"] = _wins(best_ens_d1, res_int, is_plus) if best_ens_d1 is not None else None

        # Best ensemble — D2
        ens_d2 = [row[f"{k}_d2"] for k in ens_keys if row.get(f"{k}_d2") is not None]
        best_ens_d2 = round(sum(ens_d2) / len(ens_d2), 1) if len(ens_d2) == len(ens_keys) else None
        row["best_ens_d2"] = best_ens_d2
        row["best_ens_d2_win"] = _wins(best_ens_d2, res_int, is_plus) if best_ens_d2 is not None else None

        rows.append(row)

    return {"rows": rows, "fetched_at": datetime.now(UTC).isoformat()}


def _build_leaderboard(rows: list[dict], city: str) -> pd.DataFrame:
    cfg = ACCURACY_CITIES[city]
    ens_cfg = cfg["best_ensemble"]

    strategies: list[tuple[str, str, str]] = [
        ("best_ens_d1", f"{ens_cfg['short']} D1", "🏆"),
        ("best_ens_d2", f"{ens_cfg['short']} D2", "📅"),
    ]
    for mk, (label, icon) in cfg["models"].items():
        strategies.append((f"{mk}_d1", f"{label} D1", icon))
        strategies.append((f"{mk}_d2", f"{label} D2", icon))

    records = []
    for key, label, icon in strategies:
        win_key = f"{key}_win"
        wins = sum(1 for r in rows if r.get(win_key) is True)
        total = sum(1 for r in rows if r.get(win_key) is not None)
        if total == 0:
            continue
        pct = wins / total * 100
        records.append({
            "Strategy": f"{icon} {label}",
            "Wins": wins,
            "Days": total,
            "Accuracy": pct,
            "_sort": pct,
        })
    df = pd.DataFrame(records).sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)
    df["Accuracy"] = df["Accuracy"].map(lambda x: f"{x:.1f}%")
    return df


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Weather Edge Terminal", layout="wide")
    apply_style()

    with st.sidebar:
        # Next model run trigger
        now_utc = datetime.now(UTC)
        next_dt, next_label = next_model_run_trigger(now_utc)
        mins_away = int((next_dt - now_utc).total_seconds() / 60)
        hours_away, mins_part = divmod(mins_away, 60)
        time_str = f"{hours_away}h {mins_part}m" if hours_away else f"{mins_part}m"
        st.markdown("### Next Model Run")
        st.markdown(
            f"""
<div style="background:#141A22;border:1px solid #1f2937;border-radius:8px;padding:10px 12px;margin-bottom:8px;">
  <div style="font-family:monospace;font-size:0.95rem;color:#4DA6FF;font-weight:700;">{next_label}</div>
  <div style="font-family:monospace;font-size:0.85rem;color:#E6EDF3;">{next_dt.strftime('%H:%M UTC')}</div>
  <div style="color:#888888;font-size:0.78rem;">in {time_str}</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Bot fires immediately at each trigger. Fast scans run for 30 min after each.")
        st.divider()
        st.markdown("### VM Data Sync")
        st.caption(f"Bot runs on `{VM_NAME}` ({VM_PROJECT}). Pull latest data before reading dashboard.")
        if st.button("Sync from VM", use_container_width=True):
            with st.spinner("Pulling from VM..."):
                try:
                    _, msg = sync_from_vm()
                    st.success(msg)
                    st.cache_data.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Sync failed: {exc}")
        st.divider()
        st.markdown("### Auto-refresh")

    refresh_opt = st.sidebar.selectbox("Interval", options=["off", "30s", "60s"], index=1)
    if refresh_opt != "off" and st_autorefresh is not None:
        interval_ms = 30000 if refresh_opt == "30s" else 60000
        st_autorefresh(interval=interval_ms, key="dashboard-refresh")

    now_stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    mode_env_path = DEFAULT_ENV
    paper_mode, live_mode = load_mode_from_env(mode_env_path)
    mode_text = "LIVE MODE" if live_mode else "PAPER MODE"
    mode_color = RED if live_mode else GREEN

    st.markdown(
        f"""
<div class="panel" style="display:flex;justify-content:space-between;align-items:center;">
  <div>
    <div class="banner-title">⚡ WEATHER EDGE</div>
    <div class="muted">Ensemble Mispricing Terminal</div>
  </div>
  <div style="text-align:right;">
    <div style="font-weight:700;color:{mode_color};">● {mode_text}</div>
    <div class="muted">auto-refresh: {refresh_opt} · updated: {now_stamp}</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    tab_trading, tab_accuracy = st.tabs(["⚡ Trading", "📊 Model Accuracy"])

    with tab_trading:
        _render_trading_tab()

    with tab_accuracy:
        _render_accuracy_tab()


def _render_trading_tab() -> None:
    trades_df = load_trades_df()
    _ = load_signals_df()
    positions = load_positions()
    metrics = kpis(trades_df, positions)
    _, live_mode = load_mode_from_env(DEFAULT_ENV)

    cols = st.columns(6)
    kpi_items = [
        ("Realized PnL", f"${metrics['realized_pnl']:+.2f}", GREEN if metrics["realized_pnl"] >= 0 else RED),
        ("Win Rate", f"{metrics['win_rate']*100:.1f}%", BLUE),
        ("Open Positions", f"{int(metrics['open_positions'])}", BLUE),
        ("Open Exposure", f"${metrics['open_exposure']:.2f}", BLUE),
        ("Resolving Today", f"{int(metrics['resolving_today'])}", BLUE),
        ("Resolving Tomorrow", f"{int(metrics['resolving_tomorrow'])}", BLUE),
    ]
    for col, (label, value, color) in zip(cols, kpi_items):
        with col:
            st.markdown(
                f"""
<div class="kpi-card">
  <div class="kpi-value" style="color:{color};">{value}</div>
  <div class="kpi-label">{label}</div>
</div>
                """,
                unsafe_allow_html=True,
            )

    resolved = realized_trades(trades_df)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("CUMULATIVE REALIZED PnL")
    if resolved.empty:
        st.info("No resolved trades yet — check back after markets close.")
    else:
        fig = go.Figure()
        line_color = GREEN if float(resolved["cum_pnl"].iloc[-1]) >= 0 else RED
        fig.add_trace(
            go.Scatter(
                x=resolved["timestamp"],
                y=resolved["cum_pnl"],
                mode="lines",
                line={"color": line_color, "width": 2},
                fill="tozeroy",
                fillcolor="rgba(0,255,136,0.15)" if line_color == GREEN else "rgba(255,68,68,0.15)",
                hovertemplate="%{x}<br>Cumulative PnL: $%{y:.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            plot_bgcolor=BG,
            paper_bgcolor=PANEL,
            font={"color": TEXT, "family": "Inter, Arial, sans-serif"},
            margin={"l": 20, "r": 10, "t": 10, "b": 20},
            xaxis={"gridcolor": "#1f2937"},
            yaxis={"gridcolor": "#1f2937", "title": "USD"},
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    pos_df = pd.DataFrame(positions)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader(f"OPEN POSITIONS ({len(positions)})")
    if pos_df.empty:
        st.info("No open positions.")
    else:
        show_cols = ["city", "station_icao", "date", "bucket", "side", "fill_price", "cost"]
        for col in show_cols:
            if col not in pos_df.columns:
                pos_df[col] = ""
        pos_df["fill_price"] = pd.to_numeric(pos_df["fill_price"], errors="coerce").fillna(0.0)
        pos_df["cost"] = pd.to_numeric(pos_df["cost"], errors="coerce").fillna(0.0)
        pos_df = pos_df.sort_values(["date", "city", "bucket"], ascending=True)
        st.dataframe(
            pos_df[show_cols].rename(
                columns={
                    "city": "City",
                    "station_icao": "Station",
                    "date": "Date",
                    "bucket": "Bucket",
                    "side": "Side",
                    "fill_price": "Entry Price",
                    "cost": "Cost",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Total exposure: ${metrics['open_exposure']:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader(f"RECENT RESOLVED TRADES (wins={int(metrics['wins'])}, losses={int(metrics['losses'])})")
    if resolved.empty:
        st.info("Awaiting first resolution.")
    else:
        latest = resolved.sort_values("timestamp", ascending=False).head(50).copy()
        latest["timestamp"] = latest["timestamp"].dt.strftime("%m-%d %H:%M")
        show_cols = ["timestamp", "city", "date", "bucket", "side", "fill_price", "edge", "pnl", "outcome"]
        st.dataframe(
            latest[show_cols].rename(
                columns={
                    "timestamp": "Time",
                    "city": "City",
                    "date": "Date",
                    "bucket": "Bucket",
                    "side": "Side",
                    "fill_price": "Entry",
                    "edge": "Edge",
                    "pnl": "PnL",
                    "outcome": "Result",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("TRADING MODE")
    mode_choice = st.radio("Mode", options=["Paper", "Live"], index=1 if live_mode else 0, horizontal=True)
    env_target = st.selectbox("Env file target", options=[str(DEFAULT_ENV), str(VM_ENV)], index=0)
    if env_target == str(VM_ENV):
        st.warning("Writing /etc/weather-bot.env usually requires sudo/root privileges.")
    st.warning("Live mode is still blocked by main.py safety guard unless code is changed.")

    if st.button("Apply Mode Change"):
        target_live = mode_choice == "Live"
        ok, msg = write_mode_to_env(Path(env_target), target_live=target_live)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    restart_cmd = "sudo systemctl restart weather-bot" if env_target == str(VM_ENV) else "python3 main.py"
    st.code(restart_cmd, language="bash")
    st.caption("Manual restart required. Dashboard does not execute privileged restart automatically.")

    st.markdown("### Wallet")
    st.button("Connect Polygon Wallet (Coming Soon)", disabled=True, help="Wallet integration will be added later.")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_accuracy_tab() -> None:
    st.markdown(
        f"""
<div class="panel" style="display:flex;justify-content:space-between;align-items:center;">
  <div>
    <div class="banner-title">📊 MODEL ACCURACY BACKTEST</div>
    <div class="muted">Bucket hit rate vs Polymarket resolved temperatures · Jan–Feb 2026</div>
  </div>
  <div style="text-align:right;">
    <div style="color:{BLUE};font-weight:700;font-size:0.85rem;">Data: Open-Meteo Previous Runs API</div>
    <div class="muted">Refreshes every hour · D1 = T-24h · D2 = T-48h</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    city = st.selectbox("City", options=list(ACCURACY_CITIES.keys()), index=0, key="acc_city")
    cfg = ACCURACY_CITIES[city]
    ens_cfg = cfg["best_ensemble"]

    col_refresh, col_info = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh data", key="acc_refresh"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner(f"Loading model predictions for {city}..."):
        acc = fetch_accuracy_data(city)

    rows = acc["rows"]
    fetched_at = acc.get("fetched_at", "")
    with col_info:
        st.caption(f"Last fetched: {fetched_at[:19].replace('T', ' ')} UTC · {len(rows)} market days")

    if not rows:
        st.warning("No data available. Check API connectivity.")
        return

    # ── Section 1: KPI cards ────────────────────────────────────────────────
    best_wins = sum(1 for r in rows if r.get("best_ens_d1_win") is True)
    best_n    = sum(1 for r in rows if r.get("best_ens_d1_win") is not None)
    best_pct  = best_wins / best_n * 100 if best_n else 0

    top_mk   = cfg["top_model_key"]
    top_wins = sum(1 for r in rows if r.get(f"{top_mk}_d1_win") is True)
    top_n    = sum(1 for r in rows if r.get(f"{top_mk}_d1_win") is not None)
    top_pct  = top_wins / top_n * 100 if top_n else 0

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, color in [
        (c1, f"Best Signal ({ens_cfg['short']})", f"{best_pct:.1f}%", GREEN),
        (c2, cfg["top_model_label"],              f"{top_pct:.1f}%",  BLUE),
        (c3, "Market Days Tested",                str(best_n),        BLUE),
        (c4, "Signal Lead Time",                  "T-24h / T-48h",   GRAY),
    ]:
        with col:
            st.markdown(
                f"""<div class="kpi-card">
  <div class="kpi-value" style="color:{color};">{val}</div>
  <div class="kpi-label">{label}</div>
</div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Leaderboard ──────────────────────────────────────────────
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("🏆 Model Leaderboard — Bucket Accuracy")
    st.caption("Bucket hit = model's rounded prediction matches the Polymarket winning bucket")

    lb_df = _build_leaderboard(rows, city)

    bar_colors = [GREEN if i == 0 else BLUE if i < 3 else GRAY for i in range(len(lb_df))]
    max_acc = lb_df["Accuracy"].str.replace("%", "").astype(float).max() if not lb_df.empty else 70
    fig_bar = go.Figure(go.Bar(
        x=lb_df["Strategy"],
        y=lb_df["Accuracy"].str.replace("%", "").astype(float),
        marker_color=bar_colors,
        text=lb_df["Accuracy"],
        textposition="outside",
        hovertemplate="%{x}<br>Accuracy: %{y:.1f}%<extra></extra>",
    ))
    fig_bar.add_hline(y=50, line_dash="dot", line_color=GRAY, annotation_text="50% baseline", annotation_font_color=GRAY)
    fig_bar.update_layout(
        plot_bgcolor=BG, paper_bgcolor=PANEL,
        font={"color": TEXT, "family": "Inter, Arial, sans-serif"},
        margin={"l": 10, "r": 10, "t": 10, "b": 130},
        xaxis={"gridcolor": "#1f2937", "tickangle": -35},
        yaxis={"gridcolor": "#1f2937", "title": "Bucket Accuracy %", "range": [0, min(max_acc + 15, 100)]},
        height=420, showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.dataframe(lb_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 3: Day-by-day table ─────────────────────────────────────────
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📅 Day-by-Day Predictions vs Polymarket (D1 / T-24h)")

    display_rows = []
    for r in rows:
        ens_val = r.get("best_ens_d1")
        ens_win = r.get("best_ens_d1_win")
        ens_cell = (f"{ens_val:.1f}° {'✅' if ens_win else '❌'}") if ens_val is not None else "—"

        row_d: dict = {
            "Date":           r["date"],
            "Resolved":       r["resolved"],
            ens_cfg["short"]: ens_cell,
        }
        for mk, (label, icon) in cfg["models"].items():
            val = r.get(f"{mk}_d1")
            win = r.get(f"{mk}_d1_win")
            row_d[f"{icon} {label}"] = (f"{val:.1f}° {'✅' if win else '❌'}") if val is not None else "—"

        display_rows.append(row_d)

    detail_df = pd.DataFrame(display_rows)
    st.dataframe(detail_df, use_container_width=True, hide_index=True, height=520)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 4: Rolling 10-day accuracy ──────────────────────────────────
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📈 Rolling 10-Day Accuracy")
    st.caption("How model accuracy evolves over the backtest period")

    window = 10
    _chart_palette = [BLUE, "#FF9F40", "#A855F7", GRAY, "#FF6B6B"]
    chart_model_keys = cfg.get("chart_models", list(cfg["models"].keys())[:3])

    trace_configs: list[tuple[str, str, str]] = [
        ("best_ens_d1", f"🏆 {ens_cfg['short']} D1", GREEN),
    ]
    for i, mk in enumerate(chart_model_keys):
        lbl, icon = cfg["models"][mk]
        trace_configs.append((f"{mk}_d1", f"{icon} {lbl} D1", _chart_palette[i % len(_chart_palette)]))

    fig_roll = go.Figure()
    for key, label, color in trace_configs:
        win_key = f"{key}_win"
        valid = [(r["date"], r.get(win_key)) for r in rows if r.get(win_key) is not None]
        if len(valid) < window:
            continue
        dates_v = [v[0] for v in valid]
        wins_v  = [1 if v[1] else 0 for v in valid]
        rolling = [
            sum(wins_v[max(0, i - window + 1): i + 1]) / min(i + 1, window) * 100
            for i in range(len(wins_v))
        ]
        fig_roll.add_trace(go.Scatter(
            x=dates_v, y=rolling,
            mode="lines+markers", name=label,
            line={"color": color, "width": 2},
            marker={"size": 5},
            hovertemplate=f"{label}<br>%{{x}}<br>Rolling acc: %{{y:.1f}}%<extra></extra>",
        ))

    fig_roll.add_hline(y=50, line_dash="dot", line_color=GRAY, annotation_text="50%", annotation_font_color=GRAY)
    fig_roll.update_layout(
        plot_bgcolor=BG, paper_bgcolor=PANEL,
        font={"color": TEXT, "family": "Inter, Arial, sans-serif"},
        margin={"l": 20, "r": 10, "t": 10, "b": 20},
        xaxis={"gridcolor": "#1f2937"},
        yaxis={"gridcolor": "#1f2937", "title": "Rolling accuracy %", "range": [0, 100]},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"size": 11}},
        height=340,
    )
    st.plotly_chart(fig_roll, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 5: Notes ────────────────────────────────────────────────────
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📌 Notes & Methodology")
    st.markdown(f"""
**Bucket rule:** Model prediction is rounded to nearest integer. Wins if it matches the Polymarket winning bucket.
For "X or higher" top buckets, any prediction ≥ X is a win.

**T-24h (D1):** Forecast made the day before the resolution date — this is the primary trading window.

**T-48h (D2):** Forecast made two days before — useful for earlier entry at better prices.

**Data source:** [Open-Meteo Previous Runs API](https://previous-runs-api.open-meteo.com) — `temperature_2m_previous_day1/2`

{cfg.get('notes', '')}
    """)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
