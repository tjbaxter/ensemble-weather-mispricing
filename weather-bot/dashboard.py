from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
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

    trades_df = load_trades_df()
    _ = load_signals_df()  # keep warm for future sections, currently not shown.
    positions = load_positions()
    metrics = kpis(trades_df, positions)

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


if __name__ == "__main__":
    main()
