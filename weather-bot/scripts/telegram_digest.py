#!/usr/bin/env python3
"""Telegram daily digest — push message after every trading run.

Reads today's signals.csv + resolved.csv, sends a compact summary to Telegram.

Setup:
    1. Create a bot: message @BotFather on Telegram → /newbot
    2. Get your chat ID: message @userinfobot
    3. Add to /etc/weather-bot.env:
           TELEGRAM_BOT_TOKEN=123456:ABCdef...
           TELEGRAM_CHAT_ID=-1001234567890  (group) or 123456789 (personal)

Cron (VM) — runs 15 mins after the 18:30 UTC trading trigger:
    45 18 * * * /home/tombaxter/weather-bot/venv/bin/python3 \
                /home/tombaxter/weather-bot/scripts/telegram_digest.py \
                >> /home/tombaxter/weather-bot/logs/telegram.log 2>&1
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIGNALS_CSV  = ROOT / "logs" / "signals.csv"
RESOLVED_CSV = ROOT / "logs" / "resolved.csv"
TRADES_CSV   = ROOT / "logs" / "trades.csv"

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def _read_env(key: str) -> str:
    for path in (ROOT / ".env", Path("/etc/weather-bot.env")):
        try:
            if not path.exists():
                continue
            text = path.read_text()
        except PermissionError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    # Also try system environment
    return os.getenv(key, "")


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _today_signals() -> list[dict]:
    today = date.today().isoformat()
    rows = _load_csv(SIGNALS_CSV)
    return [r for r in rows if r.get("timestamp", "").startswith(today)]


def _yesterday_resolved() -> list[dict]:
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    rows = _load_csv(RESOLVED_CSV)
    return [r for r in rows if r.get("target_date", "") == yesterday]


def _all_resolved() -> list[dict]:
    return _load_csv(RESOLVED_CSV)


def _format_message() -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    today_sigs = _today_signals()
    yesterday_res = _yesterday_resolved()
    all_res = _all_resolved()

    # ── TODAY'S TRADING RUN ────────────────────────────────────────────
    trades_today = [r for r in today_sigs if r.get("action_taken") == "trade"]
    skipped_today = [r for r in today_sigs if r.get("action_taken") not in ("trade", "already_held")]
    already_held = [r for r in today_sigs if r.get("action_taken") == "already_held"]

    lines = [
        f"🤖 *Weather Bot Daily Digest*",
        f"_{now}_",
        "",
        f"*Today's Run ({today})*",
        f"  📊 Markets scanned: {len(set(r.get('city','') for r in today_sigs))} cities",
        f"  ✅ Bets placed: {len(trades_today)}",
        f"  ⏭ Already held: {len(already_held)}",
        f"  ❌ Skipped (no edge): {len(skipped_today)}",
    ]

    if trades_today:
        lines.append("")
        lines.append("*Positions Opened:*")
        for r in trades_today[:8]:  # cap at 8 to keep message short
            city = r.get("city", "?")
            bucket = r.get("bucket", "?")
            side = r.get("side", "BUY_YES")
            price = float(r.get("market_prob") or 0)
            size = float(r.get("size_usd") or 0)
            ev = float(r.get("ev_per_bet") or 0)
            sc = r.get("spread_colour", "?")
            sc_emoji = "🟢" if sc == "GREEN" else "🔴" if sc == "RED" else "⚪"
            side_short = "YES" if side == "BUY_YES" else "NO"
            lines.append(
                f"  {sc_emoji} {city} {bucket} {side_short} "
                f"@ {price:.2f}¢ · ${size:.2f} · EV ${ev:+.2f}"
            )
        if len(trades_today) > 8:
            lines.append(f"  _...and {len(trades_today)-8} more_")

    # ── YESTERDAY'S RESOLUTION ────────────────────────────────────────
    if yesterday_res:
        lines += ["", f"*Resolved ({yesterday}):*"]
        wins = [r for r in yesterday_res if r.get("outcome") == "WIN"]
        losses = [r for r in yesterday_res if r.get("outcome") == "LOSS"]
        pnl_day = sum(float(r.get("pnl_usd") or 0) for r in yesterday_res)
        acc = len(wins) / len(yesterday_res) * 100 if yesterday_res else 0
        pnl_emoji = "📈" if pnl_day >= 0 else "📉"
        lines.append(
            f"  {pnl_emoji} {len(wins)}W / {len(losses)}L  "
            f"({acc:.0f}%)  P&L *${pnl_day:+.2f}*"
        )
        for r in yesterday_res:
            city = r.get("city", "?")
            bucket = r.get("bucket", "?")
            actual = r.get("actual_temp", "?")
            outcome = r.get("outcome", "?")
            pnl = float(r.get("pnl_usd") or 0)
            emoji = "✅" if outcome == "WIN" else "❌"
            lines.append(f"  {emoji} {city} {bucket} → actual {actual}° | {pnl:+.2f}")
    else:
        lines += ["", f"_No positions resolved yesterday ({yesterday})_"]

    # ── ALL-TIME STATS ────────────────────────────────────────────────
    if all_res:
        total = len(all_res)
        total_wins = sum(1 for r in all_res if r.get("outcome") == "WIN")
        total_pnl = sum(float(r.get("pnl_usd") or 0) for r in all_res)
        total_acc = total_wins / total * 100 if total else 0

        green_rows = [r for r in all_res if r.get("spread_colour") == "GREEN"]
        red_rows   = [r for r in all_res if r.get("spread_colour") == "RED"]
        green_acc = (
            sum(1 for r in green_rows if r["outcome"] == "WIN") / len(green_rows) * 100
            if green_rows else 0
        )
        red_acc = (
            sum(1 for r in red_rows if r["outcome"] == "WIN") / len(red_rows) * 100
            if red_rows else 0
        )

        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        lines += [
            "",
            "*All-Time Stats:*",
            f"  Resolved: {total} ({total_wins}W/{total-total_wins}L · {total_acc:.0f}%)",
            f"  Cumulative P&L: {pnl_emoji} *${total_pnl:+.2f}*",
        ]
        if green_rows:
            lines.append(f"  🟢 GREEN days: {green_acc:.0f}% (target 75%)")
        if red_rows:
            lines.append(f"  🔴 RED days:   {red_acc:.0f}% (target 55%)")

        # Calibration alert
        if total >= 20:
            if total_acc < 50:
                lines.append(
                    f"\n⚠️ *CALIBRATION ALERT*: accuracy {total_acc:.0f}% is below 50% — "
                    "review p\\_win assumptions"
                )
            elif green_rows and len(green_rows) >= 10 and green_acc < 60:
                lines.append(
                    f"\n⚠️ GREEN accuracy {green_acc:.0f}% is below 60% "
                    "(expected 75%) — model may be drifting"
                )
    else:
        lines += ["", "_No resolved trades yet — first resolution tomorrow._"]

    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    token = _read_env("TELEGRAM_BOT_TOKEN")
    chat_id = _read_env("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — skipping.")
        print("Add them to .env or /etc/weather-bot.env to enable push notifications.")
        print("\n--- DIGEST PREVIEW ---")
        print(message)
        return False

    url = TELEGRAM_API.format(token=token)
    resp = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        },
        timeout=15,
    )
    if resp.ok:
        print(f"Telegram message sent OK (msg_id={resp.json().get('result',{}).get('message_id')})")
        return True
    else:
        print(f"Telegram send failed: {resp.status_code} {resp.text}")
        return False


if __name__ == "__main__":
    print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}] Building digest...")
    msg = _format_message()
    send_telegram(msg)
