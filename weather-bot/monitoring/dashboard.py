"""Terminal dashboard for live bot state."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table


console = Console()


def render_dashboard(
    bankroll: float,
    cash: float,
    active_exposure: float,
    signals_count: int,
    open_positions: int,
    stats: dict,
) -> None:
    # When running under nohup/file redirection, skip rich terminal control
    # sequences that may raise output-stream related OSErrors on shutdown.
    if not getattr(console.file, "isatty", lambda: False)():
        return

    table = Table(title="Weather Bot State")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Bankroll", f"${bankroll:.2f}")
    table.add_row("Cash", f"${cash:.2f}")
    table.add_row("Active Exposure", f"${active_exposure:.2f}")
    table.add_row("Open Positions", str(open_positions))
    table.add_row("Signals This Scan", str(signals_count))
    table.add_row("Total Trades", str(int(stats.get("total_trades", 0))))
    table.add_row("Win Rate", f"{stats.get('win_rate', 0.0) * 100:.1f}%")
    table.add_row("Total PnL", f"${stats.get('total_pnl', 0.0):.2f}")
    table.add_row("Max Drawdown", f"{stats.get('max_drawdown', 0.0) * 100:.2f}%")
    table.add_row("ROI", f"{stats.get('roi', 0.0) * 100:.2f}%")
    try:
        console.clear()
        console.print(table)
    except OSError:
        # Keep the bot running even if terminal output fails transiently.
        return
