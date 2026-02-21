"""Historical backtest helpers against recorded WU outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    total_trades: int
    win_rate: float
    total_pnl: float
    roi: float


def load_outcomes_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "city", "bucket", "resolved_bucket", "entry_price", "size_usd"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Outcomes CSV missing columns: {sorted(missing)}")
    return df


def run_simple_backtest(df: pd.DataFrame, initial_bankroll: float = 300.0) -> BacktestResult:
    pnl_values = []
    for _, row in df.iterrows():
        won = row["bucket"] == row["resolved_bucket"]
        shares = row["size_usd"] / row["entry_price"]
        pnl = shares - row["size_usd"] if won else -row["size_usd"]
        pnl_values.append(float(pnl))

    total_pnl = sum(pnl_values)
    total_trades = len(pnl_values)
    wins = sum(1 for p in pnl_values if p > 0)
    return BacktestResult(
        total_trades=total_trades,
        win_rate=(wins / total_trades) if total_trades else 0.0,
        total_pnl=total_pnl,
        roi=(total_pnl / initial_bankroll) if initial_bankroll else 0.0,
    )
