"""Portfolio tracking with drawdown and exposure controls."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from statistics import mean, pstdev


@dataclass
class Position:
    market_id: str
    token_id: str
    side: str
    city: str
    station_icao: str
    date: str
    bucket: str
    fill_price: float
    fill_size: float
    cost: float
    timestamp: datetime


@dataclass
class ClosedTrade(Position):
    won: bool = False
    pnl: float = 0.0


@dataclass
class Portfolio:
    initial_bankroll: float
    current_cash: float = field(init=False)
    positions: list[Position] = field(default_factory=list)
    trade_history: list[ClosedTrade] = field(default_factory=list)
    daily_realized_pnl: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.current_cash = self.initial_bankroll

    def open_position(self, signal: dict, fill_price: float) -> Position:
        shares = signal["size_usd"] / fill_price
        cost = shares * fill_price
        self.current_cash -= cost
        pos = Position(
            market_id=signal["market_id"],
            token_id=signal["token_id"],
            side=signal["side"],
            city=signal["city"],
            station_icao=signal["station_icao"],
            date=signal["date"],
            bucket=signal["bucket"],
            fill_price=fill_price,
            fill_size=shares,
            cost=cost,
            timestamp=datetime.now(UTC),
        )
        self.positions.append(pos)
        return pos

    def resolve_position(self, position: Position, won: bool) -> ClosedTrade:
        payout = position.fill_size if won else 0.0
        pnl = payout - position.cost
        self.current_cash += payout

        closed = ClosedTrade(**position.__dict__, won=won, pnl=pnl)
        self.trade_history.append(closed)
        self.positions.remove(position)

        key = datetime.now(UTC).date().isoformat()
        self.daily_realized_pnl[key] = self.daily_realized_pnl.get(key, 0.0) + pnl
        return closed

    def active_exposure(self) -> float:
        return sum(p.cost for p in self.positions)

    def equity(self) -> float:
        # Mark-to-market omitted; conservative equity uses current cash only.
        return self.current_cash

    def max_drawdown_pct(self) -> float:
        curve = [self.initial_bankroll]
        running = self.initial_bankroll
        for d in sorted(self.daily_realized_pnl):
            running += self.daily_realized_pnl[d]
            curve.append(running)
        peak = curve[0]
        max_dd = 0.0
        for v in curve:
            peak = max(peak, v)
            if peak > 0:
                max_dd = max(max_dd, (peak - v) / peak)
        return max_dd

    def stats(self) -> dict[str, float]:
        if not self.trade_history:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_drawdown": self.max_drawdown_pct(),
                "sharpe_ratio": 0.0,
                "roi": 0.0,
            }

        pnls = [t.pnl for t in self.trade_history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        avg = mean(pnls)
        std = pstdev(pnls) if len(pnls) > 1 else 0.0
        sharpe = (avg / std) * (252**0.5) if std > 0 else 0.0
        total_pnl = sum(pnls)
        return {
            "total_trades": float(len(pnls)),
            "win_rate": len(wins) / len(pnls),
            "total_pnl": total_pnl,
            "avg_win": mean(wins) if wins else 0.0,
            "avg_loss": mean(losses) if losses else 0.0,
            "max_drawdown": self.max_drawdown_pct(),
            "sharpe_ratio": sharpe,
            "roi": total_pnl / self.initial_bankroll if self.initial_bankroll else 0.0,
        }
