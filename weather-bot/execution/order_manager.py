"""Order manager for paper and live execution."""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import CLOB_API_URL, PRACTICAL_MIN_ORDER_USD

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
except ImportError:  # pragma: no cover - dependency may be absent in dev.
    ClobClient = None
    OrderArgs = None
    OrderType = None


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    fill_price: float
    size_usd: float
    details: dict


class OrderManager:
    def __init__(
        self,
        live_trading: bool,
        api_key: str | None = None,
        private_key: str | None = None,
        wallet_address: str | None = None,
    ) -> None:
        self.live_trading = live_trading
        self.client = None
        if live_trading:
            if not all([api_key, private_key, wallet_address]):
                raise ValueError("Missing credentials for live trading mode.")
            if ClobClient is None:
                raise RuntimeError("py-clob-client is required for live trading.")
            self.client = ClobClient(
                host=CLOB_API_URL,
                key=api_key,
                chain_id=137,
                signature_type=1,
                funder=wallet_address,
                private_key=private_key,
            )

    def place_order(self, signal: dict) -> ExecutionResult:
        if signal["size_usd"] < PRACTICAL_MIN_ORDER_USD:
            return ExecutionResult(
                status="skipped_too_small",
                fill_price=0.0,
                size_usd=signal["size_usd"],
                details={"reason": "practical_min_order_floor"},
            )

        price = signal["market_prob"] if signal["side"] == "BUY_YES" else (1.0 - signal["market_prob"])
        price = min(max(price, 0.01), 0.99)

        if not self.live_trading:
            return ExecutionResult(
                status="paper_fill",
                fill_price=price,
                size_usd=signal["size_usd"],
                details={"mode": "paper"},
            )

        order_args = OrderArgs(
            token_id=signal["token_id"],
            price=price,
            size=signal["size_usd"] / price,
            side="BUY",
        )
        signed = self.client.create_order(order_args)
        response = self.client.post_order(signed, OrderType.GTC)
        return ExecutionResult(
            status=response.get("status", "submitted"),
            fill_price=price,
            size_usd=signal["size_usd"],
            details=response,
        )
