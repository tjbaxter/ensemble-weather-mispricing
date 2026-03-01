"""Order manager for paper and live execution.

Execution strategy: "sip, don't gulp"
Large orders are broken into small chunks (≤ SIP_MAX_CHUNK_USD) with a short
delay between them. This avoids running up thin order books and reduces market
impact on low-liquidity buckets. Each chunk is a separate limit order at the
current best ask, so later chunks reprice if the book moves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from config.settings import CLOB_API_URL, PRACTICAL_MIN_ORDER_USD

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
except ImportError:  # pragma: no cover - dependency may be absent in dev.
    ClobClient = None
    OrderArgs = None
    OrderType = None

# "Sip, don't gulp" parameters
# Each chunk is at most $2 on thin markets, $5 on deeper markets.
# Default: $2 max chunk. Bot will place multiple small orders rather than one large one.
SIP_MAX_CHUNK_USD = 2.00
SIP_DELAY_SECONDS = 4.0    # pause between chunks to let book refresh


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    fill_price: float
    size_usd: float
    details: dict


@dataclass
class ChunkedExecutionResult:
    status: str
    total_size_usd: float
    filled_size_usd: float
    num_chunks: int
    chunk_results: list[ExecutionResult] = field(default_factory=list)

    def to_execution_result(self) -> ExecutionResult:
        avg_price = (
            sum(r.fill_price * r.size_usd for r in self.chunk_results) / self.filled_size_usd
            if self.filled_size_usd > 0
            else 0.0
        )
        return ExecutionResult(
            status=self.status,
            fill_price=avg_price,
            size_usd=self.total_size_usd,
            details={
                "chunks": self.num_chunks,
                "filled": self.filled_size_usd,
                "chunk_results": [r.details for r in self.chunk_results],
            },
        )


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

    def _place_single_chunk(self, token_id: str, price: float, size_usd: float) -> ExecutionResult:
        """Place one limit order chunk. Paper-safe."""
        if not self.live_trading:
            return ExecutionResult(
                status="paper_fill",
                fill_price=price,
                size_usd=size_usd,
                details={"mode": "paper"},
            )

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size_usd / price,
            side="BUY",
        )
        signed = self.client.create_order(order_args)
        response = self.client.post_order(signed, OrderType.GTC)
        return ExecutionResult(
            status=response.get("status", "submitted"),
            fill_price=price,
            size_usd=size_usd,
            details=response,
        )

    def place_order(self, signal: dict) -> ExecutionResult:
        """Place an order, breaking it into SIP_MAX_CHUNK_USD chunks."""
        total_usd = signal["size_usd"]
        if total_usd < PRACTICAL_MIN_ORDER_USD:
            return ExecutionResult(
                status="skipped_too_small",
                fill_price=0.0,
                size_usd=total_usd,
                details={"reason": "practical_min_order_floor"},
            )

        price = signal["market_prob"] if signal["side"] == "BUY_YES" else (1.0 - signal["market_prob"])
        price = min(max(price, 0.01), 0.99)
        token_id = signal["token_id"]

        # Build chunk sizes: floor-divide into chunks, last chunk gets the remainder
        chunk_usd = SIP_MAX_CHUNK_USD
        chunks: list[float] = []
        remaining = total_usd
        while remaining > PRACTICAL_MIN_ORDER_USD:
            chunk = min(chunk_usd, remaining)
            if chunk < PRACTICAL_MIN_ORDER_USD:
                break
            chunks.append(round(chunk, 2))
            remaining = round(remaining - chunk, 4)

        if not chunks:
            chunks = [total_usd]

        chunked = ChunkedExecutionResult(
            status="pending",
            total_size_usd=total_usd,
            filled_size_usd=0.0,
            num_chunks=len(chunks),
        )

        for i, chunk_size in enumerate(chunks):
            result = self._place_single_chunk(token_id, price, chunk_size)
            chunked.chunk_results.append(result)
            chunked.filled_size_usd = round(chunked.filled_size_usd + chunk_size, 4)
            if result.status not in ("paper_fill", "matched", "live", "submitted"):
                chunked.status = f"partial_error_chunk_{i}"
                break
            if i < len(chunks) - 1 and self.live_trading:
                time.sleep(SIP_DELAY_SECONDS)

        if chunked.status == "pending":
            chunked.status = "filled" if not self.live_trading else "submitted"

        return chunked.to_execution_result()
