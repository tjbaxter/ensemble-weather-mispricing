"""Polymarket Gamma and CLOB market data utilities."""

from __future__ import annotations

import re
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from config.cities import STATIONS, find_station_for_question
from config.settings import (
    CLOB_API_URL,
    CLOB_PREFILTER_MAX_HOURS_TO_RESOLUTION,
    CLOB_PREFILTER_MIN_LIQUIDITY,
    DISCOVERY_MAX_PAGINATION_PAGES,
    ENABLE_SEARCH_FALLBACK,
    GAMMA_API_URL,
    MAX_BID_ASK_SPREAD,
    MIN_MARKET_LIQUIDITY,
    MIN_MARKET_VOLUME,
    load_clob_prefilter_priority,
    load_station_priority_filter,
)


BUCKET_PATTERN = re.compile(r"(-?\d+\s*-\s*-?\d+|-?\d+\+)")
DATE_PATTERN = re.compile(r"(\b[A-Z][a-z]{2,8}\s+\d{1,2}(?:,\s*\d{4})?\b|\d{4}-\d{2}-\d{2})")
WEATHER_KEYWORDS = ("highest temperature", "temperature in", "temperature on", "weather")


class PolymarketDataClient:
    def __init__(self, diagnostic: bool = False) -> None:
        self.http = httpx.AsyncClient(timeout=20.0)
        self.diagnostic = diagnostic
        self.priority_filter = load_station_priority_filter()
        self.clob_prefilter_priority = load_clob_prefilter_priority()
        self.reject_stats: Counter[str] = Counter()
        self.last_discovery_stats: dict[str, Any] = {}

    async def close(self) -> None:
        await self.http.aclose()

    async def discover_weather_markets(self) -> list[dict]:
        """Discover and normalize weather markets using multiple strategies."""
        self.reject_stats = Counter()
        discovered: list[dict] = []
        slugs_checked = 0
        search_hits = 0
        paginated_events = 0

        for station_icao, slug in self._iter_candidate_slugs():
            slugs_checked += 1
            event = await self._get_event_by_slug(slug)
            if event is None:
                self.reject_stats["slug_not_found"] += 1
                self._debug(f"NOT_FOUND slug={slug}")
                continue
            self._debug(f"DISCOVERED slug={slug} station={station_icao}")
            for market in event.get("markets", []):
                normalized = self._normalize_market(market, station_hint=station_icao)
                if normalized:
                    discovered.append(normalized)

        if ENABLE_SEARCH_FALLBACK:
            search_results = await self._discover_via_search()
            search_hits = len(search_results)
            discovered.extend(search_results)
        else:
            self.reject_stats["search_disabled"] += 1

        # Slug discovery is primary and typically sufficient. Only paginate when
        # diagnostics are enabled or slug pass found nothing.
        if self.diagnostic or not discovered:
            paged_results, paginated_events = await self._discover_via_events_pagination()
            discovered.extend(paged_results)
        else:
            self.reject_stats["pagination_skipped_after_slug_success"] += 1

        unique = {(m["condition_id"], m["date"]): m for m in discovered}
        out = list(unique.values())
        self.last_discovery_stats = {
            "slugs_checked": slugs_checked,
            "search_hits": search_hits,
            "paginated_events_considered": paginated_events,
            "discovered_markets": len(out),
            "reject_stats": dict(self.reject_stats),
        }
        self._debug(f"DISCOVERY_SUMMARY {self.last_discovery_stats}")
        return out

    def _normalize_market(self, raw: dict[str, Any], station_hint: str | None = None) -> dict | None:
        question = str(raw.get("question", ""))
        rules = str(raw.get("rules", "") or "")
        station_icao = station_hint or find_station_for_question(question) or find_station_for_question(rules)
        if station_icao is None or station_icao not in STATIONS:
            self.reject_stats["unknown_station"] += 1
            return None
        if STATIONS[station_icao]["priority"] not in self.priority_filter:
            self.reject_stats["priority_filtered"] += 1
            return None

        liquidity = float(raw.get("liquidity", 0.0) or 0.0)
        volume = float(raw.get("volume", 0.0) or 0.0)
        if liquidity < MIN_MARKET_LIQUIDITY or volume < MIN_MARKET_VOLUME:
            self.reject_stats["liquidity_or_volume_filtered"] += 1
            return None

        condition_id = raw.get("conditionId") or raw.get("condition_id")
        if not condition_id:
            self.reject_stats["missing_condition_id"] += 1
            return None

        end_date_iso = raw.get("endDate") or raw.get("end_date_iso") or raw.get("endDateIso")
        if not end_date_iso:
            self.reject_stats["missing_end_date"] += 1
            return None

        parsed_date = self._parse_market_date(question, end_date_iso)
        buckets = self._build_buckets_from_market(raw, question)
        if not buckets:
            self.reject_stats["missing_or_unparseable_buckets"] += 1
            return None

        self.reject_stats["accepted"] += 1
        return {
            "condition_id": condition_id,
            "question": question,
            "station_icao": station_icao,
            "city": STATIONS[station_icao]["market_label"],
            "date": parsed_date.isoformat(),
            "end_date_iso": str(end_date_iso),
            "volume": volume,
            "liquidity": liquidity,
            "buckets": buckets,
        }

    def _iter_candidate_slugs(self):
        today = datetime.now(UTC).date()
        for station_icao, station in STATIONS.items():
            city_slug = station.get("city_slug")
            if not city_slug:
                continue
            for day_offset in range(0, 3):
                d = today + timedelta(days=day_offset)
                slug = f"highest-temperature-in-{city_slug}-on-{d.strftime('%B').lower()}-{d.day}-{d.year}"
                yield station_icao, slug

    async def _get_event_by_slug(self, slug: str) -> dict[str, Any] | None:
        try:
            response = await self.http.get(f"{GAMMA_API_URL}/events/slug/{slug}")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            self.reject_stats["slug_fetch_error"] += 1
            self._debug(f"ERROR slug_fetch_error slug={slug}")
            return None

    async def _discover_via_search(self) -> list[dict]:
        discovered: list[dict] = []
        seen_ids: set[str] = set()
        for query in ("highest temperature", "temperature weather", "temperature in"):
            for page in range(0, 3):
                try:
                    response = await self.http.get(
                        f"{GAMMA_API_URL}/public-search",
                        params={
                            "q": query,
                            "limit_per_type": 100,
                            "page": page,
                            "events_status": "active",
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                except httpx.HTTPError:
                    self.reject_stats["search_fetch_error"] += 1
                    self._debug(f"ERROR search_fetch_error query={query} page={page}")
                    break

                events = payload.get("events", []) if isinstance(payload, dict) else []
                if not events:
                    break
                for event in events:
                    for market in event.get("markets", []):
                        normalized = self._normalize_market(market)
                        if normalized and normalized["condition_id"] not in seen_ids:
                            seen_ids.add(normalized["condition_id"])
                            discovered.append(normalized)

                pagination = payload.get("pagination", {}) if isinstance(payload, dict) else {}
                if not pagination.get("hasMore", False):
                    break
        return discovered

    async def _discover_via_events_pagination(self) -> tuple[list[dict], int]:
        discovered: list[dict] = []
        events_seen = 0
        for page_idx in range(DISCOVERY_MAX_PAGINATION_PAGES):
            offset = page_idx * 100
            try:
                response = await self.http.get(
                    f"{GAMMA_API_URL}/events",
                    params={
                        "active": "true",
                        "closed": "false",
                        "limit": 100,
                        "offset": offset,
                        "order": "id",
                        "ascending": "false",
                    },
                )
                response.raise_for_status()
                events = response.json()
            except httpx.HTTPError:
                self.reject_stats["events_pagination_fetch_error"] += 1
                self._debug(f"ERROR events_pagination_fetch_error page={page_idx}")
                break

            if not events:
                break
            events_seen += len(events)

            for event in events:
                blob = " ".join(
                    [
                        str(event.get("title", "")),
                        str(event.get("description", "")),
                        str(event.get("slug", "")),
                    ]
                ).lower()
                if not any(keyword in blob for keyword in WEATHER_KEYWORDS):
                    continue
                for market in event.get("markets", []):
                    market_blob = f"{market.get('question','')} {market.get('rules','')}".lower()
                    if not any(keyword in market_blob for keyword in WEATHER_KEYWORDS):
                        continue
                    normalized = self._normalize_market(market)
                    if normalized:
                        discovered.append(normalized)
        return discovered, events_seen

    def _parse_market_date(self, question: str, end_date_iso: str) -> date:
        m = DATE_PATTERN.search(question)
        if m:
            token = m.group(1).strip()
            for fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%b %d", "%B %d"):
                try:
                    parsed = datetime.strptime(token, fmt).date()
                    if parsed.year == 1900:
                        parsed = parsed.replace(year=datetime.utcnow().year)
                    return parsed
                except ValueError:
                    continue
        return datetime.fromisoformat(str(end_date_iso).replace("Z", "+00:00")).date()

    def _build_buckets_from_market(self, raw: dict[str, Any], question: str) -> dict[str, dict]:
        # Most weather events encode bucket text in the market question.
        question_bucket = self._extract_bucket_from_text(question)
        if question_bucket:
            clob_token_ids = raw.get("clobTokenIds")
            outcomes = raw.get("outcomes")
            if isinstance(clob_token_ids, str):
                try:
                    import json

                    clob_token_ids = json.loads(clob_token_ids)
                except Exception:
                    clob_token_ids = []
            if isinstance(outcomes, str):
                try:
                    import json

                    outcomes = json.loads(outcomes)
                except Exception:
                    outcomes = []

            outcome_prices = raw.get("outcomePrices")
            if isinstance(outcome_prices, str):
                try:
                    import json

                    outcome_prices = json.loads(outcome_prices)
                except Exception:
                    outcome_prices = []
            yes_price = None

            if isinstance(clob_token_ids, list) and isinstance(outcomes, list):
                yes_token_id = None
                no_token_id = None
                for idx, outcome in enumerate(outcomes):
                    if idx >= len(clob_token_ids):
                        continue
                    outcome_clean = str(outcome).strip().lower()
                    token_id = str(clob_token_ids[idx]).strip()
                    if outcome_clean == "yes":
                        yes_token_id = token_id
                        if isinstance(outcome_prices, list) and idx < len(outcome_prices):
                            try:
                                yes_price = float(outcome_prices[idx])
                            except (TypeError, ValueError):
                                yes_price = None
                    elif outcome_clean == "no":
                        no_token_id = token_id
                if yes_token_id and no_token_id:
                    return {
                        question_bucket: {
                            "yes_token_id": yes_token_id,
                            "no_token_id": no_token_id,
                            "price": yes_price if yes_price is not None else 0.0,
                            "best_bid": 0.0,
                            "best_ask": 0.0,
                            "fallback_price": yes_price if yes_price is not None else 0.0,
                        }
                    }

            tokens = raw.get("tokens", [])
            yes_token_id = None
            no_token_id = None
            for token in tokens:
                outcome = str(token.get("outcome", "")).strip().lower()
                token_id = str(token.get("token_id") or token.get("tokenId") or "")
                if not token_id:
                    continue
                if outcome == "yes":
                    yes_token_id = token_id
                elif outcome == "no":
                    no_token_id = token_id
            if yes_token_id and no_token_id:
                return {
                    question_bucket: {
                        "yes_token_id": yes_token_id,
                        "no_token_id": no_token_id,
                        "price": 0.0,
                        "best_bid": 0.0,
                        "best_ask": 0.0,
                        "fallback_price": 0.0,
                    }
                }

        tokens = raw.get("tokens", [])
        grouped: dict[str, dict[str, str]] = {}
        for token in tokens:
            outcome = str(token.get("outcome", ""))
            token_id = str(token.get("token_id") or token.get("tokenId") or "")
            if not token_id or not outcome:
                continue

            bucket_match = BUCKET_PATTERN.search(outcome)
            if not bucket_match:
                continue
            bucket = bucket_match.group(1).replace(" ", "")
            side = "yes" if "yes" in outcome.lower() else "no"
            grouped.setdefault(bucket, {})[f"{side}_token_id"] = token_id

        normalized = {}
        for bucket, ids in grouped.items():
            if "yes_token_id" in ids and "no_token_id" in ids:
                normalized[bucket] = {
                    "yes_token_id": ids["yes_token_id"],
                    "no_token_id": ids["no_token_id"],
                    "price": 0.0,
                    "best_bid": 0.0,
                    "best_ask": 0.0,
                    "fallback_price": 0.0,
                }
        return normalized

    def _extract_bucket_from_text(self, text: str) -> str | None:
        lower = text.lower()
        range_match = re.search(r"(-?\d+)\s*-\s*(-?\d+)", lower)
        if range_match:
            return f"{range_match.group(1)}-{range_match.group(2)}"

        plus_match = re.search(r"(-?\d+)\s*\+\b", lower)
        if plus_match:
            return f"{plus_match.group(1)}+"

        higher_match = re.search(r"(-?\d+)\s*(?:[°]?[fc])?\s*(?:or higher|or above|and above|or more)", lower)
        if higher_match:
            return f"{higher_match.group(1)}+"

        return None

    async def hydrate_prices(self, markets: list[dict]) -> list[dict]:
        """Populate best bid/ask-derived prices and filter wide-spread buckets."""
        hydrated: list[dict] = []
        now = datetime.now(UTC)
        token_to_book: dict[str, dict[str, Any]] = {}

        candidate_token_ids: list[str] = []
        filtered_markets: list[dict] = []
        for market in markets:
            end_dt = datetime.fromisoformat(str(market["end_date_iso"]).replace("Z", "+00:00"))
            hours_to_resolution = (end_dt - now).total_seconds() / 3600.0
            if hours_to_resolution < 0 or hours_to_resolution > CLOB_PREFILTER_MAX_HOURS_TO_RESOLUTION:
                self.reject_stats["clob_prefilter_resolution_window"] += 1
                continue
            if float(market.get("liquidity", 0.0) or 0.0) < CLOB_PREFILTER_MIN_LIQUIDITY:
                self.reject_stats["clob_prefilter_liquidity"] += 1
                continue
            station_priority = STATIONS.get(market["station_icao"], {}).get("priority", "")
            if station_priority not in self.clob_prefilter_priority:
                self.reject_stats["clob_prefilter_priority"] += 1
                continue

            filtered_markets.append(market)
            for info in market["buckets"].values():
                candidate_token_ids.append(info["yes_token_id"])

        # Batch request orderbooks to reduce API load.
        unique_token_ids = list(dict.fromkeys(candidate_token_ids))
        for chunk_start in range(0, len(unique_token_ids), 100):
            chunk = unique_token_ids[chunk_start : chunk_start + 100]
            books = await self._get_books(chunk)
            for book in books:
                token_id = str(book.get("asset_id") or book.get("token_id") or "")
                if token_id:
                    token_to_book[token_id] = book

        for market in filtered_markets:
            new_buckets = {}
            for bucket, info in market["buckets"].items():
                book = token_to_book.get(info["yes_token_id"])
                if not book:
                    self.reject_stats["book_fetch_error"] += 1
                    continue

                best_bid, best_ask = self._extract_best_bid_ask(book)
                if best_bid <= 0 or best_ask <= 0:
                    fallback = float(info.get("fallback_price", 0.0) or 0.0)
                    if 0.0 < fallback < 1.0:
                        new_buckets[bucket] = {
                            **info,
                            "price": fallback,
                            "best_bid": fallback,
                            "best_ask": fallback,
                        }
                        self.reject_stats["book_used_fallback_price"] += 1
                        continue
                    self.reject_stats["book_missing_prices"] += 1
                    continue
                spread = best_ask - best_bid
                if spread > MAX_BID_ASK_SPREAD:
                    self.reject_stats["spread_too_wide"] += 1
                    continue
                midpoint = (best_bid + best_ask) / 2.0
                new_buckets[bucket] = {
                    **info,
                    "price": midpoint,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                }
            if new_buckets:
                hydrated.append({**market, "buckets": new_buckets})
        return hydrated

    async def _get_books(self, token_ids: list[str]) -> list[dict[str, Any]]:
        if not token_ids:
            return []
        try:
            response = await self.http.post(
                f"{CLOB_API_URL}/books",
                json=[{"token_id": token_id} for token_id in token_ids],
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                return payload
            return []
        except httpx.HTTPError:
            # Fallback to legacy single-book endpoint per token if batch fails.
            out: list[dict[str, Any]] = []
            for token_id in token_ids:
                try:
                    resp = await self.http.get(f"{CLOB_API_URL}/book", params={"token_id": token_id})
                    resp.raise_for_status()
                    data = resp.json()
                    data["asset_id"] = token_id
                    out.append(data)
                except httpx.HTTPError:
                    self.reject_stats["book_fetch_error"] += 1
            return out

    def _extract_best_bid_ask(self, book: dict[str, Any]) -> tuple[float, float]:
        best_bid = float(book.get("bestBid", 0.0) or 0.0)
        best_ask = float(book.get("bestAsk", 0.0) or 0.0)
        if best_bid > 0 and best_ask > 0:
            return best_bid, best_ask

        bids = book.get("bids", [])
        asks = book.get("asks", [])
        if isinstance(bids, list) and bids:
            try:
                best_bid = max(float(b.get("price", 0.0) or 0.0) for b in bids)
            except (TypeError, ValueError):
                best_bid = 0.0
        if isinstance(asks, list) and asks:
            try:
                best_ask = min(float(a.get("price", 0.0) or 0.0) for a in asks if float(a.get("price", 0.0) or 0.0) > 0.0)
            except (TypeError, ValueError):
                best_ask = 0.0
        return best_bid, best_ask

    def _debug(self, msg: str) -> None:
        if self.diagnostic:
            print(msg)
