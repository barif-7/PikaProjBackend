from __future__ import annotations

"""
Observability middleware.

Provides:
- RequestIDMiddleware   — assigns/propagates X-Request-ID on every request
- RequestLoggingMiddleware — structured JSON logging for each request
- RateLimitMiddleware   — token-bucket per-IP rate limiter
- UserTurnQuotaMiddleware — per-user daily voice-chat turn quota
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import date, timezone, datetime
from typing import Any, Callable, Dict

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request ID
# ---------------------------------------------------------------------------

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Propagate or assign a unique ``X-Request-ID`` on every request/response.

    The client may supply its own ID (useful for end-to-end tracing); otherwise
    a fresh UUID hex is generated.  The value is stored on ``request.state.request_id``
    so it can be read by downstream middleware and route handlers.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = (
            request.headers.get("X-Request-ID", "").strip() or uuid.uuid4().hex
        )
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# Structured request logging
# ---------------------------------------------------------------------------

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Emit one structured JSON log line per request.

    Fields: event, method, path, status, latency_ms, request_id (when available).
    Responses ≥ 500 are logged at WARNING; everything else at INFO.

    Paths that are health / internal probes are logged at DEBUG to avoid noise.
    """

    _QUIET_PATHS = ("/health", "/.well-known/")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        latency_ms = round((time.monotonic() - start) * 1_000)

        record: Dict[str, Any] = {
            "event": "http_request",
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": latency_ms,
        }
        request_id = getattr(request.state, "request_id", None)
        if request_id:
            record["request_id"] = request_id

        is_quiet = any(request.url.path.startswith(p) for p in self._QUIET_PATHS)
        if is_quiet:
            level = logging.DEBUG
        elif response.status_code >= 500:
            level = logging.WARNING
        else:
            level = logging.INFO

        logger.log(level, json.dumps(record))
        return response


# ---------------------------------------------------------------------------
# Per-IP rate limiting (token bucket)
# ---------------------------------------------------------------------------

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token-bucket rate limiter keyed by client IP.

    Each IP starts with ``burst`` tokens.  Tokens refill at
    ``rate_per_minute / 60`` per second.  When tokens are exhausted the
    request receives a 429 with a ``Retry-After`` header.

    Paths under /health, /.well-known/, and /auth/google/ are exempt so that
    health probes and OAuth redirects are never throttled.
    """

    _EXEMPT_PREFIXES = ("/health", "/.well-known/", "/auth/google/")

    def __init__(self, app: ASGIApp, *, rate_per_minute: int, burst: int) -> None:
        super().__init__(app)
        self._rate_per_second = rate_per_minute / 60.0
        self._burst = float(burst)
        # ip → (tokens, last_refill_monotonic)
        self._buckets: Dict[str, tuple[float, float]] = {}
        self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if any(request.url.path.startswith(p) for p in self._EXEMPT_PREFIXES):
            return await call_next(request)

        client_ip = (request.client.host if request.client else None) or "unknown"

        async with self._lock:
            now = time.monotonic()
            tokens, last_refill = self._buckets.get(client_ip, (self._burst, now))
            elapsed = now - last_refill
            tokens = min(self._burst, tokens + elapsed * self._rate_per_second)

            if tokens < 1.0:
                wait_seconds = max(1, int((1.0 - tokens) / self._rate_per_second))
                return JSONResponse(
                    status_code=429,
                    content={"message": "Rate limit exceeded. Please slow down."},
                    headers={"Retry-After": str(wait_seconds)},
                )

            self._buckets[client_ip] = (tokens - 1.0, now)

        return await call_next(request)


# ---------------------------------------------------------------------------
# Per-user daily voice-chat turn quota
# ---------------------------------------------------------------------------

class UserTurnQuotaMiddleware(BaseHTTPMiddleware):
    """
    Enforce a per-user daily voice-chat turn limit.

    Applies only to ``POST /voice-chat/jobs`` and ``POST /voice-chat/turn``.
    Counts reset at UTC midnight.  Set ``max_turns_per_day = 0`` to disable.

    The user is identified by the first 16 characters of the bearer token, which
    is a lightweight proxy for user identity — good enough for quota without
    needing a full token validation pass before the route handler.
    """

    _QUOTA_PATHS = frozenset({"/voice-chat/jobs", "/voice-chat/turn"})

    def __init__(self, app: ASGIApp, *, max_turns_per_day: int) -> None:
        super().__init__(app)
        self._max_turns = max_turns_per_day
        # (date_iso, token_prefix) → int
        self._counts: Dict[tuple[str, str], int] = defaultdict(int)
        self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if self._max_turns <= 0:
            return await call_next(request)

        if request.method != "POST" or request.url.path not in self._QUOTA_PATHS:
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            # Unauthenticated — let the route handler deal with auth; skip quota.
            return await call_next(request)

        # Use a prefix of the token as a cheap, stable per-user key.
        token_prefix = auth[7:23]  # chars 7–22 (16-char prefix after "Bearer ")
        today = datetime.now(timezone.utc).date().isoformat()
        quota_key = (today, token_prefix)

        async with self._lock:
            current = self._counts[quota_key]
            if current >= self._max_turns:
                return JSONResponse(
                    status_code=429,
                    content={
                        "message": (
                            f"Daily voice-chat quota of {self._max_turns} turns exceeded. "
                            "Quota resets at UTC midnight."
                        )
                    },
                    headers={"Retry-After": "3600"},
                )
            self._counts[quota_key] = current + 1

        return await call_next(request)
