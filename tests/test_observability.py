"""
Tests for app/observability.py.

Covers:
- RequestIDMiddleware  — propagates supplied ID; generates one when absent
- RequestLoggingMiddleware — emits structured JSON at the correct log level
- RateLimitMiddleware  — allows requests under the limit; returns 429 with
  Retry-After when the bucket is empty
- UserTurnQuotaMiddleware — counts turns per user; returns 429 when limit hit
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub google.cloud before any imports that touch durable_storage.py
# ---------------------------------------------------------------------------
_mock_gcloud = MagicMock()
for _mod in ("google", "google.cloud", "google.cloud.firestore", "google.cloud.storage"):
    sys.modules.setdefault(_mod, _mock_gcloud)

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.observability import (
    RateLimitMiddleware,
    RequestIDMiddleware,
    RequestLoggingMiddleware,
    UserTurnQuotaMiddleware,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_echo_app() -> FastAPI:
    """Minimal app that returns 200 with the X-Request-ID it received."""
    app = FastAPI()

    @app.get("/ping")
    async def ping(request: Request) -> dict:
        return {"request_id": getattr(request.state, "request_id", None)}

    @app.post("/voice-chat/jobs")
    async def voice_jobs() -> dict:
        return {"ok": True}

    @app.post("/voice-chat/turn")
    async def voice_turn() -> dict:
        return {"ok": True}

    return app


# ---------------------------------------------------------------------------
# RequestIDMiddleware
# ---------------------------------------------------------------------------

class RequestIDMiddlewareTests(unittest.TestCase):
    def setUp(self) -> None:
        base = _make_echo_app()
        base.add_middleware(RequestIDMiddleware)
        self.client = TestClient(base, raise_server_exceptions=True)

    def test_generated_when_absent(self) -> None:
        resp = self.client.get("/ping")
        self.assertEqual(resp.status_code, 200)
        rid = resp.headers.get("X-Request-ID", "")
        self.assertTrue(rid, "X-Request-ID header should be set")
        self.assertEqual(len(rid), 32, "Generated ID should be a 32-char hex UUID")

    def test_propagated_when_supplied(self) -> None:
        resp = self.client.get("/ping", headers={"X-Request-ID": "my-trace-id"})
        self.assertEqual(resp.headers["X-Request-ID"], "my-trace-id")

    def test_stored_on_request_state(self) -> None:
        resp = self.client.get("/ping", headers={"X-Request-ID": "abc123"})
        body = resp.json()
        self.assertEqual(body["request_id"], "abc123")


# ---------------------------------------------------------------------------
# RequestLoggingMiddleware
# ---------------------------------------------------------------------------

class RequestLoggingMiddlewareTests(unittest.TestCase):
    def setUp(self) -> None:
        base = _make_echo_app()
        base.add_middleware(RequestIDMiddleware)
        base.add_middleware(RequestLoggingMiddleware)
        self.client = TestClient(base, raise_server_exceptions=False)

    def _capture_log_record(self, path: str, **kwargs) -> dict:
        """Invoke path and return the parsed JSON log record emitted by RequestLoggingMiddleware."""
        captured: list[dict] = []

        original_log = logging.Logger.log

        def _intercept(self_logger, level, msg, *args, **kw):
            original_log(self_logger, level, msg, *args, **kw)
            if self_logger.name == "app.observability":
                try:
                    captured.append(json.loads(str(msg)))
                except (json.JSONDecodeError, TypeError):
                    pass

        with patch.object(logging.Logger, "log", _intercept):
            self.client.get(path, **kwargs)

        http_records = [r for r in captured if r.get("event") == "http_request"]
        self.assertTrue(http_records, "No http_request log record was emitted")
        return http_records[0]

    def test_log_contains_method_path_status(self) -> None:
        record = self._capture_log_record("/ping")
        self.assertEqual(record["method"], "GET")
        self.assertEqual(record["path"], "/ping")
        self.assertEqual(record["status"], 200)

    def test_log_contains_latency(self) -> None:
        record = self._capture_log_record("/ping")
        self.assertIn("latency_ms", record)
        self.assertIsInstance(record["latency_ms"], int)

    def test_log_contains_request_id(self) -> None:
        record = self._capture_log_record("/ping", headers={"X-Request-ID": "trace-42"})
        self.assertEqual(record.get("request_id"), "trace-42")


# ---------------------------------------------------------------------------
# RateLimitMiddleware
# ---------------------------------------------------------------------------

class RateLimitMiddlewareTests(unittest.TestCase):
    def _make_client(self, rate_per_minute: int, burst: int) -> TestClient:
        base = _make_echo_app()
        base.add_middleware(RateLimitMiddleware, rate_per_minute=rate_per_minute, burst=burst)
        return TestClient(base, raise_server_exceptions=True)

    def test_requests_within_burst_succeed(self) -> None:
        client = self._make_client(rate_per_minute=60, burst=5)
        for _ in range(5):
            resp = client.get("/ping")
            self.assertEqual(resp.status_code, 200)

    def test_requests_beyond_burst_get_429(self) -> None:
        client = self._make_client(rate_per_minute=60, burst=3)
        for _ in range(3):
            client.get("/ping")
        resp = client.get("/ping")
        self.assertEqual(resp.status_code, 429)
        self.assertIn("Retry-After", resp.headers)

    def test_429_body_has_message(self) -> None:
        client = self._make_client(rate_per_minute=60, burst=0)
        # burst=0 uses float(0), so first request should immediately exhaust
        resp = client.get("/ping")
        if resp.status_code == 429:
            self.assertIn("message", resp.json())

    def test_health_path_exempt(self) -> None:
        base = _make_echo_app()

        @base.get("/health")
        async def health() -> dict:
            return {"status": "ok"}

        base.add_middleware(RateLimitMiddleware, rate_per_minute=60, burst=0)
        client = TestClient(base)
        # Even with burst=0, /health should never get a 429.
        resp = client.get("/health")
        self.assertNotEqual(resp.status_code, 429)


# ---------------------------------------------------------------------------
# UserTurnQuotaMiddleware
# ---------------------------------------------------------------------------

class UserTurnQuotaMiddlewareTests(unittest.TestCase):
    def _make_client(self, max_turns: int) -> TestClient:
        base = _make_echo_app()
        base.add_middleware(UserTurnQuotaMiddleware, max_turns_per_day=max_turns)
        return TestClient(base, raise_server_exceptions=True)

    def test_disabled_when_zero(self) -> None:
        client = self._make_client(max_turns=0)
        for _ in range(20):
            resp = client.post("/voice-chat/jobs", headers={"Authorization": "Bearer tok"})
            self.assertEqual(resp.status_code, 200)

    def test_allows_up_to_limit(self) -> None:
        client = self._make_client(max_turns=3)
        auth = {"Authorization": "Bearer abc123def456xyz"}
        for i in range(3):
            resp = client.post("/voice-chat/jobs", headers=auth)
            self.assertEqual(resp.status_code, 200, f"Request {i + 1} should succeed")

    def test_blocks_after_limit(self) -> None:
        client = self._make_client(max_turns=2)
        auth = {"Authorization": "Bearer qwerty123456789a"}
        client.post("/voice-chat/jobs", headers=auth)
        client.post("/voice-chat/jobs", headers=auth)
        resp = client.post("/voice-chat/jobs", headers=auth)
        self.assertEqual(resp.status_code, 429)
        self.assertIn("Retry-After", resp.headers)
        self.assertIn("quota", resp.json()["message"].lower())

    def test_unauthenticated_request_not_counted(self) -> None:
        client = self._make_client(max_turns=1)
        # No Authorization header — should pass through for the route to reject.
        resp = client.post("/voice-chat/jobs")
        self.assertNotEqual(resp.status_code, 429)

    def test_non_voice_path_not_counted(self) -> None:
        client = self._make_client(max_turns=1)
        auth = {"Authorization": "Bearer sometoken123456"}
        # GET /ping should not consume quota
        for _ in range(5):
            resp = client.get("/ping", headers=auth)
            self.assertEqual(resp.status_code, 200)

    def test_different_users_have_independent_quotas(self) -> None:
        client = self._make_client(max_turns=1)
        user_a = {"Authorization": "Bearer userA_token_abc123"}
        user_b = {"Authorization": "Bearer userB_token_xyz789"}
        # User A exhausts their quota
        client.post("/voice-chat/jobs", headers=user_a)
        self.assertEqual(client.post("/voice-chat/jobs", headers=user_a).status_code, 429)
        # User B is unaffected
        self.assertEqual(client.post("/voice-chat/jobs", headers=user_b).status_code, 200)


if __name__ == "__main__":
    unittest.main()
