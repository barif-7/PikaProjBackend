# Observability & Operations

This document describes the observability features added in Phase 5 of the North America scaling work.

## Request ID Propagation

Every request receives a unique `X-Request-ID` header in the response.

- If the client supplies `X-Request-ID` in the request, it is echoed back unchanged (useful for end-to-end distributed tracing).
- If the client does not supply one, a 32-character hex UUID is generated.
- The value is stored on `request.state.request_id` so it can be referenced in route handlers and logs.

```
→ GET /health
← 200 OK  X-Request-ID: a3f7b291c04e...
```

## Structured Request Logging

Every HTTP request emits one JSON log line:

```json
{
  "event": "http_request",
  "method": "POST",
  "path": "/voice-chat/jobs",
  "status": 200,
  "latency_ms": 83,
  "request_id": "a3f7b291c04e..."
}
```

Log levels:
- `DEBUG` — `/health` and `/.well-known/*` paths (noisy probes)
- `WARNING` — any response ≥ 500
- `INFO` — everything else

Cloud Run forwards stdout/stderr to Cloud Logging automatically. These JSON lines
can be queried with `jsonPayload.event="http_request"` in the Logs Explorer.

## Per-IP Rate Limiting

A token-bucket rate limiter is applied to all non-exempt paths.

| Env var | Default | Description |
|---|---|---|
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | `120` | Sustained request rate per IP per minute |
| `RATE_LIMIT_BURST` | `20` | Maximum burst size (tokens) |

Set `RATE_LIMIT_REQUESTS_PER_MINUTE=0` to disable rate limiting entirely (useful for local dev or trusted internal environments).

**Exempt paths** (never rate-limited):
- `/health`
- `/.well-known/*`
- `/auth/google/*` (OAuth redirects must always be reachable)

When the limit is exceeded the response is:
```
HTTP 429 Too Many Requests
Retry-After: <seconds>
{ "message": "Rate limit exceeded. Please slow down." }
```

## Per-User Daily Voice-Chat Quota

An optional per-user daily limit can be enforced on the voice-chat submission endpoints (`POST /voice-chat/jobs`, `POST /voice-chat/turn`).

| Env var | Default | Description |
|---|---|---|
| `MAX_TURNS_PER_USER_PER_DAY` | `0` | Max turns per user per UTC day; `0` = disabled |

When the quota is exceeded:
```
HTTP 429 Too Many Requests
Retry-After: 3600
{ "message": "Daily voice-chat quota of N turns exceeded. Quota resets at UTC midnight." }
```

Users are identified by a prefix of their bearer token, so quota is per-session-token-prefix. This is a lightweight approximation — it provides effective throttling without requiring a full token validation pass before rate-checking.

## Health Endpoint

`GET /health` now returns limit configuration alongside TTS status:

```json
{
  "status": "ok",
  "request_id": "a3f7b...",
  "tts": { ... },
  "limits": {
    "rate_limit_requests_per_minute": 120,
    "rate_limit_burst": 20,
    "max_turns_per_user_per_day": 0,
    "max_concurrent_voice_jobs": 10
  }
}
```

## Cloud Run Recommendations

```yaml
# cloud run service yaml snippet
env:
  - name: RATE_LIMIT_REQUESTS_PER_MINUTE
    value: "120"
  - name: RATE_LIMIT_BURST
    value: "20"
  - name: MAX_TURNS_PER_USER_PER_DAY
    value: "100"   # adjust per product requirement
  - name: MAX_CONCURRENT_VOICE_JOBS
    value: "10"    # tune to instance concurrency setting
```

For Cloud Monitoring dashboards, filter log entries with:
```
resource.type="cloud_run_revision"
jsonPayload.event="http_request"
jsonPayload.status>=500
```

## Remaining Gaps

- **Distributed rate limiting**: The current token-bucket state is in-process memory. With multiple Cloud Run instances, each instance tracks its own buckets — effective aggregate rate is `N × per-instance-limit`. For stricter enforcement, replace the in-process store with a Redis/Memorystore backend.
- **Prometheus / OpenTelemetry metrics**: The structured logs are queryable in Cloud Logging but no Prometheus endpoint is exposed. Add `starlette-prometheus` or `opentelemetry-instrumentation-fastapi` for metric scraping.
- **Alerting**: Add log-based alerting in Cloud Monitoring for `jsonPayload.status>=500 rate > X` and `jsonPayload.path="/voice-chat/jobs" latency_ms > 5000`.
