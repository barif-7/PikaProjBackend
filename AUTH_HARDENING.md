# Auth Hardening — Phase 2

## What was changed

### 1. SSRF protection for user-supplied Ollama endpoints (`app/security.py`)

`PUT /provider-connections/ollama` previously accepted any URL, allowing an
attacker to turn the backend into an SSRF proxy that reaches internal services
(GCP metadata server, Cloud Run service accounts, VPC-internal hosts).

**Fix:** `validate_ollama_endpoint()` rejects URLs that:
- Use a non-HTTP/HTTPS scheme
- Target loopback addresses (127.0.0.0/8, ::1)
- Target RFC 1918 private ranges (10/8, 172.16/12, 192.168/16)
- Target link-local / GCP+AWS metadata endpoints (169.254.0.0/16)
- Target CGNAT shared space (100.64.0.0/10)
- Match known dangerous hostnames (`localhost`, `metadata.google.internal`, etc.)
- Exceed 2048 characters

**Optional allowlist (`OLLAMA_ENDPOINT_ALLOWLIST`):** Set to a comma-separated
list of URL prefixes (e.g., `https://ollama.prod.example.com`) to enforce an
allowlist in addition to the blocklist.  This is the stronger control and
**recommended for production** because it eliminates DNS rebinding risk.

Note: DNS rebinding is not fully mitigated by hostname checks alone.  Combine
with VPC egress rules or the allowlist for full protection.

### 2. HMAC-signed OAuth state parameter (`app/security.py`, `app/auth.py`)

The `state` query parameter in the Google OAuth redirect flow previously
contained only a base64-encoded payload with no integrity protection.  An
attacker who intercepted the redirect could modify `mobile_callback` in the
state to point to an attacker-controlled URL, leaking the authorization code.

**Fix:** The state is now `{base64_payload}.{hmac_sha256_hex}`.
- `build_google_authorize_url` signs the state with `OAUTH_STATE_SECRET`.
- `decode_google_state` verifies the signature before extracting the payload.
- Tampered states raise `AuthError` and return HTTP 400.

**Required:** Set `OAUTH_STATE_SECRET` to a strong random value (≥ 32 bytes)
in Cloud Run environment variables.  If unset, a per-process random secret is
used, which breaks in-flight OAuth flows on container restart.

### 3. Path traversal fix in `fileName` fields (`app/models.py`)

`VoiceChatTurnRequest.fileName` and `VoiceProfileSubmitRequest.fileName` were
used to construct file paths inside a temp directory:

```python
audio_path = temp_path / payload.fileName   # VULNERABLE — could escape temp dir
```

A crafted `fileName` like `../../etc/passwd` would resolve to `/etc/passwd`.

**Fix:** A Pydantic `field_validator` calls `Path(v).name` which strips all
directory components, returning only the final filename component.  Characters
outside `[\w\-. ]` are rejected.

### 4. Audio payload size limits (`app/models.py`, `app/main.py`)

Audio base64 fields had no size limits.  An attacker could send arbitrarily
large payloads to exhaust memory and CPU.

**Hard ceiling** (model validator, not configurable):
- `VoiceChatTurnRequest.audioBase64`: 200 MB
- `VoiceProfileSubmitRequest.audioBase64`: 200 MB

**Configurable limit** (checked at the route level before processing):
- `MAX_AUDIO_BASE64_BYTES` env var (default 50 MB, ~37.5 MB raw audio)
- Exceeded payloads return HTTP 413

### 5. Session `expiresAt` in auth response (`app/auth.py`, `app/store_firestore.py`, `app/models.py`)

The session response did not include the expiry timestamp, so the iOS client
had no way to show a "session expires soon" warning or proactively refresh.

**Fix:** `expiresAt` (ISO-8601 UTC string) is now included in every session
response and in the `AuthSessionResponse` Pydantic model.

### 6. Optional API key middleware (`app/main.py`)

When `REQUIRE_API_KEY=1` and `API_KEY=<secret>` are set, every request (except
`/health` and `/auth/google/*`) must include the header `X-API-Key: <secret>`.

This provides a lightweight service-level guard when Cloud Run is deployed
with `--allow-unauthenticated`.  It is **not** a substitute for Cloud Run IAM
auth (see below).

### 7. Mobile callback scheme validation hardened (`app/auth.py`)

- `AUTH_MOBILE_CALLBACK_SCHEME` is now required.  If empty, `build_google_authorize_url` raises `AuthConfigurationError` instead of silently accepting any scheme.
- Web schemes (`http`, `https`, `javascript`, `data`, `vbscript`) are explicitly rejected even if they somehow matched the expected scheme.

---

## Environment variable reference

| Variable | Default | Description |
|---|---|---|
| `OAUTH_STATE_SECRET` | *(random per-process)* | HMAC-SHA256 key for OAuth state signing. **Required in production.** |
| `OLLAMA_ENDPOINT_ALLOWLIST` | *(empty — blocklist only)* | Comma-separated URL prefixes users may set as Ollama endpoints. Recommended for production. |
| `MAX_AUDIO_BASE64_BYTES` | `52428800` (50 MB) | Configurable limit on audio payloads. |
| `REQUIRE_API_KEY` | `0` | Set to `1` to enforce X-API-Key on all non-exempt routes. |
| `API_KEY` | *(unset)* | The expected API key value when `REQUIRE_API_KEY=1`. |

---

## Remaining risks and upgrade paths

### Cloud Run IAM — remove `--allow-unauthenticated`

The current `cloudbuild.yaml` deploys with `--allow-unauthenticated`, making
the service publicly reachable.  The `REQUIRE_API_KEY` guard above is a
stopgap.

**Proper fix:** Switch to Cloud Run IAM-based auth:

1. Change `_ALLOW_UNAUTHENTICATED_FLAG` in `cloudbuild.yaml` to
   `--no-allow-unauthenticated`.
2. Put a trusted token-minting layer in front of Cloud Run, or switch the API
   auth model so the native app does not need direct service-account style
   credentials.  A native iOS app should not embed a Cloud Run invoker secret
   or service account key.
3. Only after that architecture exists should the client attach whatever
   signed bearer the gateway / auth layer expects.

This is a client-side change that requires an iOS app update, so coordinate
with the mobile team before deploying.

### Universal links instead of custom URL schemes

The current OAuth flow uses a custom URL scheme (`pikatakehome://`) for the
mobile callback.  Custom schemes can be registered by any app on the device —
a malicious app can intercept the OAuth callback redirect and steal the auth
code.

**Groundwork now in place:**
- The backend now serves `/.well-known/apple-app-site-association`.
- The iOS app can now consume a full redirect URL via `AUTH_REDIRECT_URL` /
  `AuthRedirectURL`, so the auth flow no longer assumes a custom-scheme-only
  callback.

**Still required:** complete the Universal Links / App Links rollout, which is
verified by Apple/Google against `apple-app-site-association` /
`assetlinks.json` files hosted on the backend.

Migration steps:
1. Add `/.well-known/apple-app-site-association` endpoint to this backend
   with the app's Team ID and bundle identifier.
2. Update the iOS app to register the backend domain as a Universal Link.
3. Update the app to use the backend-domain redirect URL instead of the custom
   scheme fallback.
4. This is a coordinated change requiring app review re-submission.

### Bearer token lifecycle

Current TTL is still 30 days (`AUTH_SESSION_TTL_SECONDS=2592000`), but the
backend now extends `expires_at` on authenticated session reads and exposes
`POST /auth/session/refresh` so the client can proactively renew the session.

**Follow-up:** tune the refresh threshold on the client and shorten the server
TTL if you want a more aggressive session policy.

### Ollama API token storage

The user's Ollama `api_token` is stored in plaintext in Firestore /
`provider-connections.json`.

**Recommended:** Encrypt at rest using Cloud KMS.  The Firestore backend
should store `kms_encrypted_api_token` and decrypt on read using the service
account's KMS key.

### Token revocation — multi-session aware

`DELETE /auth/session` revokes only the presented session token.  If the user
has signed in from multiple devices, the other sessions remain active.

**Recommended:** Add `DELETE /auth/sessions/all` that deletes all sessions for
the authenticated user_id.

---

## Tests validated

All new security controls have unit tests in:
- `tests/test_security.py` — SSRF blocklist, allowlist, HMAC state round-trip, tampering
- `tests/test_auth.py` — signed state encode/decode, tampered state rejection, `expiresAt` in response

Run with:
```bash
python3 -m pytest tests/test_security.py tests/test_auth.py -v
```
