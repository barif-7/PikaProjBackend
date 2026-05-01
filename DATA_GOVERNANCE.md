# Data Governance

This document describes voice data ownership, retention, deletion, and access controls added in Phase 6 of the North America scaling work.

## Data Classification

| Data type | Sensitivity | Storage location | Retention |
|---|---|---|---|
| User profile (email, name, photo) | PII | Firestore `pikaUsers` | Until account deletion |
| Session tokens | Auth credential | Firestore `pikaSessions` | TTL: `AUTH_SESSION_TTL_SECONDS` (default 30 days) |
| Provider connections (Ollama endpoint) | Configuration | Firestore `pikaProviderConnections` | Until account deletion |
| Conversation history | User-generated content | Firestore `pikaConversations` | Until account deletion |
| Voice training samples | **Biometric data** | GCS `voice-profiles/{id}/samples/` | Until profile deletion |
| Voice profile artifacts (models, adapters) | **Biometric-derived** | GCS `voice-profiles/{id}/artifacts/` | Until profile deletion |
| Voice chat audio (turn recordings) | **Biometric data** | In-memory / ephemeral only | Not persisted |
| Voice chat response audio | Generated content | In-memory / job store (TTL) | `VOICE_JOB_TTL_SECONDS` (default 10 min) |

**Voice data is biometric.** Training samples and derived voice models must be treated with the same care as fingerprints or face scans. Do not retain, re-share, or use them outside of personalized TTS without explicit user consent.

## Voice Profile Ownership

Every voice profile carries a `user_id` field in its manifest. The ownership rules are:

- **Owned profiles** (`user_id` is set): Only the owning user can read, use, or delete the profile. Requests from other users get HTTP 403.
- **Anonymous profiles** (`user_id` is `null`): Any caller can read or delete them. This mode is provided for local dev/testing only and should not be used in production.

Ownership is enforced in:
- `GET /voice-profiles/{id}` (status check)
- `POST /voice-chat/turn` and `POST /voice-chat/jobs` (using the profile in a turn)
- `DELETE /voice-profiles/{id}` (deletion)

## Deleting a Voice Profile

```
DELETE /voice-profiles/{profile_id}
Authorization: Bearer <token>
```

**Response (200):**
```json
{ "status": "deleted", "profileId": "voice-profile-abc123" }
```

This permanently removes:
- Local manifest JSON (`data/profiles/{id}.json`)
- Local sample audio file (`data/samples/`)
- Local model artifacts (`.onnx`, `.wav`, `.adapter`, CosyVoice model dir)
- Firestore profile document (`voiceProfiles/{id}`)
- Firestore job documents for this profile (`voiceProfileJobs` where `profile_id == id`)
- All GCS objects under `voice-profiles/{id}/`

**Errors:**
- `401` — not authenticated
- `403` — authenticated but not the profile owner
- `404` — profile not found

## Deleting an Account

```
DELETE /auth/account
Authorization: Bearer <token>
```

**Response (200):**
```json
{ "status": "deleted", "userId": "google:123", "deletedProfiles": 2 }
```

This is a **non-reversible** operation. It permanently removes:

1. All voice profiles and their artifacts (same as calling `DELETE /voice-profiles/{id}` for each)
2. All conversation history for the user
3. The user record, all sessions, and provider connections

The bearer token used to call this endpoint is also invalidated, so the client cannot make further authenticated requests.

**Note:** If the user has no voice profiles, `deletedProfiles` is 0.

## Voice Chat Audio

Turn recordings sent to `POST /voice-chat/turn` or `POST /voice-chat/jobs` are:
- Decoded in-process for transcription
- **Never written to GCS or any persistent store**
- Discarded once the turn is complete (or when the async job is evicted after `VOICE_JOB_TTL_SECONDS`)

Operators must NOT modify the pipeline to persist raw turn audio without first obtaining explicit per-recording consent from users.

## Retention Configuration

| Env var | Default | Description |
|---|---|---|
| `AUTH_SESSION_TTL_SECONDS` | `2592000` (30 days) | Sessions expire after this many seconds of inactivity |
| `VOICE_JOB_TTL_SECONDS` | `600` (10 min) | Async voice-chat job results are evicted after this TTL |

There is currently no automated expiry for voice profiles — they live until explicitly deleted by the user or an operator. A future phase should add `VOICE_PROFILE_RETENTION_DAYS` and a scheduled Cloud Scheduler job that calls `DELETE /voice-profiles/{id}` for profiles older than that threshold.

## Encryption

- **In transit**: All Cloud Run traffic is served over TLS 1.2+. Enforced by GCP load balancer.
- **At rest**: GCS buckets and Firestore collections inherit the project-level encryption (Google-managed keys by default). For compliance requirements, configure CMEK via the GCP Console before writing any voice data.

## Access Controls (Production Checklist)

- [ ] GCS bucket for voice profiles: `allUsers` access removed, service account IAM roles only
- [ ] Firestore: production collection rules deny unauthenticated reads/writes (Cloud Run service account handles all server-side access)
- [ ] Cloud Run: deployed with `--no-allow-unauthenticated` + `REQUIRE_API_KEY=1` or IAP
- [ ] GCS bucket-level audit logging enabled (Admin Read + Data Read events)
- [ ] Firestore audit logging enabled in Cloud Audit Logs

## Remaining Gaps

- **Automated profile expiry**: No `VOICE_PROFILE_RETENTION_DAYS` policy yet. Add a Cloud Scheduler trigger that queries old Firestore manifests and calls the delete endpoint.
- **Deletion confirmation email**: After `DELETE /auth/account` succeeds, notify the user at their registered email address.
- **GDPR data export**: No `GET /auth/account/export` endpoint yet for data portability.
- **Audit log for deletions**: The deletion events are logged to Cloud Logging but not to a dedicated audit trail. Add a Firestore `deletionAuditLog` collection with `user_id`, `deleted_at`, and `resource_type` for compliance evidence.
