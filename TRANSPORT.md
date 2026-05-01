# Transport & Client Contract

This document describes the Phase 4 transport improvements for the Pika backend and iOS client.

## Audio Upload Flow

### Current path (still fully supported)

Audio is base64-encoded and embedded directly in the JSON request body:

```json
POST /voice-chat/jobs
{
  "audioBase64": "<base64>",
  "mimeType": "audio/wav",
  "fileName": "turn.wav",
  "durationSeconds": 4.2,
  ...
}
```

For audio longer than 15 seconds, the iOS client automatically splits the file into `audioChunks` (see `AudioChunking.swift`).

### New path — pre-upload then reference

For clients that want to separate the upload from the job submission (e.g., to pre-buffer audio while the user is still reviewing), audio can be uploaded via multipart first:

**Step 1: Upload audio**
```
POST /audio/uploads
Content-Type: multipart/form-data; boundary=...

--boundary
Content-Disposition: form-data; name="file"; filename="turn.wav"
Content-Type: audio/wav

<binary audio>
--boundary
Content-Disposition: form-data; name="durationSeconds"

4.2
--boundary--
```

**Response:**
```json
{
  "uploadId": "a3f7b291c04e...",
  "expiresInSeconds": 300
}
```

**Step 2: Submit job with the upload reference**
```json
POST /voice-chat/jobs
{
  "audioUploadID": "a3f7b291c04e...",
  "fileName": "turn.wav",
  "durationSeconds": 4.2,
  ...
}
```

The server looks up the upload by ID, resolves the bytes, and processes the job identically to an inline submission.

### Upload constraints

| Env var | Default | Description |
|---|---|---|
| `AUDIO_UPLOAD_TTL_SECONDS` | `300` (5 min) | Seconds before an unused upload is evicted |
| `MAX_AUDIO_BASE64_BYTES` | `52428800` (50 MB) | Ceiling for the audio file size (applies to both paths) |

Uploads are single-use — the server claims and removes the entry when the job is submitted. The store is in-process memory (per instance). For multi-instance Cloud Run deployments, replace with a short-lived GCS object + signed URL (see **Remaining Gaps** below).

## iOS Configuration

### Unified base URL

Both the voice-chat and voice-training clients now resolve their base URL from a single source: `PikaBackendConfiguration`.

| Priority | Source |
|---|---|
| 1 | `PIKA_BACKEND_BASE_URL` process environment variable |
| 2 | `PikaBackendBaseURL` Info.plist key |
| 3 | `VOICE_CHAT_BASE_URL` (legacy fallback) |
| 4 | `VOICE_TRAINING_BASE_URL` (legacy fallback) |
| 5 | `http://127.0.0.1:8080` (simulator only) |

### API Key

When the backend is deployed with `REQUIRE_API_KEY=1`, the iOS client must include an `X-API-Key` header. Set it via:

| Priority | Source |
|---|---|
| 1 | `PIKA_API_KEY` process environment variable |
| 2 | `PikaAPIKey` Info.plist key |

All HTTP services (`HTTPMessagesVoiceChatService`, `HTTPVoiceProfileTrainingService`, `HTTPMessagesConversationService`) now call `configuration.decorate(&request)` which adds `X-API-Key` when a key is configured.

## Job Polling — Exponential Backoff

The iOS voice-chat job poller previously used a fixed 1-second poll interval. It now uses exponential backoff with jitter:

| Parameter | Value |
|---|---|
| Initial interval | 0.5 s |
| Backoff factor | ×2 per poll |
| Maximum interval | 8 s |
| Jitter | ±10 % of current interval |
| Maximum total wait | 180 s (`requestTimeout`) |

Typical poll sequence for a 3-second recording: 0.5s → 1s → 2s → ready. This cuts unnecessary poll requests by ~4× compared to a fixed 1s interval.

## Remaining Gaps

- **GCS signed-URL upload**: The `POST /audio/uploads` store is in-process memory. For deployments with multiple Cloud Run instances or for uploads that outlive a single request path, replace with:
  1. `POST /audio/upload-urls` — returns a GCS signed PUT URL + an opaque `objectId`
  2. Client PUTs binary audio directly to GCS
  3. Client passes `audioObjectId` in the job submission
  4. Server reads audio from GCS before running the pipeline
- **Streaming response**: Long LLM responses block the full turn. A future phase should stream partial text back to the iOS client so it can start displaying text before TTS completes.
- **Session token refresh on 401**: The iOS client does not automatically retry with a refreshed token when it receives HTTP 401. Add a retry interceptor in `HTTPMessagesVoiceChatService` and `HTTPVoiceProfileTrainingService` that calls `POST /auth/session/refresh` and retries once on 401.
