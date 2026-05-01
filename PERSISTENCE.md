# Persistence Layer — Architecture and Migration Guide

## Overview

The Pika backend supports two persistence backends for auth/session and
conversation state.  The backend is selected at startup via the
`PERSISTENCE_BACKEND` environment variable:

| Value        | Storage                  | Suitable for                            |
|-------------|--------------------------|------------------------------------------|
| `json`       | Local JSON files on disk | Local development, single-instance dev   |
| `firestore`  | Google Cloud Firestore   | Production, horizontal Cloud Run scaling |

Voice-profile artifacts (models, adapters, reference audio) are stored in
**GCS + Firestore** regardless of this setting — that path was already
production-grade.  The `PERSISTENCE_BACKEND` setting governs only the
auth/session/conversation stores.

---

## JSON Backend (default)

### File layout

```
data/
  auth/
    users.json                  # keyed by user_id ("google:<sub>")
    sessions.json               # keyed by session_token
    provider-connections.json   # keyed by user_id → { ollama: { … } }
  conversations/
    conversations.json          # keyed by user_id → { conversation_id: { … } }
```

### Limitations that prevent horizontal scaling

- All Cloud Run instances share no file system — each instance writes to its
  own in-container path, so sessions created on instance A are invisible to
  instance B.
- Cloud Run is ephemeral: container replacement discards all local state.

### Acceptable uses

- Local development (`uvicorn app.main:app --reload`)
- Single-instance staging where state loss on redeploy is acceptable

---

## Firestore Backend (production)

### Firestore schema

```
pikaUsers/{user_id}
  user_id          string   "google:<google_sub>"
  provider         string   "google"
  google_sub       string
  email            string
  display_name     string
  photo_url        string | null
  created_at       string   ISO-8601 UTC
  updated_at       string   ISO-8601 UTC

pikaSessions/{session_token}
  session_token    string
  user_id          string
  created_at       string   ISO-8601 UTC
  expires_at       string   ISO-8601 UTC

pikaProviderConnections/{user_id}
  ollama           map
    endpoint_url   string
    model          string | null
    api_token      string | null   (stored encrypted in a future iteration)
    label          string | null
    updated_at     string   ISO-8601 UTC

pikaConversations/{user_id}/conversations/{conversation_id}
  conversation_id  string
  summary          string
  voice_profile_id string | null
  messages         array of { role: string, content: string }
```

Collection names are configurable (see env vars below) so staging and
production can share a GCP project without collisions.

### Enabling Firestore

1. Ensure the Cloud Run service account has the
   **Cloud Datastore User** IAM role (or equivalent Firestore permissions).
2. Set `PERSISTENCE_BACKEND=firestore` in the Cloud Run environment.
3. (Optional) Override collection names if you need namespace isolation.

No schema migration is required for Firestore — collections are created
lazily on first write.

---

## Migrating from JSON to Firestore

Run this one-time migration script after enabling the Firestore backend to
seed the database from existing JSON files.  It is safe to run on a live
service — the script only writes documents that do not already exist.

```python
"""
migrate_json_to_firestore.py

Usage:
    GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json python migrate_json_to_firestore.py
"""
import json
from pathlib import Path
from google.cloud import firestore

DATA_DIR = Path("data")
db = firestore.Client()

# ---- Users ----
users_path = DATA_DIR / "auth" / "users.json"
if users_path.exists():
    users = json.loads(users_path.read_text())
    for user_id, user in users.items():
        ref = db.collection("pikaUsers").document(user_id)
        if not ref.get().exists:
            ref.set(user)
            print(f"  migrated user {user_id}")

# ---- Sessions ----
sessions_path = DATA_DIR / "auth" / "sessions.json"
if sessions_path.exists():
    sessions = json.loads(sessions_path.read_text())
    for token, session in sessions.items():
        ref = db.collection("pikaSessions").document(token)
        if not ref.get().exists:
            ref.set(session)
            print(f"  migrated session {token[:8]}…")

# ---- Provider connections ----
conns_path = DATA_DIR / "auth" / "provider-connections.json"
if conns_path.exists():
    conns = json.loads(conns_path.read_text())
    for user_id, conn in conns.items():
        ref = db.collection("pikaProviderConnections").document(user_id)
        if not ref.get().exists:
            ref.set(conn)
            print(f"  migrated connection for {user_id}")

# ---- Conversations ----
convos_path = DATA_DIR / "conversations" / "conversations.json"
if convos_path.exists():
    all_convos = json.loads(convos_path.read_text())
    for user_id, user_convos in all_convos.items():
        for convo_id, convo in user_convos.items():
            ref = (
                db.collection("pikaConversations")
                .document(user_id)
                .collection("conversations")
                .document(convo_id)
            )
            if not ref.get().exists:
                ref.set(convo)
                print(f"  migrated conversation {user_id}/{convo_id}")

print("Migration complete.")
```

---

## Environment Variables

| Variable                   | Default                     | Description                                  |
|----------------------------|-----------------------------|----------------------------------------------|
| `PERSISTENCE_BACKEND`      | `json`                      | `json` or `firestore`                        |
| `AUTH_USERS_COLLECTION`    | `pikaUsers`                 | Firestore collection for user records        |
| `AUTH_SESSIONS_COLLECTION` | `pikaSessions`              | Firestore collection for sessions            |
| `AUTH_CONNECTIONS_COLLECTION` | `pikaProviderConnections` | Firestore collection for provider connections |
| `CONVERSATIONS_COLLECTION` | `pikaConversations`         | Firestore root collection for conversations  |
| `AUTH_DATA_DIR`            | `data/auth`                 | JSON backend only — directory for auth files |
| `CONVERSATION_DATA_DIR`    | `data/conversations`        | JSON backend only — directory for convo file |
| `AUTH_SESSION_TTL_SECONDS` | `2592000` (30 days)         | Session lifetime in seconds                  |

---

## Rollback

To roll back from Firestore to JSON:

1. Set `PERSISTENCE_BACKEND=json` and redeploy.
2. The JSON files in `AUTH_DATA_DIR` / `CONVERSATION_DATA_DIR` will be used.
   Active Firestore sessions become invisible but users can simply sign in again.

---

## Known Gaps and Future Work

1. **Session expiry cleanup** — Expired sessions in Firestore are deleted
   lazily on read.  For large deployments, a Cloud Scheduler job or TTL policy
   should prune the `pikaSessions` collection.

2. **api_token encryption** — Ollama API tokens are stored in plaintext in
   Firestore.  Future: encrypt with Cloud KMS or store only a hashed reference.
