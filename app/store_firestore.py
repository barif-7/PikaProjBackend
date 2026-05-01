"""
Firestore-backed storage implementations for Pika backend.

These replace the on-disk JSON stores when PERSISTENCE_BACKEND=firestore,
enabling stateless Cloud Run instances that scale horizontally without shared
local state.

Firestore schema
----------------
pikaUsers/{user_id}
    user_id, provider, google_sub, email, display_name, photo_url,
    created_at, updated_at

pikaSessions/{session_token}
    session_token, user_id, created_at, expires_at

pikaProviderConnections/{user_id}
    ollama: { endpoint_url, model, api_token, label, updated_at }

pikaConversations/{user_id}/conversations/{conversation_id}
    conversation_id, summary, voice_profile_id, messages

Collection names are configurable via Settings so staging and production
can share a GCP project without collisions.

Thread-safety
-------------
Firestore SDK calls are synchronous and thread-safe.  The lazy client init
uses a simple guard; in the async FastAPI context each call should be wrapped
in asyncio.to_thread (see main.py) to avoid blocking the event loop — this is
a follow-up improvement tracked in PERSISTENCE.md.
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .auth import AuthError

# google-cloud-firestore is a production dependency.  We import lazily inside
# _db() so that (a) the module loads cleanly in dev environments where the
# package is not installed and (b) tests can mock the client without needing
# the real SDK present.
try:
    from google.cloud import firestore as _firestore_module
except ImportError:  # pragma: no cover — only missing in stripped dev envs
    _firestore_module = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Shared time helpers (duplicated from auth.py to avoid importing private fns)
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_after_seconds_iso(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _is_expired(iso_timestamp: Optional[str]) -> bool:
    if not iso_timestamp:
        return False
    try:
        return datetime.fromisoformat(iso_timestamp) <= datetime.now(timezone.utc)
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# FirestoreAuthStore
# ---------------------------------------------------------------------------

class FirestoreAuthStore:
    """
    Firestore implementation of the AuthStoreProtocol.

    Satisfies the same public interface as app/auth.py:AuthStore so that
    main.py can hold a reference typed as AuthStoreProtocol and swap
    backends transparently.
    """

    def __init__(
        self,
        *,
        session_ttl_seconds: float,
        users_collection: str = "pikaUsers",
        sessions_collection: str = "pikaSessions",
        connections_collection: str = "pikaProviderConnections",
        gcp_project: Optional[str] = None,
    ) -> None:
        self.session_ttl_seconds = session_ttl_seconds
        self.users_collection = users_collection
        self.sessions_collection = sessions_collection
        self.connections_collection = connections_collection
        self._gcp_project = gcp_project
        self._client: Optional[Any] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _db(self) -> Any:
        if self._client is None:
            if _firestore_module is None:
                raise RuntimeError(
                    "google-cloud-firestore is not installed. "
                    "Add it to requirements.txt or set PERSISTENCE_BACKEND=json."
                )
            kwargs: dict[str, Any] = {}
            if self._gcp_project:
                kwargs["project"] = self._gcp_project
            self._client = _firestore_module.Client(**kwargs)
        return self._client

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def create_google_session(
        self,
        *,
        google_sub: str,
        email: str,
        display_name: str,
        photo_url: Optional[str] = None,
    ) -> dict[str, Any]:
        user_id = f"google:{google_sub}"
        now = _utc_now_iso()
        expires_at = _utc_after_seconds_iso(self.session_ttl_seconds)
        session_token = secrets.token_urlsafe(32)

        db = self._db()

        # Upsert user — preserve original created_at on repeat sign-ins.
        user_ref = db.collection(self.users_collection).document(user_id)
        user_snap = user_ref.get()
        created_at = (user_snap.to_dict() or {}).get("created_at", now) if user_snap.exists else now

        user_ref.set({
            "user_id": user_id,
            "provider": "google",
            "google_sub": google_sub,
            "email": email,
            "display_name": display_name,
            "photo_url": photo_url,
            "created_at": created_at,
            "updated_at": now,
        })

        db.collection(self.sessions_collection).document(session_token).set({
            "session_token": session_token,
            "user_id": user_id,
            "created_at": now,
            "expires_at": expires_at,
        })

        return self.session_response(session_token)

    def session_response(self, session_token: str) -> dict[str, Any]:
        db = self._db()

        session_ref = db.collection(self.sessions_collection).document(session_token)
        session_snap = session_ref.get()
        if not session_snap.exists:
            raise AuthError("Unknown session.")

        session = session_snap.to_dict() or {}
        if _is_expired(session.get("expires_at")):
            session_ref.delete()
            raise AuthError("Session expired.")

        session["expires_at"] = _utc_after_seconds_iso(self.session_ttl_seconds)
        session_ref.set({"expires_at": session["expires_at"]}, merge=True)

        user_snap = db.collection(self.users_collection).document(session["user_id"]).get()
        if not user_snap.exists:
            raise AuthError("Session user is missing.")

        user = user_snap.to_dict() or {}
        return {
            "sessionToken": session_token,
            "expiresAt": session.get("expires_at"),
            "user": {
                "userId": user["user_id"],
                "email": user["email"],
                "displayName": user["display_name"],
                "photoURL": user.get("photo_url"),
            },
        }

    def refresh_session(self, session_token: str) -> dict[str, Any]:
        return self.session_response(session_token)

    def revoke_session(self, session_token: str) -> None:
        # delete() is a no-op on Firestore if the doc does not exist, so this
        # is already idempotent.
        self._db().collection(self.sessions_collection).document(session_token).delete()

    # ------------------------------------------------------------------
    # Provider connections
    # ------------------------------------------------------------------

    def save_ollama_connection(
        self,
        *,
        user_id: str,
        endpoint_url: str,
        model: Optional[str],
        api_token: Optional[str],
        label: Optional[str],
    ) -> dict[str, Any]:
        self._db().collection(self.connections_collection).document(user_id).set(
            {
                "ollama": {
                    "endpoint_url": endpoint_url,
                    "model": model,
                    "api_token": api_token,
                    "label": label,
                    "updated_at": _utc_now_iso(),
                }
            },
            merge=True,
        )
        return self.ollama_connection_response(user_id)

    def ollama_connection_response(self, user_id: str) -> dict[str, Any]:
        ollama = self.ollama_connection(user_id)
        if not ollama:
            raise AuthError("No Ollama connection configured for this user.")
        return {
            "endpointURL": ollama["endpoint_url"],
            "model": ollama.get("model"),
            "hasAPIToken": bool(ollama.get("api_token")),
            "label": ollama.get("label"),
            "updatedAt": ollama.get("updated_at"),
        }

    def ollama_connection(self, user_id: str) -> Optional[dict[str, Any]]:
        snap = self._db().collection(self.connections_collection).document(user_id).get()
        if not snap.exists:
            return None
        return (snap.to_dict() or {}).get("ollama")


# ---------------------------------------------------------------------------
# FirestoreConversationStore
# ---------------------------------------------------------------------------

class FirestoreConversationStore:
    """
    Firestore implementation of the ConversationStoreProtocol.

    Conversations are stored as subcollection documents:
      {collection}/{user_id}/conversations/{conversation_id}

    This keeps per-user data isolated and allows efficient per-user queries
    without needing a composite index.
    """

    def __init__(
        self,
        *,
        collection: str = "pikaConversations",
        gcp_project: Optional[str] = None,
    ) -> None:
        self.collection = collection
        self._gcp_project = gcp_project
        self._client: Optional[Any] = None

    def _db(self) -> Any:
        if self._client is None:
            if _firestore_module is None:
                raise RuntimeError(
                    "google-cloud-firestore is not installed. "
                    "Add it to requirements.txt or set PERSISTENCE_BACKEND=json."
                )
            kwargs: dict[str, Any] = {}
            if self._gcp_project:
                kwargs["project"] = self._gcp_project
            self._client = _firestore_module.Client(**kwargs)
        return self._client

    def _doc_ref(self, user_id: str, conversation_id: str) -> Any:
        return (
            self._db()
            .collection(self.collection)
            .document(user_id)
            .collection("conversations")
            .document(conversation_id)
        )

    def fetch(self, *, user_id: str, conversation_id: str = "default") -> dict[str, Any]:
        snap = self._doc_ref(user_id, conversation_id).get()
        if not snap.exists:
            return _default_conversation_payload(conversation_id)
        data = snap.to_dict()
        return data if data else _default_conversation_payload(conversation_id)

    def save(
        self,
        *,
        user_id: str,
        conversation_id: str,
        summary: Optional[str],
        voice_profile_id: Optional[str],
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "conversation_id": conversation_id,
            "summary": summary or "",
            "voice_profile_id": voice_profile_id,
            "messages": messages,
        }
        self._doc_ref(user_id, conversation_id).set(payload)
        return payload


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _default_conversation_payload(conversation_id: str) -> dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "summary": "",
        "voice_profile_id": None,
        "messages": [],
    }


# ---------------------------------------------------------------------------
# Factories — called from make_auth_store / make_conversation_store when
# PERSISTENCE_BACKEND=firestore.
# ---------------------------------------------------------------------------

def make_firestore_auth_store(settings: Any) -> FirestoreAuthStore:
    return FirestoreAuthStore(
        session_ttl_seconds=settings.auth_session_ttl_seconds,
        users_collection=settings.auth_users_collection,
        sessions_collection=settings.auth_sessions_collection,
        connections_collection=settings.auth_connections_collection,
    )


def make_firestore_conversation_store(settings: Any) -> FirestoreConversationStore:
    return FirestoreConversationStore(
        collection=settings.conversations_collection,
    )
