"""
Storage protocol interfaces for Pika backend.

These protocols define the contracts that both the JSON file backends and
Firestore-backed backends must satisfy.  Application code in main.py should
type-hint against these protocols rather than the concrete implementations so
that the two backends are interchangeable at runtime.

Concrete implementations:
  JSON  (local dev / fallback) — app/auth.py:AuthStore
                                  app/conversations.py:ConversationStore
  Firestore (production)       — app/store_firestore.py:FirestoreAuthStore
                                  app/store_firestore.py:FirestoreConversationStore
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class AuthStoreProtocol(Protocol):
    """
    Contract for authentication, session, and provider-connection persistence.

    Both AuthStore (JSON) and FirestoreAuthStore implement this interface so
    that main.py and callers can hold a single typed reference regardless of
    which backend is active.
    """

    def create_google_session(
        self,
        *,
        google_sub: str,
        email: str,
        display_name: str,
        photo_url: Optional[str] = None,
    ) -> dict[str, Any]:
        """Upsert the Google user and create a new session token.  Returns a
        session-response dict with keys ``sessionToken`` and ``user``."""
        ...

    def session_response(self, session_token: str) -> dict[str, Any]:
        """Validate the token and return the session-response dict.
        Raises ``AuthError`` if the token is unknown or expired."""
        ...

    def refresh_session(self, session_token: str) -> dict[str, Any]:
        """Extend the session TTL and return the updated session-response dict.
        Raises ``AuthError`` if the token is unknown or expired."""
        ...

    def revoke_session(self, session_token: str) -> None:
        """Delete the session.  Must be idempotent (no error if already gone)."""
        ...

    def save_ollama_connection(
        self,
        *,
        user_id: str,
        endpoint_url: str,
        model: Optional[str],
        api_token: Optional[str],
        label: Optional[str],
    ) -> dict[str, Any]:
        """Persist the user's custom Ollama endpoint.  Returns a
        connection-response dict (never includes the raw api_token)."""
        ...

    def ollama_connection_response(self, user_id: str) -> dict[str, Any]:
        """Return the sanitised connection dict.
        Raises ``AuthError`` if no connection exists for the user."""
        ...

    def ollama_connection(self, user_id: str) -> Optional[dict[str, Any]]:
        """Return the raw connection record or ``None`` if absent."""
        ...


@runtime_checkable
class ConversationStoreProtocol(Protocol):
    """
    Contract for conversation-state persistence.

    Both ConversationStore (JSON) and FirestoreConversationStore implement
    this interface.
    """

    def fetch(self, *, user_id: str, conversation_id: str = "default") -> dict[str, Any]:
        """Return the stored conversation or a default empty payload."""
        ...

    def save(
        self,
        *,
        user_id: str,
        conversation_id: str,
        summary: Optional[str],
        voice_profile_id: Optional[str],
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Persist and return the updated conversation payload."""
        ...
