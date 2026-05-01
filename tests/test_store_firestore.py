"""
Tests for the Firestore-backed auth and conversation stores.

We mock google.cloud.firestore.Client so these tests run without a real GCP
project or network.  The goal is to verify the store logic — correct document
paths, expiry handling, idempotent revoke — not the Firestore SDK itself.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

from app.auth import AuthError
from app.store_firestore import (
    FirestoreAuthStore,
    FirestoreConversationStore,
    _is_expired,
    _utc_now_iso,
    _utc_after_seconds_iso,
)


# ---------------------------------------------------------------------------
# Helpers for building mock Firestore snapshots
# ---------------------------------------------------------------------------

def _snap(data: dict[str, Any] | None) -> MagicMock:
    """Return a mock snapshot with .exists and .to_dict()."""
    snap = MagicMock()
    snap.exists = data is not None
    snap.to_dict.return_value = data
    return snap


def _make_auth_store(**kwargs: Any) -> FirestoreAuthStore:
    return FirestoreAuthStore(
        session_ttl_seconds=kwargs.get("session_ttl_seconds", 3600.0),
        users_collection=kwargs.get("users_collection", "pikaUsers"),
        sessions_collection=kwargs.get("sessions_collection", "pikaSessions"),
        connections_collection=kwargs.get("connections_collection", "pikaProviderConnections"),
    )


def _make_convo_store(**kwargs: Any) -> FirestoreConversationStore:
    return FirestoreConversationStore(
        collection=kwargs.get("collection", "pikaConversations"),
    )


# ---------------------------------------------------------------------------
# FirestoreAuthStore tests
# ---------------------------------------------------------------------------

class FirestoreAuthStoreCreateSessionTests(unittest.TestCase):

    def test_create_session_returns_session_response(self) -> None:
        store = _make_auth_store()
        with patch("app.store_firestore._firestore_module") as mock_firestore_mod:
            mock_db = MagicMock()
            mock_firestore_mod.Client.return_value = mock_db

            # Set up full mock chain for users + sessions
            user_data = {
                "user_id": "google:sub1",
                "email": "ada@example.com",
                "display_name": "Ada Lovelace",
                "photo_url": None,
                "created_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
            }
            session_token = "fake-token-abc"
            session_data = {
                "session_token": session_token,
                "user_id": "google:sub1",
                "created_at": _utc_now_iso(),
                "expires_at": _utc_after_seconds_iso(3600),
            }

            users_doc = MagicMock()
            users_doc.get.side_effect = lambda: _snap(None)  # first call: new user
            users_doc.set = MagicMock()

            session_doc = MagicMock()
            session_doc.set = MagicMock()
            session_doc.get.return_value = _snap(session_data)

            user_doc_after = MagicMock()
            user_doc_after.get.return_value = _snap(user_data)

            call_counts: dict[str, int] = {"users": 0}

            def users_document(uid: str) -> MagicMock:
                call_counts["users"] += 1
                if call_counts["users"] == 1:
                    return users_doc  # upsert path
                return user_doc_after  # session_response lookup

            users_col = MagicMock()
            users_col.document.side_effect = users_document

            sessions_col = MagicMock()
            sessions_col.document.return_value = session_doc

            def _collection(name: str) -> MagicMock:
                if name == "pikaUsers":
                    return users_col
                if name == "pikaSessions":
                    return sessions_col
                return MagicMock()

            mock_db.collection.side_effect = _collection

            resp = store.create_google_session(
                google_sub="sub1",
                email="ada@example.com",
                display_name="Ada Lovelace",
            )
            self.assertIn("sessionToken", resp)
            self.assertEqual(resp["user"]["email"], "ada@example.com")
            self.assertEqual(resp["user"]["userId"], "google:sub1")


class FirestoreAuthStoreSessionTests(unittest.TestCase):
    def _store_with_db(self) -> tuple[FirestoreAuthStore, MagicMock]:
        store = _make_auth_store()
        mock_db = MagicMock()
        # Bypass _db() entirely by injecting the mock client directly.
        store._client = mock_db
        return store, mock_db

    def test_session_response_unknown_token_raises(self) -> None:
        store, mock_db = self._store_with_db()

        sessions_doc = MagicMock()
        sessions_doc.get.return_value = _snap(None)
        mock_db.collection.return_value.document.return_value = sessions_doc

        with self.assertRaises(AuthError):
            store.session_response("not-a-real-token")

    def test_session_response_expired_raises_and_deletes(self) -> None:
        store, mock_db = self._store_with_db()

        expired_session = {
            "session_token": "t",
            "user_id": "google:x",
            "created_at": "2020-01-01T00:00:00+00:00",
            "expires_at": "2020-01-02T00:00:00+00:00",  # past
        }
        sessions_doc = MagicMock()
        sessions_doc.get.return_value = _snap(expired_session)
        sessions_doc.delete = MagicMock()

        sessions_col = MagicMock()
        sessions_col.document.return_value = sessions_doc
        mock_db.collection.return_value = sessions_col

        with self.assertRaises(AuthError):
            store.session_response("t")
        sessions_doc.delete.assert_called_once()

    def test_session_response_slides_expiry_forward(self) -> None:
        store, mock_db = self._store_with_db()

        session_data = {
            "session_token": "t",
            "user_id": "google:x",
            "created_at": _utc_now_iso(),
            "expires_at": _utc_after_seconds_iso(300),
        }
        user_data = {
            "user_id": "google:x",
            "email": "x@example.com",
            "display_name": "X",
            "photo_url": None,
        }

        sessions_doc = MagicMock()
        sessions_doc.get.return_value = _snap(session_data)
        sessions_doc.set = MagicMock()

        user_doc = MagicMock()
        user_doc.get.return_value = _snap(user_data)

        def _collection(name: str) -> MagicMock:
            col = MagicMock()
            if name == "pikaSessions":
                col.document.return_value = sessions_doc
            elif name == "pikaUsers":
                col.document.return_value = user_doc
            return col

        mock_db.collection.side_effect = _collection

        resp = store.session_response("t")
        self.assertIn("expiresAt", resp)
        sessions_doc.set.assert_called_once()

    def test_revoke_session_calls_delete(self) -> None:
        store, mock_db = self._store_with_db()

        sessions_doc = MagicMock()
        sessions_doc.delete = MagicMock()
        mock_db.collection.return_value.document.return_value = sessions_doc

        store.revoke_session("some-token")
        sessions_doc.delete.assert_called_once()

    def test_revoke_session_is_idempotent(self) -> None:
        """Calling revoke twice must not raise (Firestore delete is a no-op if doc is gone)."""
        store, mock_db = self._store_with_db()

        sessions_doc = MagicMock()
        sessions_doc.delete = MagicMock()
        mock_db.collection.return_value.document.return_value = sessions_doc

        store.revoke_session("token")
        store.revoke_session("token")
        self.assertEqual(sessions_doc.delete.call_count, 2)


class FirestoreAuthStoreOllamaTests(unittest.TestCase):
    def _store_with_db(self) -> tuple[FirestoreAuthStore, MagicMock]:
        store = _make_auth_store()
        mock_db = MagicMock()
        store._client = mock_db
        return store, mock_db

    def test_ollama_connection_returns_none_when_absent(self) -> None:
        store, mock_db = self._store_with_db()

        conn_doc = MagicMock()
        conn_doc.get.return_value = _snap(None)
        mock_db.collection.return_value.document.return_value = conn_doc

        result = store.ollama_connection("google:nobody")
        self.assertIsNone(result)

    def test_save_and_retrieve_ollama_connection(self) -> None:
        store, mock_db = self._store_with_db()

        stored: dict = {}

        def _set(data: dict, merge: bool = False) -> None:
            stored.update(data)

        conn_doc = MagicMock()
        conn_doc.set.side_effect = _set
        conn_doc.get.side_effect = lambda: _snap(stored if stored else None)
        mock_db.collection.return_value.document.return_value = conn_doc

        store.save_ollama_connection(
            user_id="google:abc",
            endpoint_url="https://ollama.example.com",
            model="mistral",
            api_token="secret",
            label="home",
        )

        resp = store.ollama_connection_response("google:abc")
        self.assertEqual(resp["endpointURL"], "https://ollama.example.com")
        self.assertEqual(resp["model"], "mistral")
        self.assertTrue(resp["hasAPIToken"])
        self.assertNotIn("apiToken", resp)
        self.assertNotIn("api_token", resp)

    def test_missing_ollama_connection_raises(self) -> None:
        store, mock_db = self._store_with_db()

        conn_doc = MagicMock()
        conn_doc.get.return_value = _snap(None)
        mock_db.collection.return_value.document.return_value = conn_doc

        with self.assertRaises(AuthError):
            store.ollama_connection_response("google:nobody")


# ---------------------------------------------------------------------------
# FirestoreConversationStore tests
# ---------------------------------------------------------------------------

class FirestoreConversationStoreTests(unittest.TestCase):
    def _store_with_db(self) -> tuple[FirestoreConversationStore, MagicMock]:
        store = _make_convo_store()
        mock_db = MagicMock()
        store._client = mock_db
        return store, mock_db

    def _doc_ref(self, mock_db: MagicMock) -> MagicMock:
        """Return the terminal document mock at the end of the subcollection chain."""
        return (
            mock_db
            .collection.return_value
            .document.return_value
            .collection.return_value
            .document.return_value
        )

    def test_fetch_returns_default_when_missing(self) -> None:
        store, mock_db = self._store_with_db()
        self._doc_ref(mock_db).get.return_value = _snap(None)

        payload = store.fetch(user_id="u1")
        self.assertEqual(payload["conversation_id"], "default")
        self.assertEqual(payload["messages"], [])
        self.assertIsNone(payload["voice_profile_id"])

    def test_save_and_fetch_round_trip(self) -> None:
        store, mock_db = self._store_with_db()

        saved_data: dict = {}

        doc_ref = self._doc_ref(mock_db)
        doc_ref.set.side_effect = lambda d: saved_data.update(d)
        doc_ref.get.side_effect = lambda: _snap(saved_data if saved_data else None)

        saved = store.save(
            user_id="u1",
            conversation_id="default",
            summary="hello",
            voice_profile_id="vp1",
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertEqual(saved["summary"], "hello")
        self.assertEqual(saved["voice_profile_id"], "vp1")

        fetched = store.fetch(user_id="u1")
        self.assertEqual(fetched["summary"], "hello")
        self.assertEqual(len(fetched["messages"]), 1)

    def test_subcollection_path_is_correct(self) -> None:
        """Verify the store uses {collection}/{user_id}/conversations/{convo_id}."""
        store, mock_db = self._store_with_db()
        self._doc_ref(mock_db).get.return_value = _snap(None)

        store.fetch(user_id="google:user1", conversation_id="default")

        mock_db.collection.assert_called_with("pikaConversations")
        mock_db.collection.return_value.document.assert_called_with("google:user1")
        (mock_db.collection.return_value.document.return_value
             .collection.assert_called_with("conversations"))
        (mock_db.collection.return_value.document.return_value
             .collection.return_value.document.assert_called_with("default"))


# ---------------------------------------------------------------------------
# Shared utility tests
# ---------------------------------------------------------------------------

class ExpiryHelperTests(unittest.TestCase):
    def test_future_not_expired(self) -> None:
        self.assertFalse(_is_expired("9999-01-01T00:00:00+00:00"))

    def test_past_expired(self) -> None:
        self.assertTrue(_is_expired("2000-01-01T00:00:00+00:00"))

    def test_none_not_expired(self) -> None:
        self.assertFalse(_is_expired(None))

    def test_blank_not_expired(self) -> None:
        self.assertFalse(_is_expired(""))


# ---------------------------------------------------------------------------
# Helper to support mock doc creation used in create_session test
# ---------------------------------------------------------------------------

def _make_session_doc(token: str, get_fn: Any) -> MagicMock:
    doc = MagicMock()
    doc.set = MagicMock()
    doc.delete = MagicMock()
    doc.get.return_value = get_fn(token)
    return doc


def _make_user_doc(uid: str, initial_doc: MagicMock, get_fn: Any) -> MagicMock:
    doc = MagicMock()
    doc.set = MagicMock()
    doc.get.return_value = get_fn(uid)
    return doc


if __name__ == "__main__":
    unittest.main()
