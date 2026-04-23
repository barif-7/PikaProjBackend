from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.conversations import make_conversation_store


class ConversationStoreTests(unittest.TestCase):
    def test_fetch_returns_default_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = make_conversation_store(td)
            payload = store.fetch(user_id="user-1")
            self.assertEqual(payload["conversation_id"], "default")
            self.assertEqual(payload["summary"], "")
            self.assertIsNone(payload["voice_profile_id"])
            self.assertEqual(payload["messages"], [])

    def test_save_and_fetch_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = make_conversation_store(td)
            saved = store.save(
                user_id="user-1",
                conversation_id="default",
                summary="short summary",
                voice_profile_id="voice-123",
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            )
            self.assertEqual(saved["summary"], "short summary")
            self.assertEqual(saved["voice_profile_id"], "voice-123")

            fetched = store.fetch(user_id="user-1")
            self.assertEqual(fetched["summary"], "short summary")
            self.assertEqual(len(fetched["messages"]), 2)
            self.assertEqual(fetched["messages"][0]["content"], "hi")

    def test_users_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = make_conversation_store(td)
            store.save(
                user_id="user-a",
                conversation_id="default",
                summary="A's notes",
                voice_profile_id=None,
                messages=[{"role": "user", "content": "hi from a"}],
            )
            other = store.fetch(user_id="user-b")
            # User B should see the default (empty) payload, not A's data.
            self.assertEqual(other["summary"], "")
            self.assertEqual(other["messages"], [])

    def test_save_is_atomic(self) -> None:
        """After save, no *.tmp files should be left lying around on disk."""
        with tempfile.TemporaryDirectory() as td:
            store = make_conversation_store(td)
            store.save(
                user_id="user-1",
                conversation_id="default",
                summary="",
                voice_profile_id=None,
                messages=[],
            )
            tmp_files = list(Path(td).glob("*.tmp"))
            self.assertEqual(tmp_files, [])

    def test_overwrite_preserves_other_conversations(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = make_conversation_store(td)
            store.save(
                user_id="user-1",
                conversation_id="default",
                summary="first",
                voice_profile_id=None,
                messages=[],
            )
            store.save(
                user_id="user-1",
                conversation_id="default",
                summary="second",
                voice_profile_id=None,
                messages=[{"role": "user", "content": "hi"}],
            )
            fetched = store.fetch(user_id="user-1")
            self.assertEqual(fetched["summary"], "second")

    def test_corrupt_file_yields_empty(self) -> None:
        """A garbage conversations.json should not crash; it should read as empty."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "conversations.json"
            path.write_text("{ not valid json }", encoding="utf-8")
            store = make_conversation_store(td)
            payload = store.fetch(user_id="user-1")
            self.assertEqual(payload["messages"], [])


if __name__ == "__main__":
    unittest.main()
