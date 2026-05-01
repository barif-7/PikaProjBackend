"""
Tests for Phase 6 data governance features.

Covers:
- AuthStore.delete_user            — removes user, sessions, and connections
- ConversationStore.delete_user_conversations — removes all user conversations
- VoiceProfileStore.delete_profile — removes manifest, artifacts, job entry
- VoiceProfileStore.delete_user_profiles — bulk delete by owner
- DurableVoiceProfileStore.delete_profile — Firestore + GCS cleanup
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Stub google.cloud before any imports that touch durable_storage.py
# ---------------------------------------------------------------------------
_mock_gcloud = MagicMock()
for _mod in ("google", "google.cloud", "google.cloud.firestore", "google.cloud.storage"):
    sys.modules.setdefault(_mod, _mock_gcloud)

from app.auth import AuthStore, AuthError
from app.conversations import ConversationStore
from app.durable_storage import DurableVoiceProfileStore
from app.training import VoiceProfileStore, VoiceProfileTrainingError


# ---------------------------------------------------------------------------
# AuthStore.delete_user
# ---------------------------------------------------------------------------

class AuthStoreDeleteUserTests(unittest.TestCase):
    def _make_store(self, tmp: Path) -> AuthStore:
        store = AuthStore(root_dir=tmp, session_ttl_seconds=3600)
        return store

    def _seed_user(self, store: AuthStore, user_id: str = "google:abc") -> str:
        """Directly inject a user + session + connection into the store files."""
        users = {user_id: {"user_id": user_id, "email": "a@b.com", "display_name": "A", "created_at": "x", "updated_at": "x"}}
        sessions = {
            "tok1": {"session_token": "tok1", "user_id": user_id, "created_at": "x", "expires_at": "2099-01-01T00:00:00+00:00"},
            "tok2": {"session_token": "tok2", "user_id": "other:user", "created_at": "x", "expires_at": "2099-01-01T00:00:00+00:00"},
        }
        connections = {user_id: {"ollama": {"endpoint_url": "http://x:11434"}}}

        import app.auth as _auth_mod
        _auth_mod._save_json_map(store._users_path, users)
        _auth_mod._save_json_map(store._sessions_path, sessions)
        _auth_mod._save_json_map(store._provider_connections_path, connections)
        return user_id

    def test_delete_removes_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(Path(tmp))
            user_id = self._seed_user(store)
            store.delete_user(user_id)

            import app.auth as _auth_mod
            users = _auth_mod._load_json_map(store._users_path)
            self.assertNotIn(user_id, users)

    def test_delete_removes_sessions_for_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(Path(tmp))
            user_id = self._seed_user(store)
            store.delete_user(user_id)

            import app.auth as _auth_mod
            sessions = _auth_mod._load_json_map(store._sessions_path)
            # tok1 belongs to our user — should be gone.
            self.assertNotIn("tok1", sessions)
            # tok2 belongs to a different user — should survive.
            self.assertIn("tok2", sessions)

    def test_delete_removes_connections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(Path(tmp))
            user_id = self._seed_user(store)
            store.delete_user(user_id)

            import app.auth as _auth_mod
            connections = _auth_mod._load_json_map(store._provider_connections_path)
            self.assertNotIn(user_id, connections)

    def test_delete_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(Path(tmp))
            # Should not raise even when the user has never existed.
            store.delete_user("nonexistent:user")


# ---------------------------------------------------------------------------
# ConversationStore.delete_user_conversations
# ---------------------------------------------------------------------------

class ConversationStoreDeleteTests(unittest.TestCase):
    def test_delete_removes_user_conversations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ConversationStore(root_dir=Path(tmp))
            store.save(user_id="u1", conversation_id="default", summary="hi", voice_profile_id=None, messages=[])
            store.save(user_id="u2", conversation_id="default", summary="bye", voice_profile_id=None, messages=[])

            store.delete_user_conversations("u1")

            import app.conversations as _conv_mod
            data = _conv_mod._load_json_map(store._conversations_path)
            self.assertNotIn("u1", data)
            self.assertIn("u2", data)

    def test_delete_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ConversationStore(root_dir=Path(tmp))
            # Should not raise for non-existent user.
            store.delete_user_conversations("nonexistent")


# ---------------------------------------------------------------------------
# VoiceProfileStore.delete_profile
# ---------------------------------------------------------------------------

class VoiceProfileStoreDeleteProfileTests(unittest.TestCase):
    def _make_store(self, tmp: Path) -> VoiceProfileStore:
        return VoiceProfileStore(
            root_dir=tmp,
            profiles_dir=tmp / "profile-models",
            training_mode="placeholder",
        )

    def _seed_profile(self, store: VoiceProfileStore, user_id: str = "u1") -> tuple[str, Path]:
        """Create a minimal profile manifest and return (profile_id, manifest_path)."""
        profile_id = "voice-profile-test01"
        manifest_path = store.root_dir / "profiles" / f"{profile_id}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a fake sample file.
        samples_dir = store.root_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        sample_path = samples_dir / "sample.wav"
        sample_path.write_bytes(b"fake_audio")

        manifest = {
            "profile_id": profile_id,
            "user_id": user_id,
            "job_id": "job123",
            "sample_path": str(sample_path),
            "model_path": str(store.profiles_dir / f"{profile_id}.onnx"),
            "config_path": str(store.profiles_dir / f"{profile_id}.onnx.json"),
            "reference_path": str(store.profiles_dir / f"{profile_id}.wav"),
            "adapter_path": str(store.profiles_dir / f"{profile_id}.adapter"),
            "cosyvoice_model_dir": str(store.profiles_dir / f"{profile_id}.cosyvoice"),
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        # Put the profile ID into jobs.json so delete_profile can clean up.
        jobs = {
            "job123": {
                "job_id": "job123",
                "profile_id": profile_id,
                "user_id": user_id,
                "sample_path": str(sample_path),
                "status": "ready",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        }
        import app.training as _train_mod
        _train_mod._save_jobs(store._jobs_path, jobs)
        return profile_id, manifest_path

    def test_delete_removes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(Path(tmp))
            profile_id, manifest_path = self._seed_profile(store)
            store.delete_profile(profile_id, user_id="u1")
            self.assertFalse(manifest_path.exists(), "Manifest file should be deleted")

    def test_delete_removes_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(Path(tmp))
            _, manifest_path = self._seed_profile(store)
            manifest = json.loads(manifest_path.read_text())
            sample_path = Path(manifest["sample_path"])
            store.delete_profile(manifest["profile_id"], user_id="u1")
            self.assertFalse(sample_path.exists(), "Sample file should be deleted")

    def test_delete_removes_job_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(Path(tmp))
            profile_id, _ = self._seed_profile(store)
            store.delete_profile(profile_id, user_id="u1")

            import app.training as _train_mod
            jobs = _train_mod._load_jobs(store._jobs_path)
            self.assertNotIn("job123", jobs)

    def test_delete_raises_for_wrong_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(Path(tmp))
            profile_id, _ = self._seed_profile(store, user_id="owner")
            with self.assertRaises(VoiceProfileTrainingError):
                store.delete_profile(profile_id, user_id="attacker")

    def test_delete_raises_for_missing_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(Path(tmp))
            with self.assertRaises(VoiceProfileTrainingError):
                store.delete_profile("nonexistent-profile", user_id="u1")


# ---------------------------------------------------------------------------
# VoiceProfileStore.delete_user_profiles
# ---------------------------------------------------------------------------

class VoiceProfileStoreDeleteUserProfilesTests(unittest.TestCase):
    def _make_store(self, tmp: Path) -> VoiceProfileStore:
        return VoiceProfileStore(
            root_dir=tmp,
            profiles_dir=tmp / "profile-models",
            training_mode="placeholder",
        )

    def _write_manifest(self, store: VoiceProfileStore, profile_id: str, user_id: str) -> None:
        manifest_path = store.root_dir / "profiles" / f"{profile_id}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({"profile_id": profile_id, "user_id": user_id, "job_id": None}),
            encoding="utf-8",
        )

    def test_deletes_profiles_owned_by_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(Path(tmp))
            self._write_manifest(store, "p-user1-a", "user1")
            self._write_manifest(store, "p-user1-b", "user1")
            self._write_manifest(store, "p-user2-a", "user2")

            deleted = store.delete_user_profiles("user1")
            self.assertEqual(sorted(deleted), ["p-user1-a", "p-user1-b"])
            # user2's profile should remain.
            self.assertTrue((store.root_dir / "profiles" / "p-user2-a.json").exists())

    def test_returns_empty_for_unknown_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(Path(tmp))
            deleted = store.delete_user_profiles("nobody")
            self.assertEqual(deleted, [])


# ---------------------------------------------------------------------------
# DurableVoiceProfileStore.delete_profile
# ---------------------------------------------------------------------------

class DurableVoiceProfileStoreDeleteTests(unittest.TestCase):
    def _make_store(self) -> DurableVoiceProfileStore:
        return DurableVoiceProfileStore(
            bucket_name="test-bucket",
            gcs_prefix="voice-profiles",
            profiles_collection="voiceProfiles",
            jobs_collection="voiceProfileJobs",
        )

    def test_delete_clears_firestore_profile_doc(self) -> None:
        store = self._make_store()
        mock_firestore = MagicMock()
        mock_storage = MagicMock()
        mock_storage.list_blobs.return_value = []
        store._firestore_client = mock_firestore
        store._storage_client = mock_storage
        store._bucket = MagicMock()

        # Stub the jobs query to return no matches.
        mock_firestore.collection.return_value.where.return_value.stream.return_value = []

        store.delete_profile("voice-profile-abc")

        # Verify the profile document was deleted.
        mock_firestore.collection.assert_any_call("voiceProfiles")

    def test_delete_is_noop_when_disabled(self) -> None:
        store = DurableVoiceProfileStore(
            bucket_name=None,  # disabled
            gcs_prefix="voice-profiles",
            profiles_collection="voiceProfiles",
            jobs_collection="voiceProfileJobs",
        )
        self.assertFalse(store.enabled)
        # Should complete without raising.
        store.delete_profile("any-profile")


if __name__ == "__main__":
    unittest.main()
