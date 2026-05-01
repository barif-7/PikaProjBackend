from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from app.auth import (
    AuthError,
    AuthStore,
    _encode_state,
    _is_expired,
    append_query_params,
    decode_google_state,
    extract_bearer_token,
)


def _make_store(tmpdir: Path, ttl: float = 3600.0) -> AuthStore:
    return AuthStore(root_dir=tmpdir, session_ttl_seconds=ttl)


class AuthStoreTests(unittest.TestCase):
    def test_create_and_read_session(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = _make_store(Path(td))
            store.create_google_session(
                google_sub="abc123",
                email="ada@example.com",
                display_name="Ada Lovelace",
                photo_url="https://example.com/a.png",
            )
            # Now fetch by session token (we need to inspect sessions.json for it)
            sessions_path = Path(td) / "sessions.json"
            sessions = json.loads(sessions_path.read_text("utf-8"))
            self.assertEqual(len(sessions), 1)
            (session_token,) = sessions.keys()

            response = store.session_response(session_token)
            self.assertEqual(response["sessionToken"], session_token)
            self.assertEqual(response["user"]["userId"], "google:abc123")
            self.assertEqual(response["user"]["email"], "ada@example.com")
            self.assertEqual(response["user"]["displayName"], "Ada Lovelace")
            self.assertEqual(response["user"]["photoURL"], "https://example.com/a.png")

    def test_repeat_sign_in_preserves_created_at(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = _make_store(Path(td))
            store.create_google_session(
                google_sub="abc", email="e@x.com", display_name="E",
            )
            users = json.loads((Path(td) / "users.json").read_text("utf-8"))
            first_created_at = users["google:abc"]["created_at"]

            # Sleep a tick so updated_at would drift if misbehaving.
            time.sleep(0.01)
            store.create_google_session(
                google_sub="abc", email="e2@x.com", display_name="E2",
            )
            users_after = json.loads((Path(td) / "users.json").read_text("utf-8"))
            self.assertEqual(users_after["google:abc"]["created_at"], first_created_at)
            self.assertEqual(users_after["google:abc"]["email"], "e2@x.com")

    def test_revoke_session_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = _make_store(Path(td))
            store.create_google_session(
                google_sub="abc", email="e@x.com", display_name="E",
            )
            sessions = json.loads((Path(td) / "sessions.json").read_text("utf-8"))
            (token,) = sessions.keys()

            store.revoke_session(token)
            store.revoke_session(token)  # second revoke must not raise
            with self.assertRaises(AuthError):
                store.session_response(token)

    def test_session_expires(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = _make_store(Path(td))
            store.create_google_session(
                google_sub="abc", email="e@x.com", display_name="E",
            )
            sessions_path = Path(td) / "sessions.json"
            sessions = json.loads(sessions_path.read_text("utf-8"))
            (token,) = sessions.keys()
            sessions[token]["expires_at"] = "2000-01-01T00:00:00+00:00"
            sessions_path.write_text(json.dumps(sessions, indent=2), encoding="utf-8")

            with self.assertRaises(AuthError):
                store.session_response(token)

    def test_unknown_session_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = _make_store(Path(td))
            with self.assertRaises(AuthError):
                store.session_response("not-a-real-token")

    def test_ollama_connection_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = _make_store(Path(td))
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
            self.assertEqual(resp["label"], "home")

            # The wire response must never leak the api_token itself.
            self.assertNotIn("apiToken", resp)
            self.assertNotIn("api_token", resp)

    def test_missing_ollama_connection_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = _make_store(Path(td))
            with self.assertRaises(AuthError):
                store.ollama_connection_response("google:nobody")

    def test_atomic_save_leaves_no_tmp_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = _make_store(Path(td))
            store.create_google_session(
                google_sub="abc", email="e@x.com", display_name="E",
            )
            files = {p.name for p in Path(td).iterdir()}
            # Final filenames only — .tmp must have been renamed away.
            self.assertNotIn("users.json.tmp", files)
            self.assertNotIn("sessions.json.tmp", files)


class GoogleOAuthStateTests(unittest.TestCase):
    _SECRET = "test-secret-for-state-signing-!!!"

    def test_encode_decode_round_trip_no_secret(self) -> None:
        """Legacy path: decode_google_state without a secret still works for bare payloads."""
        payload = {"nonce": "xyz", "mobile_callback": "pikatakehome://cb"}
        encoded = _encode_state(payload)
        decoded = decode_google_state(encoded)  # no secret → legacy decode
        self.assertEqual(decoded, payload)

    def test_signed_encode_decode_round_trip(self) -> None:
        """Signed state: encode via build_google_authorize_url logic, decode with secret."""
        from app.security import sign_oauth_state
        payload = {"nonce": "abc", "mobile_callback": "pikatakehome://cb"}
        encoded = _encode_state(payload)
        signed = sign_oauth_state(encoded, self._SECRET)
        decoded = decode_google_state(signed, secret=self._SECRET)
        self.assertEqual(decoded, payload)

    def test_tampered_signed_state_raises(self) -> None:
        from app.security import sign_oauth_state
        payload = {"nonce": "abc", "mobile_callback": "pikatakehome://cb"}
        encoded = _encode_state(payload)
        signed = sign_oauth_state(encoded, self._SECRET)
        tampered = "X" + signed[1:]
        with self.assertRaises(AuthError):
            decode_google_state(tampered, secret=self._SECRET)

    def test_wrong_secret_raises(self) -> None:
        from app.security import sign_oauth_state
        payload = {"nonce": "abc", "mobile_callback": "pikatakehome://cb"}
        encoded = _encode_state(payload)
        signed = sign_oauth_state(encoded, self._SECRET)
        with self.assertRaises(AuthError):
            decode_google_state(signed, secret="wrong-secret")

    def test_decode_rejects_garbage(self) -> None:
        with self.assertRaises(AuthError):
            decode_google_state("this is not base64!!!")

    def test_session_response_includes_expires_at(self) -> None:
        """session_response dict must now include expiresAt so the client can track TTL."""
        with tempfile.TemporaryDirectory() as td:
            store = _make_store(Path(td))
            store.create_google_session(
                google_sub="xyz", email="e@x.com", display_name="E",
            )
            sessions_path = Path(td) / "sessions.json"
            sessions = json.loads(sessions_path.read_text("utf-8"))
            (token,) = sessions.keys()

            resp = store.session_response(token)
            self.assertIn("expiresAt", resp)
            self.assertIsNotNone(resp["expiresAt"])

    def test_session_response_slides_expiry_forward(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = _make_store(Path(td))
            store.create_google_session(
                google_sub="xyz", email="e@x.com", display_name="E",
            )
            sessions_path = Path(td) / "sessions.json"
            sessions_before = json.loads(sessions_path.read_text("utf-8"))
            (token,) = sessions_before.keys()
            before = sessions_before[token]["expires_at"]

            time.sleep(0.01)
            resp = store.session_response(token)
            after = resp["expiresAt"]
            self.assertGreater(after, before)


class ExpiryParserTests(unittest.TestCase):
    def test_future_not_expired(self) -> None:
        future = "9999-01-01T00:00:00+00:00"
        self.assertFalse(_is_expired(future))

    def test_past_expired(self) -> None:
        past = "2000-01-01T00:00:00+00:00"
        self.assertTrue(_is_expired(past))

    def test_none_or_blank_not_expired(self) -> None:
        self.assertFalse(_is_expired(None))
        self.assertFalse(_is_expired(""))


class BearerTokenExtractionTests(unittest.TestCase):
    def test_happy_path(self) -> None:
        self.assertEqual(extract_bearer_token("Bearer abc123"), "abc123")
        # Case-insensitive scheme.
        self.assertEqual(extract_bearer_token("bearer xyz"), "xyz")

    def test_missing_header_raises(self) -> None:
        with self.assertRaises(AuthError):
            extract_bearer_token(None)

    def test_non_bearer_scheme_raises(self) -> None:
        with self.assertRaises(AuthError):
            extract_bearer_token("Basic ZTp4")

    def test_empty_token_raises(self) -> None:
        with self.assertRaises(AuthError):
            extract_bearer_token("Bearer ")


class AppendQueryParamsTests(unittest.TestCase):
    def test_adds_new_param(self) -> None:
        result = append_query_params("pikatakehome://cb", {"session_token": "abc"})
        self.assertEqual(result, "pikatakehome://cb?session_token=abc")

    def test_merges_into_existing_query(self) -> None:
        result = append_query_params(
            "pikatakehome://cb?foo=1",
            {"session_token": "abc"},
        )
        self.assertIn("foo=1", result)
        self.assertIn("session_token=abc", result)

    def test_skips_empty_values(self) -> None:
        result = append_query_params(
            "pikatakehome://cb",
            {"error": "", "error_description": "boom"},
        )
        self.assertNotIn("error=", result)
        self.assertIn("error_description=boom", result)


if __name__ == "__main__":
    unittest.main()
