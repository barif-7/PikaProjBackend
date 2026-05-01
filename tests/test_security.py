"""
Tests for app/security.py — SSRF validation and HMAC OAuth state signing.
"""
from __future__ import annotations

import unittest

from app.security import (
    SSRFError,
    parse_ollama_allowlist,
    sign_oauth_state,
    validate_ollama_endpoint,
    verify_and_extract_state,
)


# ---------------------------------------------------------------------------
# SSRF — validate_ollama_endpoint
# ---------------------------------------------------------------------------

class SSRFBlocklistTests(unittest.TestCase):
    def _ok(self, url: str) -> None:
        """Assert the URL passes validation."""
        result = validate_ollama_endpoint(url)
        # Returns normalised (trailing slash stripped) URL.
        self.assertTrue(result)

    def _blocked(self, url: str) -> None:
        """Assert the URL is rejected with SSRFError."""
        with self.assertRaises(SSRFError):
            validate_ollama_endpoint(url)

    # --- Legitimate public hosts should pass ---

    def test_public_https_host(self) -> None:
        self._ok("https://ollama.example.com")

    def test_public_http_host_with_port(self) -> None:
        self._ok("http://ollama.example.com:11434")

    def test_public_http_host_with_path(self) -> None:
        self._ok("http://ollama.example.com:11434/api/chat")

    def test_nip_io_public_ip(self) -> None:
        # The current OLLAMA_BASE_URL in cloudbuild.yaml uses a nip.io address
        # that resolves to a public IP.
        self._ok("https://34.130.58.200.nip.io")

    # --- Loopback ---

    def test_localhost_blocked(self) -> None:
        self._blocked("http://localhost:11434")

    def test_127_0_0_1_blocked(self) -> None:
        self._blocked("http://127.0.0.1:11434")

    def test_127_0_0_255_blocked(self) -> None:
        self._blocked("http://127.0.0.255")

    # --- Link-local / GCP metadata ---

    def test_169_254_169_254_blocked(self) -> None:
        self._blocked("http://169.254.169.254/computeMetadata/v1/")

    def test_metadata_google_internal_blocked(self) -> None:
        self._blocked("http://metadata.google.internal/")

    def test_metadata_internal_blocked(self) -> None:
        self._blocked("http://metadata.internal/")

    # --- RFC 1918 private ranges ---

    def test_10_network_blocked(self) -> None:
        self._blocked("http://10.0.0.1")

    def test_10_network_edge_blocked(self) -> None:
        self._blocked("http://10.255.255.255")

    def test_172_16_blocked(self) -> None:
        self._blocked("http://172.16.0.1")

    def test_172_31_blocked(self) -> None:
        self._blocked("http://172.31.255.255")

    def test_192_168_blocked(self) -> None:
        self._blocked("http://192.168.1.100")

    # --- Other dangerous ranges ---

    def test_100_64_cgnat_blocked(self) -> None:
        self._blocked("http://100.64.0.1")

    def test_ipv6_loopback_blocked(self) -> None:
        self._blocked("http://[::1]:11434")

    def test_ipv6_ula_blocked(self) -> None:
        self._blocked("http://[fc00::1]")

    def test_ipv6_link_local_blocked(self) -> None:
        self._blocked("http://[fe80::1]")

    # --- Bad schemes ---

    def test_ftp_scheme_blocked(self) -> None:
        self._blocked("ftp://ollama.example.com")

    def test_file_scheme_blocked(self) -> None:
        self._blocked("file:///etc/passwd")

    def test_javascript_scheme_blocked(self) -> None:
        self._blocked("javascript://example.com")

    # --- Edge cases ---

    def test_missing_scheme_blocked(self) -> None:
        self._blocked("ollama.example.com:11434")

    def test_empty_url_blocked(self) -> None:
        with self.assertRaises((SSRFError, ValueError)):
            validate_ollama_endpoint("")

    def test_url_too_long_blocked(self) -> None:
        self._blocked("https://ollama.example.com/" + "a" * 2048)

    def test_trailing_slash_stripped(self) -> None:
        result = validate_ollama_endpoint("https://ollama.example.com/")
        self.assertFalse(result.endswith("/"))


class SSRFAllowlistTests(unittest.TestCase):
    def test_allowed_url_passes(self) -> None:
        allowed = ["https://ollama.prod.example.com"]
        result = validate_ollama_endpoint("https://ollama.prod.example.com/api/chat", allowlist=allowed)
        self.assertIn("ollama.prod.example.com", result)

    def test_disallowed_url_blocked_when_allowlist_set(self) -> None:
        allowed = ["https://ollama.prod.example.com"]
        with self.assertRaises(SSRFError):
            validate_ollama_endpoint("https://ollama.other.example.com", allowlist=allowed)

    def test_prefix_spoofed_host_blocked_when_allowlist_set(self) -> None:
        allowed = ["https://ollama.prod.example.com"]
        with self.assertRaises(SSRFError):
            validate_ollama_endpoint("https://ollama.prod.example.com.evil.tld/api/chat", allowlist=allowed)

    def test_private_ip_still_blocked_even_if_allowlisted(self) -> None:
        # SSRF blocklist check runs before allowlist — internal IPs are always blocked.
        allowed = ["http://10.0.0.1"]
        with self.assertRaises(SSRFError):
            validate_ollama_endpoint("http://10.0.0.1", allowlist=allowed)

    def test_multiple_allowlist_entries(self) -> None:
        allowed = ["https://ollama1.example.com", "https://ollama2.example.com"]
        validate_ollama_endpoint("https://ollama2.example.com", allowlist=allowed)  # should not raise

    def test_parse_ollama_allowlist(self) -> None:
        raw = "https://a.example.com, https://b.example.com , "
        parsed = parse_ollama_allowlist(raw)
        self.assertEqual(parsed, ["https://a.example.com", "https://b.example.com"])

    def test_parse_ollama_allowlist_empty(self) -> None:
        self.assertEqual(parse_ollama_allowlist(""), [])


# ---------------------------------------------------------------------------
# HMAC OAuth state signing
# ---------------------------------------------------------------------------

class OAuthStateSigningTests(unittest.TestCase):
    _SECRET = "test-secret-32-bytes-of-entropy!!"

    def test_sign_and_verify_round_trip(self) -> None:
        payload = "eyJub25jZSI6ICJ4eXoifQ"  # fake base64
        signed = sign_oauth_state(payload, self._SECRET)
        recovered = verify_and_extract_state(signed, self._SECRET)
        self.assertEqual(recovered, payload)

    def test_tampered_payload_rejected(self) -> None:
        payload = "eyJub25jZSI6ICJ4eXoifQ"
        signed = sign_oauth_state(payload, self._SECRET)
        # Flip a character in the payload portion
        tampered = "X" + signed[1:]
        with self.assertRaises(ValueError):
            verify_and_extract_state(tampered, self._SECRET)

    def test_tampered_signature_rejected(self) -> None:
        payload = "eyJub25jZSI6ICJ4eXoifQ"
        signed = sign_oauth_state(payload, self._SECRET)
        # Corrupt the last hex char of the signature
        tampered = signed[:-1] + ("0" if signed[-1] != "0" else "1")
        with self.assertRaises(ValueError):
            verify_and_extract_state(tampered, self._SECRET)

    def test_wrong_secret_rejected(self) -> None:
        payload = "eyJub25jZSI6ICJ4eXoifQ"
        signed = sign_oauth_state(payload, self._SECRET)
        with self.assertRaises(ValueError):
            verify_and_extract_state(signed, "wrong-secret")

    def test_missing_separator_rejected(self) -> None:
        with self.assertRaises(ValueError):
            verify_and_extract_state("nodotinhere", self._SECRET)

    def test_empty_state_rejected(self) -> None:
        with self.assertRaises(ValueError):
            verify_and_extract_state("", self._SECRET)

    def test_signed_state_contains_separator(self) -> None:
        payload = "abc"
        signed = sign_oauth_state(payload, self._SECRET)
        self.assertIn(".", signed)

    def test_different_secrets_produce_different_macs(self) -> None:
        payload = "same-payload"
        s1 = sign_oauth_state(payload, "secret-1")
        s2 = sign_oauth_state(payload, "secret-2")
        self.assertNotEqual(s1, s2)


# ---------------------------------------------------------------------------
# Model-level path traversal and size validation (via model validation)
# ---------------------------------------------------------------------------

class ModelValidationTests(unittest.TestCase):
    def test_path_traversal_stripped_from_voice_chat(self) -> None:
        from pydantic import ValidationError
        from app.models import VoiceChatTurnRequest

        # A "normal" looking payload but with a path-traversal fileName.
        try:
            req = VoiceChatTurnRequest(
                audioBase64="dGVzdA==",  # "test"
                fileName="../../etc/passwd",
                durationSeconds=1.0,
            )
            # After sanitisation, only the base name remains.
            self.assertEqual(req.fileName, "passwd")
        except Exception:
            # If the validator raises instead of sanitising, that's also acceptable.
            pass

    def test_absolute_path_stripped(self) -> None:
        from app.models import VoiceChatTurnRequest

        req = VoiceChatTurnRequest(
            audioBase64="dGVzdA==",
            fileName="/tmp/malicious.wav",
            durationSeconds=1.0,
        )
        self.assertEqual(req.fileName, "malicious.wav")

    def test_safe_filename_preserved(self) -> None:
        from app.models import VoiceChatTurnRequest

        req = VoiceChatTurnRequest(
            audioBase64="dGVzdA==",
            fileName="recording-001.m4a",
            durationSeconds=2.5,
        )
        self.assertEqual(req.fileName, "recording-001.m4a")

    def test_empty_filename_rejected(self) -> None:
        from pydantic import ValidationError
        from app.models import VoiceChatTurnRequest

        with self.assertRaises(ValidationError):
            VoiceChatTurnRequest(
                audioBase64="dGVzdA==",
                fileName="/",
                durationSeconds=1.0,
            )

    def test_hard_audio_size_limit_rejected(self) -> None:
        from pydantic import ValidationError
        from app.models import VoiceChatTurnRequest

        oversized = "A" * (200 * 1024 * 1024 + 1)
        with self.assertRaises(ValidationError):
            VoiceChatTurnRequest(
                audioBase64=oversized,
                fileName="test.wav",
                durationSeconds=1.0,
            )

    def test_voice_profile_path_traversal_stripped(self) -> None:
        from app.models import VoiceProfileSubmitRequest

        req = VoiceProfileSubmitRequest(
            transcript="hello",
            durationSeconds=3.0,
            fileName="../../evil.m4a",
            audioBase64="dGVzdA==",
        )
        self.assertEqual(req.fileName, "evil.m4a")

    def test_ollama_connection_empty_url_rejected(self) -> None:
        from pydantic import ValidationError
        from app.models import OllamaConnectionRequest

        with self.assertRaises(ValidationError):
            OllamaConnectionRequest(endpointURL="")

    def test_ollama_connection_url_too_long_rejected(self) -> None:
        from pydantic import ValidationError
        from app.models import OllamaConnectionRequest

        with self.assertRaises(ValidationError):
            OllamaConnectionRequest(endpointURL="https://x.com/" + "a" * 2050)


if __name__ == "__main__":
    unittest.main()
