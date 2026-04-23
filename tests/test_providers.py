from __future__ import annotations

import unittest

from app.providers import (
    OllamaRuntimeConfig,
    VoiceChatProviderError,
    _decode_audio,
    _resolve_ollama_base_url,
    _resolve_ollama_chat_url,
    default_ollama_runtime,
    ollama_runtime_for_connection,
)
from app.config import Settings


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="mistral",
        ollama_keep_alive="15m",
        google_oauth_client_id=None,
        google_oauth_client_secret=None,
        google_oauth_callback_url=None,
        auth_mobile_callback_scheme="pikatakehome",
        auth_data_dir="/tmp/auth",
        auth_session_ttl_seconds=3600.0,
        conversation_data_dir="/tmp/conv",
        prewarm_ollama_on_startup=False,
        whisper_command="whisper",
        whisper_model="base",
        whisper_language="en",
        whisper_chunk_duration_seconds=15.0,
        prewarm_whisper_on_startup=False,
        ffmpeg_command="ffmpeg",
        piper_command=None,
        piper_model_path=None,
        piper_config_path=None,
        cosyvoice_command=None,
        cosyvoice_http_url=None,
        cosyvoice_health_url=None,
        cosyvoice_language="en",
        tts_provider="auto",
        voice_profile_models_dir=None,
        voice_profile_manifests_dir=None,
        xtts_model_name=None,
        xtts_language="en",
        voice_profile_storage_bucket=None,
        voice_profile_gcs_prefix="voice-profiles",
        voice_profile_firestore_collection="voiceProfiles",
        voice_profile_jobs_firestore_collection="voiceProfileJobs",
        tts_timeout_seconds=75.0,
        http_timeout_seconds=90.0,
    )
    defaults.update(overrides)
    return Settings(**defaults)


class DecodeAudioTests(unittest.TestCase):
    def test_valid_base64(self) -> None:
        # "hello" -> base64
        self.assertEqual(_decode_audio("aGVsbG8="), b"hello")

    def test_invalid_base64_raises(self) -> None:
        with self.assertRaises(VoiceChatProviderError):
            _decode_audio("this is not valid base64!!!")


class OllamaUrlResolutionTests(unittest.TestCase):
    def test_base_url_strips_trailing_api_chat(self) -> None:
        self.assertEqual(
            _resolve_ollama_base_url("https://host.example.com/api/chat"),
            "https://host.example.com",
        )
        self.assertEqual(
            _resolve_ollama_base_url("https://host.example.com/api/chat/"),
            "https://host.example.com",
        )

    def test_base_url_strips_trailing_slash_only(self) -> None:
        self.assertEqual(
            _resolve_ollama_base_url("https://host.example.com/"),
            "https://host.example.com",
        )

    def test_chat_url_appends_api_chat(self) -> None:
        self.assertEqual(
            _resolve_ollama_chat_url("https://host.example.com"),
            "https://host.example.com/api/chat",
        )

    def test_chat_url_idempotent(self) -> None:
        self.assertEqual(
            _resolve_ollama_chat_url("https://host.example.com/api/chat"),
            "https://host.example.com/api/chat",
        )


class OllamaRuntimeTests(unittest.TestCase):
    def test_default_runtime_from_settings(self) -> None:
        settings = _make_settings(ollama_base_url="http://1.2.3.4:11434/")
        runtime = default_ollama_runtime(settings)
        self.assertEqual(runtime.base_url, "http://1.2.3.4:11434")
        self.assertEqual(runtime.chat_url, "http://1.2.3.4:11434/api/chat")
        self.assertEqual(runtime.model, "mistral")
        self.assertIsNone(runtime.api_token)

    def test_runtime_for_connection_overrides_defaults(self) -> None:
        settings = _make_settings()
        connection = {
            "endpoint_url": "https://my-remote-ollama.example.com",
            "model": "llama3",
            "api_token": "secret",
            "label": "home",
        }
        runtime = ollama_runtime_for_connection(settings, connection)
        self.assertEqual(runtime.base_url, "https://my-remote-ollama.example.com")
        self.assertEqual(runtime.chat_url, "https://my-remote-ollama.example.com/api/chat")
        self.assertEqual(runtime.model, "llama3")
        self.assertEqual(runtime.api_token, "secret")

    def test_runtime_for_connection_with_blank_endpoint_falls_back(self) -> None:
        settings = _make_settings()
        runtime = ollama_runtime_for_connection(settings, {"endpoint_url": ""})
        self.assertEqual(runtime.base_url, settings.ollama_base_url.rstrip("/"))

    def test_runtime_for_connection_with_blank_model_uses_default(self) -> None:
        settings = _make_settings(ollama_model="mistral")
        runtime = ollama_runtime_for_connection(
            settings,
            {"endpoint_url": "https://x.example.com", "model": "   "},
        )
        self.assertEqual(runtime.model, "mistral")

    def test_runtime_for_connection_with_blank_token_is_none(self) -> None:
        settings = _make_settings()
        runtime = ollama_runtime_for_connection(
            settings,
            {"endpoint_url": "https://x.example.com", "api_token": "   "},
        )
        self.assertIsNone(runtime.api_token)

    def test_runtime_config_is_frozen(self) -> None:
        """OllamaRuntimeConfig is a frozen dataclass — guards against accidental mutation."""
        cfg = OllamaRuntimeConfig(
            base_url="http://x",
            chat_url="http://x/api/chat",
            model="m",
            keep_alive="5m",
        )
        with self.assertRaises(Exception):
            cfg.model = "other"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
