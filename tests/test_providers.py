from __future__ import annotations

import unittest
import base64
import io
import wave

from app.audio_upload import AudioUploadChunk, decode_uploaded_audio
from app.models import VoiceChatTurnRequest, VoiceProfileSubmitRequest
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
        auth_mobile_callback_url=None,
        auth_data_dir="/tmp/auth",
        auth_session_ttl_seconds=3600.0,
        conversation_data_dir="/tmp/conv",
        persistence_backend="json",
        auth_users_collection="pikaUsers",
        auth_sessions_collection="pikaSessions",
        auth_connections_collection="pikaProviderConnections",
        conversations_collection="pikaConversations",
        oauth_state_secret="test-secret",
        ollama_endpoint_allowlist="",
        max_audio_base64_bytes=50 * 1024 * 1024,
        require_api_key=False,
        api_key=None,
        apple_app_site_association_app_ids="",
        universal_link_paths="/auth/google/*",
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
        voice_job_storage_bucket=None,
        voice_job_gcs_prefix="voice-jobs",
        voice_job_firestore_collection="voiceChatJobs",
        tts_timeout_seconds=75.0,
        http_timeout_seconds=90.0,
        stt_timeout_seconds=60.0,
        llm_timeout_seconds=90.0,
        max_concurrent_voice_jobs=10,
        voice_job_ttl_seconds=600.0,
        voice_job_worker_poll_seconds=1.0,
        voice_job_worker_lease_seconds=300.0,
        voice_job_worker_concurrency=1,
        audio_upload_ttl_seconds=300.0,
        rate_limit_requests_per_minute=0,
        rate_limit_burst=20,
        max_turns_per_user_per_day=0,
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


class AudioChunkUploadTests(unittest.TestCase):
    def test_chunked_audio_decodes_back_to_wav(self) -> None:
        chunk_one = self._make_wav_chunk(duration_seconds=1.0)
        chunk_two = self._make_wav_chunk(duration_seconds=1.0)
        combined = decode_uploaded_audio(
            None,
            [
                AudioUploadChunk(
                    index=0,
                    totalChunks=2,
                    fileName="chunk-0.wav",
                    mimeType="audio/wav",
                    durationSeconds=1.0,
                    audioBase64=base64.b64encode(chunk_one).decode("utf-8"),
                ),
                AudioUploadChunk(
                    index=1,
                    totalChunks=2,
                    fileName="chunk-1.wav",
                    mimeType="audio/wav",
                    durationSeconds=1.0,
                    audioBase64=base64.b64encode(chunk_two).decode("utf-8"),
                ),
            ],
        )

        with wave.open(io.BytesIO(combined), "rb") as reader:
            self.assertEqual(reader.getnchannels(), 1)
            self.assertEqual(reader.getframerate(), 16000)
            self.assertEqual(reader.getnframes(), 32000)

    def test_chunked_requests_validate_without_audio_base64(self) -> None:
        chat_request = VoiceChatTurnRequest.model_validate(
            {
                "audioChunks": [
                    {
                        "index": 0,
                        "totalChunks": 1,
                        "fileName": "turn.wav",
                        "mimeType": "audio/wav",
                        "durationSeconds": 1.0,
                        "audioBase64": base64.b64encode(self._make_wav_chunk(1.0)).decode("utf-8"),
                    }
                ],
                "fileName": "turn.wav",
                "durationSeconds": 1.0,
            }
        )
        profile_request = VoiceProfileSubmitRequest.model_validate(
            {
                "transcript": "hello",
                "audioChunks": [
                    {
                        "index": 0,
                        "totalChunks": 1,
                        "fileName": "sample.wav",
                        "mimeType": "audio/wav",
                        "durationSeconds": 1.0,
                        "audioBase64": base64.b64encode(self._make_wav_chunk(1.0)).decode("utf-8"),
                    }
                ],
                "fileName": "sample.wav",
                "durationSeconds": 1.0,
            }
        )

        self.assertEqual(chat_request.fileName, "turn.wav")
        self.assertEqual(profile_request.fileName, "sample.wav")
        self.assertEqual(len(chat_request.audioChunks), 1)
        self.assertEqual(len(profile_request.audioChunks), 1)

    def _make_wav_chunk(self, duration_seconds: float) -> bytes:
        sample_rate = 16000
        frame_count = int(sample_rate * duration_seconds)
        output = io.BytesIO()
        with wave.open(output, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.writeframes(b"\x00\x00" * frame_count)
        return output.getvalue()


if __name__ == "__main__":
    unittest.main()
