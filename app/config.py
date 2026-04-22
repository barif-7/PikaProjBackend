from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    ollama_model: str
    ollama_keep_alive: str
    google_oauth_client_id: Optional[str]
    google_oauth_client_secret: Optional[str]
    google_oauth_callback_url: Optional[str]
    auth_mobile_callback_scheme: str
    auth_data_dir: str
    auth_session_ttl_seconds: float
    conversation_data_dir: str
    prewarm_ollama_on_startup: bool
    whisper_command: str
    whisper_model: str
    whisper_language: Optional[str]
    whisper_chunk_duration_seconds: float
    prewarm_whisper_on_startup: bool
    ffmpeg_command: Optional[str]
    piper_command: Optional[str]
    piper_model_path: Optional[str]
    piper_config_path: Optional[str]
    cosyvoice_command: Optional[str]
    cosyvoice_http_url: Optional[str]
    cosyvoice_health_url: Optional[str]
    cosyvoice_language: str
    tts_provider: str
    voice_profile_models_dir: Optional[str]
    voice_profile_manifests_dir: Optional[str]
    xtts_model_name: Optional[str]
    xtts_language: str
    voice_profile_storage_bucket: Optional[str]
    voice_profile_gcs_prefix: str
    voice_profile_firestore_collection: str
    voice_profile_jobs_firestore_collection: str
    tts_timeout_seconds: float
    http_timeout_seconds: float

    @property
    def ollama_chat_url(self) -> str:
        return f"{self.ollama_base_url.rstrip('/')}/api/chat"

    @property
    def is_tts_enabled(self) -> bool:
        return bool((self.piper_command and self.piper_model_path) or self.xtts_model_name or self.cosyvoice_command)

    @property
    def is_xtts_enabled(self) -> bool:
        return bool(self.xtts_model_name)

    @property
    def is_cosyvoice_enabled(self) -> bool:
        return bool(self.cosyvoice_command)


def load_settings() -> Settings:
    prewarm_ollama_on_startup = os.getenv("PREWARM_OLLAMA_ON_STARTUP", "1").strip().lower() not in {"0", "false", "no"}
    whisper_language = os.getenv("WHISPER_LANGUAGE", "en").strip() or None
    whisper_chunk_duration_seconds = float(os.getenv("WHISPER_CHUNK_DURATION_SECONDS", "15").strip())
    prewarm_whisper_on_startup = os.getenv("PREWARM_WHISPER_ON_STARTUP", "1").strip().lower() not in {"0", "false", "no"}
    ffmpeg_command = os.getenv("FFMPEG_COMMAND", "ffmpeg").strip() or None
    piper_command = os.getenv("PIPER_COMMAND", "").strip() or None
    piper_model_path = os.getenv("PIPER_MODEL_PATH", "").strip() or None
    piper_config_path = os.getenv("PIPER_CONFIG_PATH", "").strip() or None
    cosyvoice_command = os.getenv("COSYVOICE_COMMAND", "").strip() or None
    cosyvoice_http_url = os.getenv("COSYVOICE_HTTP_URL", "").strip() or None
    cosyvoice_health_url = os.getenv("COSYVOICE_HEALTH_URL", "").strip() or None
    if cosyvoice_http_url and not cosyvoice_health_url:
        parsed = urlsplit(cosyvoice_http_url)
        path = parsed.path.rstrip("/")
        if path.endswith("/synthesize"):
            health_path = f"{path[: -len('/synthesize')]}/health" or "/health"
        else:
            health_path = f"{path}/health" if path else "/health"
        cosyvoice_health_url = urlunsplit((parsed.scheme, parsed.netloc, health_path, "", ""))
    cosyvoice_language = os.getenv("COSYVOICE_LANGUAGE", "en").strip() or "en"
    tts_provider = os.getenv("TTS_PROVIDER", "auto").strip().lower() or "auto"
    voice_profile_models_dir = os.getenv("VOICE_PROFILE_MODELS_DIR", "").strip() or str(
        Path(__file__).resolve().parent.parent / "data" / "profile-models"
    )
    voice_profile_manifests_dir = os.getenv("VOICE_PROFILE_MANIFESTS_DIR", "").strip() or str(
        Path(__file__).resolve().parent.parent / "data" / "profiles"
    )
    xtts_model_name = os.getenv("XTTS_MODEL_NAME", "").strip() or None
    xtts_language = os.getenv("XTTS_LANGUAGE", "en").strip() or "en"
    auth_data_dir = os.getenv("AUTH_DATA_DIR", "").strip() or str(
        Path(__file__).resolve().parent.parent / "data" / "auth"
    )
    conversation_data_dir = os.getenv("CONVERSATION_DATA_DIR", "").strip() or str(
        Path(__file__).resolve().parent.parent / "data" / "conversations"
    )

    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip(),
        ollama_model=os.getenv("OLLAMA_MODEL", "mistral").strip(),
        ollama_keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "15m").strip() or "15m",
        google_oauth_client_id=os.getenv("GOOGLE_OAUTH_CLIENT_ID", "").strip() or None,
        google_oauth_client_secret=os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "").strip() or None,
        google_oauth_callback_url=os.getenv("GOOGLE_OAUTH_CALLBACK_URL", "").strip() or None,
        auth_mobile_callback_scheme=os.getenv("AUTH_MOBILE_CALLBACK_SCHEME", "pikatakehome").strip() or "pikatakehome",
        auth_data_dir=auth_data_dir,
        auth_session_ttl_seconds=float(os.getenv("AUTH_SESSION_TTL_SECONDS", "2592000").strip()),
        conversation_data_dir=conversation_data_dir,
        prewarm_ollama_on_startup=prewarm_ollama_on_startup,
        whisper_command=os.getenv("WHISPER_COMMAND", "whisper").strip(),
        whisper_model=os.getenv("WHISPER_MODEL", "base").strip(),
        whisper_language=whisper_language,
        whisper_chunk_duration_seconds=max(0.0, whisper_chunk_duration_seconds),
        prewarm_whisper_on_startup=prewarm_whisper_on_startup,
        ffmpeg_command=ffmpeg_command,
        piper_command=piper_command,
        piper_model_path=piper_model_path,
        piper_config_path=piper_config_path,
        cosyvoice_command=cosyvoice_command,
        cosyvoice_http_url=cosyvoice_http_url,
        cosyvoice_health_url=cosyvoice_health_url,
        cosyvoice_language=cosyvoice_language,
        tts_provider=tts_provider,
        voice_profile_models_dir=voice_profile_models_dir,
        voice_profile_manifests_dir=voice_profile_manifests_dir,
        xtts_model_name=xtts_model_name,
        xtts_language=xtts_language,
        voice_profile_storage_bucket=os.getenv("VOICE_PROFILE_STORAGE_BUCKET", "").strip() or None,
        voice_profile_gcs_prefix=os.getenv("VOICE_PROFILE_GCS_PREFIX", "voice-profiles").strip() or "voice-profiles",
        voice_profile_firestore_collection=os.getenv("VOICE_PROFILE_FIRESTORE_COLLECTION", "voiceProfiles").strip() or "voiceProfiles",
        voice_profile_jobs_firestore_collection=os.getenv("VOICE_PROFILE_JOBS_FIRESTORE_COLLECTION", "voiceProfileJobs").strip() or "voiceProfileJobs",
        tts_timeout_seconds=float(os.getenv("TTS_TIMEOUT_SECONDS", "75").strip()),
        http_timeout_seconds=float(os.getenv("HTTP_TIMEOUT_SECONDS", "90").strip()),
    )
