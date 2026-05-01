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
    auth_mobile_callback_url: Optional[str]
    auth_data_dir: str
    auth_session_ttl_seconds: float
    conversation_data_dir: str
    # Persistence backend: "json" (default, local dev) or "firestore" (production).
    # When "firestore" is active, auth/session/conversation state is stored in
    # Firestore instead of on-disk JSON files, enabling stateless horizontal scaling.
    persistence_backend: str
    # Firestore collection names — configurable so staging and prod can share a
    # project without colliding.
    auth_users_collection: str
    auth_sessions_collection: str
    auth_connections_collection: str
    conversations_collection: str
    # ---- Security settings ------------------------------------------------
    # HMAC secret for signing the OAuth state parameter.
    # REQUIRED in production — if unset a per-process random secret is used,
    # which invalidates in-flight OAuth flows on container restart.
    oauth_state_secret: str
    # Comma-separated URL prefixes that users are allowed to set as their
    # custom Ollama endpoint.  Empty string = blocklist-only mode (less safe).
    # Example: "https://ollama.prod.example.com,https://ollama2.prod.example.com"
    ollama_endpoint_allowlist: str
    # Maximum size of an audio base64 payload in bytes.
    # Default 50 MB base64 (~37.5 MB raw audio) — enough for a 5-min recording.
    max_audio_base64_bytes: int
    # Service-level API key enforcement.  When REQUIRE_API_KEY=1 every route
    # (except /health and /auth/google/*) must include X-API-Key: <api_key>.
    require_api_key: bool
    api_key: Optional[str]
    apple_app_site_association_app_ids: str
    universal_link_paths: str
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
    voice_job_storage_bucket: Optional[str]
    voice_job_gcs_prefix: str
    voice_job_firestore_collection: str
    tts_timeout_seconds: float
    http_timeout_seconds: float
    # ---- Voice pipeline scalability settings --------------------------------
    # Per-stage timeouts used by the async job pipeline (voice_pipeline.py).
    # The synchronous /voice-chat/turn route keeps using tts_timeout_seconds.
    stt_timeout_seconds: float   # Whisper transcription stage
    llm_timeout_seconds: float   # Ollama reply-generation stage
    # Maximum concurrent voice-chat jobs in the async job store.  Requests
    # beyond this limit are rejected with HTTP 429.
    max_concurrent_voice_jobs: int
    # How long completed/failed jobs are retained in the in-memory store before
    # the background eviction pass removes them.
    voice_job_ttl_seconds: float
    # Background poll interval and claim lease for async voice-job workers.
    voice_job_worker_poll_seconds: float
    voice_job_worker_lease_seconds: float
    voice_job_worker_concurrency: int
    # ---- Transport --------------------------------------------------------
    # TTL for temporary audio upload buffers (POST /audio/uploads).
    # After this many seconds the upload is evicted and the client must re-upload.
    audio_upload_ttl_seconds: float
    # ---- Observability / rate limiting ------------------------------------
    # Per-IP token-bucket rate limit.  Set rate_limit_requests_per_minute to 0
    # to disable rate limiting entirely.
    rate_limit_requests_per_minute: int
    rate_limit_burst: int
    # Per-user daily voice-chat turn quota.  Set to 0 to disable.
    max_turns_per_user_per_day: int

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
    # Security — OAuth state secret
    _raw_state_secret = os.getenv("OAUTH_STATE_SECRET", "").strip()
    if not _raw_state_secret:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "OAUTH_STATE_SECRET is not set.  A per-process random secret will be "
            "used, which invalidates in-flight OAuth flows on container restart.  "
            "Set OAUTH_STATE_SECRET to a strong random value in production."
        )
        from .security import generate_fallback_secret as _gen
        _raw_state_secret = _gen()

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
        auth_mobile_callback_url=os.getenv("AUTH_MOBILE_CALLBACK_URL", "").strip() or None,
        auth_data_dir=auth_data_dir,
        auth_session_ttl_seconds=float(os.getenv("AUTH_SESSION_TTL_SECONDS", "2592000").strip()),
        conversation_data_dir=conversation_data_dir,
        persistence_backend=os.getenv("PERSISTENCE_BACKEND", "json").strip().lower() or "json",
        auth_users_collection=os.getenv("AUTH_USERS_COLLECTION", "pikaUsers").strip() or "pikaUsers",
        auth_sessions_collection=os.getenv("AUTH_SESSIONS_COLLECTION", "pikaSessions").strip() or "pikaSessions",
        auth_connections_collection=os.getenv("AUTH_CONNECTIONS_COLLECTION", "pikaProviderConnections").strip() or "pikaProviderConnections",
        conversations_collection=os.getenv("CONVERSATIONS_COLLECTION", "pikaConversations").strip() or "pikaConversations",
        oauth_state_secret=_raw_state_secret,
        ollama_endpoint_allowlist=os.getenv("OLLAMA_ENDPOINT_ALLOWLIST", "").strip(),
        max_audio_base64_bytes=int(os.getenv("MAX_AUDIO_BASE64_BYTES", str(50 * 1024 * 1024)).strip()),
        require_api_key=os.getenv("REQUIRE_API_KEY", "0").strip().lower() in {"1", "true", "yes"},
        api_key=os.getenv("API_KEY", "").strip() or None,
        apple_app_site_association_app_ids=os.getenv("APPLE_APP_SITE_ASSOCIATION_APP_IDS", "").strip(),
        universal_link_paths=os.getenv("UNIVERSAL_LINK_PATHS", "/auth/google/*").strip() or "/auth/google/*",
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
        voice_job_storage_bucket=(
            os.getenv("VOICE_JOB_STORAGE_BUCKET", "").strip()
            or os.getenv("VOICE_PROFILE_STORAGE_BUCKET", "").strip()
            or None
        ),
        voice_job_gcs_prefix=os.getenv("VOICE_JOB_GCS_PREFIX", "voice-jobs").strip() or "voice-jobs",
        voice_job_firestore_collection=os.getenv("VOICE_JOB_FIRESTORE_COLLECTION", "voiceChatJobs").strip() or "voiceChatJobs",
        tts_timeout_seconds=float(os.getenv("TTS_TIMEOUT_SECONDS", "75").strip()),
        http_timeout_seconds=float(os.getenv("HTTP_TIMEOUT_SECONDS", "90").strip()),
        stt_timeout_seconds=float(os.getenv("STT_TIMEOUT_SECONDS", "60").strip()),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "90").strip()),
        max_concurrent_voice_jobs=int(os.getenv("MAX_CONCURRENT_VOICE_JOBS", "10").strip()),
        voice_job_ttl_seconds=float(os.getenv("VOICE_JOB_TTL_SECONDS", "600").strip()),
        voice_job_worker_poll_seconds=float(os.getenv("VOICE_JOB_WORKER_POLL_SECONDS", "1.0").strip()),
        voice_job_worker_lease_seconds=float(os.getenv("VOICE_JOB_WORKER_LEASE_SECONDS", "300").strip()),
        voice_job_worker_concurrency=max(
            1,
            int(os.getenv("VOICE_JOB_WORKER_CONCURRENCY", "1").strip()),
        ),
        audio_upload_ttl_seconds=max(
            30.0,
            float(os.getenv("AUDIO_UPLOAD_TTL_SECONDS", "300").strip()),
        ),
        rate_limit_requests_per_minute=max(
            0,
            int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "120").strip()),
        ),
        rate_limit_burst=max(
            1,
            int(os.getenv("RATE_LIMIT_BURST", "20").strip()),
        ),
        max_turns_per_user_per_day=max(
            0,
            int(os.getenv("MAX_TURNS_PER_USER_PER_DAY", "0").strip()),
        ),
    )
