from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .audio_upload import AudioUploadChunk, _sanitize_file_name


_HARD_MAX_AUDIO_B64_BYTES = 200 * 1024 * 1024


class ChatHistoryMessage(BaseModel):
    role: str
    content: str


class AuthenticatedUser(BaseModel):
    userId: str
    email: str
    displayName: str
    photoURL: Optional[str] = None


class AuthSessionResponse(BaseModel):
    sessionToken: str
    user: AuthenticatedUser
    # Expiry timestamp (ISO-8601 UTC) so the iOS client can show a refresh UI
    # or proactively re-authenticate before the session expires.
    expiresAt: Optional[str] = None


class OllamaConnectionRequest(BaseModel):
    endpointURL: str
    model: Optional[str] = None
    apiToken: Optional[str] = None
    label: Optional[str] = None

    @field_validator("endpointURL")
    @classmethod
    def endpoint_url_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("endpointURL must not be blank.")
        if len(v) > 2048:
            raise ValueError("endpointURL is too long (max 2048 characters).")
        return v


class OllamaConnectionResponse(BaseModel):
    endpointURL: str
    model: Optional[str] = None
    hasAPIToken: bool
    label: Optional[str] = None
    updatedAt: Optional[str] = None


class ConversationMessage(BaseModel):
    role: str
    content: str


class ConversationStateResponse(BaseModel):
    conversationId: str
    summary: str = ""
    voiceProfileID: Optional[str] = None
    messages: List[ConversationMessage] = Field(default_factory=list)


class ConversationStateUpdateRequest(BaseModel):
    summary: str = ""
    voiceProfileID: Optional[str] = None
    messages: List[ConversationMessage] = Field(default_factory=list)


class VoiceChatTurnRequest(BaseModel):
    audioBase64: Optional[str] = None
    audioChunks: List[AudioUploadChunk] = Field(default_factory=list)
    # Alternatively, reference a previously uploaded audio blob by its upload ID
    # (returned by POST /audio/uploads).  Mutually exclusive with audioBase64/audioChunks.
    audioUploadID: Optional[str] = None
    mimeType: str = Field(default="audio/wav")
    fileName: str
    durationSeconds: float
    voiceProfileID: Optional[str] = None
    conversationSummary: Optional[str] = None
    history: List[ChatHistoryMessage] = Field(default_factory=list)

    @field_validator("fileName")
    @classmethod
    def sanitize_file_name(cls, v: str) -> str:
        return _sanitize_file_name(v)

    @field_validator("audioBase64")
    @classmethod
    def audio_size_limit(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if len(v) > 200 * 1024 * 1024:
            raise ValueError("audioBase64 payload exceeds the hard maximum of 200 MB.")
        return v

    @field_validator("durationSeconds")
    @classmethod
    def duration_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("durationSeconds must be non-negative.")
        return v

    @model_validator(mode="after")
    def requires_audio_source(self) -> "VoiceChatTurnRequest":
        has_inline = bool(self.audioBase64 or self.audioChunks)
        has_ref = bool(self.audioUploadID)
        if not has_inline and not has_ref:
            raise ValueError("Either audioBase64, audioChunks, or audioUploadID must be provided.")
        return self


class VoiceChatTurnResponse(BaseModel):
    transcript: str
    responseText: str
    responseAudioBase64: Optional[str] = None
    responseAudioMimeType: Optional[str] = None
    error: Optional[str] = None


class AudioUploadResponse(BaseModel):
    """Returned by POST /audio/uploads — use uploadId in subsequent job submissions."""
    uploadId: str
    expiresInSeconds: int


class VoiceProfileSubmitRequest(BaseModel):
    transcript: str
    durationSeconds: float
    fileName: str
    mimeType: str = Field(default="audio/wav")
    audioBase64: Optional[str] = None
    audioChunks: List[AudioUploadChunk] = Field(default_factory=list)
    # Reference a previously uploaded audio blob instead of sending inline audio.
    audioUploadID: Optional[str] = None
    baseProfileID: Optional[str] = None

    @field_validator("fileName")
    @classmethod
    def sanitize_file_name(cls, v: str) -> str:
        return _sanitize_file_name(v)

    @field_validator("audioBase64")
    @classmethod
    def audio_size_limit(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if len(v) > _HARD_MAX_AUDIO_B64_BYTES:
            raise ValueError(
                f"audioBase64 payload exceeds the hard maximum of "
                f"{_HARD_MAX_AUDIO_B64_BYTES // (1024 * 1024)} MB."
            )
        return v

    @field_validator("durationSeconds")
    @classmethod
    def duration_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("durationSeconds must be non-negative.")
        return v

    @model_validator(mode="after")
    def requires_audio_source(self) -> "VoiceProfileSubmitRequest":
        has_inline = bool(self.audioBase64 or self.audioChunks)
        has_ref = bool(self.audioUploadID)
        if not has_inline and not has_ref:
            raise ValueError("Either audioBase64, audioChunks, or audioUploadID must be provided.")
        return self


class VoiceProfileSubmitResponse(BaseModel):
    jobId: str
    profileId: Optional[str] = None


class VoiceProfileJobStatusResponse(BaseModel):
    status: str
    progress: Optional[float] = None
    profileId: Optional[str] = None
    message: Optional[str] = None


class VoiceProfileCapabilitiesResponse(BaseModel):
    trainingCommandConfigured: bool
    trainingMode: str
    supportsPersonalizedVoice: bool
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# Async voice-chat job models (POST /voice-chat/jobs)
# ---------------------------------------------------------------------------

class VoiceChatJobSubmitResponse(BaseModel):
    """Returned immediately when a voice-chat job is accepted."""
    jobId: str
    stage: str = "queued"


class VoiceChatJobStatusResponse(BaseModel):
    """Returned when polling GET /voice-chat/jobs/{job_id}."""
    jobId: str
    # stage: queued | transcribing | generating | synthesizing | ready | failed
    stage: str
    transcript: Optional[str] = None
    responseText: Optional[str] = None
    responseAudioBase64: Optional[str] = None
    responseAudioMimeType: Optional[str] = None
    error: Optional[str] = None
