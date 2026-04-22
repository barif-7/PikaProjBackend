from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional


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


class OllamaConnectionRequest(BaseModel):
    endpointURL: str
    model: Optional[str] = None
    apiToken: Optional[str] = None
    label: Optional[str] = None


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
    audioBase64: str
    mimeType: str = Field(default="audio/wav")
    fileName: str
    durationSeconds: float
    voiceProfileID: Optional[str] = None
    conversationSummary: Optional[str] = None
    history: List[ChatHistoryMessage] = Field(default_factory=list)


class VoiceChatTurnResponse(BaseModel):
    transcript: str
    responseText: str
    responseAudioBase64: Optional[str] = None
    responseAudioMimeType: Optional[str] = None
    error: Optional[str] = None


class VoiceProfileSubmitRequest(BaseModel):
    transcript: str
    durationSeconds: float
    fileName: str
    mimeType: str = Field(default="audio/wav")
    audioBase64: str
    baseProfileID: Optional[str] = None


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
