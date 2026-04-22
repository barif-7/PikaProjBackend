from __future__ import annotations

import asyncio
import logging

import httpx
from fastapi import FastAPI, Header, Query
from fastapi.responses import JSONResponse, RedirectResponse
from typing import Any, Dict, Optional, Union

from .auth import (
    AuthConfigurationError,
    AuthError,
    append_query_params,
    build_google_authorize_url,
    decode_google_state,
    exchange_google_code_for_user,
    extract_bearer_token,
    make_auth_store,
)
from .config import load_settings
from .conversations import make_conversation_store
from .models import (
    AuthSessionResponse,
    ConversationStateResponse,
    ConversationStateUpdateRequest,
    OllamaConnectionRequest,
    OllamaConnectionResponse,
    VoiceChatTurnRequest,
    VoiceChatTurnResponse,
    VoiceProfileCapabilitiesResponse,
    VoiceProfileJobStatusResponse,
    VoiceProfileSubmitRequest,
    VoiceProfileSubmitResponse,
)
from .providers import (
    VoiceChatProviderError,
    generate_turn_response,
    ollama_runtime_for_connection,
    prewarm_runtime,
)
from .training import (
    VoiceProfileTrainingError,
    make_voice_profile_store,
)


logger = logging.getLogger(__name__)

app = FastAPI(title="Pika Voice Chat Backend")
settings = load_settings()
voice_profile_store = make_voice_profile_store()
auth_store = make_auth_store(settings)
conversation_store = make_conversation_store(settings.conversation_data_dir)


@app.on_event("startup")
async def startup_event() -> None:
    app.state.prewarm_task = None
    if settings.prewarm_ollama_on_startup or settings.prewarm_whisper_on_startup:
        app.state.prewarm_task = asyncio.create_task(prewarm_runtime(settings))

    if settings.cosyvoice_health_url:
        cosyvoice_status = await _get_cosyvoice_health()
        logger.info(
            "[startup] cosyvoice-remote configured=%s reachable=%s health_url=%s",
            cosyvoice_status["configured"],
            cosyvoice_status["reachable"],
            cosyvoice_status.get("healthURL") or "",
        )


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "tts": {
            "provider": settings.tts_provider,
            "cosyvoice": await _get_cosyvoice_health(),
        },
    }


@app.get("/auth/google/start", response_model=None)
async def auth_google_start(mobile_callback: str = Query(...)):
    try:
        return RedirectResponse(build_google_authorize_url(settings, mobile_callback))
    except AuthConfigurationError as exc:
        return JSONResponse(status_code=503, content={"message": str(exc)})


@app.get("/auth/google/callback", response_model=None)
async def auth_google_callback(
    code: Optional[str] = Query(default=None),
    state: Optional[str] = Query(default=None),
    error: Optional[str] = Query(default=None),
):
    if not state:
        return JSONResponse(status_code=400, content={"message": "Missing OAuth state."})

    try:
        state_payload = decode_google_state(state)
        mobile_callback = state_payload["mobile_callback"]
    except AuthError as exc:
        return JSONResponse(status_code=400, content={"message": str(exc)})

    if error:
        return RedirectResponse(
            append_query_params(
                mobile_callback,
                {"error": error},
            )
        )

    if not code:
        return RedirectResponse(
            append_query_params(
                mobile_callback,
                {"error": "missing_code"},
            )
        )

    try:
        google_user = await exchange_google_code_for_user(settings, code)
        session = auth_store.create_google_session(**google_user)
        return RedirectResponse(
            append_query_params(
                mobile_callback,
                {"session_token": session["sessionToken"]},
            )
        )
    except Exception as exc:  # pragma: no cover - defensive OAuth boundary
        return RedirectResponse(
            append_query_params(
                mobile_callback,
                {"error": "google_auth_failed", "error_description": str(exc)},
            )
        )


@app.get("/auth/session", response_model=AuthSessionResponse)
async def auth_session(authorization: Optional[str] = Header(default=None)):
    try:
        session_token = extract_bearer_token(authorization)
        return auth_store.session_response(session_token)
    except AuthError as exc:
        return JSONResponse(status_code=401, content={"message": str(exc)})


@app.delete("/auth/session", response_model=None)
async def delete_auth_session(authorization: Optional[str] = Header(default=None)):
    try:
        session_token = extract_bearer_token(authorization)
        auth_store.revoke_session(session_token)
        return {"status": "signed_out"}
    except AuthError as exc:
        return JSONResponse(status_code=401, content={"message": str(exc)})


@app.get("/provider-connections/ollama", response_model=OllamaConnectionResponse)
async def get_ollama_connection(authorization: Optional[str] = Header(default=None)):
    try:
        session_token = extract_bearer_token(authorization)
        session = auth_store.session_response(session_token)
        return auth_store.ollama_connection_response(session["user"]["userId"])
    except AuthError as exc:
        status_code = 404 if "No Ollama connection" in str(exc) else 401
        return JSONResponse(status_code=status_code, content={"message": str(exc)})


@app.put("/provider-connections/ollama", response_model=OllamaConnectionResponse)
async def put_ollama_connection(
    payload: OllamaConnectionRequest,
    authorization: Optional[str] = Header(default=None),
):
    try:
        session_token = extract_bearer_token(authorization)
        session = auth_store.session_response(session_token)
        return auth_store.save_ollama_connection(
            user_id=session["user"]["userId"],
            endpoint_url=payload.endpointURL,
            model=payload.model,
            api_token=payload.apiToken,
            label=payload.label,
        )
    except AuthError as exc:
        return JSONResponse(status_code=401, content={"message": str(exc)})


@app.get("/conversations/default", response_model=ConversationStateResponse)
async def get_default_conversation(authorization: Optional[str] = Header(default=None)):
    try:
        session_token = extract_bearer_token(authorization)
        session = auth_store.session_response(session_token)
        stored = conversation_store.fetch(user_id=session["user"]["userId"], conversation_id="default")
        return ConversationStateResponse(
            conversationId=stored["conversation_id"],
            summary=stored.get("summary") or "",
            voiceProfileID=stored.get("voice_profile_id"),
            messages=stored.get("messages") or [],
        )
    except AuthError as exc:
        return JSONResponse(status_code=401, content={"message": str(exc)})


@app.put("/conversations/default", response_model=ConversationStateResponse)
async def put_default_conversation(
    payload: ConversationStateUpdateRequest,
    authorization: Optional[str] = Header(default=None),
):
    try:
        session_token = extract_bearer_token(authorization)
        session = auth_store.session_response(session_token)
        stored = conversation_store.save(
            user_id=session["user"]["userId"],
            conversation_id="default",
            summary=payload.summary,
            voice_profile_id=payload.voiceProfileID,
            messages=[message.model_dump() for message in payload.messages],
        )
        return ConversationStateResponse(
            conversationId=stored["conversation_id"],
            summary=stored.get("summary") or "",
            voiceProfileID=stored.get("voice_profile_id"),
            messages=stored.get("messages") or [],
        )
    except AuthError as exc:
        return JSONResponse(status_code=401, content={"message": str(exc)})


@app.post("/voice-chat/turn", response_model=VoiceChatTurnResponse)
async def voice_chat_turn(
    payload: VoiceChatTurnRequest,
    authorization: Optional[str] = Header(default=None),
) -> Union[JSONResponse, VoiceChatTurnResponse]:
    try:
        session = _resolve_optional_session(authorization)
        user = session["user"] if session else None

        if payload.voiceProfileID:
            voice_profile_store.assert_profile_access(
                payload.voiceProfileID,
                user_id=user["userId"] if user else None,
            )

        ollama_runtime = None
        if user:
            connection = auth_store.ollama_connection(user["userId"])
            if connection:
                ollama_runtime = ollama_runtime_for_connection(settings, connection)

        return await generate_turn_response(payload, settings, ollama_runtime=ollama_runtime)
    except AuthError as exc:
        return JSONResponse(status_code=401, content={"message": str(exc)})
    except VoiceProfileTrainingError as exc:
        return JSONResponse(status_code=403, content={"message": str(exc)})
    except VoiceChatProviderError as exc:
        return JSONResponse(
            status_code=503,
            content=VoiceChatTurnResponse(
                transcript="",
                responseText="",
                responseAudioBase64=None,
                responseAudioMimeType=None,
                error=str(exc),
            ).model_dump(),
        )


@app.post("/voice-profiles", response_model=VoiceProfileSubmitResponse)
async def submit_voice_profile(
    payload: VoiceProfileSubmitRequest,
    authorization: Optional[str] = Header(default=None),
) -> Union[JSONResponse, VoiceProfileSubmitResponse]:
    try:
        session = _resolve_optional_session(authorization)
        return voice_profile_store.submit(payload, user_id=session["user"]["userId"] if session else None)
    except AuthError as exc:
        return JSONResponse(status_code=401, content={"message": str(exc)})
    except VoiceProfileTrainingError as exc:
        return JSONResponse(status_code=400, content={"message": str(exc)})
    except Exception as exc:  # pragma: no cover - defensive server boundary
        return JSONResponse(status_code=500, content={"message": f"Unexpected voice profile failure: {exc}"})


@app.get("/voice-profiles/capabilities", response_model=VoiceProfileCapabilitiesResponse)
async def get_voice_profile_capabilities() -> Union[JSONResponse, VoiceProfileCapabilitiesResponse]:
    try:
        return voice_profile_store.capabilities()
    except Exception as exc:  # pragma: no cover - defensive server boundary
        return JSONResponse(status_code=500, content={"message": f"Unexpected voice profile failure: {exc}"})


@app.get("/voice-profiles/{job_id}", response_model=VoiceProfileJobStatusResponse)
async def get_voice_profile_status(
    job_id: str,
    authorization: Optional[str] = Header(default=None),
) -> Union[JSONResponse, VoiceProfileJobStatusResponse]:
    try:
        session = _resolve_optional_session(authorization)
        return voice_profile_store.status(job_id, user_id=session["user"]["userId"] if session else None)
    except AuthError as exc:
        return JSONResponse(status_code=401, content={"message": str(exc)})
    except VoiceProfileTrainingError as exc:
        status_code = 403 if "belongs to a signed-in user" in str(exc) or "do not have access" in str(exc) else 404
        return JSONResponse(status_code=status_code, content={"message": str(exc)})
    except Exception as exc:  # pragma: no cover - defensive server boundary
        return JSONResponse(status_code=500, content={"message": f"Unexpected voice profile failure: {exc}"})


async def _get_cosyvoice_health() -> Dict[str, Any]:
    health_url = settings.cosyvoice_health_url
    status: Dict[str, Any] = {
        "configured": bool(settings.cosyvoice_command and settings.cosyvoice_http_url),
        "mode": "remote-http" if settings.cosyvoice_http_url else "command-only",
        "commandConfigured": bool(settings.cosyvoice_command),
        "serviceURL": settings.cosyvoice_http_url,
        "healthURL": health_url,
    }
    if not health_url:
        status["reachable"] = None
        return status

    try:
        async with httpx.AsyncClient(timeout=min(settings.http_timeout_seconds, 10.0)) as client:
            response = await client.get(health_url)
        status["reachable"] = response.is_success
        if response.headers.get("content-type", "").startswith("application/json"):
            status["response"] = response.json()
        else:
            status["response"] = {"statusCode": response.status_code}
    except Exception as exc:
        status["reachable"] = False
        status["error"] = str(exc)
    return status


def _resolve_optional_session(authorization: Optional[str]) -> Optional[dict]:
    if not authorization:
        return None
    session_token = extract_bearer_token(authorization)
    return auth_store.session_response(session_token)
