from __future__ import annotations

import asyncio
import logging
import socket
import uuid

import httpx
from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
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
from .security import SSRFError, parse_ollama_allowlist, validate_ollama_endpoint
from .store import AuthStoreProtocol, ConversationStoreProtocol
from .models import (
    AuthSessionResponse,
    ConversationStateResponse,
    ConversationStateUpdateRequest,
    OllamaConnectionRequest,
    OllamaConnectionResponse,
    VoiceChatJobStatusResponse,
    VoiceChatJobSubmitResponse,
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
from .voice_job_store import ClaimedVoiceJob, JobStage, VoiceJob, VoiceJobStore, VoiceJobStoreProtocol
from .voice_job_store_firestore import FirestoreVoiceJobStore
from .voice_pipeline import run_pipeline


logger = logging.getLogger(__name__)

app = FastAPI(title="Pika Voice Chat Backend")
settings = load_settings()
voice_profile_store = make_voice_profile_store()
auth_store: AuthStoreProtocol = make_auth_store(settings)
conversation_store: ConversationStoreProtocol = make_conversation_store(
    settings.conversation_data_dir, settings=settings
)

# Initialised in startup_event; None until then.
_voice_job_store: Optional[VoiceJobStoreProtocol] = None
_voice_worker_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"

# ---------------------------------------------------------------------------
# Optional API key middleware
# ---------------------------------------------------------------------------
# Routes that are exempt from API key enforcement even when REQUIRE_API_KEY=1.
_API_KEY_EXEMPT_PREFIXES = (
    "/health",
    "/.well-known/apple-app-site-association",
    "/auth/google/",  # OAuth start + callback must be accessible to redirect
    "/auth/session",  # session lookup / refresh must remain reachable to the app
)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """
    Require the X-API-Key header on every non-exempt request when
    REQUIRE_API_KEY=1 and API_KEY is configured.

    This provides a lightweight service-level guard — useful when Cloud Run
    is deployed with --allow-unauthenticated.  It is NOT a substitute for
    proper IAM-based auth (see AUTH_HARDENING.md for the upgrade path).
    """

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        if not settings.require_api_key:
            return await call_next(request)

        path = request.url.path
        if any(path.startswith(prefix) for prefix in _API_KEY_EXEMPT_PREFIXES):
            return await call_next(request)

        if not settings.api_key:
            # Misconfiguration — key enforcement requested but no key set.
            logger.error(
                "REQUIRE_API_KEY=1 but API_KEY is not configured — "
                "rejecting all non-exempt requests."
            )
            return JSONResponse(
                status_code=503,
                content={"message": "Service is misconfigured (API key not set)."},
            )

        provided_key = request.headers.get("X-API-Key", "")
        import hmac as _hmac
        if not _hmac.compare_digest(provided_key.encode(), settings.api_key.encode()):
            return JSONResponse(
                status_code=401,
                content={"message": "Invalid or missing API key."},
            )

        return await call_next(request)


app.add_middleware(ApiKeyMiddleware)


async def _run_blocking(func: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Run synchronous store work off the FastAPI event loop."""
    return await asyncio.to_thread(func, *args, **kwargs)


@app.on_event("startup")
async def startup_event() -> None:
    global _voice_job_store
    if settings.persistence_backend == "firestore" and settings.voice_job_storage_bucket:
        _voice_job_store = FirestoreVoiceJobStore(
            bucket_name=settings.voice_job_storage_bucket,
            gcs_prefix=settings.voice_job_gcs_prefix,
            collection=settings.voice_job_firestore_collection,
            ttl_seconds=settings.voice_job_ttl_seconds,
            max_jobs=settings.max_concurrent_voice_jobs,
            lease_seconds=settings.voice_job_worker_lease_seconds,
        )
    else:
        _voice_job_store = VoiceJobStore(
            ttl_seconds=settings.voice_job_ttl_seconds,
            max_jobs=settings.max_concurrent_voice_jobs,
        )
    app.state.eviction_task = asyncio.create_task(
        _evict_expired_jobs_loop(_voice_job_store, settings.voice_job_ttl_seconds)
    )
    app.state.voice_worker_tasks = [
        asyncio.create_task(_voice_job_worker_loop(_voice_job_store, f"{_voice_worker_id}-{index}"))
        for index in range(settings.voice_job_worker_concurrency)
    ]

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


@app.get("/.well-known/apple-app-site-association", response_model=None)
async def apple_app_site_association() -> JSONResponse:
    app_ids = [
        entry.strip()
        for entry in settings.apple_app_site_association_app_ids.split(",")
        if entry.strip()
    ]
    paths = [
        entry.strip()
        for entry in settings.universal_link_paths.split(",")
        if entry.strip()
    ] or ["/auth/google/*"]
    return JSONResponse(
        content={
            "applinks": {
                "apps": [],
                "details": [{"appIDs": app_ids, "paths": paths}],
            }
        }
    )


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
        state_payload = decode_google_state(state, secret=settings.oauth_state_secret)
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
        session = await _run_blocking(auth_store.create_google_session, **google_user)
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
        return await _run_blocking(auth_store.session_response, session_token)
    except AuthError as exc:
        return JSONResponse(status_code=401, content={"message": str(exc)})


@app.post("/auth/session/refresh", response_model=AuthSessionResponse)
async def refresh_auth_session(authorization: Optional[str] = Header(default=None)):
    try:
        session_token = extract_bearer_token(authorization)
        return await _run_blocking(auth_store.refresh_session, session_token)
    except AuthError as exc:
        return JSONResponse(status_code=401, content={"message": str(exc)})


@app.delete("/auth/session", response_model=None)
async def delete_auth_session(authorization: Optional[str] = Header(default=None)):
    try:
        session_token = extract_bearer_token(authorization)
        await _run_blocking(auth_store.revoke_session, session_token)
        return {"status": "signed_out"}
    except AuthError as exc:
        return JSONResponse(status_code=401, content={"message": str(exc)})


@app.get("/provider-connections/ollama", response_model=OllamaConnectionResponse)
async def get_ollama_connection(authorization: Optional[str] = Header(default=None)):
    try:
        session_token = extract_bearer_token(authorization)
        session = await _run_blocking(auth_store.session_response, session_token)
        return await _run_blocking(auth_store.ollama_connection_response, session["user"]["userId"])
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
        session = await _run_blocking(auth_store.session_response, session_token)

        # SSRF protection — validate the endpoint URL before storing it.
        allowlist = (
            parse_ollama_allowlist(settings.ollama_endpoint_allowlist)
            if settings.ollama_endpoint_allowlist
            else None
        )
        try:
            safe_endpoint = validate_ollama_endpoint(payload.endpointURL, allowlist=allowlist)
        except SSRFError as exc:
            return JSONResponse(status_code=422, content={"message": str(exc)})

        return await _run_blocking(
            auth_store.save_ollama_connection,
            user_id=session["user"]["userId"],
            endpoint_url=safe_endpoint,
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
        session = await _run_blocking(auth_store.session_response, session_token)
        stored = await _run_blocking(
            conversation_store.fetch,
            user_id=session["user"]["userId"],
            conversation_id="default",
        )
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
        session = await _run_blocking(auth_store.session_response, session_token)
        stored = await _run_blocking(
            conversation_store.save,
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
        # Configurable size guard (the model enforces a hard ceiling; this
        # enforces the operator-configurable per-deployment limit).
        if payload.audioBase64 is not None and len(payload.audioBase64) > settings.max_audio_base64_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "message": (
                        f"Audio payload is too large.  "
                        f"Maximum allowed: {settings.max_audio_base64_bytes // (1024 * 1024)} MB."
                    )
                },
            )

        session = await _resolve_optional_session(authorization)
        user = session["user"] if session else None

        if payload.voiceProfileID:
            await _run_blocking(
                voice_profile_store.assert_profile_access,
                payload.voiceProfileID,
                user_id=user["userId"] if user else None,
            )

        ollama_runtime = None
        if user:
            connection = await _run_blocking(auth_store.ollama_connection, user["userId"])
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
        if payload.audioBase64 is not None and len(payload.audioBase64) > settings.max_audio_base64_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "message": (
                        f"Audio payload is too large.  "
                        f"Maximum allowed: {settings.max_audio_base64_bytes // (1024 * 1024)} MB."
                    )
                },
            )
        session = await _resolve_optional_session(authorization)
        return await _run_blocking(
            voice_profile_store.submit,
            payload,
            user_id=session["user"]["userId"] if session else None,
        )
    except AuthError as exc:
        return JSONResponse(status_code=401, content={"message": str(exc)})
    except VoiceProfileTrainingError as exc:
        return JSONResponse(status_code=400, content={"message": str(exc)})
    except Exception as exc:  # pragma: no cover - defensive server boundary
        return JSONResponse(status_code=500, content={"message": f"Unexpected voice profile failure: {exc}"})


@app.get("/voice-profiles/capabilities", response_model=VoiceProfileCapabilitiesResponse)
async def get_voice_profile_capabilities() -> Union[JSONResponse, VoiceProfileCapabilitiesResponse]:
    try:
        return await _run_blocking(voice_profile_store.capabilities)
    except Exception as exc:  # pragma: no cover - defensive server boundary
        return JSONResponse(status_code=500, content={"message": f"Unexpected voice profile failure: {exc}"})


@app.get("/voice-profiles/{job_id}", response_model=VoiceProfileJobStatusResponse)
async def get_voice_profile_status(
    job_id: str,
    authorization: Optional[str] = Header(default=None),
) -> Union[JSONResponse, VoiceProfileJobStatusResponse]:
    try:
        session = await _resolve_optional_session(authorization)
        return await _run_blocking(
            voice_profile_store.status,
            job_id,
            user_id=session["user"]["userId"] if session else None,
        )
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


async def _evict_expired_jobs_loop(store: VoiceJobStoreProtocol, ttl_seconds: float) -> None:
    """Background task: evict terminal jobs from the job store every ttl/2 seconds."""
    interval = max(30.0, ttl_seconds / 2)
    while True:
        await asyncio.sleep(interval)
        n = await store.evict_expired()
        if n:
            logger.debug("[voice-jobs] evicted %d expired jobs", n)


async def _run_voice_job(
    claimed_job: ClaimedVoiceJob,
    store: VoiceJobStoreProtocol,
) -> None:
    """Background coroutine that drives the voice pipeline and writes results to the job store."""

    async def _on_stage(stage: str) -> None:
        await store.update_stage(claimed_job.job_id, JobStage(stage))

    try:
        result = await run_pipeline(
            claimed_job.payload,
            settings,
            ollama_runtime=claimed_job.ollama_runtime,
            on_stage=_on_stage,
        )
        await store.complete(claimed_job.job_id, result)
    except (VoiceChatProviderError, asyncio.TimeoutError) as exc:
        await store.fail(claimed_job.job_id, str(exc))
    except Exception as exc:  # pragma: no cover — defensive boundary
        logger.exception("[voice-jobs] unexpected error in job %s", claimed_job.job_id)
        await store.fail(claimed_job.job_id, f"Unexpected pipeline error: {exc}")


async def _voice_job_worker_loop(store: VoiceJobStoreProtocol, worker_id: str) -> None:
    while True:
        try:
            claimed_job = await store.claim_next(worker_id)
            if claimed_job is None:
                await asyncio.sleep(settings.voice_job_worker_poll_seconds)
                continue
            await _run_voice_job(claimed_job, store)
        except asyncio.CancelledError:  # pragma: no cover - shutdown boundary
            raise
        except Exception:
            logger.exception("[voice-jobs] worker loop failed worker_id=%s", worker_id)
            await asyncio.sleep(settings.voice_job_worker_poll_seconds)


@app.post("/voice-chat/jobs", response_model=VoiceChatJobSubmitResponse)
async def submit_voice_chat_job(
    payload: VoiceChatTurnRequest,
    authorization: Optional[str] = Header(default=None),
) -> Union[JSONResponse, VoiceChatJobSubmitResponse]:
    """
    Submit a voice-chat turn as an async job.

    Returns immediately with a ``jobId`` and ``stage: "queued"``.  The client
    should poll ``GET /voice-chat/jobs/{job_id}`` until ``stage`` is ``"ready"``
    or ``"failed"``.

    The synchronous ``POST /voice-chat/turn`` route remains available and
    unchanged for backwards compatibility.
    """
    if _voice_job_store is None:
        return JSONResponse(status_code=503, content={"message": "Service not ready."})

    try:
        if payload.audioBase64 is not None and len(payload.audioBase64) > settings.max_audio_base64_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "message": (
                        f"Audio payload is too large.  "
                        f"Maximum allowed: {settings.max_audio_base64_bytes // (1024 * 1024)} MB."
                    )
                },
            )

        session = await _resolve_optional_session(authorization)
        user = session["user"] if session else None

        if payload.voiceProfileID:
            await _run_blocking(
                voice_profile_store.assert_profile_access,
                payload.voiceProfileID,
                user_id=user["userId"] if user else None,
            )

        ollama_runtime = None
        if user:
            connection = await _run_blocking(auth_store.ollama_connection, user["userId"])
            if connection:
                ollama_runtime = ollama_runtime_for_connection(settings, connection)

        try:
            job = await _voice_job_store.create(payload, ollama_runtime)
        except RuntimeError as exc:
            return JSONResponse(status_code=429, content={"message": str(exc)})

        return VoiceChatJobSubmitResponse(jobId=job.job_id, stage="queued")

    except AuthError as exc:
        return JSONResponse(status_code=401, content={"message": str(exc)})
    except VoiceProfileTrainingError as exc:
        return JSONResponse(status_code=403, content={"message": str(exc)})


@app.get("/voice-chat/jobs/{job_id}", response_model=VoiceChatJobStatusResponse)
async def get_voice_chat_job_status(
    job_id: str,
    authorization: Optional[str] = Header(default=None),
) -> Union[JSONResponse, VoiceChatJobStatusResponse]:
    """
    Poll the status of an async voice-chat job.

    Returns the current stage and, once ``stage == "ready"``, the full
    ``transcript``, ``responseText``, and optional ``responseAudioBase64``.
    """
    if _voice_job_store is None:
        return JSONResponse(status_code=503, content={"message": "Service not ready."})

    job: Optional[VoiceJob] = await _voice_job_store.get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"message": f"Job '{job_id}' not found or has expired."})

    resp = VoiceChatJobStatusResponse(jobId=job.job_id, stage=job.stage.value)

    if job.stage == JobStage.READY and job.result is not None:
        resp.transcript = job.result.transcript
        resp.responseText = job.result.responseText
        resp.responseAudioBase64 = job.result.responseAudioBase64
        resp.responseAudioMimeType = job.result.responseAudioMimeType

    if job.stage == JobStage.FAILED:
        resp.error = job.error

    return resp


async def _resolve_optional_session(authorization: Optional[str]) -> Optional[dict]:
    if not authorization:
        return None
    session_token = extract_bearer_token(authorization)
    return await _run_blocking(auth_store.session_response, session_token)
