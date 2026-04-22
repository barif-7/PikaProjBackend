from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import tempfile
from threading import Lock
import time
from typing import Optional

import httpx

from .config import Settings
from .durable_storage import build_durable_voice_profile_store
from .models import VoiceChatTurnRequest, VoiceChatTurnResponse
from .prompting import build_chat_messages


class VoiceChatProviderError(RuntimeError):
    pass


_XTTS_LOCK = Lock()
_XTTS_MODEL = None
_XTTS_MODEL_NAME = None
_WHISPER_LOCK = Lock()
_WHISPER_MODEL = None
_WHISPER_MODEL_NAME = None
_OLLAMA_CLIENT_LOCK = Lock()
_OLLAMA_CLIENTS: dict[str, httpx.AsyncClient] = {}
_DURABLE_VOICE_PROFILE_STORE = build_durable_voice_profile_store()


@dataclass(frozen=True)
class OllamaRuntimeConfig:
    base_url: str
    chat_url: str
    model: str
    keep_alive: str
    api_token: Optional[str] = None


async def generate_turn_response(
    payload: VoiceChatTurnRequest,
    settings: Settings,
    ollama_runtime: Optional[OllamaRuntimeConfig] = None,
) -> VoiceChatTurnResponse:
    request_started_at = time.perf_counter()
    audio_bytes = _decode_audio(payload.audioBase64)

    with tempfile.TemporaryDirectory(prefix="pika-voice-chat-") as temp_dir:
        temp_path = Path(temp_dir)
        audio_path = temp_path / payload.fileName
        audio_path.write_bytes(audio_bytes)

        transcription_started_at = time.perf_counter()
        transcript = await _transcribe_audio(audio_path, temp_path, settings, payload.durationSeconds)
        if not transcript:
            raise VoiceChatProviderError("Whisper returned an empty transcript.")

        reply_started_at = time.perf_counter()
        response_text = await _generate_reply(
            payload,
            transcript,
            settings,
            ollama_runtime or default_ollama_runtime(settings),
        )
        if not response_text:
            raise VoiceChatProviderError("The language model returned an empty response.")

        response_audio_base64: Optional[str] = None
        response_audio_mime: Optional[str] = None
        synthesis_started_at = time.perf_counter()
        if settings.is_tts_enabled:
            try:
                wav_bytes = await asyncio.wait_for(
                    _synthesize_speech(
                        response_text,
                        temp_path,
                        settings,
                        payload.voiceProfileID,
                    ),
                    timeout=settings.tts_timeout_seconds,
                )
                response_audio_base64 = base64.b64encode(wav_bytes).decode("utf-8")
                response_audio_mime = "audio/wav"
            except asyncio.TimeoutError:
                print(
                    "[voice-chat] "
                    f"tts-timeout={settings.tts_timeout_seconds:.2f}s "
                    "returning_text_only=true"
                )
            except VoiceChatProviderError as exc:
                print(f"[voice-chat] tts-failed returning_text_only=true error={exc}")

        _log_turn_timing(
            duration_seconds=payload.durationSeconds,
            transcription_seconds=reply_started_at - transcription_started_at,
            llm_seconds=synthesis_started_at - reply_started_at,
            synthesis_seconds=time.perf_counter() - synthesis_started_at if settings.is_tts_enabled else 0.0,
            total_seconds=time.perf_counter() - request_started_at,
        )
        return VoiceChatTurnResponse(
            transcript=transcript,
            responseText=response_text,
            responseAudioBase64=response_audio_base64,
            responseAudioMimeType=response_audio_mime,
            error=None,
        )


async def prewarm_runtime(settings: Settings) -> None:
    tasks: list[asyncio.Task[None]] = []

    if settings.prewarm_whisper_on_startup:
        tasks.append(asyncio.create_task(_prewarm_whisper(settings)))
    if settings.prewarm_ollama_on_startup:
        tasks.append(asyncio.create_task(_prewarm_ollama(settings)))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _decode_audio(audio_base64: str) -> bytes:
    try:
        return base64.b64decode(audio_base64, validate=True)
    except Exception as exc:  # pragma: no cover - exact exception type is not stable
        raise VoiceChatProviderError("The app sent invalid audio data.") from exc


async def _transcribe_audio(
    audio_path: Path,
    output_dir: Path,
    settings: Settings,
    duration_seconds: Optional[float],
) -> str:
    try:
        return await asyncio.to_thread(
            _transcribe_with_python_whisper,
            audio_path,
            output_dir,
            settings,
            duration_seconds,
        )
    except VoiceChatProviderError:
        raise
    except Exception:
        # Fall back to the CLI path if the in-process Whisper stack is unavailable.
        pass

    command = [
        settings.whisper_command,
        str(audio_path),
        "--model",
        settings.whisper_model,
        "--task",
        "transcribe",
        "--output_format",
        "txt",
        "--output_dir",
        str(output_dir),
        "--verbose",
        "False",
        "--fp16",
        "False",
    ]
    if settings.whisper_language:
        command.extend(["--language", settings.whisper_language])

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise VoiceChatProviderError(
            "Whisper failed to transcribe the turn. "
            f"stderr: {stderr.decode('utf-8', errors='ignore').strip()}"
        )

    transcript_path = output_dir / f"{audio_path.stem}.txt"
    if not transcript_path.exists():
        raise VoiceChatProviderError("Whisper finished without producing a transcript file.")

    return transcript_path.read_text(encoding="utf-8").strip()


def _transcribe_with_python_whisper(
    audio_path: Path,
    output_dir: Path,
    settings: Settings,
    duration_seconds: Optional[float],
) -> str:
    whisper_model = _get_whisper_model(settings)
    transcribe_kwargs = {
        "fp16": False,
        "verbose": False,
        "condition_on_previous_text": False,
    }
    if settings.whisper_language:
        transcribe_kwargs["language"] = settings.whisper_language

    if (
        duration_seconds
        and settings.whisper_chunk_duration_seconds > 0
        and duration_seconds > settings.whisper_chunk_duration_seconds
        and settings.ffmpeg_command
    ):
        chunks_dir = output_dir / "whisper-chunks"
        chunk_paths = _split_audio_into_chunks(
            audio_path,
            chunks_dir,
            settings,
        )
        segments = [
            ((whisper_model.transcribe(str(chunk_path), **transcribe_kwargs) or {}).get("text") or "").strip()
            for chunk_path in chunk_paths
        ]
        return " ".join(segment for segment in segments if segment).strip()

    result = whisper_model.transcribe(str(audio_path), **transcribe_kwargs) or {}
    return (result.get("text") or "").strip()


async def _generate_reply(
    payload: VoiceChatTurnRequest,
    transcript: str,
    settings: Settings,
    ollama_runtime: OllamaRuntimeConfig,
) -> str:
    request_body = {
        "model": ollama_runtime.model,
        "stream": False,
        "keep_alive": ollama_runtime.keep_alive,
        "messages": build_chat_messages(
            payload.history,
            transcript,
            conversation_summary=payload.conversationSummary,
        ),
        "options": {"temperature": 0.7},
    }

    headers = {}
    if ollama_runtime.api_token:
        headers["Authorization"] = f"Bearer {ollama_runtime.api_token}"

    client = _get_ollama_client(ollama_runtime.base_url, settings.http_timeout_seconds)
    response = await client.post(ollama_runtime.chat_url, json=request_body, headers=headers or None)

    if response.status_code >= 400:
        raise VoiceChatProviderError(
            f"Ollama returned {response.status_code}: {response.text.strip()}"
        )

    body = response.json()
    return ((body.get("message") or {}).get("content") or "").strip()


def default_ollama_runtime(settings: Settings) -> OllamaRuntimeConfig:
    base_url = settings.ollama_base_url.strip().rstrip("/")
    return OllamaRuntimeConfig(
        base_url=base_url,
        chat_url=_resolve_ollama_chat_url(base_url),
        model=settings.ollama_model,
        keep_alive=settings.ollama_keep_alive,
        api_token=None,
    )


def ollama_runtime_for_connection(settings: Settings, connection: dict[str, str]) -> OllamaRuntimeConfig:
    endpoint_url = (connection.get("endpoint_url") or "").strip().rstrip("/")
    if not endpoint_url:
        return default_ollama_runtime(settings)

    return OllamaRuntimeConfig(
        base_url=_resolve_ollama_base_url(endpoint_url),
        chat_url=_resolve_ollama_chat_url(endpoint_url),
        model=(connection.get("model") or settings.ollama_model).strip() or settings.ollama_model,
        keep_alive=settings.ollama_keep_alive,
        api_token=(connection.get("api_token") or "").strip() or None,
    )


async def _synthesize_speech(
    response_text: str,
    output_dir: Path,
    settings: Settings,
    voice_profile_id: Optional[str],
) -> bytes:
    profile_manifest = _load_voice_profile_manifest(settings, voice_profile_id)
    if profile_manifest:
        provider = (profile_manifest.get("provider") or "").strip().lower()
        reference_audio_path = (profile_manifest.get("reference_audio_path") or "").strip()
        adapter_path = (profile_manifest.get("adapter_path") or "").strip()
        cosyvoice_model_dir = (profile_manifest.get("cosyvoice_model_dir") or "").strip()
        if provider in {"xtts-reference", "cosyvoice-reference", "cosyvoice-adapter", "cosyvoice-finetuned"}:
            if provider == "cosyvoice-adapter":
                if not adapter_path or not Path(adapter_path).is_dir():
                    raise VoiceChatProviderError("The saved CosyVoice adapter artifact is missing.")
                if not settings.is_cosyvoice_enabled:
                    raise VoiceChatProviderError(
                        "A personalized voice profile is ready for CosyVoice, but CosyVoice is not configured."
                    )
                return await _synthesize_with_cosyvoice_adapter(
                    response_text,
                    output_dir,
                    settings,
                    adapter_path,
                    reference_audio_path,
                    voice_profile_id,
                    (profile_manifest.get("base_model") or "").strip() or None,
                )
            if provider == "cosyvoice-finetuned":
                if not cosyvoice_model_dir or not Path(cosyvoice_model_dir).is_dir():
                    raise VoiceChatProviderError("The saved CosyVoice fine-tuned model directory is missing.")
                if not settings.is_cosyvoice_enabled:
                    raise VoiceChatProviderError(
                        "A personalized voice profile is ready for CosyVoice, but CosyVoice is not configured."
                    )
                return await _synthesize_with_cosyvoice_finetuned(
                    response_text,
                    output_dir,
                    settings,
                    cosyvoice_model_dir,
                    reference_audio_path,
                    voice_profile_id,
                    (profile_manifest.get("speaker_id") or "").strip() or None,
                )
            if not reference_audio_path or not Path(reference_audio_path).exists():
                raise VoiceChatProviderError("The saved personalized voice reference audio is missing.")
            if provider == "cosyvoice-reference":
                if not settings.is_cosyvoice_enabled:
                    raise VoiceChatProviderError(
                        "A personalized voice profile is ready for CosyVoice, but CosyVoice is not configured."
                    )
                return await _synthesize_with_cosyvoice(
                    response_text,
                    output_dir,
                    settings,
                    reference_audio_path,
                    voice_profile_id,
                )
            if not settings.is_xtts_enabled:
                raise VoiceChatProviderError(
                    "A personalized voice profile is ready, but XTTS is not configured on the backend."
                )
            return await _synthesize_with_xtts(
                response_text,
                output_dir,
                settings,
                reference_audio_path,
            )

    if profile_manifest and settings.tts_provider == "cosyvoice" and settings.is_cosyvoice_enabled:
        reference_audio_path = (profile_manifest.get("reference_audio_path") or "").strip()
        if reference_audio_path and Path(reference_audio_path).exists():
            return await _synthesize_with_cosyvoice(
                response_text,
                output_dir,
                settings,
                reference_audio_path,
                voice_profile_id,
            )

    if settings.tts_provider == "cosyvoice" and settings.is_cosyvoice_enabled:
        if voice_profile_id:
            raise VoiceChatProviderError(
                "CosyVoice is configured, but the selected voice profile is missing its reference audio."
            )
        raise VoiceChatProviderError(
            "CosyVoice is configured for TTS, but this request does not include a personalized voice profile."
        )

    if not settings.piper_command or not settings.piper_model_path:
        raise VoiceChatProviderError("No compatible TTS provider is configured for this request.")

    piper_model_path, piper_config_path = _resolve_voice_profile_paths(settings, voice_profile_id)
    return await _synthesize_with_piper(
        response_text,
        output_dir,
        settings,
        piper_model_path,
        piper_config_path,
    )


async def _synthesize_with_piper(
    response_text: str,
    output_dir: Path,
    settings: Settings,
    piper_model_path: str,
    piper_config_path: Optional[str],
) -> bytes:
    output_path = output_dir / "response.wav"
    command = [
        settings.piper_command,
        "--model",
        piper_model_path,
        "--output_file",
        str(output_path),
    ]
    if piper_config_path:
        command.extend(["--config", piper_config_path])

    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate(response_text.encode("utf-8"))
    if process.returncode != 0:
        raise VoiceChatProviderError(
            "Piper failed to synthesize the response. "
            f"stderr: {stderr.decode('utf-8', errors='ignore').strip()}"
        )

    if not output_path.exists():
        raise VoiceChatProviderError("Piper finished without producing a WAV file.")

    return output_path.read_bytes()


async def _synthesize_with_xtts(
    response_text: str,
    output_dir: Path,
    settings: Settings,
    reference_audio_path: str,
) -> bytes:
    output_path = output_dir / "response.wav"
    await asyncio.to_thread(
        _xtts_to_file,
        settings,
        response_text,
        reference_audio_path,
        output_path,
    )

    if not output_path.exists():
        raise VoiceChatProviderError("XTTS finished without producing a WAV file.")

    return output_path.read_bytes()


async def _synthesize_with_cosyvoice(
    response_text: str,
    output_dir: Path,
    settings: Settings,
    reference_audio_path: str,
    voice_profile_id: Optional[str],
) -> bytes:
    assert settings.cosyvoice_command is not None

    output_path = output_dir / "response.wav"

    def _run() -> None:
        env = os.environ.copy()
        env.update(
            {
                "COSYVOICE_TEXT": response_text,
                "COSYVOICE_REFERENCE_AUDIO_PATH": reference_audio_path,
                "COSYVOICE_OUTPUT_PATH": str(output_path),
                "COSYVOICE_LANGUAGE": settings.cosyvoice_language,
                "VOICE_PROFILE_ID": voice_profile_id or "",
            }
        )
        completed = subprocess.run(
            settings.cosyvoice_command,
            shell=True,
            env=env,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "CosyVoice command failed").strip()
            raise VoiceChatProviderError(f"CosyVoice failed to synthesize the response: {detail}")

    await asyncio.to_thread(_run)
    if not output_path.exists():
        raise VoiceChatProviderError("CosyVoice finished without producing a WAV file.")

    return output_path.read_bytes()


async def _synthesize_with_cosyvoice_adapter(
    response_text: str,
    output_dir: Path,
    settings: Settings,
    adapter_path: str,
    reference_audio_path: str,
    voice_profile_id: Optional[str],
    base_model: Optional[str],
) -> bytes:
    assert settings.cosyvoice_command is not None

    output_path = output_dir / "response.wav"

    def _run() -> None:
        env = os.environ.copy()
        env.update(
            {
                "COSYVOICE_TEXT": response_text,
                "COSYVOICE_REFERENCE_AUDIO_PATH": reference_audio_path,
                "COSYVOICE_OUTPUT_PATH": str(output_path),
                "COSYVOICE_LANGUAGE": settings.cosyvoice_language,
                "COSYVOICE_ADAPTER_PATH": adapter_path,
                "COSYVOICE_INFERENCE_MODE": "adapter",
                "VOICE_PROFILE_ID": voice_profile_id or "",
            }
        )
        if base_model:
            env["COSYVOICE_ADAPTER_BASE_MODEL"] = base_model
        completed = subprocess.run(
            settings.cosyvoice_command,
            shell=True,
            env=env,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "CosyVoice adapter command failed").strip()
            raise VoiceChatProviderError(f"CosyVoice adapter failed to synthesize the response: {detail}")

    await asyncio.to_thread(_run)
    if not output_path.exists():
        raise VoiceChatProviderError("CosyVoice adapter finished without producing a WAV file.")

    return output_path.read_bytes()


async def _synthesize_with_cosyvoice_finetuned(
    response_text: str,
    output_dir: Path,
    settings: Settings,
    cosyvoice_model_dir: str,
    reference_audio_path: str,
    voice_profile_id: Optional[str],
    speaker_id: Optional[str],
) -> bytes:
    assert settings.cosyvoice_command is not None

    output_path = output_dir / "response.wav"

    def _run() -> None:
        env = os.environ.copy()
        env.update(
            {
                "COSYVOICE_TEXT": response_text,
                "COSYVOICE_OUTPUT_PATH": str(output_path),
                "COSYVOICE_LANGUAGE": settings.cosyvoice_language,
                "COSYVOICE_MODEL_DIR_OVERRIDE": cosyvoice_model_dir,
                "COSYVOICE_INFERENCE_MODE": "sft",
                "VOICE_PROFILE_ID": voice_profile_id or "",
            }
        )
        if reference_audio_path:
            env["COSYVOICE_REFERENCE_AUDIO_PATH"] = reference_audio_path
        if speaker_id:
            env["COSYVOICE_SPK_ID"] = speaker_id
        completed = subprocess.run(
            settings.cosyvoice_command,
            shell=True,
            env=env,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "CosyVoice fine-tuned command failed").strip()
            raise VoiceChatProviderError(f"CosyVoice fine-tuned synthesis failed: {detail}")

    await asyncio.to_thread(_run)
    if not output_path.exists():
        raise VoiceChatProviderError("CosyVoice fine-tuned synthesis finished without producing a WAV file.")

    return output_path.read_bytes()


def _resolve_voice_profile_paths(
    settings: Settings,
    voice_profile_id: Optional[str],
) -> tuple[str, Optional[str]]:
    default_model_path = settings.piper_model_path
    assert default_model_path is not None

    if not voice_profile_id or not settings.voice_profile_models_dir:
        return default_model_path, settings.piper_config_path

    profiles_dir = Path(settings.voice_profile_models_dir)
    candidate_model_path = profiles_dir / f"{voice_profile_id}.onnx"
    candidate_config_path = profiles_dir / f"{voice_profile_id}.onnx.json"
    if candidate_model_path.exists():
        return (
            str(candidate_model_path),
            str(candidate_config_path) if candidate_config_path.exists() else settings.piper_config_path,
        )

    return default_model_path, settings.piper_config_path


def _load_voice_profile_manifest(
    settings: Settings,
    voice_profile_id: Optional[str],
) -> Optional[dict]:
    if not voice_profile_id or not settings.voice_profile_manifests_dir:
        return None

    manifest_path = Path(settings.voice_profile_manifests_dir) / f"{voice_profile_id}.json"
    try:
        if _DURABLE_VOICE_PROFILE_STORE.enabled:
            manifest = _DURABLE_VOICE_PROFILE_STORE.ensure_manifest_local(voice_profile_id, manifest_path)
            if manifest:
                return _DURABLE_VOICE_PROFILE_STORE.ensure_artifacts_local(manifest)
        if not manifest_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _DURABLE_VOICE_PROFILE_STORE.enabled:
            manifest = _DURABLE_VOICE_PROFILE_STORE.ensure_artifacts_local(manifest)
        return manifest
    except Exception as exc:
        raise VoiceChatProviderError(f"Failed to read the saved voice profile manifest: {exc}") from exc


def _xtts_to_file(
    settings: Settings,
    response_text: str,
    reference_audio_path: str,
    output_path: Path,
) -> None:
    xtts = _get_xtts_model(settings)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xtts.tts_to_file(
        text=response_text,
        speaker_wav=reference_audio_path,
        language=settings.xtts_language,
        file_path=str(output_path),
    )


def _get_xtts_model(settings: Settings):
    global _XTTS_MODEL, _XTTS_MODEL_NAME

    if not settings.xtts_model_name:
        raise VoiceChatProviderError("XTTS model name is not configured.")

    with _XTTS_LOCK:
        if _XTTS_MODEL is None or _XTTS_MODEL_NAME != settings.xtts_model_name:
            try:
                from TTS.api import TTS
            except Exception as exc:
                raise VoiceChatProviderError(
                    "XTTS dependencies are not available. Install the TTS stack in the backend venv."
                ) from exc

            try:
                _XTTS_MODEL = TTS(settings.xtts_model_name, progress_bar=False).to("cpu")
                _XTTS_MODEL_NAME = settings.xtts_model_name
            except Exception as exc:
                raise VoiceChatProviderError(
                    f"XTTS failed to initialize with model '{settings.xtts_model_name}': {exc}"
                ) from exc

    return _XTTS_MODEL


def _get_whisper_model(settings: Settings):
    global _WHISPER_MODEL, _WHISPER_MODEL_NAME

    with _WHISPER_LOCK:
        if _WHISPER_MODEL is None or _WHISPER_MODEL_NAME != settings.whisper_model:
            try:
                import whisper
            except Exception as exc:
                raise VoiceChatProviderError(
                    "Whisper dependencies are not available in-process. Falling back to the CLI is required."
                ) from exc

            try:
                _WHISPER_MODEL = whisper.load_model(settings.whisper_model)
                _WHISPER_MODEL_NAME = settings.whisper_model
            except Exception as exc:
                raise VoiceChatProviderError(
                    f"Whisper failed to initialize with model '{settings.whisper_model}': {exc}"
                ) from exc

    return _WHISPER_MODEL


def _split_audio_into_chunks(
    audio_path: Path,
    chunks_dir: Path,
    settings: Settings,
) -> list[Path]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_pattern = chunks_dir / "chunk-%03d.wav"
    command = [
        settings.ffmpeg_command,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-segment_time",
        str(settings.whisper_chunk_duration_seconds),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(chunk_pattern),
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        raise VoiceChatProviderError(
            "ffmpeg failed while splitting the audio for chunked transcription. "
            f"stderr: {completed.stderr.decode('utf-8', errors='ignore').strip()}"
        )

    chunk_paths = sorted(chunks_dir.glob("chunk-*.wav"))
    if not chunk_paths:
        raise VoiceChatProviderError("ffmpeg finished without producing any audio chunks.")
    return chunk_paths


def _get_ollama_client(base_url: str, timeout_seconds: float) -> httpx.AsyncClient:
    with _OLLAMA_CLIENT_LOCK:
        client = _OLLAMA_CLIENTS.get(base_url)
        if client is None:
            client = httpx.AsyncClient(
                timeout=timeout_seconds,
                headers={"Connection": "keep-alive"},
            )
            _OLLAMA_CLIENTS[base_url] = client
        return client


async def _prewarm_whisper(settings: Settings) -> None:
    started_at = time.perf_counter()
    try:
        await asyncio.to_thread(_get_whisper_model, settings)
        print(f"[startup] whisper-prewarm={time.perf_counter() - started_at:.2f}s model={settings.whisper_model}")
    except Exception as exc:
        print(f"[startup] whisper-prewarm-failed model={settings.whisper_model} error={exc}")


async def _prewarm_ollama(settings: Settings) -> None:
    started_at = time.perf_counter()
    try:
        runtime = default_ollama_runtime(settings)
        client = _get_ollama_client(runtime.base_url, settings.http_timeout_seconds)
        response = await client.post(
            runtime.chat_url,
            json={
                "model": runtime.model,
                "keep_alive": runtime.keep_alive,
                "stream": False,
                "messages": [{"role": "user", "content": "Ping."}],
                "options": {
                    "temperature": 0,
                    "num_predict": 1,
                },
            },
        )
        if response.status_code >= 400:
            print(
                "[startup] ollama-prewarm-failed "
                f"model={runtime.model} status={response.status_code} "
                f"body={response.text.strip()}"
            )
            return
        print(f"[startup] ollama-prewarm={time.perf_counter() - started_at:.2f}s model={runtime.model}")
    except Exception as exc:
        print(f"[startup] ollama-prewarm-failed model={settings.ollama_model} error={exc}")


def _resolve_ollama_base_url(endpoint_url: str) -> str:
    trimmed = endpoint_url.strip().rstrip("/")
    if trimmed.endswith("/api/chat"):
        return trimmed[:-9].rstrip("/")
    return trimmed


def _resolve_ollama_chat_url(endpoint_url: str) -> str:
    trimmed = endpoint_url.strip().rstrip("/")
    if trimmed.endswith("/api/chat"):
        return trimmed
    return f"{trimmed}/api/chat"


def _log_turn_timing(
    duration_seconds: float,
    transcription_seconds: float,
    llm_seconds: float,
    synthesis_seconds: float,
    total_seconds: float,
) -> None:
    print(
        "[voice-chat] "
        f"audio={duration_seconds:.2f}s "
        f"transcribe={transcription_seconds:.2f}s "
        f"llm={llm_seconds:.2f}s "
        f"tts={synthesis_seconds:.2f}s "
        f"total={total_seconds:.2f}s"
    )
