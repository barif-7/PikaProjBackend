"""
Async voice pipeline with per-stage timeouts and explicit TTS fallback.

This module wraps the core inference functions from providers.py to add:

- **Stage callbacks** for job-store integration (TRANSCRIBING → GENERATING →
  SYNTHESIZING).
- **Per-stage timeouts** via ``asyncio.wait_for`` so that a slow Whisper run
  cannot block the LLM slot indefinitely, and a stalled TTS call does not
  prevent the reply text from being returned.
- **Explicit TTS fallback chain**: primary provider (CosyVoice / XTTS /
  default Piper) → Piper safety net (when a non-Piper primary is configured)
  → text-only.

The synchronous ``/voice-chat/turn`` route in ``main.py`` keeps using
``providers.generate_turn_response`` unchanged.  ``run_pipeline`` is called
exclusively from the async job path (``POST /voice-chat/jobs``).
"""
from __future__ import annotations

import asyncio
import base64
import tempfile
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

from .audio_upload import AudioUploadError, decode_uploaded_audio
from .config import Settings
from .models import VoiceChatTurnRequest, VoiceChatTurnResponse
from .providers import (
    OllamaRuntimeConfig,
    VoiceChatProviderError,
    _decode_audio,
    _generate_reply,
    _log_turn_timing,
    _resolve_voice_profile_paths,
    _stream_reply,
    _synthesize_speech,
    _synthesize_with_piper,
    _transcribe_audio,
    default_ollama_runtime,
)

# Characters that mark the end of a sentence worth synthesizing on its own.
_SENTENCE_TERMINATORS = ".!?\n"

# Async callable that receives the stage name string, e.g. "transcribing".
StageCallback = Optional[Callable[[str], Awaitable[None]]]


async def run_pipeline(
    payload: VoiceChatTurnRequest,
    settings: Settings,
    ollama_runtime: Optional[OllamaRuntimeConfig] = None,
    on_stage: StageCallback = None,
) -> VoiceChatTurnResponse:
    """
    Execute the full voice pipeline: decode → transcribe → generate → synthesize.

    Parameters
    ----------
    payload:
        The validated ``VoiceChatTurnRequest``.
    settings:
        Application settings.  The ``stt_timeout_seconds``,
        ``llm_timeout_seconds``, and ``tts_timeout_seconds`` fields control
        per-stage timeouts.
    ollama_runtime:
        Optional Ollama endpoint override.  Defaults to
        ``default_ollama_runtime(settings)``.
    on_stage:
        Optional *async* callback invoked at the start of each stage with the
        stage name string (``"transcribing"``, ``"generating"``,
        ``"synthesizing"``).  Used to advance the job store stage without
        coupling this module to ``VoiceJobStore``.

    Returns
    -------
    VoiceChatTurnResponse
        Always returns a response.  If TTS fails at every level the response
        will have ``responseAudioBase64=None`` (text-only) rather than raising.

    Raises
    ------
    VoiceChatProviderError
        If the STT or LLM stage fails.
    asyncio.TimeoutError
        If the STT or LLM stage exceeds its configured timeout.
    """
    request_started_at = time.perf_counter()
    runtime = ollama_runtime or default_ollama_runtime(settings)

    try:
        audio_bytes = decode_uploaded_audio(payload.audioBase64, payload.audioChunks)
    except AudioUploadError as exc:
        raise VoiceChatProviderError(str(exc)) from exc

    with tempfile.TemporaryDirectory(prefix="pika-pipeline-") as temp_dir:
        temp_path = Path(temp_dir)
        audio_path = temp_path / payload.fileName
        audio_path.write_bytes(audio_bytes)

        # ── Stage: TRANSCRIBING ──────────────────────────────────────────
        if on_stage is not None:
            await on_stage("transcribing")
        transcription_started_at = time.perf_counter()

        transcript = await asyncio.wait_for(
            _transcribe_audio(audio_path, temp_path, settings, payload.durationSeconds),
            timeout=settings.stt_timeout_seconds,
        )
        if not transcript:
            raise VoiceChatProviderError("Whisper returned an empty transcript.")

        # ── Stage: GENERATING ────────────────────────────────────────────
        if on_stage is not None:
            await on_stage("generating")
        reply_started_at = time.perf_counter()

        response_text = await asyncio.wait_for(
            _generate_reply(payload, transcript, settings, runtime),
            timeout=settings.llm_timeout_seconds,
        )
        if not response_text:
            raise VoiceChatProviderError("The language model returned an empty response.")

        # ── Stage: SYNTHESIZING ──────────────────────────────────────────
        if on_stage is not None:
            await on_stage("synthesizing")
        synthesis_started_at = time.perf_counter()

        response_audio_base64: Optional[str] = None
        response_audio_mime: Optional[str] = None

        if settings.is_tts_enabled:
            response_audio_base64, response_audio_mime = await _synthesize_with_fallback(
                response_text, temp_path, settings, payload.voiceProfileID
            )

        _log_turn_timing(
            duration_seconds=payload.durationSeconds,
            transcription_seconds=reply_started_at - transcription_started_at,
            llm_seconds=synthesis_started_at - reply_started_at,
            synthesis_seconds=(
                time.perf_counter() - synthesis_started_at if settings.is_tts_enabled else 0.0
            ),
            total_seconds=time.perf_counter() - request_started_at,
        )

        return VoiceChatTurnResponse(
            transcript=transcript,
            responseText=response_text,
            responseAudioBase64=response_audio_base64,
            responseAudioMimeType=response_audio_mime,
            error=None,
        )


def _split_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """Split ``buffer`` into complete sentences and a trailing remainder.

    A sentence is considered complete once a terminator (``.!?`` or newline)
    is seen.  The remainder (text after the last terminator) is returned so the
    caller can keep accumulating until the next terminator arrives.
    """
    sentences: list[str] = []
    start = 0
    for index, char in enumerate(buffer):
        if char in _SENTENCE_TERMINATORS:
            sentence = buffer[start : index + 1].strip()
            if sentence:
                sentences.append(sentence)
            start = index + 1
    return sentences, buffer[start:]


async def run_pipeline_streaming(
    payload: VoiceChatTurnRequest,
    settings: Settings,
    ollama_runtime: Optional[OllamaRuntimeConfig] = None,
):
    """Run the voice pipeline incrementally, yielding events as they are ready.

    Yields dicts with a ``type`` discriminator:

    - ``{"type": "transcript", "transcript": str}`` once STT completes.
    - ``{"type": "text", "delta": str}`` for each LLM token delta.
    - ``{"type": "audio", "audioBase64": str, "mimeType": str, "text": str}``
      for each synthesized sentence (omitted when TTS is disabled or fails).
    - ``{"type": "done", "responseText": str}`` at the end.

    Unlike :func:`run_pipeline`, this begins synthesizing speech sentence by
    sentence as the LLM streams, so the first audio reaches the client well
    before the full reply is generated.  The job/poll path is unaffected.
    """
    runtime = ollama_runtime or default_ollama_runtime(settings)

    try:
        audio_bytes = decode_uploaded_audio(payload.audioBase64, payload.audioChunks)
    except AudioUploadError as exc:
        raise VoiceChatProviderError(str(exc)) from exc

    with tempfile.TemporaryDirectory(prefix="pika-pipeline-stream-") as temp_dir:
        temp_path = Path(temp_dir)
        audio_path = temp_path / payload.fileName
        audio_path.write_bytes(audio_bytes)

        transcript = await asyncio.wait_for(
            _transcribe_audio(audio_path, temp_path, settings, payload.durationSeconds),
            timeout=settings.stt_timeout_seconds,
        )
        if not transcript:
            raise VoiceChatProviderError("Whisper returned an empty transcript.")
        yield {"type": "transcript", "transcript": transcript}

        full_reply_parts: list[str] = []
        buffer = ""
        sentence_index = 0

        async def _emit_sentence(sentence: str):
            nonlocal sentence_index
            if not settings.is_tts_enabled:
                return None
            sentence_dir = temp_path / f"sentence-{sentence_index}"
            sentence_dir.mkdir(parents=True, exist_ok=True)
            sentence_index += 1
            audio_b64, mime = await _synthesize_with_fallback(
                sentence, sentence_dir, settings, payload.voiceProfileID
            )
            if audio_b64 is None:
                return None
            return {
                "type": "audio",
                "audioBase64": audio_b64,
                "mimeType": mime,
                "text": sentence,
            }

        async for delta in _stream_reply(payload, transcript, settings, runtime):
            full_reply_parts.append(delta)
            buffer += delta
            yield {"type": "text", "delta": delta}

            sentences, buffer = _split_complete_sentences(buffer)
            for sentence in sentences:
                event = await _emit_sentence(sentence)
                if event is not None:
                    yield event

        remainder = buffer.strip()
        if remainder:
            event = await _emit_sentence(remainder)
            if event is not None:
                yield event

        response_text = "".join(full_reply_parts).strip()
        if not response_text:
            raise VoiceChatProviderError("The language model returned an empty response.")

        yield {"type": "done", "responseText": response_text}


async def _synthesize_with_fallback(
    response_text: str,
    output_dir: Path,
    settings: Settings,
    voice_profile_id: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """
    Attempt TTS synthesis with an explicit provider fallback chain.

    Tries in order:
    1. Primary provider via ``_synthesize_speech`` (CosyVoice / XTTS /
       default Piper).
    2. Piper as a safety net — *only* when a non-Piper primary is configured
       and it failed (retrying Piper after Piper already failed is pointless).
    3. Text-only: return ``(None, None)`` — the reply text is still delivered
       to the client.

    Returns ``(base64_wav, "audio/wav")`` or ``(None, None)``.
    """
    # Attempt 1 — primary TTS provider
    try:
        wav_bytes = await asyncio.wait_for(
            _synthesize_speech(response_text, output_dir, settings, voice_profile_id),
            timeout=settings.tts_timeout_seconds,
        )
        return base64.b64encode(wav_bytes).decode("utf-8"), "audio/wav"
    except asyncio.TimeoutError:
        print(
            "[pipeline] tts-primary-timeout "
            f"timeout={settings.tts_timeout_seconds:.1f}s "
            "trying_piper_fallback=true"
        )
    except VoiceChatProviderError as exc:
        print(f"[pipeline] tts-primary-failed error={exc!r} trying_piper_fallback=true")

    # Attempt 2 — Piper safety net
    # Only useful when the primary provider was CosyVoice or XTTS.
    # If Piper itself was the primary (is_tts_enabled=True via piper_command
    # only), retrying it won't help.
    has_non_piper_primary = settings.is_cosyvoice_enabled or settings.is_xtts_enabled
    if has_non_piper_primary and settings.piper_command and settings.piper_model_path:
        try:
            piper_model, piper_config = _resolve_voice_profile_paths(settings, voice_profile_id)
            wav_bytes = await asyncio.wait_for(
                _synthesize_with_piper(
                    response_text, output_dir, settings, piper_model, piper_config
                ),
                timeout=settings.tts_timeout_seconds,
            )
            print("[pipeline] tts-piper-fallback-succeeded")
            return base64.b64encode(wav_bytes).decode("utf-8"), "audio/wav"
        except (asyncio.TimeoutError, VoiceChatProviderError) as exc:
            print(f"[pipeline] tts-piper-fallback-failed error={exc!r} returning_text_only=true")

    # Attempt 3 — text-only (no audio)
    print("[pipeline] tts-all-providers-failed returning_text_only=true")
    return None, None
