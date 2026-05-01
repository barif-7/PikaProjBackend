"""
Tests for Phase 3 voice-pipeline scalability work:

- app.voice_job_store.VoiceJobStore — lifecycle, TTL, concurrent-job limit
- app.voice_pipeline.run_pipeline — stage callbacks, per-stage timeouts,
  TTS fallback chain
- app.voice_pipeline._synthesize_with_fallback — explicit provider fallback

Google Cloud libraries are not installed in the local dev environment, so we
inject stub modules into sys.modules before any imports that transitively pull
in durable_storage.py (which has a hard ``from google.cloud import ...``).
"""
from __future__ import annotations

import asyncio
import dataclasses
import sys
import time
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Stub out google.cloud before any app imports that touch durable_storage.py
# ---------------------------------------------------------------------------
_mock_gcloud = MagicMock()
for _mod in ("google", "google.cloud", "google.cloud.firestore", "google.cloud.storage"):
    sys.modules.setdefault(_mod, _mock_gcloud)

from app.voice_job_store import JobStage, VoiceJob, VoiceJobStore  # noqa: E402
from app.models import VoiceChatTurnResponse  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(text: str = "hello", with_audio: bool = False) -> VoiceChatTurnResponse:
    return VoiceChatTurnResponse(
        transcript="hi",
        responseText=text,
        responseAudioBase64="abc" if with_audio else None,
        responseAudioMimeType="audio/wav" if with_audio else None,
        error=None,
    )


# ---------------------------------------------------------------------------
# VoiceJobStore tests
# ---------------------------------------------------------------------------

class VoiceJobStoreCreateTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_returns_queued_job(self) -> None:
        store = VoiceJobStore()
        job = await store.create()
        self.assertIsInstance(job, VoiceJob)
        self.assertEqual(job.stage, JobStage.QUEUED)
        self.assertTrue(job.job_id)

    async def test_create_ids_are_unique(self) -> None:
        store = VoiceJobStore()
        ids = {(await store.create()).job_id for _ in range(10)}
        self.assertEqual(len(ids), 10)

    async def test_max_jobs_raises_when_limit_reached(self) -> None:
        store = VoiceJobStore(max_jobs=2)
        await store.create()
        await store.create()
        with self.assertRaises(RuntimeError):
            await store.create()

    async def test_max_jobs_not_counted_against_terminal_jobs(self) -> None:
        """READY/FAILED jobs don't count toward the active-job limit."""
        store = VoiceJobStore(max_jobs=2)
        j1 = await store.create()
        j2 = await store.create()
        await store.complete(j1.job_id, _make_response())
        await store.fail(j2.job_id, "boom")
        # Both are now terminal — creating a new job must succeed.
        j3 = await store.create()
        self.assertEqual(j3.stage, JobStage.QUEUED)


class VoiceJobStoreStateTests(unittest.IsolatedAsyncioTestCase):
    async def _store_with_job(self) -> tuple[VoiceJobStore, VoiceJob]:
        store = VoiceJobStore()
        job = await store.create()
        return store, job

    async def test_update_stage_transcribing(self) -> None:
        store, job = await self._store_with_job()
        await store.update_stage(job.job_id, JobStage.TRANSCRIBING)
        fetched = await store.get(job.job_id)
        self.assertEqual(fetched.stage, JobStage.TRANSCRIBING)

    async def test_update_stage_silently_ignores_missing_job(self) -> None:
        store = VoiceJobStore()
        # Must not raise.
        await store.update_stage("no-such-id", JobStage.GENERATING)

    async def test_complete_stores_result(self) -> None:
        store, job = await self._store_with_job()
        result = _make_response("nice answer", with_audio=True)
        await store.complete(job.job_id, result)
        fetched = await store.get(job.job_id)
        self.assertEqual(fetched.stage, JobStage.READY)
        self.assertIs(fetched.result, result)
        self.assertIsNone(fetched.error)

    async def test_fail_stores_error(self) -> None:
        store, job = await self._store_with_job()
        await store.fail(job.job_id, "whisper crashed")
        fetched = await store.get(job.job_id)
        self.assertEqual(fetched.stage, JobStage.FAILED)
        self.assertEqual(fetched.error, "whisper crashed")
        self.assertIsNone(fetched.result)

    async def test_get_returns_none_for_unknown_job(self) -> None:
        store = VoiceJobStore()
        self.assertIsNone(await store.get("ghost-id"))

    async def test_count_active_excludes_terminal(self) -> None:
        store = VoiceJobStore()
        j1 = await store.create()
        j2 = await store.create()
        j3 = await store.create()
        await store.complete(j1.job_id, _make_response())
        await store.fail(j2.job_id, "err")
        # j3 is still QUEUED — active count should be 1.
        self.assertEqual(await store.count_active(), 1)

    async def test_count_active_in_progress_stages(self) -> None:
        store = VoiceJobStore()
        j1 = await store.create()
        j2 = await store.create()
        await store.update_stage(j1.job_id, JobStage.TRANSCRIBING)
        await store.update_stage(j2.job_id, JobStage.SYNTHESIZING)
        self.assertEqual(await store.count_active(), 2)


class VoiceJobStoreEvictionTests(unittest.IsolatedAsyncioTestCase):
    async def test_evict_removes_expired_terminal_jobs(self) -> None:
        store = VoiceJobStore(ttl_seconds=10.0)
        job = await store.create()
        await store.complete(job.job_id, _make_response())

        # Backdate updated_at to simulate expiry.
        store._jobs[job.job_id].updated_at = time.monotonic() - 20.0

        n = await store.evict_expired()
        self.assertEqual(n, 1)
        self.assertIsNone(await store.get(job.job_id))

    async def test_evict_keeps_active_jobs_regardless_of_age(self) -> None:
        store = VoiceJobStore(ttl_seconds=0.001)
        job = await store.create()
        # Backdate an *active* job — it must NOT be evicted.
        store._jobs[job.job_id].updated_at = time.monotonic() - 1000.0

        n = await store.evict_expired()
        self.assertEqual(n, 0)
        self.assertIsNotNone(await store.get(job.job_id))

    async def test_evict_keeps_recent_terminal_jobs(self) -> None:
        store = VoiceJobStore(ttl_seconds=3600.0)
        job = await store.create()
        await store.complete(job.job_id, _make_response())

        n = await store.evict_expired()
        self.assertEqual(n, 0)
        self.assertIsNotNone(await store.get(job.job_id))

    async def test_create_succeeds_after_terminal_jobs_free_slots(self) -> None:
        store = VoiceJobStore(ttl_seconds=10.0, max_jobs=1)
        job = await store.create()
        await store.complete(job.job_id, _make_response())
        # Terminal job doesn't count — new slot is immediately available.
        new_job = await store.create()
        self.assertEqual(new_job.stage, JobStage.QUEUED)


# ---------------------------------------------------------------------------
# voice_pipeline tests
# ---------------------------------------------------------------------------

def _base_settings():
    """Load settings and strip all TTS to get a clean base."""
    from app.config import load_settings
    s = load_settings()
    return dataclasses.replace(
        s,
        piper_command=None,
        piper_model_path=None,
        cosyvoice_command=None,
        cosyvoice_http_url=None,
        xtts_model_name=None,
    )


def _make_payload() -> MagicMock:
    payload = MagicMock()
    payload.audioBase64 = "dGVzdA=="  # "test"
    payload.fileName = "test.wav"
    payload.durationSeconds = 1.0
    payload.voiceProfileID = None
    payload.conversationSummary = None
    payload.history = []
    return payload


class VoicePipelineStageCallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_stage_callbacks_fired_in_order(self) -> None:
        from app.voice_pipeline import run_pipeline

        stages: list[str] = []

        async def _on_stage(stage: str) -> None:
            stages.append(stage)

        settings = _base_settings()
        payload = _make_payload()

        with (
            patch("app.voice_pipeline._decode_audio", return_value=b"fake"),
            patch("app.voice_pipeline._transcribe_audio", new_callable=AsyncMock, return_value="hello"),
            patch("app.voice_pipeline._generate_reply", new_callable=AsyncMock, return_value="world"),
        ):
            result = await run_pipeline(payload, settings, on_stage=_on_stage)

        # TTS disabled by base_settings (no piper/cosyvoice/xtts) — still reaches synthesizing.
        self.assertEqual(stages, ["transcribing", "generating", "synthesizing"])
        self.assertEqual(result.transcript, "hello")
        self.assertEqual(result.responseText, "world")

    async def test_stt_timeout_propagates(self) -> None:
        from app.voice_pipeline import run_pipeline

        settings = dataclasses.replace(_base_settings(), stt_timeout_seconds=0.001)
        payload = _make_payload()

        async def _slow(*args, **kwargs):
            await asyncio.sleep(10)

        with (
            patch("app.voice_pipeline._decode_audio", return_value=b"fake"),
            patch("app.voice_pipeline._transcribe_audio", side_effect=_slow),
        ):
            with self.assertRaises(asyncio.TimeoutError):
                await run_pipeline(payload, settings)

    async def test_llm_timeout_propagates(self) -> None:
        from app.voice_pipeline import run_pipeline

        settings = dataclasses.replace(_base_settings(), llm_timeout_seconds=0.001)
        payload = _make_payload()

        async def _slow(*args, **kwargs):
            await asyncio.sleep(10)

        with (
            patch("app.voice_pipeline._decode_audio", return_value=b"fake"),
            patch("app.voice_pipeline._transcribe_audio", new_callable=AsyncMock, return_value="hello"),
            patch("app.voice_pipeline._generate_reply", side_effect=_slow),
        ):
            with self.assertRaises(asyncio.TimeoutError):
                await run_pipeline(payload, settings)

    async def test_no_on_stage_callback_is_fine(self) -> None:
        """run_pipeline must not raise when on_stage is None."""
        from app.voice_pipeline import run_pipeline

        settings = _base_settings()
        payload = _make_payload()

        with (
            patch("app.voice_pipeline._decode_audio", return_value=b"fake"),
            patch("app.voice_pipeline._transcribe_audio", new_callable=AsyncMock, return_value="hi"),
            patch("app.voice_pipeline._generate_reply", new_callable=AsyncMock, return_value="hey"),
        ):
            result = await run_pipeline(payload, settings, on_stage=None)

        self.assertEqual(result.responseText, "hey")


class SynthesizeWithFallbackTests(unittest.IsolatedAsyncioTestCase):
    """_synthesize_with_fallback: explicit TTS provider fallback chain."""

    async def test_primary_succeeds_returns_audio(self) -> None:
        from app.voice_pipeline import _synthesize_with_fallback

        settings = _base_settings()

        with patch("app.voice_pipeline._synthesize_speech",
                   new_callable=AsyncMock, return_value=b"\x00\x01"):
            b64, mime = await _synthesize_with_fallback("hello", Path("/tmp"), settings, None)

        self.assertIsNotNone(b64)
        self.assertEqual(mime, "audio/wav")

    async def test_primary_fails_piper_fallback_used(self) -> None:
        from app.providers import VoiceChatProviderError
        from app.voice_pipeline import _synthesize_with_fallback

        # CosyVoice is configured → non-piper primary → Piper fallback eligible.
        settings = dataclasses.replace(
            _base_settings(),
            cosyvoice_command="cosyvoice",
            piper_command="piper",
            piper_model_path="/model.onnx",
        )

        with (
            patch("app.voice_pipeline._synthesize_speech", new_callable=AsyncMock,
                  side_effect=VoiceChatProviderError("cosyvoice failed")),
            patch("app.voice_pipeline._synthesize_with_piper", new_callable=AsyncMock,
                  return_value=b"\x00\x02"),
            patch("app.voice_pipeline._resolve_voice_profile_paths",
                  return_value=("/model.onnx", None)),
        ):
            b64, mime = await _synthesize_with_fallback("hello", Path("/tmp"), settings, None)

        self.assertIsNotNone(b64)
        self.assertEqual(mime, "audio/wav")

    async def test_primary_and_piper_both_fail_returns_text_only(self) -> None:
        from app.providers import VoiceChatProviderError
        from app.voice_pipeline import _synthesize_with_fallback

        settings = dataclasses.replace(
            _base_settings(),
            cosyvoice_command="cosyvoice",
            piper_command="piper",
            piper_model_path="/model.onnx",
        )

        with (
            patch("app.voice_pipeline._synthesize_speech", new_callable=AsyncMock,
                  side_effect=VoiceChatProviderError("cosyvoice failed")),
            patch("app.voice_pipeline._synthesize_with_piper", new_callable=AsyncMock,
                  side_effect=VoiceChatProviderError("piper failed")),
            patch("app.voice_pipeline._resolve_voice_profile_paths",
                  return_value=("/model.onnx", None)),
        ):
            b64, mime = await _synthesize_with_fallback("hello", Path("/tmp"), settings, None)

        self.assertIsNone(b64)
        self.assertIsNone(mime)

    async def test_piper_fallback_skipped_when_piper_is_only_provider(self) -> None:
        """When only Piper is configured, failing once gives text-only without a second attempt."""
        from app.providers import VoiceChatProviderError
        from app.voice_pipeline import _synthesize_with_fallback

        # No cosyvoice, no xtts — Piper is the only TTS.
        settings = dataclasses.replace(
            _base_settings(),
            piper_command="piper",
            piper_model_path="/model.onnx",
        )

        piper_call_count = 0

        async def _count_piper(*args, **kwargs):
            nonlocal piper_call_count
            piper_call_count += 1
            raise VoiceChatProviderError("piper fail")

        with (
            patch("app.voice_pipeline._synthesize_speech", new_callable=AsyncMock,
                  side_effect=VoiceChatProviderError("piper via primary")),
            patch("app.voice_pipeline._synthesize_with_piper", side_effect=_count_piper),
        ):
            b64, mime = await _synthesize_with_fallback("hello", Path("/tmp"), settings, None)

        # Piper must not be tried a second time.
        self.assertEqual(piper_call_count, 0)
        self.assertIsNone(b64)
        self.assertIsNone(mime)

    async def test_primary_timeout_tries_piper_fallback(self) -> None:
        from app.voice_pipeline import _synthesize_with_fallback

        settings = dataclasses.replace(
            _base_settings(),
            cosyvoice_command="cosyvoice",
            piper_command="piper",
            piper_model_path="/model.onnx",
            tts_timeout_seconds=0.001,
        )

        async def _slow(*args, **kwargs):
            await asyncio.sleep(10)

        with (
            patch("app.voice_pipeline._synthesize_speech", side_effect=_slow),
            patch("app.voice_pipeline._synthesize_with_piper", new_callable=AsyncMock,
                  return_value=b"\x00\x03"),
            patch("app.voice_pipeline._resolve_voice_profile_paths",
                  return_value=("/m.onnx", None)),
        ):
            b64, mime = await _synthesize_with_fallback("hello", Path("/tmp"), settings, None)

        self.assertIsNotNone(b64)
        self.assertEqual(mime, "audio/wav")


if __name__ == "__main__":
    unittest.main()
