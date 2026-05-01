"""
Voice-chat job store abstractions.

The default ``VoiceJobStore`` remains an in-memory implementation for local
development and tests, but this module now exposes the common async interface
used by both the in-memory and Firestore-backed distributed store.
"""
from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol

from .models import VoiceChatTurnRequest, VoiceChatTurnResponse
from .providers import OllamaRuntimeConfig


class JobStage(str, Enum):
    QUEUED = "queued"
    TRANSCRIBING = "transcribing"
    GENERATING = "generating"
    SYNTHESIZING = "synthesizing"
    READY = "ready"
    FAILED = "failed"


_TERMINAL_STAGES = {JobStage.READY, JobStage.FAILED}


@dataclass
class VoiceJob:
    job_id: str
    stage: JobStage = JobStage.QUEUED
    result: Optional[VoiceChatTurnResponse] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)


@dataclass
class ClaimedVoiceJob:
    job_id: str
    payload: VoiceChatTurnRequest
    ollama_runtime: Optional[OllamaRuntimeConfig] = None


class VoiceJobStoreProtocol(Protocol):
    async def create(
        self,
        payload: Optional[VoiceChatTurnRequest] = None,
        ollama_runtime: Optional[OllamaRuntimeConfig] = None,
    ) -> VoiceJob:
        ...

    async def claim_next(self, worker_id: str) -> Optional[ClaimedVoiceJob]:
        ...

    async def update_stage(self, job_id: str, stage: JobStage) -> None:
        ...

    async def complete(self, job_id: str, result: VoiceChatTurnResponse) -> None:
        ...

    async def fail(self, job_id: str, error: str) -> None:
        ...

    async def get(self, job_id: str) -> Optional[VoiceJob]:
        ...

    async def count_active(self) -> int:
        ...

    async def evict_expired(self) -> int:
        ...


class VoiceJobStore:
    """
    Thread-safe (asyncio.Lock) in-memory store for voice pipeline jobs.

    This powers local development and unit tests.  Jobs are submitted into the
    store, then background workers call ``claim_next()`` to execute them.
    """

    def __init__(self, ttl_seconds: float = 600.0, max_jobs: int = 10) -> None:
        self._jobs: dict[str, VoiceJob] = {}
        self._requests: dict[str, ClaimedVoiceJob] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self._ttl = ttl_seconds
        self._max_jobs = max_jobs

    async def create(
        self,
        payload: Optional[VoiceChatTurnRequest] = None,
        ollama_runtime: Optional[OllamaRuntimeConfig] = None,
    ) -> VoiceJob:
        async with self._lock:
            active = sum(1 for job in self._jobs.values() if job.stage not in _TERMINAL_STAGES)
            if active >= self._max_jobs:
                raise RuntimeError(
                    f"Too many concurrent voice jobs ({active}/{self._max_jobs}).  "
                    "Retry after a moment."
                )
            job_id = secrets.token_urlsafe(16)
            job = VoiceJob(job_id=job_id)
            self._jobs[job_id] = job
            if payload is not None:
                self._requests[job_id] = ClaimedVoiceJob(
                    job_id=job_id,
                    payload=payload,
                    ollama_runtime=ollama_runtime,
                )
            return job

    async def claim_next(self, worker_id: str) -> Optional[ClaimedVoiceJob]:
        del worker_id  # worker ownership is implicit in-process for local mode
        async with self._lock:
            queued = sorted(
                (job for job in self._jobs.values() if job.stage == JobStage.QUEUED),
                key=lambda job: job.created_at,
            )
            if not queued:
                return None
            job = queued[0]
            request = self._requests.get(job.job_id)
            if request is None:
                job.stage = JobStage.FAILED
                job.error = "Job payload is missing."
                job.updated_at = time.monotonic()
                return None
            job.stage = JobStage.TRANSCRIBING
            job.updated_at = time.monotonic()
            return request

    async def update_stage(self, job_id: str, stage: JobStage) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.stage = stage
            job.updated_at = time.monotonic()

    async def complete(self, job_id: str, result: VoiceChatTurnResponse) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.stage = JobStage.READY
            job.result = result
            job.error = None
            job.updated_at = time.monotonic()
            self._requests.pop(job_id, None)

    async def fail(self, job_id: str, error: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.stage = JobStage.FAILED
            job.error = error
            job.updated_at = time.monotonic()
            self._requests.pop(job_id, None)

    async def get(self, job_id: str) -> Optional[VoiceJob]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def count_active(self) -> int:
        async with self._lock:
            return sum(1 for job in self._jobs.values() if job.stage not in _TERMINAL_STAGES)

    async def evict_expired(self) -> int:
        cutoff = time.monotonic() - self._ttl
        async with self._lock:
            expired = [
                job_id
                for job_id, job in self._jobs.items()
                if job.stage in _TERMINAL_STAGES and job.updated_at < cutoff
            ]
            for job_id in expired:
                del self._jobs[job_id]
                self._requests.pop(job_id, None)
            return len(expired)
