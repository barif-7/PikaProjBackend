from __future__ import annotations

import asyncio
import json
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .models import VoiceChatTurnRequest, VoiceChatTurnResponse
from .providers import OllamaRuntimeConfig
from .voice_job_store import ClaimedVoiceJob, JobStage, VoiceJob

try:
    from google.cloud import firestore as _firestore_module
    from google.cloud import storage as _storage_module
except ImportError:  # pragma: no cover - optional in local stripped envs
    _firestore_module = None  # type: ignore[assignment]
    _storage_module = None  # type: ignore[assignment]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_after_seconds_iso(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _runtime_to_dict(runtime: Optional[OllamaRuntimeConfig]) -> Optional[dict[str, Any]]:
    if runtime is None:
        return None
    return {
        "base_url": runtime.base_url,
        "chat_url": runtime.chat_url,
        "model": runtime.model,
        "keep_alive": runtime.keep_alive,
        "api_token": runtime.api_token,
    }


def _runtime_from_dict(data: Optional[dict[str, Any]]) -> Optional[OllamaRuntimeConfig]:
    if not data:
        return None
    return OllamaRuntimeConfig(
        base_url=str(data["base_url"]),
        chat_url=str(data["chat_url"]),
        model=str(data["model"]),
        keep_alive=str(data["keep_alive"]),
        api_token=data.get("api_token"),
    )


class FirestoreVoiceJobStore:
    """
    Firestore + GCS-backed distributed voice-job queue.

    Firestore stores job state and claim metadata; GCS stores the request and
    final response payloads so the queue can handle larger audio attachments
    than Firestore's document size limit permits.
    """

    def __init__(
        self,
        *,
        bucket_name: str,
        gcs_prefix: str,
        collection: str,
        ttl_seconds: float,
        max_jobs: int,
        lease_seconds: float,
    ) -> None:
        self.bucket_name = bucket_name.strip()
        self.gcs_prefix = gcs_prefix.strip().strip("/") or "voice-jobs"
        self.collection = collection.strip() or "voiceChatJobs"
        self.ttl_seconds = ttl_seconds
        self.max_jobs = max_jobs
        self.lease_seconds = lease_seconds
        self._firestore_client: Any = None
        self._storage_client: Any = None
        self._bucket_ref: Any = None

    async def create(
        self,
        payload: VoiceChatTurnRequest,
        ollama_runtime: Optional[OllamaRuntimeConfig],
    ) -> VoiceJob:
        return await asyncio.to_thread(self._create_sync, payload, ollama_runtime)

    async def claim_next(self, worker_id: str) -> Optional[ClaimedVoiceJob]:
        return await asyncio.to_thread(self._claim_next_sync, worker_id)

    async def update_stage(self, job_id: str, stage: JobStage) -> None:
        await asyncio.to_thread(self._update_stage_sync, job_id, stage)

    async def complete(self, job_id: str, result: VoiceChatTurnResponse) -> None:
        await asyncio.to_thread(self._complete_sync, job_id, result)

    async def fail(self, job_id: str, error: str) -> None:
        await asyncio.to_thread(self._fail_sync, job_id, error)

    async def get(self, job_id: str) -> Optional[VoiceJob]:
        return await asyncio.to_thread(self._get_sync, job_id)

    async def count_active(self) -> int:
        return await asyncio.to_thread(self._count_active_sync)

    async def evict_expired(self) -> int:
        return await asyncio.to_thread(self._evict_expired_sync)

    def _create_sync(
        self,
        payload: VoiceChatTurnRequest,
        ollama_runtime: Optional[OllamaRuntimeConfig],
    ) -> VoiceJob:
        if payload is None:
            raise RuntimeError("Voice job payload is required for Firestore-backed execution.")
        active_count = self._count_active_sync()
        if active_count >= self.max_jobs:
            raise RuntimeError(
                f"Too many concurrent voice jobs ({active_count}/{self.max_jobs}).  "
                "Retry after a moment."
            )

        job_id = secrets.token_urlsafe(16)
        now = _utc_now_iso()
        request_object = f"{self.gcs_prefix}/{job_id}/request.json"
        self._upload_json(
            request_object,
            {
                "payload": payload.model_dump(mode="json"),
                "ollama_runtime": _runtime_to_dict(ollama_runtime),
            },
        )
        self._jobs().document(job_id).set(
            {
                "job_id": job_id,
                "stage": JobStage.QUEUED.value,
                "error": None,
                "created_at": now,
                "updated_at": now,
                "request_gcs_object": request_object,
                "result_gcs_object": None,
                "worker_id": None,
                "lease_expires_at": None,
            }
        )
        return VoiceJob(job_id=job_id)

    def _claim_next_sync(self, worker_id: str) -> Optional[ClaimedVoiceJob]:
        self._requeue_stale_claims_sync()
        for snapshot in (
            self._jobs()
            .where("stage", "==", JobStage.QUEUED.value)
            .order_by("created_at")
            .limit(5)
            .stream()
        ):
            data = self._claim_snapshot(snapshot.reference, worker_id)
            if not isinstance(data, dict):
                continue
            request_data = self._download_json(str(data["request_gcs_object"]))
            payload = VoiceChatTurnRequest.model_validate(request_data["payload"])
            runtime = _runtime_from_dict(request_data.get("ollama_runtime"))
            return ClaimedVoiceJob(job_id=str(data.get("job_id") or snapshot.id), payload=payload, ollama_runtime=runtime)
        return None

    def _claim_snapshot(self, reference: Any, worker_id: str) -> Optional[dict[str, Any]]:
        transactional = getattr(_firestore_module, "transactional", None)
        if callable(transactional):
            transaction = self._firestore().transaction()

            @transactional
            def _claim_in_transaction(txn: Any) -> Optional[dict[str, Any]]:
                snapshot = reference.get(transaction=txn)
                if not snapshot.exists:
                    return None
                data = snapshot.to_dict() or {}
                if not isinstance(data, dict) or data.get("stage") != JobStage.QUEUED.value:
                    return None
                txn.set(
                    reference,
                    {
                        "stage": JobStage.TRANSCRIBING.value,
                        "error": None,
                        "worker_id": worker_id,
                        "lease_expires_at": _utc_after_seconds_iso(self.lease_seconds),
                        "updated_at": _utc_now_iso(),
                    },
                    merge=True,
                )
                return data

            return _claim_in_transaction(transaction)

        snapshot = reference.get()
        if not snapshot.exists:
            return None
        data = snapshot.to_dict() or {}
        if not isinstance(data, dict) or data.get("stage") != JobStage.QUEUED.value:
            return None
        reference.set(
            {
                "stage": JobStage.TRANSCRIBING.value,
                "error": None,
                "worker_id": worker_id,
                "lease_expires_at": _utc_after_seconds_iso(self.lease_seconds),
                "updated_at": _utc_now_iso(),
            },
            merge=True,
        )
        return data

    def _update_stage_sync(self, job_id: str, stage: JobStage) -> None:
        self._jobs().document(job_id).set(
            {
                "stage": stage.value,
                "updated_at": _utc_now_iso(),
                "lease_expires_at": _utc_after_seconds_iso(self.lease_seconds),
            },
            merge=True,
        )

    def _complete_sync(self, job_id: str, result: VoiceChatTurnResponse) -> None:
        result_object = f"{self.gcs_prefix}/{job_id}/result.json"
        self._upload_json(result_object, result.model_dump(mode="json"))
        self._jobs().document(job_id).set(
            {
                "stage": JobStage.READY.value,
                "error": None,
                "result_gcs_object": result_object,
                "worker_id": None,
                "lease_expires_at": None,
                "updated_at": _utc_now_iso(),
            },
            merge=True,
        )

    def _fail_sync(self, job_id: str, error: str) -> None:
        self._jobs().document(job_id).set(
            {
                "stage": JobStage.FAILED.value,
                "error": error,
                "worker_id": None,
                "lease_expires_at": None,
                "updated_at": _utc_now_iso(),
            },
            merge=True,
        )

    def _get_sync(self, job_id: str) -> Optional[VoiceJob]:
        snapshot = self._jobs().document(job_id).get()
        if not snapshot.exists:
            return None
        data = snapshot.to_dict() or {}
        if not isinstance(data, dict):
            return None

        result = None
        result_object = str(data.get("result_gcs_object") or "").strip()
        if data.get("stage") == JobStage.READY.value and result_object:
            result = VoiceChatTurnResponse.model_validate(self._download_json(result_object))

        return VoiceJob(
            job_id=str(data.get("job_id") or snapshot.id),
            stage=JobStage(str(data.get("stage") or JobStage.QUEUED.value)),
            result=result,
            error=data.get("error"),
        )

    def _count_active_sync(self) -> int:
        active = 0
        for snapshot in self._jobs().stream():
            if not snapshot.exists:
                continue
            stage = str((snapshot.to_dict() or {}).get("stage") or "")
            if stage not in {JobStage.READY.value, JobStage.FAILED.value}:
                active += 1
        return active

    def _evict_expired_sync(self) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.ttl_seconds)
        removed = 0
        for snapshot in self._jobs().stream():
            if not snapshot.exists:
                continue
            data = snapshot.to_dict() or {}
            if str(data.get("stage") or "") not in {JobStage.READY.value, JobStage.FAILED.value}:
                continue
            updated_at = _parse_iso(data.get("updated_at"))
            if updated_at is None or updated_at >= cutoff:
                continue
            request_object = str(data.get("request_gcs_object") or "").strip()
            result_object = str(data.get("result_gcs_object") or "").strip()
            if request_object:
                self._delete_blob(request_object)
            if result_object:
                self._delete_blob(result_object)
            snapshot.reference.delete()
            removed += 1
        return removed

    def _requeue_stale_claims_sync(self) -> None:
        now = datetime.now(timezone.utc)
        for snapshot in self._jobs().stream():
            if not snapshot.exists:
                continue
            data = snapshot.to_dict() or {}
            stage = str(data.get("stage") or "")
            if stage not in {
                JobStage.TRANSCRIBING.value,
                JobStage.GENERATING.value,
                JobStage.SYNTHESIZING.value,
            }:
                continue
            lease_expires_at = _parse_iso(data.get("lease_expires_at"))
            if lease_expires_at is None or lease_expires_at > now:
                continue
            snapshot.reference.set(
                {
                    "stage": JobStage.QUEUED.value,
                    "worker_id": None,
                    "lease_expires_at": None,
                    "updated_at": _utc_now_iso(),
                    "error": "Recovered from stale worker lease.",
                },
                merge=True,
            )

    def _jobs(self) -> Any:
        return self._firestore().collection(self.collection)

    def _firestore(self) -> Any:
        if _firestore_module is None:
            raise RuntimeError("google-cloud-firestore is not installed.")
        if self._firestore_client is None:
            self._firestore_client = _firestore_module.Client()
        return self._firestore_client

    def _storage(self) -> Any:
        if _storage_module is None:
            raise RuntimeError("google-cloud-storage is not installed.")
        if self._storage_client is None:
            self._storage_client = _storage_module.Client()
        return self._storage_client

    def _bucket(self) -> Any:
        if self._bucket_ref is None:
            self._bucket_ref = self._storage().bucket(self.bucket_name)
        return self._bucket_ref

    def _upload_json(self, object_name: str, payload: dict[str, Any]) -> None:
        self._bucket().blob(object_name).upload_from_string(
            json.dumps(payload),
            content_type="application/json",
        )

    def _download_json(self, object_name: str) -> dict[str, Any]:
        raw = self._bucket().blob(object_name).download_as_bytes()
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise RuntimeError(f"Expected JSON object in {object_name}.")
        return data

    def _delete_blob(self, object_name: str) -> None:
        self._bucket().blob(object_name).delete()
