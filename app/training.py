from __future__ import annotations

import base64
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from time import sleep
from typing import Any
from uuid import uuid4

from .audio_upload import AudioUploadError, decode_uploaded_audio
from .durable_storage import build_durable_voice_profile_store
from .models import (
    VoiceProfileCapabilitiesResponse,
    VoiceProfileJobStatusResponse,
    VoiceProfileSubmitRequest,
    VoiceProfileSubmitResponse,
)


class VoiceProfileTrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class VoiceProfileStore:
    root_dir: Path
    profiles_dir: Path
    training_command: str | None = None
    training_mode: str = "placeholder"
    polling_interval_seconds: float = 1.0
    training_timeout_seconds: float = 900.0
    durable_store: Any = None

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        (self.root_dir / "samples").mkdir(parents=True, exist_ok=True)
        (self.root_dir / "profiles").mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._jobs_path.touch(exist_ok=True)
        worker = Thread(target=self._run_worker_loop, name="voice-profile-worker", daemon=True)
        worker.start()

    @property
    def _jobs_path(self) -> Path:
        return self.root_dir / "jobs.json"

    def _load_jobs_state(self) -> dict[str, Any]:
        if self.durable_store and self.durable_store.enabled:
            jobs = self.durable_store.list_jobs()
            if jobs:
                _save_jobs(self._jobs_path, jobs)
                return jobs
        return _load_jobs(self._jobs_path)

    def _save_jobs_state(self, jobs: dict[str, Any]) -> None:
        _save_jobs(self._jobs_path, jobs)
        if self.durable_store and self.durable_store.enabled:
            for job in jobs.values():
                self.durable_store.save_job(job)

    def submit(
        self,
        payload: VoiceProfileSubmitRequest,
        user_id: Any = None,
    ) -> VoiceProfileSubmitResponse:
        try:
            audio_bytes = decode_uploaded_audio(payload.audioBase64, payload.audioChunks)
        except AudioUploadError as exc:
            raise VoiceProfileTrainingError(str(exc)) from exc
        job_id = uuid4().hex
        base_manifest = self._load_profile_manifest(payload.baseProfileID, user_id) if payload.baseProfileID else None
        profile_family_id = (base_manifest or {}).get("profile_family_id") or (base_manifest or {}).get("profile_id") or f"voice-profile-{job_id[:8]}"
        profile_version = int((base_manifest or {}).get("profile_version") or 0) + 1
        profile_id = profile_family_id if profile_version == 1 else f"{profile_family_id}-v{profile_version}"
        suffix = Path(payload.fileName).suffix or ".m4a"
        sample_path = self.root_dir / "samples" / f"{job_id}{suffix}"
        sample_path.write_bytes(audio_bytes)

        model_path = self.profiles_dir / f"{profile_id}.onnx"
        config_path = self.profiles_dir / f"{profile_id}.onnx.json"
        reference_path = self.profiles_dir / f"{profile_id}.wav"
        adapter_path = self.profiles_dir / f"{profile_id}.adapter"
        cosyvoice_model_dir = self.profiles_dir / f"{profile_id}.cosyvoice"
        manifest_path = self.root_dir / "profiles" / f"{profile_id}.json"

        with _STORE_LOCK:
            jobs = self._load_jobs_state()
            jobs[job_id] = {
                "job_id": job_id,
                "profile_id": profile_id,
                "created_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
                "status": "queued",
                "progress": 0.0,
                "message": "Queued for voice profile training.",
                "transcript": payload.transcript,
                "duration_seconds": payload.durationSeconds,
                "file_name": payload.fileName,
                "mime_type": payload.mimeType,
                "sample_path": str(sample_path),
                "model_path": str(model_path),
                "config_path": str(config_path),
                "reference_path": str(reference_path),
                "adapter_path": str(adapter_path),
                "cosyvoice_model_dir": str(cosyvoice_model_dir),
                "manifest_path": str(manifest_path),
                "trainer_command": self.training_command,
                "training_mode": self.training_mode,
                "user_id": user_id,
                "profile_family_id": profile_family_id,
                "profile_version": profile_version,
                "sample_history": self._sample_history_with_append(base_manifest, sample_path, payload),
                "previous_profile_id": (base_manifest or {}).get("profile_id"),
                "base_model": (base_manifest or {}).get("base_model"),
                "artifact_type": (base_manifest or {}).get("artifact_type"),
                "eval_status": (base_manifest or {}).get("eval_status"),
                "eval_metrics": (base_manifest or {}).get("eval_metrics"),
                "promoted_at": (base_manifest or {}).get("promoted_at"),
            }
            self._save_jobs_state(jobs)

        self._write_profile_manifest(jobs[job_id], state="queued")
        self._sync_job_to_durable_store(jobs[job_id])
        return VoiceProfileSubmitResponse(jobId=job_id, profileId=profile_id)

    def status(self, job_id: str, user_id: Any = None) -> VoiceProfileJobStatusResponse:
        with _STORE_LOCK:
            jobs = self._load_jobs_state()
            job = jobs.get(job_id)

        if not job and self.durable_store and self.durable_store.enabled:
            job = self.durable_store.load_job(job_id)
            if job:
                with _STORE_LOCK:
                    jobs = self._load_jobs_state()
                    jobs[job_id] = job
                    self._save_jobs_state(jobs)

        if not job:
            raise VoiceProfileTrainingError(f"Unknown voice training job: {job_id}")
        self._assert_job_access(job, user_id)

        self._refresh_artifact_state(job)
        with _STORE_LOCK:
            refreshed_job = self._load_jobs_state().get(job_id) or job
        return VoiceProfileJobStatusResponse(
            status=refreshed_job["status"],
            progress=refreshed_job.get("progress"),
            profileId=refreshed_job["profile_id"] if refreshed_job["status"] == "ready" else None,
            message=refreshed_job.get("message"),
        )

    def capabilities(self) -> VoiceProfileCapabilitiesResponse:
        has_command = bool(self.training_command)
        mode = (self.training_mode or "placeholder").strip() or "placeholder"
        coqui_tos_agreed = os.getenv("COQUI_TOS_AGREED", "").strip() == "1"
        supports_personalized_voice = has_command and (
            (mode == "xtts-reference" and coqui_tos_agreed)
            or mode in {"cosyvoice-reference", "cosyvoice-adapter", "cosyvoice-finetuned"}
        )

        message: str | None = None
        if not has_command:
            message = (
                "Voice-profile training is not configured on the backend yet. "
                "Add VOICE_PROFILE_TRAINING_COMMAND to enable real model generation."
            )
        elif mode == "placeholder":
            message = (
                "The backend is using placeholder training mode. "
                "It accepts samples but cannot generate a personalized voice yet."
            )
        elif mode == "copy-default-for-smoke-test":
            message = (
                "The backend is in smoke-test mode. "
                "It can validate the artifact pipeline, but it will not sound like the user."
            )
        elif mode == "xtts-reference":
            if coqui_tos_agreed:
                message = "The backend is ready to create XTTS reference profiles from recorded speech samples."
            else:
                message = (
                    "XTTS is installed, but the Coqui CPML license must be accepted before personalized voice "
                    "training and synthesis can be enabled. Set COQUI_TOS_AGREED=1 only if you agree to those terms."
                )
        elif mode == "cosyvoice-reference":
            message = (
                "The backend is ready to create CosyVoice reference profiles from recorded speech samples."
            )
        elif mode == "cosyvoice-adapter":
            message = (
                "The backend is configured for CosyVoice adapter training. "
                "Training must produce an adapter artifact plus a manifest before the profile becomes ready."
            )
        elif mode == "cosyvoice-finetuned":
            message = (
                "The backend is configured for CosyVoice fine-tuned model training. "
                "Training must produce an inference-loadable CosyVoice model directory before the profile becomes ready."
            )
        else:
            message = (
                f"The backend is in unsupported training mode '{mode}'. "
                "Training requests are unlikely to succeed until that mode is implemented."
            )

        return VoiceProfileCapabilitiesResponse(
            trainingCommandConfigured=has_command,
            trainingMode=mode,
            supportsPersonalizedVoice=supports_personalized_voice,
            message=message,
        )

    def assert_profile_access(self, profile_id: str, user_id: Any = None) -> None:
        manifest = self._load_profile_manifest(profile_id, user_id=None)
        self._assert_owner(manifest.get("user_id"), user_id)

    def _load_profile_manifest(self, profile_id: str | None, user_id: Any = None) -> dict[str, Any] | None:
        if not profile_id:
            return None

        manifest_path = self.root_dir / "profiles" / f"{profile_id}.json"
        if self.durable_store and self.durable_store.enabled:
            manifest = self.durable_store.ensure_manifest_local(profile_id, manifest_path)
        elif manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise VoiceProfileTrainingError("The voice profile manifest is unreadable.") from exc
        else:
            manifest = None

        if not manifest:
            raise VoiceProfileTrainingError(f"Unknown voice profile: {profile_id}")

        if self.durable_store and self.durable_store.enabled:
            manifest = self.durable_store.ensure_artifacts_local(manifest)

        self._assert_owner(manifest.get("user_id"), user_id)
        return manifest

    def _run_worker_loop(self) -> None:
        while True:
            try:
                self._tick_jobs()
            except Exception:
                pass
            sleep(self.polling_interval_seconds)

    def _tick_jobs(self) -> None:
        with _STORE_LOCK:
            jobs = self._load_jobs_state()
            dirty = False

            for job in jobs.values():
                if self._refresh_artifact_state(job):
                    dirty = True

            processing_jobs = [job for job in jobs.values() if job["status"] == "processing"]
            if processing_jobs:
                if dirty:
                    self._save_jobs_state(jobs)
                return

            queued_job = next((job for job in jobs.values() if job["status"] == "queued"), None)
            if not queued_job:
                if dirty:
                    self._save_jobs_state(jobs)
                return

            queued_job["status"] = "processing"
            queued_job["progress"] = 0.2
            queued_job["message"] = "Starting voice profile training."
            queued_job["updated_at"] = _utc_now_iso()
            self._save_jobs_state(jobs)

        self._run_training_job(queued_job["job_id"])

    def _run_training_job(self, job_id: str) -> None:
        with _STORE_LOCK:
            jobs = self._load_jobs_state()
            job = jobs.get(job_id)
            if not job:
                return

        if self._profile_artifacts_exist(job):
            self._mark_ready(job)
            return

        if not self.training_command:
            self._mark_failed(
                job,
                "No VOICE_PROFILE_TRAINING_COMMAND is configured. Add a real trainer command or manually place the profile model artifact.",
            )
            return

        try:
            self._execute_training_command(job)
        except VoiceProfileTrainingError as exc:
            self._mark_failed(job, str(exc))
            return

        if self._profile_artifacts_exist(job):
            self._mark_ready(job)
        else:
            self._mark_failed(
                job,
                "Training finished without producing the expected profile model artifact.",
            )

    def _execute_training_command(self, job: dict[str, Any]) -> None:
        self._mark_processing(job, progress=0.6, message="Running voice profile trainer.")

        env = os.environ.copy()
        env.update(
            {
                "VOICE_PROFILE_JOB_ID": job["job_id"],
                "VOICE_PROFILE_ID": job["profile_id"],
                "VOICE_PROFILE_SAMPLE_PATH": job["sample_path"],
                "VOICE_PROFILE_TRANSCRIPT": job["transcript"],
                "VOICE_PROFILE_OUTPUT_MODEL_PATH": job["model_path"],
                "VOICE_PROFILE_OUTPUT_CONFIG_PATH": job["config_path"],
                "VOICE_PROFILE_OUTPUT_REFERENCE_PATH": job["reference_path"],
                "VOICE_PROFILE_OUTPUT_ADAPTER_PATH": job["adapter_path"],
                "VOICE_PROFILE_OUTPUT_COSYVOICE_MODEL_DIR": job["cosyvoice_model_dir"],
                "VOICE_PROFILE_MANIFEST_PATH": job["manifest_path"],
                "VOICE_PROFILE_SAMPLE_HISTORY_JSON": json.dumps(job.get("sample_history") or []),
            }
        )

        try:
            completed = subprocess.run(
                self.training_command,
                shell=True,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.training_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise VoiceProfileTrainingError(
                f"Voice trainer timed out after {self.training_timeout_seconds:.0f} seconds."
            ) from exc

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or f"Trainer exited with code {completed.returncode}."
            raise VoiceProfileTrainingError(f"Voice trainer failed: {detail}")

    def _refresh_artifact_state(self, job: dict[str, Any]) -> bool:
        if self.durable_store and self.durable_store.enabled:
            manifest = self._load_profile_manifest(job.get("profile_id"), user_id=None)
            if manifest:
                self.durable_store.ensure_artifacts_local(manifest)

        if job["status"] == "ready":
            return False

        if self._profile_artifacts_exist(job):
            self._mark_ready(job)
            return True

        return False

    def _profile_artifacts_exist(self, job: dict[str, Any]) -> bool:
        training_mode = (job.get("training_mode") or self.training_mode or "placeholder").strip() or "placeholder"
        if training_mode in {"xtts-reference", "cosyvoice-reference"}:
            reference_path = Path(job["reference_path"])
            manifest_path = Path(job["manifest_path"])
            return reference_path.exists() and manifest_path.exists()
        if training_mode == "cosyvoice-adapter":
            adapter_path = Path(job["adapter_path"])
            manifest_path = Path(job["manifest_path"])
            adapter_weights_path = adapter_path / "adapter.bin"
            return adapter_path.is_dir() and adapter_weights_path.exists() and manifest_path.exists()
        if training_mode == "cosyvoice-finetuned":
            cosyvoice_model_dir = Path(job["cosyvoice_model_dir"])
            manifest_path = Path(job["manifest_path"])
            return _is_cosyvoice_model_dir(cosyvoice_model_dir) and manifest_path.exists()

        model_path = Path(job["model_path"])
        return model_path.exists()

    def _mark_processing(self, job: dict[str, Any], progress: float, message: str) -> None:
        with _STORE_LOCK:
            jobs = self._load_jobs_state()
            stored_job = jobs.get(job["job_id"])
            if not stored_job:
                return
            stored_job["status"] = "processing"
            stored_job["progress"] = progress
            stored_job["message"] = message
            stored_job["updated_at"] = _utc_now_iso()
            self._save_jobs_state(jobs)
            self._write_profile_manifest(stored_job, state="processing")
            self._sync_job_to_durable_store(stored_job)

    def _mark_ready(self, job: dict[str, Any]) -> None:
        with _STORE_LOCK:
            jobs = self._load_jobs_state()
            stored_job = jobs.get(job["job_id"])
            if not stored_job:
                return
            stored_job["status"] = "ready"
            stored_job["progress"] = 1.0
            stored_job["message"] = "Voice profile ready."
            stored_job["updated_at"] = _utc_now_iso()
            self._save_jobs_state(jobs)
            self._write_profile_manifest(stored_job, state="ready")
            self._sync_job_to_durable_store(stored_job)

    def _mark_failed(self, job: dict[str, Any], message: str) -> None:
        with _STORE_LOCK:
            jobs = self._load_jobs_state()
            stored_job = jobs.get(job["job_id"])
            if not stored_job:
                return
            stored_job["status"] = "failed"
            stored_job["progress"] = None
            stored_job["message"] = message
            stored_job["updated_at"] = _utc_now_iso()
            self._save_jobs_state(jobs)
            self._write_profile_manifest(stored_job, state="failed")
            self._sync_job_to_durable_store(stored_job)

    def _write_profile_manifest(self, job: dict[str, Any], state: str) -> None:
        manifest_path = Path(job["manifest_path"])
        existing_doc: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                existing_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing_doc = {}

        profile_doc = {
            "profile_id": job["profile_id"],
            "profile_family_id": job.get("profile_family_id") or job["profile_id"],
            "profile_version": job.get("profile_version") or 1,
            "job_id": job["job_id"],
            "user_id": job.get("user_id"),
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "status": state,
            "sample_path": job["sample_path"],
            "transcript": job["transcript"],
            "duration_seconds": job["duration_seconds"],
            "model_path": job["model_path"],
            "config_path": job["config_path"],
            "reference_path": job.get("reference_path"),
            "adapter_path": job.get("adapter_path"),
            "cosyvoice_model_dir": job.get("cosyvoice_model_dir"),
            "training_mode": job.get("training_mode", self.training_mode),
            "training_command": job.get("trainer_command"),
            "sample_history": job.get("sample_history") or [],
            "previous_profile_id": job.get("previous_profile_id"),
            "base_model": job.get("base_model"),
            "artifact_type": job.get("artifact_type"),
            "eval_status": job.get("eval_status"),
            "eval_metrics": job.get("eval_metrics"),
            "promoted_at": job.get("promoted_at"),
        }
        profile_doc = {**existing_doc, **profile_doc}
        manifest_path.write_text(json.dumps(profile_doc, indent=2), encoding="utf-8")

        if self.durable_store and self.durable_store.enabled:
            self.durable_store.save_manifest(job["profile_id"], profile_doc)
            self.durable_store.sync_job_artifacts(job, profile_doc)

    def _sync_job_to_durable_store(self, job: dict[str, Any]) -> None:
        if not self.durable_store or not self.durable_store.enabled:
            return
        self.durable_store.save_job(job)

    def _assert_job_access(self, job: dict[str, Any], user_id: Any) -> None:
        self._assert_owner(job.get("user_id"), user_id)

    def _assert_owner(self, owner_user_id: Any, requesting_user_id: Any) -> None:
        if not owner_user_id:
            return
        if not requesting_user_id:
            raise VoiceProfileTrainingError("This voice profile belongs to a signed-in user.")
        if owner_user_id != requesting_user_id:
            raise VoiceProfileTrainingError("You do not have access to this voice profile.")

    def _sample_history_with_append(
        self,
        base_manifest: dict[str, Any] | None,
        sample_path: Path,
        payload: VoiceProfileSubmitRequest,
    ) -> list[dict[str, Any]]:
        history = list((base_manifest or {}).get("sample_history") or [])
        history.append(
            {
                "sample_path": str(sample_path),
                "transcript": payload.transcript,
                "duration_seconds": payload.durationSeconds,
                "file_name": payload.fileName,
                "mime_type": payload.mimeType,
                "recorded_at": _utc_now_iso(),
            }
        )
        return history[-6:]


_STORE_LOCK = Lock()


def make_voice_profile_store() -> VoiceProfileStore:
    root_dir = Path(__file__).resolve().parent.parent / "data"
    profiles_dir = Path(
        os.getenv("VOICE_PROFILE_MODELS_DIR", "").strip() or root_dir / "profile-models"
    )
    training_command = os.getenv("VOICE_PROFILE_TRAINING_COMMAND", "").strip() or None
    training_mode = os.getenv("VOICE_PROFILE_TRAINING_MODE", "placeholder").strip() or "placeholder"
    training_timeout_seconds = float(os.getenv("VOICE_PROFILE_TRAINING_TIMEOUT_SECONDS", "900").strip())
    return VoiceProfileStore(
        root_dir=root_dir,
        profiles_dir=profiles_dir,
        training_command=training_command,
        training_mode=training_mode,
        training_timeout_seconds=training_timeout_seconds,
        durable_store=build_durable_voice_profile_store(),
    )


def _decode_audio(audio_base64: str) -> bytes:
    try:
        return base64.b64decode(audio_base64, validate=True)
    except Exception as exc:
        raise VoiceProfileTrainingError("The app sent invalid voice training audio data.") from exc


def _load_jobs(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}

    try:
        jobs = json.loads(raw)
    except json.JSONDecodeError:
        # Tolerate a corrupted jobs file rather than crashing the worker loop.
        # A fresh empty state will be written on the next save.
        return {}
    root_dir = path.parent
    profiles_dir = root_dir / "profile-models"
    for job_id, job in jobs.items():
        profile_id = job.get("profile_id") or f"voice-profile-{job_id[:8]}"
        job.setdefault("job_id", job_id)
        job.setdefault("profile_id", profile_id)
        job.setdefault("updated_at", job.get("created_at", _utc_now_iso()))
        job.setdefault("status", "queued")
        job.setdefault("progress", 0.0 if job["status"] in {"queued", "processing"} else None)
        job.setdefault("message", "Queued for voice profile training.")
        job.setdefault("model_path", str(profiles_dir / f"{profile_id}.onnx"))
        job.setdefault("config_path", str(profiles_dir / f"{profile_id}.onnx.json"))
        job.setdefault("reference_path", str(profiles_dir / f"{profile_id}.wav"))
        job.setdefault("adapter_path", str(profiles_dir / f"{profile_id}.adapter"))
        job.setdefault("cosyvoice_model_dir", str(profiles_dir / f"{profile_id}.cosyvoice"))
        job.setdefault("manifest_path", str(root_dir / "profiles" / f"{profile_id}.json"))
        job.setdefault("trainer_command", None)
        job.setdefault("training_mode", "placeholder")
        job.setdefault("profile_family_id", profile_id)
        job.setdefault("profile_version", 1)
        job.setdefault("sample_history", [])
        job.setdefault("previous_profile_id", None)
        job.setdefault("base_model", None)
        job.setdefault("artifact_type", None)
        job.setdefault("eval_status", None)
        job.setdefault("eval_metrics", None)
        job.setdefault("promoted_at", None)
    return jobs


def _save_jobs(path: Path, jobs: dict[str, Any]) -> None:
    # Atomic write — the worker loop is constantly rewriting this file, and a
    # partial write would corrupt every in-flight training job until the file
    # is repaired manually.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(jobs, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _is_cosyvoice_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / name).exists() for name in ("cosyvoice.yaml", "cosyvoice2.yaml", "cosyvoice3.yaml"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()
