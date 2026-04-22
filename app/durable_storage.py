from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from google.cloud import firestore, storage


class DurableVoiceProfileStore:
    def __init__(
        self,
        *,
        bucket_name: str | None,
        gcs_prefix: str,
        profiles_collection: str,
        jobs_collection: str,
    ) -> None:
        self.bucket_name = (bucket_name or "").strip() or None
        self.gcs_prefix = gcs_prefix.strip().strip("/") or "voice-profiles"
        self.profiles_collection = profiles_collection.strip() or "voiceProfiles"
        self.jobs_collection = jobs_collection.strip() or "voiceProfileJobs"
        self._firestore_client: firestore.Client | None = None
        self._storage_client: storage.Client | None = None
        self._bucket: storage.Bucket | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.bucket_name)

    def save_job(self, job: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._firestore().collection(self.jobs_collection).document(str(job["job_id"])).set(_json_safe(job), merge=True)

    def load_job(self, job_id: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        snapshot = self._firestore().collection(self.jobs_collection).document(job_id).get()
        if not snapshot.exists:
            return None
        data = snapshot.to_dict() or {}
        if not isinstance(data, dict):
            return None
        return data

    def save_manifest(self, profile_id: str, manifest: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._firestore().collection(self.profiles_collection).document(profile_id).set(_json_safe(manifest), merge=True)

    def load_manifest(self, profile_id: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        snapshot = self._firestore().collection(self.profiles_collection).document(profile_id).get()
        if not snapshot.exists:
            return None
        data = snapshot.to_dict() or {}
        if not isinstance(data, dict):
            return None
        return data

    def sync_job_artifacts(self, job: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return manifest

        profile_id = str(job["profile_id"])
        prefix = self._profile_prefix(profile_id)
        updates: dict[str, Any] = {
            "gcs_bucket": f"gs://{self.bucket_name}",
            "gcs_prefix": prefix,
        }

        sample_path = Path(job.get("sample_path") or "")
        if sample_path.exists() and sample_path.is_file():
            object_name = f"{prefix}/samples/{sample_path.name}"
            self._upload_file(sample_path, object_name)
            updates["sample_gcs_object"] = object_name

        manifest_path = Path(job.get("manifest_path") or "")
        if manifest_path.exists() and manifest_path.is_file():
            object_name = f"{prefix}/manifest.json"
            self._upload_file(manifest_path, object_name, content_type="application/json")
            updates["manifest_gcs_object"] = object_name

        reference_path = Path(job.get("reference_path") or "")
        if reference_path.exists() and reference_path.is_file():
            object_name = f"{prefix}/reference/{reference_path.name}"
            self._upload_file(reference_path, object_name)
            updates["reference_audio_gcs_object"] = object_name

        model_path = Path(job.get("model_path") or "")
        if model_path.exists() and model_path.is_file():
            object_name = f"{prefix}/artifacts/{model_path.name}"
            self._upload_file(model_path, object_name)
            updates["model_gcs_object"] = object_name

        config_path = Path(job.get("config_path") or "")
        if config_path.exists() and config_path.is_file():
            object_name = f"{prefix}/artifacts/{config_path.name}"
            self._upload_file(config_path, object_name, content_type="application/json")
            updates["config_gcs_object"] = object_name

        adapter_path = Path(job.get("adapter_path") or "")
        if adapter_path.exists() and adapter_path.is_dir():
            gcs_prefix = f"{prefix}/adapter"
            self._upload_dir(adapter_path, gcs_prefix)
            updates["adapter_gcs_prefix"] = gcs_prefix

        cosyvoice_model_dir = Path(job.get("cosyvoice_model_dir") or "")
        if cosyvoice_model_dir.exists() and cosyvoice_model_dir.is_dir():
            gcs_prefix = f"{prefix}/cosyvoice"
            self._upload_dir(cosyvoice_model_dir, gcs_prefix)
            updates["cosyvoice_model_gcs_prefix"] = gcs_prefix

        manifest_with_storage = {**manifest, **updates}
        self.save_manifest(profile_id, manifest_with_storage)
        if manifest_path.exists() and manifest_path.is_file():
            manifest_path.write_text(json.dumps(manifest_with_storage, indent=2), encoding="utf-8")
            self._upload_file(manifest_path, f"{prefix}/manifest.json", content_type="application/json")
        return manifest_with_storage

    def ensure_manifest_local(self, profile_id: str, local_manifest_path: Path) -> dict[str, Any] | None:
        if local_manifest_path.exists():
            return json.loads(local_manifest_path.read_text(encoding="utf-8"))
        manifest = self.load_manifest(profile_id)
        if not manifest:
            return None
        local_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        local_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def ensure_artifacts_local(self, manifest: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return manifest

        updated = dict(manifest)

        reference_path = Path(str(updated.get("reference_audio_path") or updated.get("reference_path") or "").strip())
        reference_object = str(updated.get("reference_audio_gcs_object") or "").strip()
        if reference_path and str(reference_path) and reference_object and not reference_path.exists():
            self._download_file(reference_object, reference_path)

        model_path = Path(str(updated.get("model_path") or "").strip())
        model_object = str(updated.get("model_gcs_object") or "").strip()
        if model_path and str(model_path) and model_object and not model_path.exists():
            self._download_file(model_object, model_path)

        config_path = Path(str(updated.get("config_path") or "").strip())
        config_object = str(updated.get("config_gcs_object") or "").strip()
        if config_path and str(config_path) and config_object and not config_path.exists():
            self._download_file(config_object, config_path)

        adapter_path_raw = str(updated.get("adapter_path") or "").strip()
        adapter_prefix = str(updated.get("adapter_gcs_prefix") or "").strip()
        if adapter_path_raw and adapter_prefix and not Path(adapter_path_raw).is_dir():
            self._download_prefix(adapter_prefix, Path(adapter_path_raw))

        cosyvoice_path_raw = str(updated.get("cosyvoice_model_dir") or "").strip()
        cosyvoice_prefix = str(updated.get("cosyvoice_model_gcs_prefix") or "").strip()
        if cosyvoice_path_raw and cosyvoice_prefix and not Path(cosyvoice_path_raw).is_dir():
            self._download_prefix(cosyvoice_prefix, Path(cosyvoice_path_raw))

        return updated

    def _firestore(self) -> firestore.Client:
        if self._firestore_client is None:
            self._firestore_client = firestore.Client()
        return self._firestore_client

    def _storage(self) -> storage.Client:
        if self._storage_client is None:
            self._storage_client = storage.Client()
        return self._storage_client

    def _get_bucket(self) -> storage.Bucket:
        if self._bucket is None:
            assert self.bucket_name is not None
            self._bucket = self._storage().bucket(self.bucket_name)
        return self._bucket

    def _profile_prefix(self, profile_id: str) -> str:
        return f"{self.gcs_prefix}/{profile_id}"

    def _upload_file(self, path: Path, object_name: str, *, content_type: str | None = None) -> None:
        blob = self._get_bucket().blob(object_name)
        blob.upload_from_filename(str(path), content_type=content_type)

    def _upload_dir(self, directory: Path, object_prefix: str) -> None:
        for child in directory.rglob("*"):
            if not child.is_file():
                continue
            relative = child.relative_to(directory).as_posix()
            self._upload_file(child, f"{object_prefix}/{relative}")

    def _download_file(self, object_name: str, dest_path: Path) -> None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        self._get_bucket().blob(object_name).download_to_filename(str(dest_path))

    def _download_prefix(self, object_prefix: str, dest_dir: Path) -> None:
        bucket = self._get_bucket()
        blobs = list(self._storage().list_blobs(bucket, prefix=f"{object_prefix.rstrip('/')}/"))
        if not blobs:
            return
        if dest_dir.exists() and dest_dir.is_file():
            dest_dir.unlink()
        dest_dir.mkdir(parents=True, exist_ok=True)
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            relative = blob.name[len(object_prefix.rstrip('/') + '/') :]
            target = dest_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(target))


def build_durable_voice_profile_store() -> DurableVoiceProfileStore:
    return DurableVoiceProfileStore(
        bucket_name=os.getenv("VOICE_PROFILE_STORAGE_BUCKET", "").strip() or None,
        gcs_prefix=os.getenv("VOICE_PROFILE_GCS_PREFIX", "voice-profiles"),
        profiles_collection=os.getenv("VOICE_PROFILE_FIRESTORE_COLLECTION", "voiceProfiles"),
        jobs_collection=os.getenv("VOICE_PROFILE_JOBS_FIRESTORE_COLLECTION", "voiceProfileJobs"),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
