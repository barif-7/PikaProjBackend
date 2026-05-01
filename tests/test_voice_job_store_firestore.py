from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from app.models import VoiceChatTurnRequest, VoiceChatTurnResponse
from app.voice_job_store import JobStage
from app.voice_job_store_firestore import FirestoreVoiceJobStore, _utc_now_iso


class _FakeBlob:
    def __init__(self, store: dict[str, bytes], name: str) -> None:
        self._store = store
        self.name = name

    def upload_from_string(self, data: str, content_type: str | None = None) -> None:
        del content_type
        self._store[self.name] = data.encode("utf-8")

    def download_as_bytes(self) -> bytes:
        return self._store[self.name]

    def delete(self) -> None:
        self._store.pop(self.name, None)


class _FakeBucket:
    def __init__(self, store: dict[str, bytes]) -> None:
        self._store = store

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self._store, name)


class _FakeStorageClient:
    def __init__(self, store: dict[str, bytes]) -> None:
        self._store = store

    def bucket(self, name: str) -> _FakeBucket:
        del name
        return _FakeBucket(self._store)


class _FakeSnapshot:
    def __init__(self, reference: "_FakeDocumentReference", data: dict[str, Any] | None) -> None:
        self.reference = reference
        self.id = reference.doc_id
        self._data = data

    @property
    def exists(self) -> bool:
        return self._data is not None

    def to_dict(self) -> dict[str, Any] | None:
        return None if self._data is None else dict(self._data)


class _FakeDocumentReference:
    def __init__(self, backing: dict[str, dict[str, Any]], doc_id: str) -> None:
        self._backing = backing
        self.doc_id = doc_id

    def set(self, data: dict[str, Any], merge: bool = False) -> None:
        if merge and self.doc_id in self._backing:
            merged = dict(self._backing[self.doc_id])
            merged.update(data)
            self._backing[self.doc_id] = merged
        else:
            self._backing[self.doc_id] = dict(data)

    def get(self) -> _FakeSnapshot:
        return _FakeSnapshot(self, self._backing.get(self.doc_id))

    def delete(self) -> None:
        self._backing.pop(self.doc_id, None)


class _FakeQuery:
    def __init__(self, backing: dict[str, dict[str, Any]], rows: list[tuple[str, dict[str, Any]]]) -> None:
        self._backing = backing
        self._rows = rows

    def where(self, field: str, op: str, value: Any) -> "_FakeQuery":
        assert op == "=="
        rows = [(doc_id, data) for doc_id, data in self._rows if data.get(field) == value]
        return _FakeQuery(self._backing, rows)

    def order_by(self, field: str) -> "_FakeQuery":
        rows = sorted(self._rows, key=lambda item: str(item[1].get(field) or ""))
        return _FakeQuery(self._backing, rows)

    def limit(self, n: int) -> "_FakeQuery":
        return _FakeQuery(self._backing, self._rows[:n])

    def stream(self) -> list[_FakeSnapshot]:
        return [
            _FakeSnapshot(_FakeDocumentReference(self._backing, doc_id), data)
            for doc_id, data in self._rows
        ]


class _FakeCollection:
    def __init__(self, backing: dict[str, dict[str, Any]]) -> None:
        self._backing = backing

    def document(self, doc_id: str) -> _FakeDocumentReference:
        return _FakeDocumentReference(self._backing, doc_id)

    def stream(self) -> list[_FakeSnapshot]:
        return _FakeQuery(self._backing, list(self._backing.items())).stream()

    def where(self, field: str, op: str, value: Any) -> _FakeQuery:
        return _FakeQuery(self._backing, list(self._backing.items())).where(field, op, value)

    def order_by(self, field: str) -> _FakeQuery:
        return _FakeQuery(self._backing, list(self._backing.items())).order_by(field)

    def limit(self, n: int) -> _FakeQuery:
        return _FakeQuery(self._backing, list(self._backing.items())).limit(n)


class _FakeFirestoreClient:
    def __init__(self) -> None:
        self._collections: dict[str, dict[str, dict[str, Any]]] = {}

    def collection(self, name: str) -> _FakeCollection:
        backing = self._collections.setdefault(name, {})
        return _FakeCollection(backing)


class FirestoreVoiceJobStoreTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.fake_db = _FakeFirestoreClient()
        self.fake_blobs: dict[str, bytes] = {}
        firestore_mod = type("FakeFirestoreModule", (), {"Client": lambda *args, **kwargs: self.fake_db})
        storage_mod = type("FakeStorageModule", (), {"Client": lambda *args, **kwargs: _FakeStorageClient(self.fake_blobs)})

        self.firestore_patcher = patch("app.voice_job_store_firestore._firestore_module", firestore_mod)
        self.storage_patcher = patch("app.voice_job_store_firestore._storage_module", storage_mod)
        self.firestore_patcher.start()
        self.storage_patcher.start()

    def tearDown(self) -> None:
        self.firestore_patcher.stop()
        self.storage_patcher.stop()

    def _store(self) -> FirestoreVoiceJobStore:
        return FirestoreVoiceJobStore(
            bucket_name="voice-bucket",
            gcs_prefix="voice-jobs",
            collection="voiceChatJobs",
            ttl_seconds=60.0,
            max_jobs=10,
            lease_seconds=120.0,
        )

    def _payload(self) -> VoiceChatTurnRequest:
        return VoiceChatTurnRequest(
            audioBase64="dGVzdA==",
            mimeType="audio/wav",
            fileName="sample.wav",
            durationSeconds=1.0,
            voiceProfileID=None,
            conversationSummary=None,
            history=[],
        )

    async def test_create_and_claim_job(self) -> None:
        store = self._store()
        job = await store.create(self._payload(), None)
        claimed = await store.claim_next("worker-a")

        self.assertEqual(job.stage, JobStage.QUEUED)
        self.assertIsNotNone(claimed)
        assert claimed is not None
        self.assertEqual(claimed.job_id, job.job_id)
        self.assertEqual(claimed.payload.fileName, "sample.wav")

    async def test_complete_persists_result_for_polling(self) -> None:
        store = self._store()
        job = await store.create(self._payload(), None)
        await store.complete(
            job.job_id,
            VoiceChatTurnResponse(
                transcript="hello",
                responseText="hi",
                responseAudioBase64=None,
                responseAudioMimeType=None,
                error=None,
            ),
        )

        fetched = await store.get(job.job_id)
        self.assertIsNotNone(fetched)
        assert fetched is not None
        self.assertEqual(fetched.stage, JobStage.READY)
        self.assertEqual(fetched.result.responseText, "hi")

    async def test_evict_expired_removes_firestore_doc_and_blobs(self) -> None:
        store = self._store()
        job = await store.create(self._payload(), None)
        await store.complete(
            job.job_id,
            VoiceChatTurnResponse(
                transcript="hello",
                responseText="hi",
                responseAudioBase64=None,
                responseAudioMimeType=None,
                error=None,
            ),
        )

        doc = self.fake_db.collection("voiceChatJobs").document(job.job_id)
        doc.set({"updated_at": "2000-01-01T00:00:00+00:00"}, merge=True)

        removed = await store.evict_expired()
        self.assertEqual(removed, 1)
        self.assertIsNone(await store.get(job.job_id))
        self.assertEqual(self.fake_blobs, {})
