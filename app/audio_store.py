from __future__ import annotations

"""
Temporary in-process audio upload store.

Clients that cannot send large base64 JSON payloads can POST audio via
multipart/form-data to ``POST /audio/uploads``, receive an ``uploadId``, and
then reference that ID in subsequent ``POST /voice-chat/jobs`` or
``POST /voice-profiles`` calls instead of sending the audio inline.

Uploads are held in-process memory with a configurable TTL.  They are
automatically evicted once claimed (single-use) or once the TTL expires.
This design keeps the API stateless-compatible: if the server restarts
between the upload and the job submission the upload is gone and the client
must re-upload.  For production, replace this with a GCS signed-URL flow
(see TRANSPORT.md).
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class _UploadEntry:
    upload_id: str
    audio_bytes: bytes
    mime_type: str
    file_name: str
    duration_seconds: float
    stored_at: float = field(default_factory=time.monotonic)


class AudioUploadStore:
    """
    In-memory store for temporary audio upload buffers.

    Thread-safe via asyncio lock.
    """

    def __init__(self, *, ttl_seconds: float = 300.0) -> None:
        self._ttl = ttl_seconds
        self._entries: Dict[str, _UploadEntry] = {}
        self._lock = asyncio.Lock()

    async def store(
        self,
        *,
        audio_bytes: bytes,
        mime_type: str,
        file_name: str,
        duration_seconds: float,
    ) -> str:
        """Store audio bytes and return a unique upload ID."""
        upload_id = uuid.uuid4().hex
        async with self._lock:
            self._entries[upload_id] = _UploadEntry(
                upload_id=upload_id,
                audio_bytes=audio_bytes,
                mime_type=mime_type,
                file_name=file_name,
                duration_seconds=duration_seconds,
            )
        return upload_id

    async def claim(self, upload_id: str) -> Optional[_UploadEntry]:
        """
        Retrieve and remove an upload entry.

        Returns ``None`` if the upload does not exist or has expired.
        Single-use — the entry is deleted on first successful claim.
        """
        async with self._lock:
            entry = self._entries.get(upload_id)
            if entry is None:
                return None
            if time.monotonic() - entry.stored_at > self._ttl:
                self._entries.pop(upload_id, None)
                return None
            self._entries.pop(upload_id)
            return entry

    async def evict_expired(self) -> int:
        """Remove all entries whose TTL has elapsed.  Returns the count evicted."""
        now = time.monotonic()
        async with self._lock:
            expired = [
                uid
                for uid, entry in self._entries.items()
                if now - entry.stored_at > self._ttl
            ]
            for uid in expired:
                self._entries.pop(uid, None)
        if expired:
            logger.debug("[audio-store] evicted %d expired uploads", len(expired))
        return len(expired)
