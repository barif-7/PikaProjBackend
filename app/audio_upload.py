from __future__ import annotations

import base64
import io
import re
import wave
from pathlib import Path
from typing import Optional, Sequence

from pydantic import BaseModel, Field, field_validator, model_validator


_SAFE_FILENAME_RE = re.compile(r"^[\w\-. ]+$")


class AudioUploadError(ValueError):
    pass


def _sanitize_file_name(value: str) -> str:
    name = Path(value).name
    if not name:
        raise ValueError("fileName must not be empty after stripping path components.")
    if not _SAFE_FILENAME_RE.match(name):
        raise ValueError(
            "fileName contains characters that are not permitted. "
            "Use only letters, digits, hyphens, underscores, dots, and spaces."
        )
    return name[:255]


class AudioUploadChunk(BaseModel):
    index: int = Field(ge=0)
    totalChunks: int = Field(ge=1)
    fileName: str
    mimeType: str
    durationSeconds: float = Field(ge=0)
    audioBase64: str

    @field_validator("fileName")
    @classmethod
    def sanitize_file_name(cls, value: str) -> str:
        return _sanitize_file_name(value)

    @field_validator("mimeType")
    @classmethod
    def validate_mime_type(cls, value: str) -> str:
        mime = value.strip().lower()
        if not mime:
            raise ValueError("mimeType must not be blank.")
        return mime

    @field_validator("audioBase64")
    @classmethod
    def validate_audio_base64(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("audioBase64 must not be blank.")
        return value

    @model_validator(mode="after")
    def validate_chunk_shape(self) -> "AudioUploadChunk":
        if self.index >= self.totalChunks:
            raise ValueError("Chunk index must be smaller than totalChunks.")
        return self


def decode_uploaded_audio(
    audio_base64: Optional[str],
    audio_chunks: Sequence[AudioUploadChunk] | None,
) -> bytes:
    if audio_base64:
        return base64.b64decode(audio_base64, validate=True)

    if not audio_chunks:
        raise AudioUploadError("Either audioBase64 or audioChunks must be provided.")

    ordered_chunks = sorted(audio_chunks, key=lambda chunk: chunk.index)
    expected_total = ordered_chunks[0].totalChunks
    if any(chunk.totalChunks != expected_total for chunk in ordered_chunks):
        raise AudioUploadError("audioChunks totalChunks values must all match.")
    if expected_total != len(ordered_chunks):
        raise AudioUploadError("audioChunks totalChunks does not match the number of chunks provided.")
    if [chunk.index for chunk in ordered_chunks] != list(range(expected_total)):
        raise AudioUploadError("audioChunks must be indexed from 0 with no gaps.")

    decoded_chunks = []
    reference_params = None

    for chunk in ordered_chunks:
        chunk_bytes = base64.b64decode(chunk.audioBase64, validate=True)
        if chunk.mimeType not in {"audio/wav", "audio/x-wav", "audio/wave"}:
            raise AudioUploadError("Chunked uploads currently support WAV audio only.")

        with wave.open(io.BytesIO(chunk_bytes), "rb") as reader:
            params = reader.getparams()
            if reference_params is None:
                reference_params = params
            elif params[:4] != reference_params[:4]:
                raise AudioUploadError("audioChunks must use the same audio format.")
            decoded_chunks.append(reader.readframes(reader.getnframes()))

    if reference_params is None:
        raise AudioUploadError("audioChunks were empty.")

    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setparams(reference_params)
        for frames in decoded_chunks:
            writer.writeframes(frames)

    return output.getvalue()
