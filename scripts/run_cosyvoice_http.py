from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request


def main() -> int:
    endpoint_url = required_env("COSYVOICE_HTTP_URL").rstrip("/")
    text = required_env("COSYVOICE_TEXT")
    output_path = Path(required_env("COSYVOICE_OUTPUT_PATH"))
    reference_audio_raw = os.getenv("COSYVOICE_REFERENCE_AUDIO_PATH", "").strip()
    reference_audio_path = Path(reference_audio_raw) if reference_audio_raw else None

    payload = {
        "text": text,
        "language": os.getenv("COSYVOICE_LANGUAGE", "en").strip() or "en",
        "voiceProfileID": os.getenv("VOICE_PROFILE_ID", "").strip() or None,
        "promptText": os.getenv("COSYVOICE_PROMPT_TEXT", "").strip() or None,
        "inferenceMode": os.getenv("COSYVOICE_INFERENCE_MODE", "").strip() or None,
        "referenceAudioBase64": None,
    }

    if reference_audio_path:
        if not reference_audio_path.exists():
            print(f"CosyVoice reference audio does not exist: {reference_audio_path}", file=sys.stderr)
            return 2
        payload["referenceAudioBase64"] = base64.b64encode(reference_audio_path.read_bytes()).decode("utf-8")

    request = urllib.request.Request(
        f"{endpoint_url}/synthesize",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=float(os.getenv("COSYVOICE_HTTP_TIMEOUT_SECONDS", "180"))) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        print(f"CosyVoice HTTP bridge returned {exc.code}: {detail}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"CosyVoice HTTP bridge request failed: {exc}", file=sys.stderr)
        return 4

    audio_base64 = str(body.get("audioBase64") or "")
    if not audio_base64:
        print(f"CosyVoice HTTP bridge returned no audio: {body}", file=sys.stderr)
        return 5

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(audio_base64))
    return 0


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
