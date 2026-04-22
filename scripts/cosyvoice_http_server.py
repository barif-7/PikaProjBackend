from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REPO_DIR = ROOT_DIR.parent / ".vendor" / "CosyVoice"
DEFAULT_MODEL_DIR = DEFAULT_REPO_DIR / "pretrained_models" / "CosyVoice-300M"
DEFAULT_PYTHON = ROOT_DIR / ".venv-cosyvoice" / "bin" / "python"


class CosyVoiceHandler(BaseHTTPRequestHandler):
    server_version = "PikaCosyVoiceHTTP/1.0"

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._write_json({"status": "ok"})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/synthesize":
            self.send_error(404)
            return

        try:
            payload = self._read_json()
            audio = synthesize(payload)
            self._write_json({"audioBase64": base64.b64encode(audio).decode("utf-8"), "mimeType": "audio/wav"})
        except Exception as exc:
            self._write_json({"message": str(exc)}, status=500)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[cosyvoice-http] {self.address_string()} {format % args}")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def synthesize(payload: dict[str, Any]) -> bytes:
    text = str(payload.get("text") or "").strip()
    if not text:
        raise RuntimeError("Missing text.")

    with tempfile.TemporaryDirectory(prefix="pika-cosyvoice-http-") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        output_path = temp_dir / "output.wav"
        reference_audio_path = temp_dir / "reference.wav"

        reference_audio_base64 = str(payload.get("referenceAudioBase64") or "")
        if not reference_audio_base64:
            raise RuntimeError("Missing referenceAudioBase64.")
        reference_audio_path.write_bytes(base64.b64decode(reference_audio_base64))

        env = os.environ.copy()
        repo_dir = Path(os.getenv("COSYVOICE_REPO_DIR", "").strip() or DEFAULT_REPO_DIR)
        python_bin = os.getenv("COSYVOICE_PYTHON", "").strip() or str(DEFAULT_PYTHON)
        env.update(
            {
                "PYTHONPATH": f"{repo_dir}:{repo_dir / 'third_party' / 'Matcha-TTS'}:{env.get('PYTHONPATH', '')}",
                "MPLCONFIGDIR": str(temp_dir / "matplotlib"),
                "COSYVOICE_TEXT": text,
                "COSYVOICE_OUTPUT_PATH": str(output_path),
                "COSYVOICE_REFERENCE_AUDIO_PATH": str(reference_audio_path),
                "COSYVOICE_MODEL_DIR": os.getenv("COSYVOICE_MODEL_DIR", "").strip() or str(DEFAULT_MODEL_DIR),
                "COSYVOICE_REPO_DIR": str(repo_dir),
                "COSYVOICE_LANGUAGE": str(payload.get("language") or "en"),
            }
        )

        prompt_text = str(payload.get("promptText") or "").strip()
        if prompt_text:
            env["COSYVOICE_PROMPT_TEXT"] = prompt_text
        inference_mode = str(payload.get("inferenceMode") or "").strip()
        if inference_mode:
            env["COSYVOICE_INFERENCE_MODE"] = inference_mode

        completed = subprocess.run(
            [python_bin, str(ROOT_DIR / "scripts" / "cosyvoice_local_runtime.py")],
            env=env,
            capture_output=True,
            text=True,
            timeout=float(os.getenv("COSYVOICE_LOCAL_TIMEOUT_SECONDS", "180")),
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "CosyVoice synthesis failed.").strip()
            raise RuntimeError(detail)
        if not output_path.exists():
            raise RuntimeError("CosyVoice synthesis finished without output.wav.")
        return output_path.read_bytes()


def main() -> int:
    host = os.getenv("COSYVOICE_HTTP_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("COSYVOICE_HTTP_PORT", "8765").strip())
    server = ThreadingHTTPServer((host, port), CosyVoiceHandler)
    print(f"[cosyvoice-http] listening on http://{host}:{port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
