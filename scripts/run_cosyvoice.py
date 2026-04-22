from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    text = required_env("COSYVOICE_TEXT")
    output_path = Path(required_env("COSYVOICE_OUTPUT_PATH"))
    reference_audio_raw = os.getenv("COSYVOICE_REFERENCE_AUDIO_PATH", "").strip()
    reference_audio_path = Path(reference_audio_raw) if reference_audio_raw else None
    model_dir_override_raw = os.getenv("COSYVOICE_MODEL_DIR_OVERRIDE", "").strip()
    model_dir_override = Path(model_dir_override_raw) if model_dir_override_raw else None
    adapter_path_raw = os.getenv("COSYVOICE_ADAPTER_PATH", "").strip()
    adapter_path = Path(adapter_path_raw) if adapter_path_raw else None
    language = os.getenv("COSYVOICE_LANGUAGE", "en").strip() or "en"
    inference_command = os.getenv("COSYVOICE_INFERENCE_COMMAND", "").strip()

    if reference_audio_path and not reference_audio_path.exists():
        print(
            f"CosyVoice reference audio does not exist: {reference_audio_path}",
            file=sys.stderr,
        )
        return 2
    if adapter_path and not adapter_path.is_dir():
        print(
            f"CosyVoice adapter artifact does not exist: {adapter_path}",
            file=sys.stderr,
        )
        return 2
    if model_dir_override and not model_dir_override.is_dir():
        print(
            f"CosyVoice model directory override does not exist: {model_dir_override}",
            file=sys.stderr,
        )
        return 2
    if not reference_audio_path and not adapter_path and not model_dir_override:
        print(
            "CosyVoice requires a reference clip, adapter artifact, or model directory override.",
            file=sys.stderr,
        )
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not inference_command:
        local_model_dir = model_dir_override_raw or os.getenv("COSYVOICE_MODEL_DIR", "").strip()
        if local_model_dir:
            local_python = os.getenv("COSYVOICE_PYTHON", "").strip() or sys.executable
            local_runtime = Path(__file__).with_name("cosyvoice_local_runtime.py")
            inference_command = f"{local_python} {local_runtime}"
        else:
            print(
                "COSYVOICE_INFERENCE_COMMAND is not configured. "
                "Set COSYVOICE_COMMAND to point at this script and then either "
                "set COSYVOICE_INFERENCE_COMMAND to your actual CosyVoice inference command, "
                "or configure COSYVOICE_MODEL_DIR / COSYVOICE_MODEL_DIR_OVERRIDE for the built-in local runtime.",
                file=sys.stderr,
            )
            return 3

    env = os.environ.copy()
    env.update(
        {
            "COSYVOICE_TEXT": text,
            "COSYVOICE_OUTPUT_PATH": str(output_path),
            "COSYVOICE_LANGUAGE": language,
        }
    )
    if reference_audio_path:
        env["COSYVOICE_REFERENCE_AUDIO_PATH"] = str(reference_audio_path)
    if adapter_path:
        env["COSYVOICE_ADAPTER_PATH"] = str(adapter_path)
    if model_dir_override:
        env["COSYVOICE_MODEL_DIR_OVERRIDE"] = str(model_dir_override)

    completed = subprocess.run(
        inference_command,
        shell=True,
        env=env,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "CosyVoice inference failed").strip()
        print(detail, file=sys.stderr)
        return completed.returncode or 1

    if not output_path.exists():
        print(
            f"CosyVoice inference finished without producing output: {output_path}",
            file=sys.stderr,
        )
        return 4

    return 0


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
