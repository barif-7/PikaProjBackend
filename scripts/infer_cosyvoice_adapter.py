from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    text = required_env("COSYVOICE_TEXT")
    output_path = Path(required_env("COSYVOICE_OUTPUT_PATH"))
    adapter_dir = Path(required_env("COSYVOICE_ADAPTER_PATH"))
    adapter_weights_path = adapter_dir / "adapter.bin"
    adapter_config_path = adapter_dir / "config.json"
    reference_audio_raw = os.getenv("COSYVOICE_REFERENCE_AUDIO_PATH", "").strip()
    reference_audio_path = Path(reference_audio_raw) if reference_audio_raw else None
    base_model = os.getenv("COSYVOICE_ADAPTER_BASE_MODEL", "").strip() or None

    if not text:
        print("COSYVOICE_TEXT is empty.", file=sys.stderr)
        return 2
    if not adapter_dir.is_dir():
        print(f"Adapter directory not found: {adapter_dir}", file=sys.stderr)
        return 3
    if not adapter_weights_path.exists():
        print(f"Adapter weights not found: {adapter_weights_path}", file=sys.stderr)
        return 4
    if reference_audio_path and not reference_audio_path.exists():
        print(f"Reference audio not found: {reference_audio_path}", file=sys.stderr)
        return 5

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # This wrapper defines the runtime contract for a real adapter-capable CosyVoice inference stack.
    # Replace this section with model loading plus synthesis to output_path.
    print(
        "infer_cosyvoice_adapter.py is a contract wrapper only. "
        "Implement the actual CosyVoice adapter inference pipeline here.",
        file=sys.stderr,
    )
    print(
        json.dumps(
            {
                "base_model": base_model,
                "adapter_dir": str(adapter_dir),
                "adapter_weights_path": str(adapter_weights_path),
                "adapter_config_path": str(adapter_config_path),
                "reference_audio_path": str(reference_audio_path) if reference_audio_path else None,
                "output_path": str(output_path),
            },
            indent=2,
        ),
        file=sys.stderr,
    )
    return 6


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
