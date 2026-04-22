from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    profile_id = required_env("VOICE_PROFILE_ID")
    dataset_path = Path(required_env("VOICE_PROFILE_DATASET_PATH"))
    output_adapter_dir = Path(required_env("VOICE_PROFILE_OUTPUT_ADAPTER_PATH"))
    output_weights_path = Path(
        os.getenv("VOICE_PROFILE_OUTPUT_ADAPTER_WEIGHTS_PATH", "").strip()
        or output_adapter_dir / "adapter.bin"
    )
    output_config_path = Path(
        os.getenv("VOICE_PROFILE_OUTPUT_ADAPTER_CONFIG_PATH", "").strip()
        or output_adapter_dir / "config.json"
    )
    output_eval_path = Path(
        os.getenv("VOICE_PROFILE_OUTPUT_ADAPTER_EVAL_PATH", "").strip()
        or output_adapter_dir / "eval.json"
    )
    base_model = os.getenv("COSYVOICE_ADAPTER_BASE_MODEL", "").strip() or None

    if not dataset_path.exists():
        print(f"Dataset manifest not found: {dataset_path}", file=sys.stderr)
        return 2

    try:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Dataset manifest is unreadable: {exc}", file=sys.stderr)
        return 3

    samples = list(dataset.get("sample_history") or [])
    if not samples:
        print("Dataset manifest contains no approved samples.", file=sys.stderr)
        return 4

    output_adapter_dir.mkdir(parents=True, exist_ok=True)
    output_weights_path.parent.mkdir(parents=True, exist_ok=True)

    # This wrapper defines the integration contract for a real GPU-backed CosyVoice trainer.
    # Replace this section with your actual fine-tune/LoRA export pipeline.
    print(
        "train_cosyvoice_adapter.py is a contract wrapper only. "
        "Implement the actual CosyVoice adapter training pipeline here.",
        file=sys.stderr,
    )
    print(
        json.dumps(
            {
                "profile_id": profile_id,
                "base_model": base_model,
                "sample_count": len(samples),
                "dataset_path": str(dataset_path),
                "output_adapter_dir": str(output_adapter_dir),
                "expected_weights_path": str(output_weights_path),
            },
            indent=2,
        ),
        file=sys.stderr,
    )
    return 5


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
