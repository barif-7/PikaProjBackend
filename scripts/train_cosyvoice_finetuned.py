from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    profile_id = required_env("VOICE_PROFILE_ID")
    dataset_path = Path(required_env("VOICE_PROFILE_DATASET_PATH"))
    output_model_dir = Path(required_env("VOICE_PROFILE_OUTPUT_COSYVOICE_MODEL_DIR"))
    base_model_dir = Path(
        os.getenv("COSYVOICE_FINETUNED_BASE_MODEL_DIR", "").strip()
        or os.getenv("COSYVOICE_MODEL_DIR", "").strip()
    )
    config_path = os.getenv("COSYVOICE_FINETUNED_CONFIG_PATH", "").strip()
    train_data = os.getenv("COSYVOICE_FINETUNED_TRAIN_DATA", "").strip()
    cv_data = os.getenv("COSYVOICE_FINETUNED_CV_DATA", "").strip()
    component = os.getenv("COSYVOICE_FINETUNED_COMPONENT", "").strip()
    speaker_id = os.getenv("COSYVOICE_FINETUNED_SPK_ID", "").strip() or None

    if not dataset_path.exists():
        print(f"Dataset manifest not found: {dataset_path}", file=sys.stderr)
        return 2
    if not base_model_dir.is_dir():
        print(
            "COSYVOICE_FINETUNED_BASE_MODEL_DIR must point at an existing CosyVoice model directory.",
            file=sys.stderr,
        )
        return 3

    try:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Dataset manifest is unreadable: {exc}", file=sys.stderr)
        return 4

    samples = list(dataset.get("sample_history") or [])
    if not samples:
        print("Dataset manifest contains no approved samples.", file=sys.stderr)
        return 5

    output_model_dir.mkdir(parents=True, exist_ok=True)

    print(
        "train_cosyvoice_finetuned.py is a contract wrapper around upstream CosyVoice training. "
        "You still need to provide the exact training config and dataset files that match your base model.",
        file=sys.stderr,
    )
    print(
        json.dumps(
            {
                "profile_id": profile_id,
                "base_model_dir": str(base_model_dir),
                "output_model_dir": str(output_model_dir),
                "dataset_path": str(dataset_path),
                "sample_count": len(samples),
                "speaker_id": speaker_id,
                "required_env": {
                    "COSYVOICE_FINETUNED_CONFIG_PATH": config_path or "<required>",
                    "COSYVOICE_FINETUNED_TRAIN_DATA": train_data or "<required>",
                    "COSYVOICE_FINETUNED_CV_DATA": cv_data or "<required>",
                    "COSYVOICE_FINETUNED_COMPONENT": component or "<required>",
                },
                "upstream_train_entrypoint": "cosyvoice/bin/train.py",
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
