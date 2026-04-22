from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    mode = os.getenv("VOICE_PROFILE_TRAINING_MODE", "placeholder").strip() or "placeholder"
    profile_id = required_env("VOICE_PROFILE_ID")
    sample_path = Path(required_env("VOICE_PROFILE_SAMPLE_PATH"))
    output_model_path = Path(required_env("VOICE_PROFILE_OUTPUT_MODEL_PATH"))
    output_config_path = Path(required_env("VOICE_PROFILE_OUTPUT_CONFIG_PATH"))
    output_reference_path = Path(required_env("VOICE_PROFILE_OUTPUT_REFERENCE_PATH"))
    output_adapter_path = Path(required_env("VOICE_PROFILE_OUTPUT_ADAPTER_PATH"))
    output_cosyvoice_model_dir = Path(required_env("VOICE_PROFILE_OUTPUT_COSYVOICE_MODEL_DIR"))
    manifest_path = Path(required_env("VOICE_PROFILE_MANIFEST_PATH"))
    sample_history = load_sample_history()

    if mode == "placeholder":
        print(
            "No real voice trainer is configured yet. "
            "Set VOICE_PROFILE_TRAINING_MODE=copy-default-for-smoke-test to exercise the artifact path, "
            "or replace this script with a real model-training pipeline.",
            file=sys.stderr,
        )
        return 1

    if mode == "copy-default-for-smoke-test":
        default_model = Path(required_env("PIPER_MODEL_PATH"))
        default_config = Path(required_env("PIPER_CONFIG_PATH"))
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(default_model, output_model_path)
        shutil.copyfile(default_config, output_config_path)

        manifest = {
            "profile_id": profile_id,
            "mode": mode,
            "sample_path": str(sample_path),
            "output_model_path": str(output_model_path),
            "output_config_path": str(output_config_path),
            "warning": "Smoke-test mode copied the default Piper voice. This is not a personalized model.",
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Created smoke-test voice profile artifacts for {profile_id}")
        return 0

    if mode in {"xtts-reference", "cosyvoice-reference"}:
        ffmpeg_command = os.getenv("FFMPEG_COMMAND", "ffmpeg").strip() or "ffmpeg"
        output_reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_inputs = [
            Path(entry["sample_path"])
            for entry in sample_history
            if str(entry.get("sample_path") or "").strip()
        ] or [sample_path]

        try:
            build_reference_audio(ffmpeg_command, reference_inputs, output_reference_path)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 3

        manifest = {
            "profile_id": profile_id,
            "mode": mode,
            "provider": mode,
            "sample_path": str(sample_path),
            "reference_audio_path": str(output_reference_path),
            "sample_history": sample_history,
            "reference_sample_count": len(reference_inputs),
            "warning": None,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Created {mode} profile for {profile_id} using {len(reference_inputs)} sample(s)")
        return 0

    if mode == "cosyvoice-adapter":
        ffmpeg_command = os.getenv("FFMPEG_COMMAND", "ffmpeg").strip() or "ffmpeg"
        adapter_train_command = os.getenv("COSYVOICE_ADAPTER_TRAIN_COMMAND", "").strip()
        base_model = (
            os.getenv("COSYVOICE_ADAPTER_BASE_MODEL", "").strip()
            or os.getenv("COSYVOICE_MODEL_DIR", "").strip()
            or None
        )
        output_reference_path.parent.mkdir(parents=True, exist_ok=True)
        output_adapter_path.mkdir(parents=True, exist_ok=True)
        reference_inputs = [
            Path(entry["sample_path"])
            for entry in sample_history
            if str(entry.get("sample_path") or "").strip()
        ] or [sample_path]

        try:
            build_reference_audio(ffmpeg_command, reference_inputs, output_reference_path)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 3

        if not adapter_train_command:
            print(
                "COSYVOICE_ADAPTER_TRAIN_COMMAND is not configured. "
                "Set it to a GPU-capable trainer that writes the adapter artifact "
                f"to {output_adapter_path / 'adapter.bin'}.",
                file=sys.stderr,
            )
            return 4

        training_dataset_path = output_adapter_path / "dataset.json"
        dataset_manifest = {
            "profile_id": profile_id,
            "sample_history": sample_history,
            "base_model": base_model,
            "reference_audio_path": str(output_reference_path),
        }
        training_dataset_path.write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")

        env = os.environ.copy()
        env.update(
            {
                "VOICE_PROFILE_ID": profile_id,
                "VOICE_PROFILE_OUTPUT_ADAPTER_PATH": str(output_adapter_path),
                "VOICE_PROFILE_OUTPUT_ADAPTER_WEIGHTS_PATH": str(output_adapter_path / "adapter.bin"),
                "VOICE_PROFILE_OUTPUT_ADAPTER_CONFIG_PATH": str(output_adapter_path / "config.json"),
                "VOICE_PROFILE_OUTPUT_ADAPTER_EVAL_PATH": str(output_adapter_path / "eval.json"),
                "VOICE_PROFILE_OUTPUT_REFERENCE_PATH": str(output_reference_path),
                "VOICE_PROFILE_DATASET_PATH": str(training_dataset_path),
                "VOICE_PROFILE_SAMPLE_HISTORY_JSON": json.dumps(sample_history),
                "COSYVOICE_ADAPTER_BASE_MODEL": base_model or "",
            }
        )
        completed = subprocess.run(
            adapter_train_command,
            shell=True,
            env=env,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "CosyVoice adapter training failed").strip()
            print(detail, file=sys.stderr)
            return completed.returncode or 5

        adapter_weights_path = output_adapter_path / "adapter.bin"
        if not adapter_weights_path.exists():
            print(
                "CosyVoice adapter training finished without producing an adapter artifact "
                f"at {adapter_weights_path}.",
                file=sys.stderr,
            )
            return 6

        manifest = {
            "profile_id": profile_id,
            "mode": mode,
            "provider": mode,
            "artifact_type": "adapter",
            "sample_path": str(sample_path),
            "reference_audio_path": str(output_reference_path),
            "adapter_path": str(output_adapter_path),
            "adapter_weights_path": str(adapter_weights_path),
            "adapter_config_path": str(output_adapter_path / "config.json"),
            "adapter_eval_path": str(output_adapter_path / "eval.json"),
            "base_model": base_model,
            "sample_history": sample_history,
            "trained_on_sample_count": len(reference_inputs),
            "eval_status": "pending",
            "eval_metrics": None,
            "warning": None,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            f"Created cosyvoice-adapter profile for {profile_id} using {len(reference_inputs)} sample(s)"
        )
        return 0

    if mode == "cosyvoice-finetuned":
        ffmpeg_command = os.getenv("FFMPEG_COMMAND", "ffmpeg").strip() or "ffmpeg"
        finetuned_train_command = os.getenv("COSYVOICE_FINETUNED_TRAIN_COMMAND", "").strip()
        speaker_id = os.getenv("COSYVOICE_FINETUNED_SPK_ID", "").strip() or None
        base_model = (
            os.getenv("COSYVOICE_FINETUNED_BASE_MODEL_DIR", "").strip()
            or os.getenv("COSYVOICE_MODEL_DIR", "").strip()
            or None
        )
        output_reference_path.parent.mkdir(parents=True, exist_ok=True)
        output_cosyvoice_model_dir.mkdir(parents=True, exist_ok=True)
        reference_inputs = [
            Path(entry["sample_path"])
            for entry in sample_history
            if str(entry.get("sample_path") or "").strip()
        ] or [sample_path]

        try:
            build_reference_audio(ffmpeg_command, reference_inputs, output_reference_path)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 3

        if not finetuned_train_command:
            print(
                "COSYVOICE_FINETUNED_TRAIN_COMMAND is not configured. "
                "Set it to a trainer that writes an inference-loadable CosyVoice model directory "
                f"to {output_cosyvoice_model_dir}.",
                file=sys.stderr,
            )
            return 4

        training_dataset_path = output_cosyvoice_model_dir / "dataset.json"
        dataset_manifest = {
            "profile_id": profile_id,
            "sample_history": sample_history,
            "base_model": base_model,
            "reference_audio_path": str(output_reference_path),
        }
        training_dataset_path.write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")

        env = os.environ.copy()
        env.update(
            {
                "VOICE_PROFILE_ID": profile_id,
                "VOICE_PROFILE_OUTPUT_COSYVOICE_MODEL_DIR": str(output_cosyvoice_model_dir),
                "VOICE_PROFILE_OUTPUT_REFERENCE_PATH": str(output_reference_path),
                "VOICE_PROFILE_DATASET_PATH": str(training_dataset_path),
                "VOICE_PROFILE_SAMPLE_HISTORY_JSON": json.dumps(sample_history),
                "COSYVOICE_FINETUNED_BASE_MODEL_DIR": base_model or "",
            }
        )
        completed = subprocess.run(
            finetuned_train_command,
            shell=True,
            env=env,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "CosyVoice fine-tuned model training failed").strip()
            print(detail, file=sys.stderr)
            return completed.returncode or 5

        if not is_cosyvoice_model_dir(output_cosyvoice_model_dir):
            print(
                "CosyVoice fine-tuned training finished without producing an inference-loadable model directory "
                f"at {output_cosyvoice_model_dir}.",
                file=sys.stderr,
            )
            return 6

        manifest = {
            "profile_id": profile_id,
            "mode": mode,
            "provider": mode,
            "artifact_type": "model_dir",
            "sample_path": str(sample_path),
            "reference_audio_path": str(output_reference_path),
            "cosyvoice_model_dir": str(output_cosyvoice_model_dir),
            "base_model": base_model,
            "speaker_id": speaker_id,
            "sample_history": sample_history,
            "trained_on_sample_count": len(reference_inputs),
            "eval_status": "pending",
            "eval_metrics": None,
            "warning": None,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            f"Created cosyvoice-finetuned profile for {profile_id} using {len(reference_inputs)} sample(s)"
        )
        return 0

    print(f"Unsupported VOICE_PROFILE_TRAINING_MODE: {mode}", file=sys.stderr)
    return 2


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_sample_history() -> list[dict]:
    raw = os.getenv("VOICE_PROFILE_SAMPLE_HISTORY_JSON", "").strip()
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def is_cosyvoice_model_dir(path: Path) -> bool:
    return path.is_dir() and any((path / name).exists() for name in ("cosyvoice.yaml", "cosyvoice2.yaml", "cosyvoice3.yaml"))


def build_reference_audio(ffmpeg_command: str, sample_paths: list[Path], output_reference_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="voice-profile-reference-") as temp_dir:
        temp_root = Path(temp_dir)
        normalized_paths: list[Path] = []

        for index, source_path in enumerate(sample_paths):
            normalized_path = temp_root / f"sample-{index}.wav"
            completed = subprocess.run(
                [
                    ffmpeg_command,
                    "-y",
                    "-i",
                    str(source_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "24000",
                    str(normalized_path),
                ],
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "ffmpeg failed").strip()
                raise RuntimeError(f"Failed to normalize reference audio: {detail}")
            normalized_paths.append(normalized_path)

        if len(normalized_paths) == 1:
            shutil.copyfile(normalized_paths[0], output_reference_path)
            return

        concat_list_path = temp_root / "concat.txt"
        concat_list_path.write_text(
            "\n".join(f"file '{path}'" for path in normalized_paths),
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                ffmpeg_command,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list_path),
                "-c",
                "copy",
                str(output_reference_path),
            ],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "ffmpeg failed").strip()
            raise RuntimeError(f"Failed to build merged reference audio: {detail}")


if __name__ == "__main__":
    raise SystemExit(main())
