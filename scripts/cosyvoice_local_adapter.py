from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    text = required_env("COSYVOICE_TEXT")
    output_path = Path(required_env("COSYVOICE_OUTPUT_PATH"))
    model_dir = Path(required_env("COSYVOICE_MODEL_DIR"))
    reference_audio_raw = os.getenv("COSYVOICE_REFERENCE_AUDIO_PATH", "").strip()
    reference_audio_path = Path(reference_audio_raw) if reference_audio_raw else None
    adapter_path_raw = os.getenv("COSYVOICE_ADAPTER_PATH", "").strip()
    adapter_path = Path(adapter_path_raw) if adapter_path_raw else None
    repo_dir_raw = os.getenv("COSYVOICE_REPO_DIR", "").strip()
    repo_dir = Path(repo_dir_raw) if repo_dir_raw else None
    prompt_text = (
        os.getenv("COSYVOICE_PROMPT_TEXT", "").strip()
        or "This voice sample should guide the speaker identity of the synthesized reply."
    )
    inference_mode = os.getenv("COSYVOICE_INFERENCE_MODE", "zero_shot").strip().lower() or "zero_shot"
    language = os.getenv("COSYVOICE_LANGUAGE", "en").strip() or "en"

    if repo_dir:
        sys.path.insert(0, str(repo_dir))

    try:
        import torch
        import soundfile as sf
        from cosyvoice.cli.cosyvoice import CosyVoice
    except Exception as exc:  # pragma: no cover - import surface depends on local install
        print(
            "Failed to import CosyVoice runtime. "
            "Set COSYVOICE_REPO_DIR to a valid CosyVoice checkout and install its Python dependencies. "
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 2

    if inference_mode == "adapter":
        print(
            "The bundled local CosyVoice runner does not implement adapter-backed inference yet. "
            "Point COSYVOICE_INFERENCE_COMMAND at an adapter-capable runtime.",
            file=sys.stderr,
        )
        return 8

    if not reference_audio_path:
        print("COSYVOICE_REFERENCE_AUDIO_PATH is required for non-adapter inference.", file=sys.stderr)
        return 3

    if not reference_audio_path.exists():
        print(f"Reference audio not found: {reference_audio_path}", file=sys.stderr)
        return 3

    if adapter_path and not adapter_path.is_dir():
        print(f"Adapter artifact not found: {adapter_path}", file=sys.stderr)
        return 3

    if not model_dir.exists():
        print(f"CosyVoice model directory not found: {model_dir}", file=sys.stderr)
        return 4

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        cosyvoice = CosyVoice(str(model_dir))
    except Exception as exc:  # pragma: no cover - runtime depends on local install
        print(f"Failed to initialize CosyVoice from {model_dir}: {exc}", file=sys.stderr)
        return 5

    prompt_speech = str(reference_audio_path)

    try:
        generated = synthesize(
            cosyvoice=cosyvoice,
            text=text,
            prompt_text=prompt_text,
            prompt_speech=prompt_speech,
            inference_mode=inference_mode,
            language=language,
        )
    except Exception as exc:  # pragma: no cover - runtime depends on local install
        print(f"CosyVoice inference failed: {exc}", file=sys.stderr)
        return 6

    sample_rate = int(getattr(cosyvoice, "sample_rate", 22050))
    try:
        tensor = torch.as_tensor(generated)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 2 and tensor.shape[0] != 1 and tensor.shape[1] == 1:
            tensor = tensor.transpose(0, 1)
        tensor = tensor.detach().cpu().to(dtype=torch.float32)
        audio = tensor.squeeze(0).numpy()
        sf.write(str(output_path), audio, sample_rate)
    except Exception as exc:  # pragma: no cover - runtime depends on local install
        print(f"Failed to write CosyVoice output WAV: {exc}", file=sys.stderr)
        return 7

    return 0


def synthesize(
    *,
    cosyvoice,
    text: str,
    prompt_text: str,
    prompt_speech,
    inference_mode: str,
    language: str,
):
    if inference_mode == "cross_lingual":
        result = cosyvoice.inference_cross_lingual(text, prompt_speech, stream=False)
    elif inference_mode == "instruct":
        instruct_text = (
            os.getenv("COSYVOICE_INSTRUCT_TEXT", "").strip()
            or f"Respond naturally in {language} while preserving the speaker identity from the prompt audio."
        )
        result = cosyvoice.inference_instruct(text, instruct_text, prompt_speech, stream=False)
    else:
        result = cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech, stream=False)

    audio = first_audio_chunk(result)
    if audio is None:
        raise RuntimeError("CosyVoice returned no audio payload.")
    return audio


def first_audio_chunk(result):
    try:
        iterator = iter(result)
    except TypeError:
        iterator = iter([result])

    for item in iterator:
        if isinstance(item, dict):
            for key in ("tts_speech", "audio", "speech"):
                payload = item.get(key)
                if payload is not None:
                    return payload
        elif item is not None:
            return item
    return None


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
