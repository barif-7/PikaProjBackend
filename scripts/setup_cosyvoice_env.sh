#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COSYVOICE_REPO_DIR="${COSYVOICE_REPO_DIR:-$ROOT_DIR/../.vendor/CosyVoice}"
COSYVOICE_VENV_DIR="${COSYVOICE_VENV_DIR:-$ROOT_DIR/.venv-cosyvoice}"
COSYVOICE_PYTHON_BIN="${COSYVOICE_PYTHON_BIN:-python3.11}"

if ! command -v "$COSYVOICE_PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing Python interpreter: $COSYVOICE_PYTHON_BIN" >&2
  echo "Install Python 3.10-3.12 and rerun with COSYVOICE_PYTHON_BIN=<path-to-python>." >&2
  exit 2
fi

PYTHON_VERSION="$("$COSYVOICE_PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PYTHON_VERSION" in
  3.10|3.11|3.12)
    ;;
  *)
    echo "Unsupported Python version for CosyVoice: $PYTHON_VERSION" >&2
    echo "Use Python 3.10, 3.11, or 3.12." >&2
    exit 3
    ;;
esac

mkdir -p "$(dirname "$COSYVOICE_REPO_DIR")"

if [ ! -d "$COSYVOICE_REPO_DIR/.git" ]; then
  git clone https://github.com/FunAudioLLM/CosyVoice "$COSYVOICE_REPO_DIR"
fi

"$COSYVOICE_PYTHON_BIN" -m venv "$COSYVOICE_VENV_DIR"
source "$COSYVOICE_VENV_DIR/bin/activate"

python -m pip install --upgrade pip "setuptools<81" wheel
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.3.1 torchaudio==2.3.1
python -m pip install \
  conformer==0.3.2 \
  diffusers==0.29.0 \
  fastapi==0.115.6 \
  fastapi-cli==0.0.4 \
  gdown==5.1.0 \
  gradio==5.4.0 \
  grpcio==1.57.0 \
  grpcio-tools==1.57.0 \
  hydra-core==1.3.2 \
  HyperPyYAML==1.2.3 \
  inflect==7.3.1 \
  librosa==0.10.2 \
  lightning==2.2.4 \
  matplotlib==3.7.5 \
  modelscope==1.20.0 \
  networkx==3.1 \
  numpy==1.26.4 \
  omegaconf==2.3.0 \
  onnx==1.16.0 \
  onnxruntime==1.18.0 \
  protobuf==4.25 \
  pyarrow==18.1.0 \
  pydantic==2.7.0 \
  pyworld==0.3.4 \
  rich==13.7.1 \
  soundfile==0.12.1 \
  tensorboard==2.14.0 \
  transformers==4.51.3 \
  x-transformers==2.11.24 \
  uvicorn==0.30.0 \
  wetext==0.0.4 \
  wget==3.2

python -m pip install --no-build-isolation openai-whisper==20231117

cat <<EOF
CosyVoice environment is ready.

Suggested env:
  COSYVOICE_COMMAND=python3 $ROOT_DIR/scripts/run_cosyvoice.py
  COSYVOICE_PYTHON=$COSYVOICE_VENV_DIR/bin/python
  COSYVOICE_REPO_DIR=$COSYVOICE_REPO_DIR
  COSYVOICE_MODEL_DIR=/absolute/path/to/CosyVoice/pretrained_models/CosyVoice-300M
  COSYVOICE_INFERENCE_MODE=zero_shot
  TTS_PROVIDER=cosyvoice
  VOICE_PROFILE_TRAINING_MODE=cosyvoice-reference
EOF
