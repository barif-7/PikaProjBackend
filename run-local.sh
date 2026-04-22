#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
REQUIREMENTS_FILE="$ROOT_DIR/requirements.txt"
STAMP_FILE="$VENV_DIR/.requirements.sha256"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

requirements_hash="$(
  python3 - <<'PY' "$REQUIREMENTS_FILE"
import hashlib
import pathlib
import sys

print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"

installed_hash=""
if [ -f "$STAMP_FILE" ]; then
  installed_hash="$(cat "$STAMP_FILE")"
fi

if [ "$requirements_hash" != "$installed_hash" ]; then
  pip install -r "$REQUIREMENTS_FILE"
  printf '%s' "$requirements_hash" > "$STAMP_FILE"
fi

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

exec uvicorn app.main:app --app-dir "$ROOT_DIR" --host 0.0.0.0 --port 8080
