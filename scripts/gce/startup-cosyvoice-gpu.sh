#!/usr/bin/env bash
set -euo pipefail

exec > >(tee -a /var/log/pika-tts-startup.log) 2>&1

METADATA='http://metadata.google.internal/computeMetadata/v1/instance/attributes'
md() {
  curl -fsH 'Metadata-Flavor: Google' "$METADATA/$1"
}

ARCHIVE_URL="$(md backend_archive_url)"
HOSTNAME_VALUE="$(md public_hostname)"
APP_DIR="/opt/pika-tts"
ARCHIVE_PATH="/tmp/pika-backend.tgz"
NGINX_SITE_PATH="/etc/nginx/sites-available/pika-cosyvoice-http"
ENV_PATH="/etc/pika-cosyvoice-http.env"
SERVICE_PATH="/etc/systemd/system/pika-cosyvoice-http.service"
PYTHON_BIN="/usr/bin/python3.11"
VENV_PATH="$APP_DIR/.venv-cosyvoice"
REPO_DIR="$APP_DIR/.vendor/CosyVoice"
MODEL_DIR="$REPO_DIR/pretrained_models/CosyVoice-300M"
PATH_PREFIX="$(md path_prefix)"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3.11 python3.11-dev python3.11-venv git ffmpeg curl nginx build-essential

mkdir -p "$APP_DIR"
gcloud storage cp "$ARCHIVE_URL" "$ARCHIVE_PATH"
tar -xzf "$ARCHIVE_PATH" -C /opt
if [ -d /opt/PikaProjBackend ] && [ ! -d "$APP_DIR/scripts" ]; then
  rm -rf "$APP_DIR"
  mv /opt/PikaProjBackend "$APP_DIR"
fi
mkdir -p "$APP_DIR/.vendor"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone https://github.com/FunAudioLLM/CosyVoice "$REPO_DIR"
fi

rm -rf "$VENV_PATH"
"$PYTHON_BIN" -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip 'setuptools<81' wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchaudio==2.5.1
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
  onnxruntime-gpu==1.18.0 \
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
python - <<'PY'
from pathlib import Path
from modelscope.hub.snapshot_download import snapshot_download
repo_dir = Path('/opt/pika-tts/.vendor/CosyVoice')
model_dir = repo_dir / 'pretrained_models' / 'CosyVoice-300M'
if not model_dir.exists():
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download('iic/CosyVoice-300M', local_dir=str(model_dir))
PY

cat > "$ENV_PATH" <<EOF
COSYVOICE_HTTP_HOST=127.0.0.1
COSYVOICE_HTTP_PORT=8765
COSYVOICE_REPO_DIR=$REPO_DIR
COSYVOICE_MODEL_DIR=$MODEL_DIR
COSYVOICE_PYTHON=$VENV_PATH/bin/python
COSYVOICE_LANGUAGE=en
COSYVOICE_INFERENCE_MODE=zero_shot
COSYVOICE_LOCAL_TIMEOUT_SECONDS=180
EOF

cat > "$SERVICE_PATH" <<EOF
[Unit]
Description=Pika CosyVoice HTTP service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
EnvironmentFile=$ENV_PATH
ExecStart=$VENV_PATH/bin/python $APP_DIR/scripts/cosyvoice_http_server.py
Restart=always
RestartSec=5
User=root

[Install]
WantedBy=multi-user.target
EOF

cat > "$NGINX_SITE_PATH" <<EOF
server {
    listen 80;
    server_name $HOSTNAME_VALUE;

    location /$PATH_PREFIX/ {
        proxy_pass http://127.0.0.1:8765/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location / {
        return 404;
    }
}
EOF
ln -sf "$NGINX_SITE_PATH" /etc/nginx/sites-enabled/pika-cosyvoice-http
rm -f /etc/nginx/sites-enabled/default

systemctl daemon-reload
systemctl enable --now pika-cosyvoice-http.service
systemctl enable --now nginx
systemctl restart nginx
