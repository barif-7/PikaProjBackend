# PikaProjBackend

Reference backend for the Pika iOS app and `Open Messages` screen.

Related repo:
- iOS app: `https://github.com/barif-7/PikaProjiOS`

It exposes:
- `GET /health`
- `GET /auth/google/start`
- `GET /auth/google/callback`
- `GET /auth/session`
- `DELETE /auth/session`
- `GET /conversations/default`
- `PUT /conversations/default`
- `GET /provider-connections/ollama`
- `PUT /provider-connections/ollama`
- `POST /voice-chat/turn`
- `POST /voice-profiles`
- `GET /voice-profiles/capabilities`
- `GET /voice-profiles/{jobId}`

The service is intentionally self-hosted and open-stack:
- Whisper CLI for transcription
- Ollama for the LLM
- Piper for optional TTS
- XTTS v2 for profile-based voice cloning from a reference sample
- CosyVoice via an external command hook for speaker-adapted synthesis

No API keys are required in the iOS app.

## Google auth and provider connections

The backend can now mint an app session from Google OAuth and associate provider settings with that signed-in user.

Required env vars for Google sign-in:
- `GOOGLE_OAUTH_CLIENT_ID`
- `GOOGLE_OAUTH_CLIENT_SECRET`
- `GOOGLE_OAUTH_CALLBACK_URL`

Optional auth env vars:
- `AUTH_MOBILE_CALLBACK_SCHEME`
- `AUTH_DATA_DIR`
- `AUTH_SESSION_TTL_SECONDS`

The iOS app opens `GET /auth/google/start?mobile_callback=pikatakehome://auth/google`, the backend handles the Google callback, then redirects back to the app with a session token. The app stores that session token in Keychain and reuses it across launches.

Current provider connection support:
- `GET /provider-connections/ollama`
- `PUT /provider-connections/ollama`

`PUT /provider-connections/ollama`

```json
{
  "endpointURL": "https://ollama.example.com",
  "model": "qwen2.5:3b",
  "apiToken": "optional-token",
  "label": "My hosted Ollama"
}
```

These endpoints are authenticated with `Authorization: Bearer <session-token>`.

## Conversation persistence

The backend can now persist the default SEMI conversation per authenticated user.

- `GET /conversations/default`
- `PUT /conversations/default`

`PUT /conversations/default`

```json
{
  "summary": "User has been discussing burnout and focus.",
  "voiceProfileID": "voice-profile-1234",
  "messages": [
    { "role": "assistant", "content": "I am here." },
    { "role": "user", "content": "I feel scattered." }
  ]
}
```

The iOS app uses this to rehydrate the message thread across launches once an authenticated app session exists.

## Voice profile evolution

Voice profiles are now structured as a versioned lineage rather than a single one-off artifact.

- the first approved sample creates profile version `1`
- later approved samples can append to the existing profile family
- each new training job writes a new profile version manifest
- XTTS reference generation can merge multiple stored user samples into one stronger reference clip

This is the foundation for a future "gets better over time" pipeline. The current system still uses XTTS reference conditioning rather than full continual fine-tuning, but it now preserves sample history and profile versions so a stronger trainer can be introduced later without losing user data.

## CosyVoice provider

The backend now supports a CosyVoice-oriented synthesis path by shelling out to an external command.

Required env vars for the command-based integration:
- `COSYVOICE_COMMAND`
- optionally `COSYVOICE_HTTP_URL`
- optionally `COSYVOICE_HEALTH_URL`
- optionally `COSYVOICE_LANGUAGE`
- optionally `TTS_PROVIDER=cosyvoice`

`COSYVOICE_HTTP_URL` should point at a remote CosyVoice service, not a machine-specific assumption. That lets Cloud Run treat CosyVoice as an external dependency that can later move from Basil's Mac to any dedicated host without backend code changes.

If `COSYVOICE_HEALTH_URL` is omitted, the backend derives it automatically from `COSYVOICE_HTTP_URL` by replacing `/synthesize` with `/health`.

At synthesis time the backend will export:
- `COSYVOICE_TEXT`
- `COSYVOICE_REFERENCE_AUDIO_PATH`
- `COSYVOICE_OUTPUT_PATH`
- `COSYVOICE_LANGUAGE`
- `VOICE_PROFILE_ID`

Your command is responsible for producing a WAV file at `COSYVOICE_OUTPUT_PATH`.

For a repo-local setup, point `COSYVOICE_COMMAND` at the bundled runner:

```env
COSYVOICE_COMMAND=python3 scripts/run_cosyvoice.py
COSYVOICE_MODEL_DIR=/absolute/path/to/CosyVoice/pretrained_models/CosyVoice-300M
COSYVOICE_REPO_DIR=/absolute/path/to/CosyVoice
COSYVOICE_INFERENCE_MODE=zero_shot
COSYVOICE_LANGUAGE=en
TTS_PROVIDER=cosyvoice
```

With that configuration, `run_cosyvoice.py` automatically falls back to the bundled local adapter at `scripts/cosyvoice_local_adapter.py`. The adapter expects a real CosyVoice checkout and Python environment with the CosyVoice dependencies installed.

To bootstrap a dedicated CosyVoice environment without polluting the main backend venv, use:

```bash
bash scripts/setup_cosyvoice_env.sh
```

The setup script intentionally requires Python `3.10`, `3.11`, or `3.12`. It will fail fast on older interpreters such as the repo's current Python `3.9` backend venv.

Optional local-adapter env vars:
- `COSYVOICE_PYTHON`
- `COSYVOICE_PROMPT_TEXT`
- `COSYVOICE_INSTRUCT_TEXT`

If you already have your own CosyVoice wrapper command, skip the local adapter and set:

```env
COSYVOICE_INFERENCE_COMMAND=/absolute/path/to/your/cosyvoice_wrapper.sh
```

If you want new profiles to default to CosyVoice instead of XTTS, set:

```env
VOICE_PROFILE_TRAINING_MODE=cosyvoice-reference
```

That keeps the current multi-sample reference-building pipeline, but marks the resulting profile for CosyVoice-backed synthesis instead of XTTS-backed synthesis.

## Request contract

`POST /voice-chat/turn`

```json
{
  "audioBase64": "<base64 m4a>",
  "mimeType": "audio/m4a",
  "fileName": "turn.m4a",
  "durationSeconds": 4.2,
  "voiceProfileID": "voice-profile-1234",
  "history": [
    { "role": "assistant", "content": "I am here." },
    { "role": "user", "content": "I feel scattered." }
  ]
}
```

Successful response:

```json
{
  "transcript": "I feel scattered.",
  "responseText": "Then narrow the next move. Pick one meaningful task and finish it cleanly.",
  "responseAudioBase64": "<optional base64 wav>",
  "responseAudioMimeType": "audio/wav",
  "error": null
}
```

Error response shape is the same JSON with `error` populated and an HTTP `5xx` status.

## Prerequisites

1. Python 3.11+
2. Ollama with an open model pulled locally, for example:
   - `ollama pull mistral`
3. Whisper CLI installed:
   - `pip install openai-whisper`
4. Optional Piper binary and model if you want stock spoken playback:
   - set `PIPER_COMMAND`
   - set `PIPER_MODEL_PATH`
   - optionally set `PIPER_CONFIG_PATH`
   - optionally set `VOICE_PROFILE_MODELS_DIR`
5. XTTS stack for personalized voice:
   - install `TTS`, `torchaudio`, and `soundfile` in the backend venv
   - set `XTTS_MODEL_NAME`
   - optionally set `XTTS_LANGUAGE`
   - set `COQUI_TOS_AGREED=1` only if you accept Coqui's CPML / license terms for the model download
6. Voice profile training worker:
   - set `VOICE_PROFILE_TRAINING_COMMAND`
   - set `VOICE_PROFILE_TRAINING_MODE`
   - optionally set `VOICE_PROFILE_TRAINING_TIMEOUT_SECONDS`
   - optionally set `FFMPEG_COMMAND`

## Voice profile training contract

`GET /voice-profiles/capabilities`

```json
{
  "trainingCommandConfigured": true,
  "trainingMode": "placeholder|copy-default-for-smoke-test|xtts-reference",
  "supportsPersonalizedVoice": true,
  "message": "..."
}
```

The iOS app uses this endpoint to distinguish:
- backend is not configured at all
- backend is in smoke-test mode only
- backend is capable of generating a real personalized voice

Only when `supportsPersonalizedVoice` is `true` should the product claim the trained agent voice can sound like the user.
For `xtts-reference`, this also requires `COQUI_TOS_AGREED=1`.

`POST /voice-profiles`

```json
{
  "transcript": "My best self is just ahead.",
  "durationSeconds": 6.4,
  "fileName": "voice-sample.m4a",
  "mimeType": "audio/m4a",
  "audioBase64": "<base64 m4a>"
}
```

Success response:

```json
{
  "jobId": "abc123..."
}
```

`GET /voice-profiles/{jobId}`

```json
{
  "status": "queued|processing|ready|failed",
  "progress": 0.45,
  "profileId": "voice-profile-1234",
  "message": "..."
}
```

Jobs become `ready` when the selected training mode produces its expected artifact:
- `xtts-reference`: a normalized reference WAV and manifest
- `copy-default-for-smoke-test`: a copied Piper `.onnx`

The backend can optionally invoke an external trainer command. It runs with these env vars:
- `VOICE_PROFILE_JOB_ID`
- `VOICE_PROFILE_ID`
- `VOICE_PROFILE_SAMPLE_PATH`
- `VOICE_PROFILE_TRANSCRIPT`
- `VOICE_PROFILE_OUTPUT_MODEL_PATH`
- `VOICE_PROFILE_OUTPUT_CONFIG_PATH`
- `VOICE_PROFILE_OUTPUT_REFERENCE_PATH`
- `VOICE_PROFILE_MANIFEST_PATH`
- `FFMPEG_COMMAND`
- `PIPER_MODEL_PATH`
- `PIPER_CONFIG_PATH`

Trainer scaffold:

- `scripts/train_voice_profile.py`
- `VOICE_PROFILE_TRAINING_MODE=placeholder`
  - default, fails honestly because no real trainer is implemented yet
- `VOICE_PROFILE_TRAINING_MODE=copy-default-for-smoke-test`
  - copies the default Piper voice into the profile artifact location
  - useful only for validating the end-to-end job/artifact path
  - not a personalized voice
- `VOICE_PROFILE_TRAINING_MODE=xtts-reference`
  - normalizes the recorded sample to a mono 24 kHz WAV
  - marks the profile ready for XTTS-driven synthesis
  - this is the first mode that can actually trend toward the user's voice
- `VOICE_PROFILE_TRAINING_MODE=cosyvoice-reference`
  - normalizes and merges the user's approved samples into a 24 kHz reference WAV
  - marks the profile ready for CosyVoice-driven synthesis
  - this is the preferred path when you have an external CosyVoice runtime available
- `VOICE_PROFILE_TRAINING_MODE=cosyvoice-finetuned`
  - builds a normalized reference WAV as a fallback artifact
  - expects `COSYVOICE_FINETUNED_TRAIN_COMMAND` to produce an inference-loadable CosyVoice model directory at `VOICE_PROFILE_OUTPUT_COSYVOICE_MODEL_DIR`
  - is the recommended path when you want a real versioned custom model instead of reference-audio cloning
  - the model directory must contain one of `cosyvoice.yaml`, `cosyvoice2.yaml`, or `cosyvoice3.yaml`
- `VOICE_PROFILE_TRAINING_MODE=cosyvoice-adapter`
  - builds the same normalized reference WAV as a fallback artifact
  - expects `COSYVOICE_ADAPTER_TRAIN_COMMAND` to write an adapter artifact directory to `VOICE_PROFILE_OUTPUT_ADAPTER_PATH`
  - the directory should contain at least `adapter.bin`, and may also contain `config.json` and `eval.json`
  - marks the profile ready only when the adapter artifact and manifest both exist
  - is the correct path for versioned per-user CosyVoice fine-tunes or LoRA-style adapters

Starter wrappers:

- `scripts/train_cosyvoice_finetuned.py`
  - contract wrapper for the upstream CosyVoice `cosyvoice/bin/train.py` path
  - reads `VOICE_PROFILE_DATASET_PATH`
  - expects you to provide a matching base model, config, and train/cv data files
- `scripts/cosyvoice_local_runtime.py`
  - real local inference wrapper around `AutoModel`
  - can load `COSYVOICE_MODEL_DIR_OVERRIDE` and synthesize via SFT or zero-shot modes
- `scripts/train_cosyvoice_adapter.py`
  - contract wrapper for a real GPU-backed training job
  - reads `VOICE_PROFILE_DATASET_PATH`
  - should export weights to `VOICE_PROFILE_OUTPUT_ADAPTER_WEIGHTS_PATH`
- `scripts/infer_cosyvoice_adapter.py`
  - contract wrapper for adapter-backed inference
  - reads `COSYVOICE_ADAPTER_PATH`
  - should synthesize to `COSYVOICE_OUTPUT_PATH`

Example local env:

```bash
VOICE_PROFILE_TRAINING_MODE=cosyvoice-finetuned
VOICE_PROFILE_TRAINING_COMMAND="$PWD/.venv/bin/python $PWD/scripts/train_voice_profile.py"
COSYVOICE_FINETUNED_TRAIN_COMMAND="$PWD/.venv/bin/python $PWD/scripts/train_cosyvoice_finetuned.py"
COSYVOICE_COMMAND="$PWD/.venv/bin/python $PWD/scripts/run_cosyvoice.py"
COSYVOICE_FINETUNED_BASE_MODEL_DIR=/absolute/path/to/CosyVoice/pretrained_models/CosyVoice-300M
COSYVOICE_FINETUNED_CONFIG_PATH=/absolute/path/to/your/finetune-config.yaml
COSYVOICE_FINETUNED_TRAIN_DATA=/absolute/path/to/train.parquet
COSYVOICE_FINETUNED_CV_DATA=/absolute/path/to/cv.parquet
COSYVOICE_FINETUNED_COMPONENT=flow
COSYVOICE_FINETUNED_SPK_ID=my_voice
```

Experimental adapter env:

```bash
VOICE_PROFILE_TRAINING_MODE=cosyvoice-adapter
VOICE_PROFILE_TRAINING_COMMAND="$PWD/.venv/bin/python $PWD/scripts/train_voice_profile.py"
COSYVOICE_ADAPTER_TRAIN_COMMAND="$PWD/.venv/bin/python $PWD/scripts/train_cosyvoice_adapter.py"
COSYVOICE_COMMAND="$PWD/.venv/bin/python $PWD/scripts/run_cosyvoice.py"
COSYVOICE_INFERENCE_COMMAND="$PWD/.venv/bin/python $PWD/scripts/infer_cosyvoice_adapter.py"
COSYVOICE_ADAPTER_BASE_MODEL=/absolute/path/to/CosyVoice/pretrained_models/CosyVoice-300M
```

Example command:

```bash
export VOICE_PROFILE_TRAINING_COMMAND="$PWD/.venv/bin/python $PWD/scripts/train_voice_profile.py"
```

## Local run

```bash
cd voice-backend
cp .env.example .env
./run-local.sh
```

The script will:
- create `.venv` if needed
- install `requirements.txt`
- load `.env` if present
- start the backend on `http://0.0.0.0:8080`

## Cloud Run prep

This repo now includes:
- `Dockerfile`
- `.dockerignore`
- `.env.example`

The container listens on `0.0.0.0:$PORT`, which matches Cloud Run's runtime contract.

### Recommended deployment shape

Use Cloud Run for:
- Google OAuth callback handling
- app sessions
- provider connections
- conversation persistence
- the API surface

Do not treat this exact local-model runtime as the final production shape for:
- Ollama on-box inference
- Whisper CLI on-box inference
- Piper / XTTS on-box synthesis

Those can work for early hosted experiments, but they are not the long-term scalable form factor. The right next architecture is:
- Cloud Run for the API layer
- separate managed or worker-backed STT / LLM / TTS services later

### Build and deploy

This repo includes `cloudbuild.yaml` so Cloud Build can build and deploy the service in one path.

Example Cloud Build submit from Cloud Shell:

```bash
gcloud builds submit --config cloudbuild.yaml
```

Useful substitution overrides:

```bash
gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions _SERVICE_NAME=pika-voice-backend,_REGION=us-central1,_REPOSITORY=pika,_IMAGE_NAME=voice-backend
```

`cloudbuild.yaml` currently deploys with:
- port `8080`
- `AUTH_MOBILE_CALLBACK_SCHEME=pikatakehome`
- `--allow-unauthenticated`

After deploy, Cloud Run gives you a stable HTTPS URL ending in `run.app`.

### Recommended custom domain

Use a real domain such as:

```text
https://api.yourdomain.com
```

Then set:

```env
GOOGLE_OAUTH_CALLBACK_URL=https://api.yourdomain.com/auth/google/callback
```

And in the iOS app set:
- `AuthBaseURL = https://api.yourdomain.com`
- `VoiceChatBaseURL = https://api.yourdomain.com`
- `VoiceTrainingBaseURL = https://api.yourdomain.com`

For custom domains on Cloud Run, Google currently recommends a global external Application Load Balancer in front of Cloud Run rather than relying on Cloud Run domain mapping preview.

### Cloud Run env vars

Minimum auth/session config:

```env
GOOGLE_OAUTH_CLIENT_ID=...
GOOGLE_OAUTH_CLIENT_SECRET=...
GOOGLE_OAUTH_CALLBACK_URL=https://api.yourdomain.com/auth/google/callback
AUTH_MOBILE_CALLBACK_SCHEME=pikatakehome
```

You can set these after the first deploy with:

```bash
gcloud run services update pika-voice-backend \
  --region us-central1 \
  --set-env-vars GOOGLE_OAUTH_CLIENT_ID=...,GOOGLE_OAUTH_CLIENT_SECRET=...,GOOGLE_OAUTH_CALLBACK_URL=https://api.yourdomain.com/auth/google/callback,AUTH_MOBILE_CALLBACK_SCHEME=pikatakehome
```

Useful runtime defaults:

```env
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_KEEP_ALIVE=15m
WHISPER_MODEL=base
WHISPER_CHUNK_DURATION_SECONDS=15
HTTP_TIMEOUT_SECONDS=90
```

If you are only deploying the auth/session/API layer first, leave the voice-model env vars unset and avoid routing production traffic through `/voice-chat/turn` until the inference layer is moved to an appropriate runtime.

## iOS app configuration

Set one of:
- `VOICE_CHAT_BASE_URL`
- `VoiceChatBaseURL` in the iOS app's `Info.plist`

Examples:
- Simulator: `http://127.0.0.1:8080`
- Physical device: `http://<your-mac-lan-ip>:8080`

## Notes

- If Piper is not configured, the backend still returns `transcript` and `responseText`; the app will show text-only replies.
- If you want a different open model, change `OLLAMA_MODEL` in `.env`.
- `OLLAMA_KEEP_ALIVE` controls how long Ollama keeps the chat model hot between turns. The backend defaults this to `15m`.
- `WHISPER_CHUNK_DURATION_SECONDS` controls when long recordings are split into smaller transcription chunks. The backend defaults this to `15`.
- If `voiceProfileID` is provided and its manifest declares `provider: xtts-reference`, the backend will synthesize with XTTS using the saved reference clip.
- If `voiceProfileID` is provided and its manifest declares `provider: cosyvoice-reference`, the backend will synthesize by running `COSYVOICE_COMMAND`.
- If `voiceProfileID` is provided and its manifest declares `provider: cosyvoice-finetuned`, the backend will synthesize by running `COSYVOICE_COMMAND` with `COSYVOICE_MODEL_DIR_OVERRIDE` set to the fine-tuned model directory. If a single speaker exists in `spk2info.pt`, the local runtime will use it automatically.
- If `voiceProfileID` is provided and its manifest declares `provider: cosyvoice-adapter`, the backend will synthesize by running `COSYVOICE_COMMAND` with `COSYVOICE_ADAPTER_PATH` and optional `COSYVOICE_ADAPTER_BASE_MODEL`.
- Adapter artifacts are directory-based. The current contract expects at least `adapter.bin` inside `COSYVOICE_ADAPTER_PATH`.
- The first XTTS request may download model weights for `XTTS_MODEL_NAME`, so expect a slow cold start.
- XTTS model download is gated by Coqui's license prompt. In non-interactive backend runs, set `COQUI_TOS_AGREED=1` only if you have explicitly accepted those terms.
- If a profile manifest is missing or XTTS is unavailable, the backend falls back to the default Piper voice.
