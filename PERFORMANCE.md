# Voice Backend Performance Notes

This document captures the latency work applied to the local voice backend and the measured impact on the end-to-end `POST /voice-chat/turn` path.

## Scope

The measured path includes:

1. base64 audio decode
2. speech transcription
3. LLM response generation through Ollama
4. audio synthesis for the reply

These measurements were taken locally on the current development machine against the same 3-second sample clip.

## Changes Made

### 1. Reused Whisper in-process

Before:
- the backend spawned the Whisper CLI on every turn
- each request paid subprocess startup cost and transcript file I/O cost

After:
- the backend loads Whisper in-process and caches the model
- repeated turns reuse the already-loaded model

Implementation:
- [app/providers.py](/Users/basilarif/Documents/GitHub/PikaProj/voice-backend/app/providers.py)

### 2. Added optional audio chunking for long recordings

Before:
- all audio was transcribed as one unit

After:
- longer recordings can be split into smaller chunks before transcription
- chunking is controlled by `WHISPER_CHUNK_DURATION_SECONDS`
- default is `15` seconds

Why this helps:
- reduces worst-case latency on longer recordings
- makes long-turn transcription less brittle than one large decode pass

Implementation:
- [app/config.py](/Users/basilarif/Documents/GitHub/PikaProj/voice-backend/app/config.py)
- [app/providers.py](/Users/basilarif/Documents/GitHub/PikaProj/voice-backend/app/providers.py)

### 3. Reused the Ollama HTTP client

Before:
- the backend created a new `httpx.AsyncClient` for every turn

After:
- the backend reuses a persistent client
- keep-alive is explicitly sent to Ollama

Why this helps:
- reduces connection overhead
- improves warm-turn behavior

Implementation:
- [app/providers.py](/Users/basilarif/Documents/GitHub/PikaProj/voice-backend/app/providers.py)

### 4. Switched the local chat model from `mistral` to `qwen2.5:3b`

Before:
- `OLLAMA_MODEL=mistral`

After:
- `OLLAMA_MODEL=qwen2.5:3b`

Why this helps:
- much smaller local model
- faster cold start
- materially faster generation on the same hardware

Configuration:
- `voice-backend/.env`

### 5. Added per-stage timing logs

The backend now logs:
- audio duration
- transcription time
- LLM time
- TTS time
- total turn time

This makes it easy to see which stage is actually dominating latency.

Example log format:

```text
[voice-chat] audio=3.00s transcribe=0.52s llm=0.70s tts=0.86s total=2.09s
```

## Measured Results

### Baseline after backend code optimization, before model switch

Using the optimized backend with the prior local model:

- warm turn: `transcribe=2.84s llm=4.26s tts=1.65s total=8.76s`

Observation:
- LLM generation was the dominant cost

### After switching to `qwen2.5:3b`

Cold first turn after model load:

- `transcribe=3.18s llm=2.72s tts=1.11s total=7.01s`

Immediate warm turn:

- `transcribe=0.52s llm=0.70s tts=0.86s total=2.09s`

## Practical Gains

Compared with the prior measured warm path:

- total latency improved from `8.76s` to `2.09s`
- LLM latency improved from `4.26s` to `0.70s`
- transcription also improved on the warm path due to in-process reuse

## Remaining Bottlenecks

The remaining latency is now mostly one of:

1. first-turn cold loads
2. transcription on longer recordings
3. TTS synthesis cost on every reply

If more speed is needed later, the next options are:

- stream partial text responses to the client instead of waiting for the full reply
- skip or defer TTS for some turns
- reduce reply length
- use a smaller fallback model for low-priority turns
- add VAD or incremental chunk upload from the client for longer speech

## Current Recommendation

For this project's local setup:

- use `qwen2.5:3b` as the default local model
- keep `OLLAMA_KEEP_ALIVE=15m`
- keep `WHISPER_CHUNK_DURATION_SECONDS=15`

That gives a good balance between latency and response quality while keeping the local stack simple.
