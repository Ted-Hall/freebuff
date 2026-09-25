# Sotto

Sotto is an offline-first voice-to-text MVP designed for older, low-memory mobile phones. It records mono 16 kHz audio, applies the browser's echo cancellation, noise suppression, and automatic gain controls, then transcribes the recording locally with a quantized Whisper Tiny English model.

## Why this stack

- **Transformers.js + ONNX Runtime** runs the recognizer in a Web Worker without sending audio to a server.
- **Fast mode (default)** uses quantized Moonshine Tiny English, roughly 28 MB of ONNX weights, for the shortest first-run download on old phones. **Accuracy mode** uses Whisper Tiny English (~41 MB) when extra robustness matters.
- **WebGPU** is used when the browser exposes a compatible adapter. A quantized **WASM** path is the automatic fallback for older phones.
- **Mono 16 kHz capture** reduces memory and compute while retaining the range the speech models expect.
- **Browser audio constraints** request echo cancellation, noise suppression, and automatic gain control before recording.
- **The app shell is installable and cacheable.** The model is cached by Transformers.js after its first download; the first transcription needs a connection unless the model has already been cached.

## Run locally

```bash
bun install
bun run dev
```

Open the preview on a phone over HTTPS, allow microphone access, record, and tap **Make transcript**. The default Fast model is about 28 MB and is cached by the browser after the first download. Sotto also requests persistent browser storage so the model is less likely to be evicted.

## Verify

```bash
bun run typecheck
bun test
bun run build
```

## Privacy and limits

Audio and transcripts stay in the browser. Sotto has no account, analytics, upload endpoint, or cloud transcription API. The browser downloads the open model from Hugging Face on first use. Very noisy audio remains challenging: place the phone 20–30 cm from the speaker, use a quiet side or windscreen, and record one speaker at a time.
