/// <reference lib="webworker" />

import { env, pipeline } from "@huggingface/transformers";

const workerScope = self as unknown as DedicatedWorkerGlobalScope;
const MODEL_IDS = {
  fast: "onnx-community/moonshine-tiny-ONNX",
  accurate: "onnx-community/whisper-tiny.en",
} as const;

type ModelMode = keyof typeof MODEL_IDS;

type ProgressEvent = {
  status: string;
  progress?: number;
  file?: string;
};

type Recognizer = (
  audio: Float32Array,
  options: Record<string, unknown>,
) => Promise<{ text: string }>;

env.allowLocalModels = false;
env.useBrowserCache = true;
env.useWasmCache = true;

async function loadRecognizer(preferWebGpu: boolean, modelMode: ModelMode): Promise<Recognizer> {
  const modelSize = modelMode === "fast" ? "~28 MB fast" : "~41 MB accuracy";
  const report = (message: string, progress = 0) => {
    workerScope.postMessage({ type: "loading", message, progress });
  };

  if (preferWebGpu && modelMode === "accurate") {
    try {
      report("Starting the faster on-device engine…", 4);
      const recognizer = await pipeline("automatic-speech-recognition", MODEL_IDS[modelMode], {
        device: "webgpu",
        dtype: { encoder_model: "fp16", decoder_model_merged: "fp16" },
        progress_callback: (progress: ProgressEvent) => {
          if ((progress.status === "progress" || progress.status === "progress_total") && progress.progress !== undefined) {
            report("Preparing the faster on-device engine…", Math.round(progress.progress));
          }
        },
      });
      workerScope.postMessage({ type: "mode", mode: "WebGPU" });
      return recognizer as unknown as Recognizer;
    } catch {
      report("WebGPU is unavailable here. Switching to the low-memory engine…", 5);
    }
  }

  if (modelMode === "fast") {
    report("Using the compact 28 MB model…", 3);
  }
  const recognizer = await pipeline("automatic-speech-recognition", MODEL_IDS[modelMode], {
    device: "wasm",
    dtype: "q8",
    progress_callback: (progress: ProgressEvent) => {
      if ((progress.status === "progress" || progress.status === "progress_total") && progress.progress !== undefined) {
        report(`Downloading the ${modelSize} speech model…`, Math.round(progress.progress));
      }
    },
  });
  workerScope.postMessage({ type: "mode", mode: "WASM" });
  return recognizer as unknown as Recognizer;
}

workerScope.onmessage = async (event: MessageEvent) => {
  const message = event.data as {
    type: "transcribe";
    id: number;
    samples: ArrayBuffer;
    preferWebGpu: boolean;
    modelMode: ModelMode;
  };
  if (message.type !== "transcribe") return;

  try {
    const recognizer = await loadRecognizer(message.preferWebGpu, message.modelMode);
    workerScope.postMessage({ type: "progress", progress: 100 });
    const samples = new Float32Array(message.samples);

    if (message.modelMode === "fast") {
      // Moonshine is fast and small but expects short windows. Keep each window
      // below its training context so long field notes do not fail or drift.
      const windowLength = 30 * 16_000;
      const parts: string[] = [];
      const windowCount = Math.max(1, Math.ceil(samples.length / windowLength));
      for (let start = 0; start < samples.length; start += windowLength) {
        const window = samples.slice(start, Math.min(start + windowLength, samples.length));
        const result = await recognizer(window, { return_timestamps: false });
        if (result.text.trim()) parts.push(result.text.trim());
        workerScope.postMessage({ type: "progress", progress: Math.round(100 * ((start / samples.length) + 1 / windowCount)) });
      }
      const text = parts.join(" ").replace(/\s+/g, " ").replace(/(.{72}[.!?])\s+/g, "$1\n\n").trim();
      workerScope.postMessage({ type: "result", text });
      return;
    }

    const result = await recognizer(samples, {
      chunk_length_s: 30,
      stride_length_s: 5,
      return_timestamps: false,
      do_sample: false,
      num_beams: 1,
      max_new_tokens: 256,
    });
    const text = result.text.replace(/\s+/g, " ").replace(/(.{72}[.!?])\s+/g, "$1\n\n").trim();
    workerScope.postMessage({ type: "result", text });
  } catch (error) {
    const messageText = error instanceof Error ? error.message : "Local transcription failed.";
    workerScope.postMessage({ type: "error", message: messageText });
  }
};

export {};
