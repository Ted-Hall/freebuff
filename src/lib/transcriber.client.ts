export type ModelMode = "fast" | "accurate";

export interface TranscriberCallbacks {
  onStatus: (message: string) => void;
  onProgress: (progress: number) => void;
  onMode: (mode: "WebGPU" | "WASM") => void;
}

interface PendingRequest {
  resolve: (text: string) => void;
  reject: (error: Error) => void;
}

type WorkerMessage =
  | { type: "loading"; message: string; progress: number }
  | { type: "mode"; mode: "WebGPU" | "WASM" }
  | { type: "progress"; progress: number }
  | { type: "result"; text: string }
  | { type: "error"; message: string };

export async function detectWebGpu(): Promise<boolean> {
  const gpu = (navigator as Navigator & {
    gpu?: { requestAdapter: (options?: { powerPreference?: "low-power" | "high-performance" }) => Promise<unknown> };
  }).gpu;
  if (!gpu) return false;
  try {
    return Boolean(await gpu.requestAdapter({ powerPreference: "low-power" }));
  } catch {
    return false;
  }
}

export class LocalTranscriber {
  private worker: Worker | null = null;
  private pending: PendingRequest | null = null;
  private requestId = 0;
  private stallTimer: number | null = null;

  constructor(
    private readonly callbacks: TranscriberCallbacks,
    private readonly stallTimeoutMs = 45_000,
  ) {}

  async transcribe(samples: Float32Array, preferWebGpu: boolean, modelMode: ModelMode): Promise<string> {
    this.dispose(false);
    const id = ++this.requestId;
    const result = new Promise<string>((resolve, reject) => {
      this.pending = { resolve, reject };
    });

    const beginAttempt = (attempt: number): void => {
      const worker = new Worker(new URL("../workers/transcriber.worker.ts", import.meta.url), {
        type: "module",
      });
      this.worker = worker;
      let modelIsLoading = true;

      const clearStallTimer = (): void => {
        if (this.stallTimer !== null) window.clearTimeout(this.stallTimer);
        this.stallTimer = null;
      };
      const armStallTimer = (): void => {
        clearStallTimer();
        this.stallTimer = window.setTimeout(() => {
          if (this.worker !== worker || !modelIsLoading) return;
          worker.terminate();
          this.worker = null;
          clearStallTimer();
          if (attempt < 1 && navigator.onLine !== false) {
            this.callbacks.onStatus("The model download stalled. Restarting it…");
            this.callbacks.onProgress(0);
            beginAttempt(attempt + 1);
            return;
          }
          const error = new Error(
            navigator.onLine === false
              ? "You appear to be offline. Connect once to download the speech model."
              : "The model download did not respond. Check your connection and try again.",
          );
          const pending = this.pending;
          this.pending = null;
          pending?.reject(error);
        }, this.stallTimeoutMs);
      };

      worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
        if (this.worker !== worker) return;
        const message = event.data;
        if (message.type === "loading") {
          this.callbacks.onStatus(message.message);
          this.callbacks.onProgress(message.progress);
          armStallTimer();
        } else if (message.type === "mode") {
          this.callbacks.onMode(message.mode);
        } else if (message.type === "progress") {
          this.callbacks.onProgress(message.progress);
          if (message.progress >= 100) {
            modelIsLoading = false;
            clearStallTimer();
          }
        } else if (message.type === "result") {
          clearStallTimer();
          const pending = this.pending;
          this.pending = null;
          pending?.resolve(message.text);
          this.dispose(false);
        } else {
          clearStallTimer();
          const error = new Error(message.message);
          const pending = this.pending;
          this.pending = null;
          pending?.reject(error);
          this.dispose(false);
        }
      };
      worker.onerror = (event) => {
        if (this.worker !== worker) return;
        clearStallTimer();
        const error = new Error(event.message || "The local transcription worker stopped unexpectedly.");
        const pending = this.pending;
        this.pending = null;
        pending?.reject(error);
        this.dispose(false);
      };

      const copy = samples.slice();
      worker.postMessage(
        { type: "transcribe", id, samples: copy.buffer, preferWebGpu, modelMode },
        [copy.buffer],
      );
    };

    beginAttempt(0);
    return result;
  }

  dispose(rejectPending = true): void {
    if (this.stallTimer !== null) window.clearTimeout(this.stallTimer);
    this.stallTimer = null;
    if (rejectPending && this.pending) {
      this.pending.reject(new Error("Transcription was cancelled."));
    }
    this.pending = null;
    this.worker?.terminate();
    this.worker = null;
  }
}
