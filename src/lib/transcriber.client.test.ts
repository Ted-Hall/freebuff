import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LocalTranscriber, type TranscriberCallbacks } from "./transcriber.client";

class FakeWorker {
  static instances: FakeWorker[] = [];
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  postMessage = vi.fn();
  terminate = vi.fn(() => {
    this.terminated = true;
  });
  terminated = false;

  constructor() {
    FakeWorker.instances.push(this);
  }

  emit(data: unknown): void {
    this.onmessage?.({ data } as MessageEvent);
  }
}

const callbacks: TranscriberCallbacks = {
  onStatus: vi.fn(),
  onProgress: vi.fn(),
  onMode: vi.fn(),
};

const originalWorker = globalThis.Worker;
const originalWindow = globalThis.window;
const originalNavigator = globalThis.navigator;

function replaceGlobal(name: string, value: unknown): void {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}

describe("LocalTranscriber", () => {
  beforeEach(() => {
    FakeWorker.instances = [];
    replaceGlobal("Worker", FakeWorker);
    replaceGlobal("window", globalThis);
    replaceGlobal("navigator", { onLine: true });
  });

  afterEach(() => {
    vi.useRealTimers();
    replaceGlobal("Worker", originalWorker);
    replaceGlobal("window", originalWindow);
    replaceGlobal("navigator", originalNavigator);
  });

  it("resolves with the worker's transcript", async () => {
    const transcriber = new LocalTranscriber(callbacks);
    const result = transcriber.transcribe(new Float32Array(1_600), false, "fast");
    const worker = FakeWorker.instances[0];
    worker.emit({ type: "progress", progress: 100 });
    worker.emit({ type: "result", text: "Local transcript" });
    await expect(result).resolves.toBe("Local transcript");
  });

  it("passes the selected engine and model to the worker", () => {
    const transcriber = new LocalTranscriber(callbacks);
    void transcriber.transcribe(new Float32Array(1_600), true, "fast");
    expect(FakeWorker.instances[0].postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ preferWebGpu: true, modelMode: "fast" }),
      expect.anything(),
    );
    transcriber.dispose(false);
  });

  it("rejects when the worker reports a terminal error", async () => {
    const transcriber = new LocalTranscriber(callbacks);
    const result = transcriber.transcribe(new Float32Array(1_600), false, "fast");
    FakeWorker.instances[0].emit({ type: "error", message: "Model unavailable" });
    await expect(result).rejects.toThrow("Model unavailable");
  });
});
