export const TARGET_SAMPLE_RATE = 16_000;
export const MAX_RECORDING_SECONDS = 5 * 60;
export const MAX_RECORDING_SAMPLES = TARGET_SAMPLE_RATE * MAX_RECORDING_SECONDS;

export type SignalQuality = "clear" | "noisy" | "very-noisy" | "quiet";

export interface SignalAnalysis {
  peak: number;
  noiseFloor: number;
  quality: SignalQuality;
}

export function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

export function formatDuration(totalSeconds: number): string {
  const safeSeconds = Math.max(0, Math.floor(totalSeconds));
  const minutes = Math.floor(safeSeconds / 60);
  const seconds = safeSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

export function calculateRms(samples: Float32Array): number {
  if (samples.length === 0) return 0;
  let sum = 0;
  for (const sample of samples) sum += sample * sample;
  return Math.sqrt(sum / samples.length);
}

export function mixToMono(channels: Float32Array[]): Float32Array {
  if (channels.length === 0) return new Float32Array();
  if (channels.length === 1) return channels[0].slice();

  const output = new Float32Array(channels[0].length);
  for (const channel of channels) {
    for (let index = 0; index < output.length; index += 1) {
      output[index] += channel[index] ?? 0;
    }
  }
  for (let index = 0; index < output.length; index += 1) output[index] /= channels.length;
  return output;
}

export function resampleAudio(
  input: Float32Array,
  inputSampleRate: number,
  outputSampleRate = TARGET_SAMPLE_RATE,
): Float32Array {
  if (input.length === 0 || inputSampleRate === outputSampleRate) return input.slice();
  if (inputSampleRate <= 0 || outputSampleRate <= 0) return input.slice();

  if (inputSampleRate > outputSampleRate) {
    const outputLength = Math.max(1, Math.round((input.length * outputSampleRate) / inputSampleRate));
    const output = new Float32Array(outputLength);
    for (let index = 0; index < outputLength; index += 1) {
      const start = (index * input.length) / outputLength;
      const end = ((index + 1) * input.length) / outputLength;
      const firstSample = Math.floor(start);
      const lastSample = Math.min(input.length - 1, Math.ceil(end) - 1);
      let sum = 0;
      let weight = 0;
      for (let sourceIndex = firstSample; sourceIndex <= lastSample; sourceIndex += 1) {
        const overlap = Math.max(0, Math.min(end, sourceIndex + 1) - Math.max(start, sourceIndex));
        sum += input[sourceIndex] * overlap;
        weight += overlap;
      }
      output[index] = weight > 0 ? sum / weight : 0;
    }
    return output;
  }

  const ratio = inputSampleRate / outputSampleRate;
  const output = new Float32Array(Math.max(1, Math.floor(input.length * ratio)));
  for (let index = 0; index < output.length; index += 1) {
    const position = index / ratio;
    const left = Math.floor(position);
    const right = Math.min(input.length - 1, left + 1);
    const fraction = position - left;
    output[index] = input[left] * (1 - fraction) + input[right] * fraction;
  }
  return output;
}

export function enhanceForSpeech(input: Float32Array, sampleRate = TARGET_SAMPLE_RATE): Float32Array {
  if (input.length === 0) return input;

  const cutoffHz = 65;
  const timeConstant = 1 / (2 * Math.PI * cutoffHz);
  const alpha = timeConstant / (timeConstant + 1 / sampleRate);
  let previousInput = input[0];
  let previousOutput = 0;
  const filtered = new Float32Array(input.length);

  for (let index = 0; index < input.length; index += 1) {
    const sample = input[index];
    previousOutput = alpha * (previousOutput + sample - previousInput);
    filtered[index] = previousOutput;
    previousInput = sample;
  }

  let peak = 0;
  for (const sample of filtered) peak = Math.max(peak, Math.abs(sample));
  const gain = peak > 0.008 ? Math.min(4, 0.92 / peak) : 1;
  for (let index = 0; index < filtered.length; index += 1) {
    filtered[index] = clamp(filtered[index] * gain, -1, 1);
  }
  return filtered;
}

export function analyzeSignal(samples: Float32Array): SignalAnalysis {
  if (samples.length === 0) return { peak: 0, noiseFloor: 0, quality: "quiet" };

  const frameSize = 1_600;
  const levels: number[] = [];
  let peak = 0;
  for (let start = 0; start < samples.length; start += frameSize) {
    const frame = samples.subarray(start, Math.min(start + frameSize, samples.length));
    levels.push(calculateRms(frame));
    for (const sample of frame) peak = Math.max(peak, Math.abs(sample));
  }
  levels.sort((a, b) => a - b);
  const percentile = (position: number) => levels[Math.min(levels.length - 1, Math.floor(levels.length * position))];
  const noiseFloor = percentile(0.2);
  const speechLevel = percentile(0.9);
  const peakLevel = percentile(0.98);

  if (peak < 0.012 || speechLevel < 0.006) return { peak, noiseFloor, quality: "quiet" };
  if (noiseFloor < 0.0015) return { peak, noiseFloor, quality: "clear" };
  const ratioDb = 20 * Math.log10(Math.max(noiseFloor, 1e-6) / Math.max(peakLevel, noiseFloor * 1.01));
  return { peak, noiseFloor, quality: ratioDb > -15 ? "noisy" : "very-noisy" };
}

export async function decodeAudioFile(file: File): Promise<Float32Array> {
  const AudioContextClass = window.AudioContext ??
    (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AudioContextClass) throw new Error("This browser cannot decode audio files.");

  const context = new AudioContextClass();
  try {
    const decoded = await context.decodeAudioData(await file.arrayBuffer());
    const mono = mixToMono(Array.from({ length: decoded.numberOfChannels }, (_, channel) => decoded.getChannelData(channel)));
    return enhanceForSpeech(resampleAudio(mono, decoded.sampleRate));
  } finally {
    await context.close();
  }
}

export class VoiceRecorder {
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private processor: ScriptProcessorNode | null = null;
  private silentOutput: GainNode | null = null;
  private chunks: Float32Array[] = [];
  private recordedSamples = 0;

  constructor(
    private readonly onLevel: (level: number) => void,
    private readonly noiseGuard: boolean,
  ) {}

  async start(): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Microphone recording needs HTTPS or localhost.");
    }
    if (this.context) throw new Error("A recording is already active.");

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: this.noiseGuard,
        autoGainControl: this.noiseGuard,
      },
      video: false,
    });

    const AudioContextClass = window.AudioContext ??
      (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextClass) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
      throw new Error("This browser does not support local audio capture.");
    }

    try {
      this.context = new AudioContextClass({ latencyHint: "interactive" });
      if (this.context.state === "suspended") await this.context.resume();
      this.chunks = [];
      this.recordedSamples = 0;
      this.source = this.context.createMediaStreamSource(this.stream);
      this.processor = this.context.createScriptProcessor(4_096, 1, 1);
      this.silentOutput = this.context.createGain();
      this.silentOutput.gain.value = 0;

      this.processor.onaudioprocess = (event) => {
        if (this.recordedSamples >= MAX_RECORDING_SAMPLES) return;
        const input = event.inputBuffer.getChannelData(0);
        const remaining = MAX_RECORDING_SAMPLES - this.recordedSamples;
        const resampled = resampleAudio(input, this.context?.sampleRate ?? TARGET_SAMPLE_RATE);
        const chunk = resampled.length > remaining ? resampled.slice(0, remaining) : resampled.slice();
        this.chunks.push(chunk);
        this.recordedSamples += chunk.length;
        this.onLevel(clamp(calculateRms(chunk) * 7, 0, 1));
      };

      this.source.connect(this.processor);
      this.processor.connect(this.silentOutput);
      this.silentOutput.connect(this.context.destination);
    } catch (error) {
      await this.cleanup();
      throw error;
    }
  }

  async stop(): Promise<Float32Array> {
    const sampleCount = this.recordedSamples;
    const samples = new Float32Array(sampleCount);
    let offset = 0;
    for (const chunk of this.chunks) {
      samples.set(chunk, offset);
      offset += chunk.length;
    }
    this.chunks = [];
    await this.cleanup();
    return enhanceForSpeech(samples);
  }

  async cancel(): Promise<void> {
    this.chunks = [];
    this.recordedSamples = 0;
    await this.cleanup();
  }

  private async cleanup(): Promise<void> {
    if (this.processor) {
      this.processor.onaudioprocess = null;
      this.processor.disconnect();
    }
    this.source?.disconnect();
    this.silentOutput?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    if (this.context && this.context.state !== "closed") await this.context.close();
    this.processor = null;
    this.source = null;
    this.silentOutput = null;
    this.stream = null;
    this.context = null;
    this.onLevel(0);
  }
}
