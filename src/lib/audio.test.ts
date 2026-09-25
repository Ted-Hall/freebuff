import { describe, expect, it } from "vitest";
import {
  analyzeSignal,
  calculateRms,
  enhanceForSpeech,
  formatDuration,
  mixToMono,
  resampleAudio,
} from "./audio";

describe("audio utilities", () => {
  it("formats recording time for a compact mobile display", () => {
    expect(formatDuration(0)).toBe("0:00");
    expect(formatDuration(65.9)).toBe("1:05");
  });

  it("mixes stereo channels into mono", () => {
    const mixed = mixToMono([new Float32Array([1, 0]), new Float32Array([0, 1])]);
    expect(Array.from(mixed)).toEqual([0.5, 0.5]);
  });

  it("downsamples 48 kHz audio to 16 kHz with area averaging", () => {
    const input = new Float32Array([0, 0, 1, 1]);
    const output = resampleAudio(input, 48_000, 16_000);
    expect(output).toHaveLength(1);
    expect(output[0]).toBe(0.5);
  });

  it("preserves samples when the sample rate already matches", () => {
    const input = new Float32Array([0.1, -0.2, 0.3]);
    expect(Array.from(resampleAudio(input, 16_000))).toEqual(Array.from(input));
  });

  it("removes low-frequency rumble and safely raises quiet speech", () => {
    const makeTone = (frequency: number) => {
      const tone = new Float32Array(16_000);
      for (let index = 0; index < tone.length; index += 1) {
        tone[index] = Math.sin((2 * Math.PI * frequency * index) / 16_000) * 0.02;
      }
      return tone;
    };
    const enhancedRumble = enhanceForSpeech(makeTone(20));
    const enhancedSpeech = enhanceForSpeech(makeTone(1_000));
    const rumbleLevel = calculateRms(enhancedRumble.slice(3_200, 4_800));
    const speechLevel = calculateRms(enhancedSpeech.slice(3_200, 4_800));
    expect(speechLevel).toBeGreaterThan(rumbleLevel * 5);
    expect(Math.max(...Array.from(enhancedSpeech, Math.abs))).toBeLessThanOrEqual(1);
  });

  it("labels quiet, noisy, and clear recordings from measured signal levels", () => {
    expect(analyzeSignal(new Float32Array(3_200)).quality).toBe("quiet");

    const noisy = new Float32Array(16_000).fill(0.02);
    for (let index = 8_000; index < 12_000; index += 1) noisy[index] = 0.5;
    expect(["noisy", "very-noisy"]).toContain(analyzeSignal(noisy).quality);

    const clear = new Float32Array(16_000).fill(0.0001);
    for (let index = 4_000; index < 12_000; index += 1) clear[index] = 0.4;
    expect(analyzeSignal(clear).quality).toBe("clear");
  });
});
