import { useEffect, useMemo, useRef, useState } from "react";
import {
  analyzeSignal,
  decodeAudioFile,
  formatDuration,
  MAX_RECORDING_SECONDS,
  MAX_RECORDING_SAMPLES,
  TARGET_SAMPLE_RATE,
  VoiceRecorder,
  type SignalAnalysis,
} from "./lib/audio";
import { detectWebGpu, LocalTranscriber, type ModelMode } from "./lib/transcriber.client";

type Phase = "idle" | "requesting" | "recording" | "decoding" | "ready" | "transcribing" | "complete" | "error";
type EngineMode = "WebGPU" | "WASM";

type InstallPromptEvent = Event & {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed" }>;
};

const qualityCopy: Record<SignalAnalysis["quality"], { label: string; detail: string }> = {
  clear: { label: "Clear input", detail: "Strong speech signal" },
  noisy: { label: "Noisy input", detail: "Cleanup will be working hard" },
  "very-noisy": { label: "Very noisy", detail: "Move closer to the speaker if you can" },
  quiet: { label: "Very quiet", detail: "Speak a little closer to the phone" },
};

function Icon({ name, size = 20 }: { name: "mic" | "stop" | "upload" | "copy" | "share" | "download" | "check" | "spark" | "lock" | "install" | "trash"; size?: number }) {
  const common = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true };
  if (name === "mic") return <svg {...common}><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" /><path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v3M8 22h8" /></svg>;
  if (name === "stop") return <svg {...common}><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none" /></svg>;
  if (name === "upload") return <svg {...common}><path d="M12 16V4M7 9l5-5 5 5" /><path d="M5 14v5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-5" /></svg>;
  if (name === "copy") return <svg {...common}><rect x="8" y="8" width="11" height="12" rx="2" /><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h2" /></svg>;
  if (name === "share") return <svg {...common}><circle cx="18" cy="5" r="2.5" /><circle cx="6" cy="12" r="2.5" /><circle cx="18" cy="19" r="2.5" /><path d="m8.2 10.8 7.6-4.5M8.2 13.2l7.6 4.5" /></svg>;
  if (name === "download") return <svg {...common}><path d="M12 4v12M7 11l5 5 5-5" /><path d="M5 20h14" /></svg>;
  if (name === "check") return <svg {...common}><path d="m5 12 4 4L19 6" /></svg>;
  if (name === "spark") return <svg {...common}><path d="m12 2 1.5 5.2L19 9l-5.5 1.8L12 16l-1.5-5.2L5 9l5.5-1.8L12 2Z" /><path d="m19 15 .7 2.3L22 18l-2.3.7L19 21l-.7-2.3L16 18l2.3-.7L19 15Z" /></svg>;
  if (name === "lock") return <svg {...common}><rect x="5" y="10" width="14" height="11" rx="2" /><path d="M8 10V7a4 4 0 0 1 8 0v3" /></svg>;
  if (name === "install") return <svg {...common}><path d="M12 3v12M7 10l5 5 5-5" /><path d="M5 20h14" /></svg>;
  return <svg {...common}><path d="M4 7h16M9 7V4h6v3M7 7l1 14h8l1-14M10 11v6M14 11v6" /></svg>;
}

function initialTranscript(): string {
  try {
    return window.localStorage.getItem("sotto:last-transcript") ?? "";
  } catch {
    return "";
  }
}

export default function App() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [elapsed, setElapsed] = useState(0);
  const [audio, setAudio] = useState<Float32Array | null>(null);
  const [analysis, setAnalysis] = useState<SignalAnalysis | null>(null);
  const [transcript, setTranscript] = useState(initialTranscript);
  const [status, setStatus] = useState("Ready when you are");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const [noiseGuard, setNoiseGuard] = useState(true);
  const [modelMode, setModelMode] = useState<ModelMode>("fast");
  const [waveform, setWaveform] = useState<number[]>(() => Array.from({ length: 24 }, (_, index) => 0.08 + (index % 4) * 0.025));
  const [sourceName, setSourceName] = useState("");
  const [webGpuAvailable, setWebGpuAvailable] = useState<boolean | null>(null);
  const [engineMode, setEngineMode] = useState<EngineMode | null>(null);
  const [copied, setCopied] = useState(false);
  const [installPrompt, setInstallPrompt] = useState<InstallPromptEvent | null>(null);

  const recorderRef = useRef<VoiceRecorder | null>(null);
  const transcriberRef = useRef<LocalTranscriber | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    void detectWebGpu().then(setWebGpuAvailable);
    void navigator.storage?.persist?.();
    const handlePrompt = (event: Event) => {
      event.preventDefault();
      setInstallPrompt(event as InstallPromptEvent);
    };
    window.addEventListener("beforeinstallprompt", handlePrompt);
    return () => window.removeEventListener("beforeinstallprompt", handlePrompt);
  }, []);

  useEffect(() => {
    if (phase !== "recording") return;
    const timer = window.setInterval(() => {
      setElapsed((current) => {
        const next = Math.min(MAX_RECORDING_SECONDS, current + 0.25);
        if (next >= MAX_RECORDING_SECONDS) void finishRecording();
        return next;
      });
    }, 250);
    return () => window.clearInterval(timer);
  }, [phase]);

  useEffect(() => () => {
    void recorderRef.current?.cancel();
    transcriberRef.current?.dispose();
  }, []);

  const duration = audio ? audio.length / TARGET_SAMPLE_RATE : elapsed;
  const words = useMemo(() => transcript.trim() ? transcript.trim().split(/\s+/).length : 0, [transcript]);
  const isTranscribing = phase === "transcribing";
  const isPreparing = phase === "requesting" || phase === "decoding";
  const hasRecording = Boolean(audio);
  const canStartRecording = phase === "idle" || phase === "ready" || phase === "complete" || phase === "error";

  async function beginRecording(): Promise<void> {
    if (!canStartRecording) return;
    setError("");
    setPhase("requesting");
    setStatus("Opening the microphone…");
    setElapsed(0);
    setProgress(0);
    const recorder = new VoiceRecorder((level) => {
      setWaveform((current) => [...current.slice(1), Math.max(0.08, level)]);
    }, noiseGuard);
    recorderRef.current = recorder;
    try {
      await recorder.start();
      setPhase("recording");
      setStatus(noiseGuard ? "Recording · noise guard active" : "Recording");
    } catch (reason) {
      recorderRef.current = null;
      const message = reason instanceof Error ? reason.message : "The microphone could not be opened.";
      setError(message);
      setStatus("Microphone unavailable");
      setPhase("error");
    }
  }

  async function finishRecording(): Promise<void> {
    const recorder = recorderRef.current;
    if (!recorder) return;
    recorderRef.current = null;
    setPhase("decoding");
    setStatus("Preparing your recording…");
    try {
      const recording = await recorder.stop();
      if (recording.length < TARGET_SAMPLE_RATE * 0.35) throw new Error("That recording was too short. Hold closer and try for at least a second.");
      setAudio(recording);
      setAnalysis(analyzeSignal(recording));
      setSourceName("Microphone recording");
      setTranscript("");
      setWaveform(Array.from({ length: 24 }, (_, index) => 0.08 + (index % 4) * 0.025));
      setPhase("ready");
      setStatus("Recording ready");
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "The recording could not be prepared.";
      setError(message);
      setStatus("Try again");
      setPhase("error");
    }
  }

  function toggleRecording(): void {
    if (phase === "recording") void finishRecording();
    else if (canStartRecording) void beginRecording();
  }

  async function handleFile(file: File | undefined): Promise<void> {
    if (!file) return;
    setError("");
    if (file.size > 25 * 1024 * 1024) {
      setError("For this low-memory mode, choose an audio file smaller than 25 MB.");
      setPhase("error");
      return;
    }
    setPhase("decoding");
    setStatus("Reading audio on this device…");
    setProgress(8);
    try {
      const decoded = await decodeAudioFile(file);
      if (decoded.length < TARGET_SAMPLE_RATE * 0.35) throw new Error("That audio file is too short to transcribe.");
      setAudio(decoded);
      setAnalysis(analyzeSignal(decoded));
      setElapsed(decoded.length / TARGET_SAMPLE_RATE);
      setSourceName(file.name);
      setTranscript("");
      setProgress(0);
      setPhase("ready");
      setStatus("Audio ready");
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "This audio format could not be read on the device.";
      setError(message);
      setStatus("Try a different recording");
      setPhase("error");
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function makeTranscript(): Promise<void> {
    if (!audio || isTranscribing) return;
    setError("");
    setPhase("transcribing");
    setProgress(1);
    setStatus("Preparing the private speech model…");
    setEngineMode(null);
    const client = new LocalTranscriber({
      onStatus: setStatus,
      onProgress: setProgress,
      onMode: setEngineMode,
    });
    transcriberRef.current = client;
    try {
      const text = await client.transcribe(audio, webGpuAvailable === true, modelMode);
      const cleanText = text || "No speech was clear enough to transcribe. Try moving closer to the speaker.";
      setTranscript(cleanText);
      setProgress(100);
      setStatus("Transcript ready");
      setPhase("complete");
      try {
        window.localStorage.setItem("sotto:last-transcript", cleanText);
      } catch {
        // The transcript remains available even when private browsing blocks storage.
      }
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "The local model could not finish this recording.";
      setError(message);
      setStatus("Transcription stopped");
      setPhase("error");
    } finally {
      transcriberRef.current = null;
    }
  }

  function resetNote(): void {
    void recorderRef.current?.cancel();
    recorderRef.current = null;
    transcriberRef.current?.dispose();
    transcriberRef.current = null;
    setAudio(null);
    setAnalysis(null);
    setTranscript("");
    setElapsed(0);
    setProgress(0);
    setError("");
    setSourceName("");
    setEngineMode(null);
    setWaveform(Array.from({ length: 24 }, (_, index) => 0.08 + (index % 4) * 0.025));
    setStatus("Ready when you are");
    setPhase("idle");
  }

  async function copyTranscript(): Promise<void> {
    if (!transcript) return;
    try {
      await navigator.clipboard.writeText(transcript);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1_800);
    } catch {
      setError("Copy is unavailable in this browser. Select the transcript text to copy it.");
    }
  }

  async function shareTranscript(): Promise<void> {
    if (!transcript) return;
    if (navigator.share) {
      try {
        await navigator.share({ title: "Sotto voice note", text: transcript });
      } catch {
        // Closing the native share sheet is not an app error.
      }
    } else {
      await copyTranscript();
    }
  }

  function downloadTranscript(): void {
    if (!transcript) return;
    const blob = new Blob([transcript], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `sotto-${new Date().toISOString().slice(0, 10)}.txt`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function installApp(): Promise<void> {
    if (!installPrompt) return;
    await installPrompt.prompt();
    await installPrompt.userChoice;
    setInstallPrompt(null);
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Sotto home">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /><i /><i /></span>
          <span>Sotto</span>
        </a>
        <div className="topbar-actions">
          <span className="privacy-chip"><Icon name="lock" size={14} /> On-device</span>
          {installPrompt && <button className="install-link" onClick={() => void installApp()}><Icon name="install" size={15} /> Install</button>}
        </div>
      </header>

      <main id="top">
        <section className="hero" aria-labelledby="page-title">
          <div className="eyebrow"><span>Local voice memo</span><span className="eyebrow-dot" /><span>No account needed</span></div>
          <h1 id="page-title">Voice, without<br /><em>the static.</em></h1>
          <p>Turn field notes, lectures, and loud-room moments into clean text. Your audio stays on your phone.</p>
          <div className="hero-proof" aria-label="Product qualities">
            <span><Icon name="check" size={16} /> Works offline after setup</span>
            <span><Icon name="check" size={16} /> Built for low memory</span>
          </div>
        </section>

        <section className={`recorder-card phase-${phase}`} aria-labelledby="recorder-title">
          <div className="card-heading">
            <div>
              <p className="section-kicker">Your private recorder</p>
              <h2 id="recorder-title">{isTranscribing ? "Listening closely…" : hasRecording ? "Your recording is ready" : "Say what matters."}</h2>
            </div>
            <div className="time-display" aria-label={`${formatDuration(duration)} recorded`}>
              <span className={phase === "recording" ? "live-dot" : ""} />
              {formatDuration(duration)}
            </div>
          </div>

          <div className="waveform" aria-hidden="true">
            {waveform.map((level, index) => (
              <i key={index} style={{ height: `${Math.round((0.12 + level * 0.88) * 100)}%` }} />
            ))}
          </div>

          <div className="record-control">
            <button
              className={`record-button ${phase === "recording" ? "is-recording" : ""}`}
              onClick={toggleRecording}
              disabled={isPreparing || isTranscribing}
              aria-label={phase === "recording" ? "Stop recording" : hasRecording ? "Record again" : "Start recording"}
            >
              <span className="record-button-inner"><Icon name={phase === "recording" ? "stop" : "mic"} size={30} /></span>
            </button>
            <p>{phase === "recording" ? "Tap to finish" : isPreparing ? "One moment…" : hasRecording ? "Tap to record again" : "Tap to record"}</p>
          </div>

          <div className="recorder-status" aria-live="polite">
            <span className={`status-beacon ${phase === "recording" || isTranscribing ? "active" : ""}`} />
            <span>{status}</span>
            {phase === "recording" && <span className="limit-note">max {MAX_RECORDING_SECONDS / 60} min</span>}
          </div>

          <div className="capture-tools">
            <button
              className="noise-toggle"
              role="switch"
              aria-checked={noiseGuard}
              onClick={() => setNoiseGuard((value) => !value)}
              disabled={phase === "recording" || isPreparing || isTranscribing}
            >
              <span className="toggle-track"><span /></span>
              <span><strong>Noise guard</strong><small>Phone mic cleanup + gain</small></span>
            </button>
            <button className="upload-button" onClick={() => fileInputRef.current?.click()} disabled={phase === "recording" || isPreparing || isTranscribing}>
              <Icon name="upload" size={18} /> Import audio
            </button>
            <input ref={fileInputRef} type="file" accept="audio/*,.m4a,.mp3,.wav,.webm,.ogg" onChange={(event) => void handleFile(event.target.files?.[0])} hidden />
          </div>

          <label className="model-picker" htmlFor="model-mode">
            <span><strong>Speech model</strong><small>{modelMode === "fast" ? "Fastest download · tuned for this phone" : "Higher accuracy · larger download"}</small></span>
            <select id="model-mode" value={modelMode} onChange={(event) => setModelMode(event.target.value as ModelMode)} disabled={phase === "recording" || isPreparing || isTranscribing}>
              <option value="fast">Fast · ~28 MB</option>
              <option value="accurate">Accuracy · ~41 MB</option>
            </select>
          </label>

          {sourceName && phase !== "transcribing" && (
            <div className="source-row">
              <span><Icon name="mic" size={15} /> {sourceName}</span>
              <button onClick={resetNote} aria-label="Discard recording"><Icon name="trash" size={16} /> Discard</button>
            </div>
          )}

          {analysis && phase === "ready" && (
            <div className={`signal-card signal-${analysis.quality}`}>
              <span className="signal-icon"><Icon name="spark" size={18} /></span>
              <span><strong>{qualityCopy[analysis.quality].label}</strong><small>{qualityCopy[analysis.quality].detail}</small></span>
              <span className="signal-bars" aria-hidden="true"><i /><i /><i /><i /></span>
            </div>
          )}

          {phase === "ready" && (
            <button className="primary-action" onClick={() => void makeTranscript()}>
              <Icon name="spark" size={19} /> Make transcript <span>→</span>
            </button>
          )}

          {isTranscribing && (
            <div className="progress-card" role="status" aria-live="polite">
              <div className="progress-copy">
                <span>{status}</span>
                <strong>{progress < 100 ? `${progress}%` : "Working…"}</strong>
              </div>
              <div className="progress-track"><span style={{ width: `${Math.max(4, progress)}%` }} /></div>
              <div className="engine-row">
                <span>{engineMode ? `${engineMode} engine` : "Choosing the best engine"} · {modelMode === "fast" ? "Fast 28 MB" : "Accuracy 41 MB"}</span>
                <span>{modelMode === "fast" ? "Compact engine" : webGpuAvailable ? "WebGPU ready" : "Low-memory mode"}</span>
              </div>
              <p>First use downloads the speech model once, then the browser keeps it for offline reuse.</p>
              <button className="text-button" onClick={resetNote}>Cancel</button>
            </div>
          )}

          {error && phase === "error" && (
            <div className="error-card" role="alert">
              <strong>That didn’t go through.</strong>
              <p>{error}</p>
              <button onClick={() => audio ? void makeTranscript() : resetNote()}>{audio ? "Try the download again" : "Start fresh"}</button>
            </div>
          )}

          {transcript && (phase === "complete" || phase === "idle" || phase === "error") && (
            <div className="transcript-section">
              <div className="transcript-heading">
                <div>
                  <p className="section-kicker">Transcript</p>
                  <h3>Your words, cleaned up.</h3>
                </div>
                <span>{words} {words === 1 ? "word" : "words"}</span>
              </div>
              <label className="sr-only" htmlFor="transcript">Editable transcript</label>
              <textarea id="transcript" value={transcript} onChange={(event) => setTranscript(event.target.value)} rows={9} />
              <div className="transcript-meta">
                <span><Icon name="lock" size={14} /> Stored only on this device</span>
                {engineMode && <span>{engineMode} · local</span>}
              </div>
              <div className="export-actions">
                <button className="export-primary" onClick={() => void copyTranscript()}><Icon name={copied ? "check" : "copy"} size={17} /> {copied ? "Copied" : "Copy"}</button>
                <button onClick={() => void shareTranscript()}><Icon name="share" size={17} /> Share</button>
                <button onClick={downloadTranscript}><Icon name="download" size={17} /> .txt</button>
              </div>
            </div>
          )}
        </section>

        <section className="privacy-strip" aria-label="Privacy and performance details">
          <div><span>01</span><strong>Private by design</strong><p>No recording is uploaded. There is no account or tracking.</p></div>
          <div><span>02</span><strong>Made for old phones</strong><p>Small model, mono audio, and a lightweight fallback engine.</p></div>
          <div><span>03</span><strong>Ready offline</strong><p>Install Sotto and reuse the cached model without signal.</p></div>
        </section>

        <section className="field-section" aria-labelledby="field-title">
          <div className="section-intro">
            <p className="section-kicker">Built for the real world</p>
            <h2 id="field-title">From pocket recording<br />to useful text.</h2>
          </div>
          <div className="field-grid">
            <article><span>01 / FIELD NOTES</span><h3>Capture the thought before it moves on.</h3><p>Speak naturally, even with traffic, wind, or room echo around you.</p></article>
            <article><span>02 / LECTURES</span><h3>Keep the useful parts. Lose the filler.</h3><p>Import a longer class recording and turn it into an editable reference.</p></article>
            <article><span>03 / CHECK-INS</span><h3>Send a clear update without typing.</h3><p>Record a quick walkaround, clean it up, then copy or share.</p></article>
          </div>
        </section>

        <section className="how-section">
          <div className="how-card">
            <p className="section-kicker">Three simple moves</p>
            <h2>From sound to words.<br />Without the cloud.</h2>
            <ol>
              <li><span>1</span><div><strong>Record or import</strong><p>Use the cleaned microphone or choose an existing audio file.</p></div></li>
              <li><span>2</span><div><strong>Transcribe locally</strong><p>A compact Whisper model listens entirely inside your browser.</p></div></li>
              <li><span>3</span><div><strong>Edit and take it with you</strong><p>Correct a word, then copy, share, or save a plain-text note.</p></div></li>
            </ol>
          </div>
          <aside className="tip-card">
            <span className="tip-label"><Icon name="spark" size={15} /> Noisy room?</span>
            <h3>Get closer to the words.</h3>
            <p>Put the phone 20–30 cm from the speaker, cover it with a scarf, or record one voice at a time. Cleanup helps, but distance still wins.</p>
            <details>
              <summary>Why Tiny?</summary>
              <p>Fast uses a compact Moonshine model tuned for quick first-run and low memory. Accuracy uses Whisper Tiny English for a slower download and more forgiving transcripts. Sotto also applies phone noise suppression, conservative gain, and clean 16 kHz mono audio.</p>
            </details>
          </aside>
        </section>
      </main>

      <footer>
        <a className="brand footer-brand" href="#top"><span className="brand-mark" aria-hidden="true"><i /><i /><i /><i /><i /></span><span>Sotto</span></a>
        <p>Your voice stays yours.</p>
        <span>Local MVP · v0.1</span>
      </footer>
    </div>
  );
}
