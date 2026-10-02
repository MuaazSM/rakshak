// FR-3: MediaRecorder (webm/opus), max 60 s. Live level samples drive the waveform.
export const MAX_RECORD_MS = 60_000;

export interface Recording {
  stop: () => Promise<Blob>;
  cancel: () => void;
  level: () => number; // 0..1, instantaneous
  startedAt: number;
}

export function recordingSupported(): boolean {
  return typeof MediaRecorder !== "undefined" && !!navigator.mediaDevices?.getUserMedia;
}

function pickMime(): string | undefined {
  for (const m of ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"]) {
    if (MediaRecorder.isTypeSupported(m)) return m;
  }
  return undefined;
}

export async function startRecording(): Promise<Recording> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const mime = pickMime();
  const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  const chunks: Blob[] = [];
  rec.ondataavailable = (e) => {
    if (e.data.size) chunks.push(e.data);
  };

  let analyser: AnalyserNode | null = null;
  let ctx: AudioContext | null = null;
  const buf = new Uint8Array(256);
  try {
    ctx = new AudioContext();
    const src = ctx.createMediaStreamSource(stream);
    analyser = ctx.createAnalyser();
    analyser.fftSize = 256;
    src.connect(analyser);
  } catch {
    analyser = null;
  }

  const release = () => {
    stream.getTracks().forEach((t) => t.stop());
    void ctx?.close().catch(() => undefined);
  };

  rec.start(250);
  return {
    startedAt: Date.now(),
    level: () => {
      if (!analyser) return 0;
      analyser.getByteTimeDomainData(buf);
      let peak = 0;
      for (const b of buf) peak = Math.max(peak, Math.abs(b - 128));
      return Math.min(1, peak / 100);
    },
    stop: () =>
      new Promise<Blob>((resolve) => {
        rec.onstop = () => {
          release();
          resolve(new Blob(chunks, { type: rec.mimeType || "audio/webm" }));
        };
        if (rec.state !== "inactive") rec.stop();
        else rec.onstop?.(new Event("stop"));
      }),
    cancel: () => {
      rec.ondataavailable = null;
      rec.onstop = null;
      if (rec.state !== "inactive") rec.stop();
      release();
    },
  };
}
