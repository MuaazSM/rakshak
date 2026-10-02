import { useCallback, useEffect, useRef, useState } from "react";
import { sendVoice } from "../api/client";
import { Icon } from "../components/Icon";
import { TopBar } from "../components/TopBar";
import { t } from "../i18n";
import { MAX_RECORD_MS, recordingSupported, startRecording, type Recording } from "../lib/recorder";
import { useSubmit } from "../lib/useSubmit";
import { Checking } from "./Checking";

const BARS = 40;
const CANCEL_PX = 80; // slide left past this to cancel
const MIN_MS = 700;

type Phase = "idle" | "recording" | "cancelled";

const fmt = (ms: number) => {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

/** B10 Describe a call: hold to talk, slide left to cancel, max 60 s, then B5. */
export function Talk() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [elapsed, setElapsed] = useState(0);
  const [levels, setLevels] = useState<number[]>(() => Array(BARS).fill(0.06));
  const [note, setNote] = useState<string | null>(null);

  const rec = useRef<Recording | null>(null);
  const startX = useRef(0);
  const raf = useRef(0);
  const phaseRef = useRef<Phase>("idle");
  const busy = useRef(false); // getUserMedia in flight
  const wantStop = useRef(false);
  const lastElapsed = useRef(0);

  const submit = useSubmit(sendVoice);

  const setP = (p: Phase) => {
    phaseRef.current = p;
    setPhase(p);
  };

  const stopLoop = () => cancelAnimationFrame(raf.current);

  const loop = useCallback(() => {
    const r = rec.current;
    if (!r) return;
    const ms = Date.now() - r.startedAt;
    lastElapsed.current = ms;
    setElapsed(ms);
    setLevels((prev) => [...prev.slice(1), Math.max(0.06, r.level())]);
    if (ms >= MAX_RECORD_MS) {
      void finish();
      return;
    }
    raf.current = requestAnimationFrame(loop);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function finish() {
    const r = rec.current;
    if (!r) return;
    rec.current = null;
    stopLoop();
    const ms = Date.now() - r.startedAt;
    const blob = await r.stop();
    setP("idle");
    if (ms < MIN_MS) {
      setNote(t("B10.tooShort"));
      return;
    }
    submit.mutate(blob);
  }

  function cancel() {
    const r = rec.current;
    rec.current = null;
    stopLoop();
    r?.cancel();
    setP("cancelled");
    navigator.vibrate?.(15);
  }

  async function begin(e: React.PointerEvent | React.KeyboardEvent) {
    if (phaseRef.current === "recording" || busy.current) return;
    if (!recordingSupported()) {
      setNote(t("B10.micMissing"));
      return;
    }
    if ("pointerId" in e) {
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      startX.current = e.clientX;
    }
    setNote(null);
    wantStop.current = false;
    busy.current = true;
    try {
      const r = await startRecording();
      busy.current = false;
      if (wantStop.current) {
        r.cancel(); // released before the mic opened
        setP("idle");
        return;
      }
      rec.current = r;
      setElapsed(0);
      setLevels(Array(BARS).fill(0.06));
      setP("recording");
      raf.current = requestAnimationFrame(loop);
    } catch {
      busy.current = false;
      setNote(t("B10.micDenied"));
    }
  }

  function move(e: React.PointerEvent) {
    if (phaseRef.current !== "recording") return;
    if (e.clientX - startX.current < -CANCEL_PX) cancel();
  }

  function release() {
    wantStop.current = true;
    if (phaseRef.current === "recording") void finish();
    else if (phaseRef.current === "cancelled") {
      /* stays cancelled until the next hold */
    }
  }

  useEffect(
    () => () => {
      stopLoop();
      rec.current?.cancel();
      rec.current = null;
    },
    [],
  );

  if (submit.isPending) return <Checking />;

  const recording = phase === "recording";
  const cancelled = phase === "cancelled";
  const hold = {
    onPointerDown: begin,
    onPointerMove: move,
    onPointerUp: release,
    onPointerCancel: release,
    onContextMenu: (e: React.MouseEvent) => e.preventDefault(),
    onKeyDown: (e: React.KeyboardEvent) => {
      if ((e.key === " " || e.key === "Enter") && !e.repeat) {
        e.preventDefault();
        void begin(e);
      }
    },
    onKeyUp: (e: React.KeyboardEvent) => {
      if (e.key === " " || e.key === "Enter") release();
    },
  };

  return (
    <main className="screen">
      <TopBar title={t("B10.title")} />

      <div className="talk-body">
        {recording ? (
          <>
            <h2 className="talk-big">{t("B10.listening")}</h2>
            <div className="wave" aria-hidden="true">
              {levels.map((l, i) => (
                <span key={i} style={{ height: `${Math.round(14 + l * 82)}px` }} className="wave-bar" />
              ))}
            </div>
            <p className="talk-hint">{t("B10.releaseHint")}</p>
          </>
        ) : cancelled ? (
          <>
            <h2 className="talk-big">{t("B10.cancelled")}</h2>
            <p className="talk-sub">{t("B10.cancelledBody")}</p>
          </>
        ) : (
          <>
            <h2 className="talk-big">{t("B10.prompt")}</h2>
            {note && (
              <p className="talk-note" role="alert">
                {note}
              </p>
            )}
          </>
        )}
      </div>

      <div className={`talk-dock ${recording ? "is-rec" : ""}`}>
        <div className="talk-pill">
          {recording ? (
            <>
              <span className="rec-dot" aria-hidden="true" />
              <span className="talk-time">{fmt(elapsed)}</span>
              <span className="talk-slide">{t("B10.slideCancel")}</span>
            </>
          ) : cancelled ? (
            <>
              <Icon name="trash-2" size={22} />
              <span className="talk-cancelled">{t("B10.cancelled")}</span>
              <span className="talk-struck">{fmt(lastElapsed.current)}</span>
            </>
          ) : (
            <span className="talk-idle">{t("B10.holdToTalk")}</span>
          )}
        </div>
        <button
          className="mic"
          aria-label={t("B10.holdToTalk")}
          aria-pressed={recording}
          {...hold}
        >
          <Icon name="mic" color="white" size={recording ? 34 : 28} />
        </button>
      </div>
    </main>
  );
}
