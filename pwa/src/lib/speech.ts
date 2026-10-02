// FR-5: on-device TTS via speechSynthesis. en-IN by default; hi-IN only when the explanation is Hindi.
// No matching voice -> text only. iOS Safari blocks auto-play, so callers detect "never started".
import { useCallback, useEffect, useRef, useState } from "react";

const norm = (l: string) => l.replace("_", "-").toLowerCase();

function hasSpeech(): boolean {
  return typeof window !== "undefined" && "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
}

function findVoice(lang: string): SpeechSynthesisVoice | null {
  const want = norm(lang);
  return window.speechSynthesis.getVoices().find((v) => norm(v.lang) === want) ?? null;
}

export type SpeechState =
  | "checking" // waiting for the voice list
  | "none" // no usable voice -> text only
  | "idle"
  | "speaking"
  | "blocked"; // auto-play was refused (iOS Safari)

export function useSpeech(text: string, explanationLang: "en" | "hi") {
  const ttsLang = explanationLang === "hi" ? "hi-IN" : "en-IN";
  const [state, setState] = useState<SpeechState>(hasSpeech() ? "checking" : "none");
  const started = useRef(false);

  const speak = useCallback(
    (auto: boolean) => {
      if (!hasSpeech() || !text) return;
      const synth = window.speechSynthesis;
      synth.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = ttsLang;
      const voice = findVoice(ttsLang);
      if (voice) u.voice = voice;
      u.rate = 0.95;
      let began = false;
      u.onstart = () => {
        began = true;
        started.current = true;
        setState("speaking");
      };
      u.onend = () => setState("idle");
      u.onerror = () => setState(began ? "idle" : auto ? "blocked" : "idle");
      synth.speak(u);
      if (auto) {
        // Safari may silently ignore speak() without a gesture.
        window.setTimeout(() => {
          if (!began) {
            synth.cancel();
            setState("blocked");
          }
        }, 1800);
      }
    },
    [text, ttsLang],
  );

  useEffect(() => {
    if (!hasSpeech() || !text) {
      setState("none");
      return;
    }
    const synth = window.speechSynthesis;
    let cancelled = false;
    let autoDone = false;

    const begin = () => {
      if (cancelled || autoDone) return;
      if (!findVoice(ttsLang)) return;
      autoDone = true;
      setState("idle");
      speak(true);
    };

    begin();
    synth.addEventListener?.("voiceschanged", begin);
    // Voices can take a moment on Android; give up on the voice after 2.5 s.
    const giveUp = window.setTimeout(() => {
      if (!cancelled && !autoDone) setState("none");
    }, 2500);

    return () => {
      cancelled = true;
      window.clearTimeout(giveUp);
      synth.removeEventListener?.("voiceschanged", begin);
      synth.cancel();
    };
  }, [text, ttsLang, speak]);

  return { state, listenAgain: () => speak(false) };
}
