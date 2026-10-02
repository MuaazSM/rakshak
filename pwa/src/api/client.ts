// Same-origin API paths per PRD §5.3. Metadata only is ever stored client-side.
import { getLang, getParentId } from "../lib/storage";
import { FIXTURES, fixtureById, fixtureForText } from "./fixtures";
import type { Verdict } from "./types";

export const MOCK = import.meta.env.VITE_MOCK === "1";
const CHECK_TIMEOUT_MS = 20_000; // B5: after 20 s with no answer -> B9

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function request(input: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(input, { ...init, signal: AbortSignal.timeout(timeoutMs) });
  } catch {
    throw new ApiError(0, "network");
  }
  if (!res.ok) throw new ApiError(res.status, `HTTP ${res.status}`);
  return res;
}

export async function checkText(text: string): Promise<Verdict> {
  if (MOCK) {
    await sleep(2800);
    return fixtureForText(text);
  }
  const body = {
    parent_id: getParentId(),
    text,
    channel: "sms", // PRD §8 channels have no "pasted"; sms is the nearest
    lang: getLang(),
  };
  const res = await request(
    "/api/check",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
    CHECK_TIMEOUT_MS,
  );
  return (await res.json()) as Verdict;
}

export async function checkImage(file: File): Promise<Verdict> {
  if (MOCK) {
    await sleep(3200);
    return FIXTURES.SCAM;
  }
  const form = new FormData();
  form.set("parent_id", getParentId());
  form.set("lang", getLang());
  form.set("image", file, file.name || "screenshot");
  const res = await request("/api/share", { method: "POST", body: form }, CHECK_TIMEOUT_MS);
  return (await res.json()) as Verdict;
}

export async function sendVoice(blob: Blob): Promise<Verdict> {
  if (MOCK) {
    await sleep(3200);
    return FIXTURES.SUSPICIOUS;
  }
  const form = new FormData();
  form.set("parent_id", getParentId());
  form.set("lang", getLang());
  form.set("audio", blob, blob.type.includes("webm") ? "voice.webm" : "voice");
  const res = await request("/voice", { method: "POST", body: form }, CHECK_TIMEOUT_MS);
  return (await res.json()) as Verdict;
}

export async function getVerdict(id: string): Promise<Verdict> {
  if (MOCK) {
    await sleep(400);
    const v = fixtureById(id);
    if (!v) throw new ApiError(404, "not found");
    return v;
  }
  const res = await request(`/api/verdict/${encodeURIComponent(id)}`, {}, 10_000);
  return (await res.json()) as Verdict;
}

/** FR-9: persist the explanation language on the parent's profile (204). */
export async function patchLanguage(language: "en" | "hi"): Promise<void> {
  if (MOCK) {
    await sleep(150);
    return;
  }
  await request(
    `/api/parents/${encodeURIComponent(getParentId())}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ language }),
    },
    10_000,
  );
}

/** Local stand-in shown when the hub can't be reached (B9). Never a guess. */
export function offlineVerdict(): Verdict {
  return { ...FIXTURES.UNKNOWN, event_id: "offline", explanation: "", parent_id: getParentId() };
}
