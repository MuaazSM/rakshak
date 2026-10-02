// localStorage wrapped in try/catch with an in-memory fallback (private windows, blocked storage).
const mem = new Map<string, string>();

function read(key: string): string | null {
  try {
    const v = window.localStorage.getItem(key);
    if (v !== null) return v;
  } catch {
    /* storage unavailable */
  }
  return mem.get(key) ?? null;
}

function write(key: string, value: string): void {
  mem.set(key, value);
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* storage unavailable */
  }
}

export type Lang = "en" | "hi";

const PARENT = "rk.parent";
const LANG = "rk.lang";
const RECENT = "rk.recent";
const INSTALL_SEEN = "rk.installSeen";

/** FR-7: `?parent=mom` on first open is remembered; every request carries it. */
export function captureParentFromUrl(): void {
  try {
    const p = new URLSearchParams(window.location.search).get("parent");
    if (p && /^[a-z][a-z0-9_]*$/.test(p)) write(PARENT, p);
  } catch {
    /* ignore */
  }
}

/** Falls back to "mom": this PWA is the Android app and Mom is its only user. */
export function getParentId(): string {
  return read(PARENT) ?? "mom";
}

export function getLang(): Lang {
  return read(LANG) === "hi" ? "hi" : "en";
}

export function setLang(lang: Lang): void {
  write(LANG, lang);
}

export function installSeen(): boolean {
  return read(INSTALL_SEEN) === "1";
}

export function markInstallSeen(): void {
  write(INSTALL_SEEN, "1");
}

/** Recently checked: verdict word + time only, never message text. */
export interface RecentItem {
  id: string;
  verdict: "SCAM" | "SUSPICIOUS" | "SAFE" | "UNKNOWN";
  ts: number;
}

export function getRecent(): RecentItem[] {
  try {
    const raw = read(RECENT);
    const list = raw ? (JSON.parse(raw) as RecentItem[]) : [];
    return Array.isArray(list) ? list.filter((r) => r && typeof r.id === "string") : [];
  } catch {
    return [];
  }
}

export function addRecent(item: RecentItem): void {
  const list = getRecent().filter((r) => r.id !== item.id);
  list.unshift(item);
  write(RECENT, JSON.stringify(list.slice(0, 5)));
}
