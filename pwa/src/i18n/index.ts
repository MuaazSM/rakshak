import en from "./en.json";
import hi from "./hi.json";
import { getLang } from "../lib/storage";

type Dict = { [key: string]: string | Dict };

function lookup(dict: Dict, path: string): string | undefined {
  let cur: string | Dict | undefined = dict;
  for (const part of path.split(".")) {
    if (cur === undefined || typeof cur === "string") return undefined;
    cur = cur[part];
  }
  return typeof cur === "string" ? cur : undefined;
}

/** Name used in "Call {son}" copy. Configure via VITE_SON_NAME. */
export const SON_NAME: string = import.meta.env.VITE_SON_NAME?.trim() || "Muaaz";

/**
 * Look up UI copy by dotted key (e.g. "B6.headline"). English is the default;
 * hi.json only wins when the setting is Hindi and the entry is translated
 * (entries still marked "TODO" fall back to English).
 */
export function t(key: string, vars: Record<string, string | number> = {}): string {
  let s: string | undefined;
  if (getLang() === "hi") {
    const h = lookup(hi as Dict, key);
    if (h && h !== "TODO") s = h;
  }
  s ??= lookup(en as Dict, key) ?? key;
  const all: Record<string, string | number> = { son: SON_NAME, ...vars };
  return s.replace(/\{(\w+)\}/g, (_, k: string) => String(all[k] ?? ""));
}

export function hasKey(key: string): boolean {
  return lookup(en as Dict, key) !== undefined;
}
