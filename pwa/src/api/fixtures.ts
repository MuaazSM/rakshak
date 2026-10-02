import type { Verdict, VerdictLabel } from "./types";

// Mock-mode fixtures (VITE_MOCK=1). Copy follows docs/design frames B6-B9.
const base = { category: null, p_scam: null, language: "en" as const, parent_id: "mom" };

export const FIXTURES: Record<VerdictLabel, Verdict> = {
  SCAM: {
    ...base,
    event_id: "mock-scam",
    verdict: "SCAM",
    category: "kyc_account_block",
    p_scam: 0.97,
    red_flags: [
      { quote: "sbi-yono-kyc.in", reason: "lookalike_link", source: "rule" },
      { quote: "blocked today", reason: "threat_or_arrest", source: "model" },
    ],
    explanation:
      "Real banks never ask you to update KYC through a link. Don't tap the link and don't send any money. Call Muaaz.",
  },
  SUSPICIOUS: {
    ...base,
    event_id: "mock-careful",
    verdict: "SUSPICIOUS",
    p_scam: 0.55,
    red_flags: [
      { quote: "+91 70••• ••412", reason: "unregistered_sender", source: "rule" },
      { quote: "reply urgently", reason: "urgency_deadline", source: "model" },
    ],
    explanation:
      "This is from an unknown number and is rushing you to reply. Check with Muaaz before you answer.",
  },
  SAFE: {
    ...base,
    event_id: "mock-safe",
    verdict: "SAFE",
    category: "genuine_otp",
    p_scam: 0.02,
    red_flags: [],
    explanation: "This is a real OTP from your bank. Never share an OTP with anyone.",
  },
  UNKNOWN: {
    ...base,
    event_id: "mock-unknown",
    verdict: "UNKNOWN",
    red_flags: [],
    explanation: "The home computer is offline. Don't open any links, and call Muaaz.",
  },
};

const BY_ID: Record<string, Verdict> = Object.fromEntries(
  Object.values(FIXTURES).map((v) => [v.event_id, v]),
);

export function fixtureById(id: string): Verdict | undefined {
  return BY_ID[id];
}

/** Cheap keyword stand-in for the hub so the Check screen can be exercised offline. */
export function fixtureForText(text: string): Verdict {
  const s = text.toLowerCase();
  if (/\botp\b/.test(s) && /(do not share|never share)/.test(s)) return FIXTURES.SAFE;
  if (/(kyc|blocked|arrest|\.apk)/.test(s)) return FIXTURES.SCAM;
  if (/(unknown|urgent)/.test(s)) return FIXTURES.SUSPICIOUS;
  return FIXTURES.SAFE;
}
