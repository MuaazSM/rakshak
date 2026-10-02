// PRD §8.1 Verdict (API response).
export type VerdictLabel = "SCAM" | "SUSPICIOUS" | "SAFE" | "UNKNOWN";

export interface RedFlag {
  quote: string;
  reason: string;
  source: "model" | "rule";
}

export interface Verdict {
  event_id: string;
  verdict: VerdictLabel;
  category: string | null;
  p_scam: number | null;
  red_flags: RedFlag[];
  explanation: string;
  language: "en" | "hi";
  parent_id: string;
  timings_ms?: Record<string, number>;
  model_versions?: Record<string, string>;
}
