import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { offlineVerdict } from "../api/client";
import type { Verdict } from "../api/types";
import { addRecent } from "./storage";

/** Run a check; show B5 while pending, then open /v/:id. Any failure -> B9 (never a guess). */
export function useSubmit<A>(fn: (arg: A) => Promise<Verdict>) {
  const qc = useQueryClient();
  const nav = useNavigate();
  return useMutation({
    mutationFn: fn,
    onSuccess: (v) => {
      qc.setQueryData(["verdict", v.event_id], v);
      addRecent({ id: v.event_id, verdict: v.verdict, ts: Date.now() });
      nav(`/v/${encodeURIComponent(v.event_id)}`, { replace: true });
    },
    onError: () => {
      qc.setQueryData(["verdict", "offline"], offlineVerdict());
      nav("/v/offline", { replace: true });
    },
  });
}
