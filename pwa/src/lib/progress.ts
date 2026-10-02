import { useEffect, useState } from "react";

/** Which of Reading / Checking / Explaining is active. Timed: the hub reports no stage events. */
export function useStage(): 0 | 1 | 2 {
  const [stage, setStage] = useState<0 | 1 | 2>(0);
  useEffect(() => {
    const a = window.setTimeout(() => setStage(1), 1500);
    const b = window.setTimeout(() => setStage(2), 5500);
    return () => {
      window.clearTimeout(a);
      window.clearTimeout(b);
    };
  }, []);
  return stage;
}
