import { useEffect, useMemo } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ApiError, getVerdict, offlineVerdict } from "../api/client";
import type { RedFlag, Verdict as VerdictT, VerdictLabel } from "../api/types";
import { Button, LinkButton } from "../components/Button";
import { Icon, type IconColor } from "../components/Icon";
import { TopBar } from "../components/TopBar";
import { hasKey, t } from "../i18n";
import { isIOS } from "../lib/install";
import { useSpeech } from "../lib/speech";
import { addRecent, getRecent } from "../lib/storage";
import { Checking } from "./Checking";

interface Look {
  key: "scam" | "careful" | "safe" | "unknown";
  icon: string;
  color: IconColor; // icon-set color name
}

const LOOK: Record<VerdictLabel, Look> = {
  SCAM: { key: "scam", icon: "hand", color: "scam" },
  SUSPICIOUS: { key: "careful", icon: "triangle-alert", color: "careful" },
  SAFE: { key: "safe", icon: "check", color: "normal" },
  UNKNOWN: { key: "unknown", icon: "phone", color: "unknown" },
};

const LINK_REASONS = new Set(["lookalike_link", "lookalike_domain", "apk_link", "shortener_link"]);

const SON_PHONE = (import.meta.env.VITE_SON_PHONE ?? "").replace(/[^\d+]/g, "");

/** Links and numbers are shown raw; phrases get typographic quotes (design A5). */
function formatQuote(f: RedFlag): string {
  const q = f.quote.trim();
  const rawLike = !/\s/.test(q) && /[.\d@]/.test(q);
  return rawLike || LINK_REASONS.has(f.reason) ? q : `‘${q}’`;
}

function flagLabel(reason: string): string {
  const key = `flags.${reason}`;
  return hasKey(key) ? t(key) : t("verdict.flagFallback");
}

function Chips({ flags, look }: { flags: RedFlag[]; look: Look }) {
  const seen = new Set<string>();
  const list = flags.filter((f) => f.quote && !seen.has(f.quote) && seen.add(f.quote)).slice(0, 4);
  if (list.length === 0) return null;
  return (
    <ul className="chips">
      {list.map((f) => (
        <li key={f.quote} className={`chip chip-${look.key}`}>
          <Icon
            name={LINK_REASONS.has(f.reason) ? "link" : "flag"}
            color={look.color}
            size={20}
            className="chip-icon"
          />
          <div className="chip-text">
            <span className="chip-label">{flagLabel(f.reason)}:</span>{" "}
            <span className="chip-quote">{formatQuote(f)}</span>
          </div>
        </li>
      ))}
    </ul>
  );
}

/** B6-B9 (Android) and C4 (iOS Safari view): the verdict card, speech, and next actions. */
export function VerdictScreen() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const ios = isIOS();

  const q = useQuery<VerdictT>({
    queryKey: ["verdict", id],
    queryFn: () => getVerdict(id),
    enabled: id !== "offline",
    initialData: id === "offline" ? offlineVerdict() : undefined,
    staleTime: Infinity,
    retry: (n, err) => n < 6 && (!(err instanceof ApiError) || err.status === 404 || err.status === 0),
    retryDelay: 1500,
  });

  // Couldn't fetch at all (hub down / unknown id): show B9, never a guess.
  const v: VerdictT | undefined = q.data ?? (q.isError ? offlineVerdict() : undefined);
  const label: VerdictLabel = v?.verdict ?? "UNKNOWN";
  const look = LOOK[label];
  const offline = id === "offline" || (q.isError && !q.data);
  const explanation = v ? (v.explanation || t("verdict.UNKNOWN.explanation")) : "";
  const lang = v?.language === "hi" ? "hi" : "en";

  const { state: speech, listenAgain } = useSpeech(offline ? "" : explanation, lang);

  useEffect(() => {
    if (v && !offline && !getRecent().some((r) => r.id === v.event_id)) addRecent({ id: v.event_id, verdict: v.verdict, ts: Date.now() });
  }, [v, offline]);

  const showFlags = useMemo(
    () => (label === "SCAM" || label === "SUSPICIOUS" ? (v?.red_flags ?? []) : []),
    [label, v],
  );

  if (!v) return <Checking />;

  const headline = t(`verdict.${label}.headline`);
  const long = headline.length > 20;
  const canListen = speech !== "none" && speech !== "checking" && !offline;
  const blocked = speech === "blocked";

  return (
    <main className={`screen verdict ${ios ? "verdict-ios" : ""}`}>
      <TopBar wordmark hideBack={ios} onBack={() => nav("/", { replace: true })} />
      <div className="verdict-body">
        <section className={`vcard vcard-${look.key}`} aria-labelledby="vhead">
          <div className="vcard-rule" aria-hidden="true" />
          <div className="vcard-finial" aria-hidden="true" />
          <div className="vcard-head">
            <div className="vcard-icon">
              <Icon name={look.icon} color="white" size={34} />
            </div>
            <h1 id="vhead" className={`vcard-title ${long ? "vcard-title-long" : ""}`}>
              {headline}
            </h1>
          </div>
          <p className="vcard-text" lang={lang === "hi" ? "hi" : "en"}>
            {explanation}
          </p>
          <Chips flags={showFlags} look={look} />
        </section>
      </div>

      <div className="verdict-actions">
        {label === "UNKNOWN" ? (
          <Button variant="secondary" icon="rotate-ccw" onClick={() => nav("/check", { replace: true })}>
            {t("verdict.tryAgain")}
          </Button>
        ) : (
          canListen && (
            <Button
              variant="secondary"
              icon="volume-2"
              className={blocked ? "btn-focused" : ""}
              autoFocus={blocked}
              onClick={listenAgain}
            >
              {t("verdict.listenAgain")}
            </Button>
          )
        )}
        {label === "SAFE" ? (
          <Button icon="check" onClick={() => nav("/", { replace: true })}>
            {t("verdict.ok")}
          </Button>
        ) : (
          SON_PHONE && (
            <LinkButton href={`tel:${SON_PHONE}`} icon="phone">
              {t("app.callSon")}
            </LinkButton>
          )
        )}
      </div>
    </main>
  );
}
