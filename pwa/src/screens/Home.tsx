import { Link } from "react-router-dom";
import { Icon, type IconColor } from "../components/Icon";
import { MOCK } from "../api/client";
import { t } from "../i18n";
import { addRecent, getRecent, type RecentItem } from "../lib/storage";

const STYLE: Record<RecentItem["verdict"], { icon: string; color: IconColor; cls: string }> = {
  SCAM: { icon: "hand", color: "scam", cls: "scam" },
  SUSPICIOUS: { icon: "triangle-alert", color: "careful", cls: "careful" },
  SAFE: { icon: "check", color: "normal", cls: "safe" },
  UNKNOWN: { icon: "phone", color: "unknown", cls: "unknown" },
};

function seedMock(): void {
  if (!MOCK || getRecent().length) return;
  const now = Date.now();
  const at = (h: number, m: number, daysAgo = 0) => {
    const d = new Date(now - daysAgo * 86_400_000);
    d.setHours(h, m, 0, 0);
    return d.getTime();
  };
  addRecent({ id: "mock-careful", verdict: "SUSPICIOUS", ts: at(18, 5, 1) });
  addRecent({ id: "mock-safe", verdict: "SAFE", ts: at(11, 30) });
  addRecent({ id: "mock-scam", verdict: "SCAM", ts: at(16, 12) });
}

function whenLabel(ts: number): string {
  const d = new Date(ts);
  const time = d.toLocaleTimeString("en-IN", { hour: "numeric", minute: "2-digit", hour12: true }).toUpperCase();
  const startToday = new Date();
  startToday.setHours(0, 0, 0, 0);
  const day = 86_400_000;
  if (ts >= startToday.getTime()) return t("B2.today", { time });
  if (ts >= startToday.getTime() - day) return t("B2.yesterday", { time });
  const date = d.toLocaleDateString("en-IN", { day: "numeric", month: "short" });
  return t("B2.earlier", { date, time });
}

function BlockPrintStrip() {
  return <div className="strip" aria-hidden="true" />;
}

/** B2. Home: block-print strip, wordmark + gear, two big actions, recently checked. */
export function Home() {
  seedMock();
  const recent = getRecent();
  return (
    <main className="screen home">
      <BlockPrintStrip />
      <header className="home-head">
        <img src="/brand/mark-light.svg" alt="" width={40} height={40} />
        <span className="home-wordmark">{t("B2.wordmark")}</span>
        <Link className="icon-btn" to="/settings" aria-label={t("B2.settings")}>
          <Icon name="settings" />
        </Link>
      </header>

      <nav className="home-actions">
        <Link className="action-card" to="/check">
          <span className="action-disc">
            <Icon name="message-square-text" size={30} />
          </span>
          <span className="action-label">{t("B2.check")}</span>
          <Icon name="chevron-right" color="muted" />
        </Link>
        <Link className="action-card" to="/talk">
          <span className="action-disc">
            <Icon name="mic" size={30} />
          </span>
          <span className="action-label">{t("B2.talk")}</span>
          <Icon name="chevron-right" color="muted" />
        </Link>
      </nav>

      <h2 className="section-title">{t("B2.recent")}</h2>
      {recent.length === 0 ? (
        <p className="recent-empty">{t("B2.recentEmpty")}</p>
      ) : (
        <ul className="recent">
          {recent.map((r) => {
            const s = STYLE[r.verdict];
            return (
              <li key={r.id}>
                <Link className="recent-row" to={`/v/${encodeURIComponent(r.id)}`}>
                  <span className={`recent-dot bg-${s.cls}`}>
                    <Icon name={s.icon} color={s.color} size={18} />
                  </span>
                  <span className={`recent-word fg-${s.cls}`}>{t(`verdict.${r.verdict}.recent`)}</span>
                  <span className="recent-time">{whenLabel(r.ts)}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      )}

      <div className="spacer" />
      <p className="home-private">
        <Icon name="house" color="muted" size={20} />
        <span>{t("app.private")}</span>
      </p>
    </main>
  );
}
