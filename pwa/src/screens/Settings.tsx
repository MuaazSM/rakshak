import { useState } from "react";
import { patchLanguage } from "../api/client";
import { Icon } from "../components/Icon";
import { TopBar } from "../components/TopBar";
import { t } from "../i18n";
import { getLang, setLang, type Lang } from "../lib/storage";

const OPTIONS: { value: Lang; key: string }[] = [
  { value: "en", key: "B11.english" },
  { value: "hi", key: "B11.hindi" },
];

/** B11 Settings: one row, "Explanation language". Saved on the profile (FR-9). */
export function Settings() {
  const [lang, setLocal] = useState<Lang>(getLang());
  const [failed, setFailed] = useState(false);

  const choose = async (value: Lang) => {
    if (value === lang) return;
    setFailed(false);
    const prev = lang;
    setLang(value);
    setLocal(value);
    try {
      await patchLanguage(value);
    } catch {
      setLang(prev);
      setLocal(prev);
      setFailed(true);
    }
  };

  return (
    <main className="screen">
      <TopBar title={t("B11.title")} />
      <h2 className="settings-label" id="lang-label">
        {t("B11.languageLabel")}
      </h2>
      <div className="settings-card" role="radiogroup" aria-labelledby="lang-label">
        {OPTIONS.map((o) => {
          const on = lang === o.value;
          return (
            <button
              key={o.value}
              role="radio"
              aria-checked={on}
              className={`settings-row ${on ? "is-on" : ""}`}
              onClick={() => choose(o.value)}
            >
              <span className={`settings-name ${o.value === "hi" ? "is-hindi" : ""}`} lang={o.value}>
                {t(o.key)}
              </span>
              {on ? (
                <span className="settings-check">
                  <Icon name="check" size={18} alt={t("B11.selected")} />
                </span>
              ) : (
                <span className="settings-radio" />
              )}
            </button>
          );
        })}
      </div>
      {failed && (
        <p className="settings-error" role="alert">
          {t("B11.saveFailed")}
        </p>
      )}
    </main>
  );
}
