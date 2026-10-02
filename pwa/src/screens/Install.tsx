import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "../components/Button";
import { t } from "../i18n";
import { canPrompt, onInstallChange, promptInstall } from "../lib/install";
import { markInstallSeen } from "../lib/storage";

/** B1. Shown once, when the app is opened in the browser from ?parent=mom. */
export function Install() {
  const nav = useNavigate();
  const [, force] = useState(0);
  const [noPrompt, setNoPrompt] = useState(false);

  useEffect(() => onInstallChange(() => force((n) => n + 1)), []);

  const done = () => {
    markInstallSeen();
    nav("/", { replace: true });
  };

  const onInstall = async () => {
    if (!canPrompt()) {
      setNoPrompt(true);
      return;
    }
    await promptInstall();
    done();
  };

  return (
    <main className="screen install">
      <div className="install-hero">
        <img className="install-mark" src="/brand/mark-light.svg" alt="" width={112} height={112} />
        <div className="install-wordmark">{t("B1.wordmark")}</div>
        <div className="install-rule" />
        <h1 className="install-tagline">{t("B1.tagline")}</h1>
      </div>
      <div className="install-actions">
        {noPrompt && <p className="install-hint">{t("B1.hint")}</p>}
        <Button className="btn-focused" icon="square-plus" onClick={onInstall} autoFocus>
          {t("B1.install")}
        </Button>
        {noPrompt && (
          <button className="link-btn" onClick={done}>
            {t("B1.continue")}
          </button>
        )}
      </div>
    </main>
  );
}
