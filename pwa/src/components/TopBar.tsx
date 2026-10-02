import { useNavigate } from "react-router-dom";
import { t } from "../i18n";
import { Icon } from "./Icon";

interface Props {
  title?: string;
  wordmark?: boolean;
  onBack?: () => void;
  hideBack?: boolean;
}

/** 64-72px bar: 56px back target + title (Mukta 700 24px) or the Yatra One wordmark. */
export function TopBar({ title, wordmark, onBack, hideBack }: Props) {
  const nav = useNavigate();
  return (
    <header className={`topbar ${hideBack ? "topbar-noback" : ""}`}>
      {!hideBack && (
        <button className="icon-btn" aria-label={t("app.back")} onClick={onBack ?? (() => nav(-1))}>
          <Icon name="arrow-left" />
        </button>
      )}
      {wordmark ? (
        <span className="topbar-wordmark">{t("app.name")}</span>
      ) : (
        <h1 className="topbar-title">{title}</h1>
      )}
    </header>
  );
}
