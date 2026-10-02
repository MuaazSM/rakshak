import { useRef, useState } from "react";
import { checkImage, checkText } from "../api/client";
import { Button } from "../components/Button";
import { Icon } from "../components/Icon";
import { TopBar } from "../components/TopBar";
import { t } from "../i18n";
import { useSubmit } from "../lib/useSubmit";
import { Checking } from "./Checking";

/** B4 Check a message (paste text or pick a screenshot) and B5 while it runs. */
export function Check() {
  const [text, setText] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const textRef = useRef<HTMLTextAreaElement>(null);

  const sendText = useSubmit(checkText);
  const sendImage = useSubmit(checkImage);
  const pending = sendText.isPending || sendImage.isPending;

  if (pending) return <Checking />;

  const ready = text.trim().length > 0 || image !== null;
  const submit = () => {
    if (image) sendImage.mutate(image);
    else if (text.trim()) sendText.mutate(text.trim());
  };

  return (
    <main className="screen">
      <TopBar title={t("B4.title")} />
      <form
        className="check-body"
        onSubmit={(e) => {
          e.preventDefault();
          if (ready) submit();
        }}
      >
        <label className="check-label" htmlFor="msg">
          {t("B4.pasteLabel")}
        </label>
        <div className="paste">
          <textarea
            id="msg"
            ref={textRef}
            className="paste-input"
            value={text}
            placeholder={t("B4.pastePlaceholder")}
            onChange={(e) => setText(e.target.value)}
            rows={6}
            autoFocus
          />
          <div className="paste-foot">
            {text && (
              <button
                type="button"
                className="chip-btn"
                onClick={() => {
                  setText("");
                  textRef.current?.focus();
                }}
              >
                <Icon name="x" color="muted" size={18} />
                {t("B4.clear")}
              </button>
            )}
          </div>
        </div>

        <div className="or-row" aria-hidden="true">
          <span />
          {t("B4.or")}
          <span />
        </div>

        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => setImage(e.target.files?.[0] ?? null)}
        />
        {image ? (
          <div className="picked">
            <Icon name="image" size={22} />
            <span className="picked-name">{t("B4.screenshotChosen")}</span>
            <button
              type="button"
              className="chip-btn"
              onClick={() => {
                setImage(null);
                if (fileRef.current) fileRef.current.value = "";
              }}
            >
              <Icon name="x" color="muted" size={18} />
              {t("B4.removeScreenshot")}
            </button>
          </div>
        ) : (
          <Button type="button" variant="secondary" icon="image" onClick={() => fileRef.current?.click()}>
            {t("B4.chooseScreenshot")}
          </Button>
        )}

        <div className="grow" />
        <Button type="submit" disabled={!ready}>
          {t("B4.submit")}
        </Button>
      </form>
    </main>
  );
}
