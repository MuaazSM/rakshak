import { Icon } from "../components/Icon";
import { t } from "../i18n";
import { useStage } from "../lib/progress";

/** B5. One motion only: the yellow bar glides under the top edge. */
export function Checking() {
  const stage = useStage();
  const steps = ["B5.stepReading", "B5.stepChecking", "B5.stepExplaining"];
  return (
    <main className="screen" role="status" aria-live="polite">
      <div className="progress" aria-hidden="true">
        <div className="progress-glide" />
      </div>
      <div className="checking-body">
        <div className="checking-badge">
          <Icon name="house" size={40} />
        </div>
        <h1 className="checking-title">{t("B5.title")}</h1>
        <ol className="steps">
          {steps.map((key, i) => {
            const state = i < stage ? "done" : i === stage ? "active" : "todo";
            return (
              <li key={key} className={`step step-${state}`}>
                <span className="step-dot">
                  {state === "done" && <Icon name="check" color="white" size={16} />}
                  {state === "active" && <span className="step-pulse" />}
                </span>
                <span className="step-label">{t(key)}</span>
              </li>
            );
          })}
        </ol>
      </div>
      <p className="checking-eta">{t("B5.eta")}</p>
    </main>
  );
}
