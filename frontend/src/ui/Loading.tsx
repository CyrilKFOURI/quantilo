import { useEffect, useState, type CSSProperties } from "react";

import { t } from "../lib/i18n";
import { LITE } from "../lite/flag";

/** The time shown once a wait is no longer short, and the note that says it is still working (042). */
const SHOW_SECONDS = 3;
const SLOW_SECONDS = 8;

type LoadingProps = {
  /** What is being computed: "Computing the trend…". */
  label?: string;
  /** The size of the content it stands for (`h-72 rounded-lg`), so nothing jumps when it arrives. */
  className?: string;
  style?: CSSProperties;
  /** A small spot (a card, a line): no "still working" note. */
  compact?: boolean;
};

/** Stands where content will appear (042): a slow sheen, three small bars and what is being computed;
 *  after a few seconds, the time it has taken and a word that it is still working. */
export function Loading({ label = t("loading.default"), className = "h-40 rounded-lg", style, compact = false }: LoadingProps) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const started = Date.now();
    const timer = setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(timer);
  }, []);
  return (
    <div
      role="status"
      aria-live="polite"
      aria-busy="true"
      style={style}
      className={`loading-sheen relative flex flex-col items-center justify-center gap-1 overflow-hidden border border-line bg-surface-1 px-4 text-center ${className}`}
    >
      <span className="flex flex-wrap items-center justify-center gap-x-2.5 gap-y-1 text-[13px] text-ink-2">
        <span className="loading-bars" aria-hidden>
          <i />
          <i />
          <i />
        </span>
        <span>{label}</span>
        {seconds >= SHOW_SECONDS && <span className="text-xs text-ink-3 tabular-nums">{t("loading.seconds", { seconds: String(seconds) })}</span>}
      </span>
      {!compact && seconds >= SLOW_SECONDS && (
        <span className="max-w-sm text-xs text-ink-3">{LITE ? t("loading.slowLite") : t("loading.slow")}</span>
      )}
    </div>
  );
}
