import { Lock } from "lucide-react";

import { t } from "../lib/i18n";
import { FULL_APP_URL } from "./flag";

/** Small lock shown next to what the browser-only version cannot do. */
export function LockBadge({ className = "" }: { className?: string }) {
  return (
    <Lock
      size={12}
      strokeWidth={1.75}
      aria-label={t("lite.locked")}
      className={`shrink-0 text-ink-3 ${className}`}
    />
  );
}

/** The page of a module the browser-only version cannot run, named so one knows what is missing. */
export function LockedPage({ page }: { page: string }) {
  return (
    <div className="mx-auto max-w-xl px-(--gutter) py-16">
      <div className="mb-4 inline-flex size-10 items-center justify-center rounded-full bg-surface-2 text-ink-2">
        <Lock size={18} strokeWidth={1.5} aria-hidden />
      </div>
      <h1 className="text-xl font-semibold tracking-tight">
        {t("lite.lockedTitle", { page })}
      </h1>
      <p className="mt-2 text-ink-2">{t("lite.lockedBody")}</p>
      <p className="mt-4 text-[13px] text-ink-2">{t("lite.lockedHow")}</p>
      <pre className="mt-2 overflow-x-auto rounded-md border border-line bg-surface-1 px-3 py-2 font-mono text-xs text-ink-1">
        git clone https://github.com/CyrilKFOURI/quantilo.git; cd
        quantilo; .\run.bat
      </pre>
      <a
        href={FULL_APP_URL}
        target="_blank"
        rel="noreferrer"
        className="mt-3 inline-block text-[13px] text-accent hover:underline"
      >
        {t("lite.lockedLink")}
      </a>
    </div>
  );
}
