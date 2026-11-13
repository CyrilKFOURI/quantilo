import { useSyncExternalStore, type ReactNode } from "react";

import { t } from "../lib/i18n";
import { ProgressBar } from "../ui/fields";
import { engineState, onEngineState } from "./bridge";

/** Holds the app back while Python loads in the page (about a minute the first time). */
export function EngineGate({ children }: { children: ReactNode }) {
  const state = useSyncExternalStore(onEngineState, engineState);
  if (state.status === "ready") return children;
  return (
    <div className="flex min-h-dvh items-center justify-center bg-surface-0 px-6">
      <div className="w-full max-w-md">
        <h1 className="text-lg font-semibold tracking-tight">
          {t("lite.loadingTitle")}
        </h1>
        {state.status === "failed" ? (
          <p className="mt-3 text-[13px] text-bad">
            {t("lite.failed", { error: state.reason ? t(state.reason.key, state.reason.vars) : state.error })}
          </p>
        ) : (
          <>
            <p className="mt-1 text-[13px] text-ink-2">{t("lite.loadingBody")}</p>
            <div className="mt-5">
              <ProgressBar value={state.fraction} label={t(state.step)} />
              <p className="mt-1.5 text-xs text-ink-3">{t(state.step)}…</p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
