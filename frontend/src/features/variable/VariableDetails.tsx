import { Link } from "@tanstack/react-router";
import { ArrowUpRight, CircleAlert, CircleCheck, CircleMinus } from "lucide-react";

import type { NumericProfile } from "../../api/client";
import { useVariable } from "../../api/hooks";
import { t, type MessageKey } from "../../lib/i18n";
import { categoricalMetrics, dateMetrics } from "./profilePanels";
import { Metrics } from "./Metrics";
import { normalitySummary, rejects, testPLabel } from "./normality";
import { GROUPS, StatTable, statRow } from "./StatTable";
import { Loading } from "../../ui/Loading";

const ALPHA = 0.05;

/** The fullscreen view's details: the variable page in brief, with a link to it. */
export function VariableDetails({ datasetId, column }: { datasetId: string; column: string }) {
  const profile = useVariable(datasetId, column);
  const data = profile.data;

  return (
    <section className="space-y-3 border-t border-line pt-4">
      <div className="flex justify-end">
        <Link
          to="/d/$datasetId/variable/$column"
          params={{ datasetId, column }}
          className="inline-flex h-(--control-h) items-center gap-1.5 rounded-md border border-line bg-surface-1 px-3 text-[13px] text-ink-1 hover:bg-surface-2"
        >
          {t("details.open")}
          <ArrowUpRight size={15} strokeWidth={1.5} aria-hidden />
        </Link>
      </div>
      {!data ? (
        profile.isError ? (
          <p className="text-bad">{profile.error.message}</p>
        ) : (
          <Loading label={t("loading.stats")} className="h-40 rounded-md" />
        )
      ) : (
        <>
          {data.numeric && <NumericDetails profile={data.numeric} />}
          {data.categorical && !data.numeric && <Metrics items={categoricalMetrics(data.categorical)} />}
          {data.date && <Metrics items={dateMetrics(data.date)} />}
        </>
      )}
    </section>
  );
}

function NumericDetails({ profile }: { profile: NumericProfile }) {
  const summary = profile.describe;
  const normality = profile.normality;
  return (
    <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
      <div className="grid gap-3 sm:grid-cols-2">
        {GROUPS.map((group) => (
          <StatTable key={group.title} title={t(group.title)} rows={group.keys.map((key) => statRow(summary, key))} />
        ))}
      </div>
      <section className="min-w-0">
        <h3 className="mb-1.5 text-xs font-medium text-ink-3">{t("variable.tab.tests")} · α = {ALPHA}</h3>
        <div className="rounded-md border border-line bg-surface-1">
          <div className="space-y-1 border-b border-line px-3 py-2">
            {normalitySummary(normality, ALPHA).map((sentence) => (
              <p key={sentence} className="text-ink-1 first:font-medium [&+p]:text-xs [&+p]:text-ink-2">
                {sentence}
              </p>
            ))}
          </div>
          <ul className="divide-y divide-line">
            {normality.tests.map((test) => {
              const rejected = rejects(test, ALPHA);
              const Icon = rejected === null ? CircleMinus : rejected ? CircleAlert : CircleCheck;
              const tone = rejected === null ? "text-ink-3" : rejected ? "text-warn" : "text-ok";
              return (
                <li key={test.key} className="flex items-center justify-between gap-2 px-3 py-1.5">
                  <span className="flex min-w-0 items-center gap-1.5">
                    <Icon size={14} strokeWidth={1.75} className={`shrink-0 ${tone}`} aria-hidden />
                    <span className="truncate text-ink-2">{t(`test.${test.key}.name` as MessageKey)}</span>
                  </span>
                  <span className="text-xs text-ink-1 tabular-nums">{testPLabel(test)}</span>
                </li>
              );
            })}
          </ul>
        </div>
      </section>
    </div>
  );
}
