import { Filter, Users, X } from "lucide-react";
import { useId, useState } from "react";

import { ApiError, type CohortRequest, type FunnelRequest, type KpiAnalysis } from "../../api/client";
import { useKpiCohort, useKpiFunnel } from "../../api/hooks";
import { Plot } from "../../charts/Plot";
import { t } from "../../lib/i18n";
import { ErrorNote } from "../../ui/fields";
import { TipBox } from "../../ui/InfoTip";
import { Loading } from "../../ui/Loading";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { cohortFigure, conversions, funnelFigure, newReturningFigure } from "./cohortFigures";
import { formatShare } from "./format";
import { splitReason } from "./SuggestionChips";

type Grain = CohortRequest["grain"];

function Notes({ lines }: { lines: string[] }) {
  if (lines.length === 0) return null;
  return (
    <ul className="space-y-0.5">
      {lines.map((line) => (
        <li key={line} className="border-l-2 border-accent pl-3 text-xs text-ink-2">
          {line}
        </li>
      ))}
    </ul>
  );
}

function Why({ lines }: { lines: string[] }) {
  return (
    <p className="text-xs text-ink-3">
      <span className="font-medium text-ink-2">{t("kpi.why")} </span>
      {lines.map((line) => splitReason(line).text).join(" · ")}
    </p>
  );
}

function Failure({ error }: { error: unknown }) {
  return <ErrorNote>{error instanceof ApiError ? error.message : t("kpi.analysis.failed")}</ErrorNote>;
}

/** 060: retention by cohort (triangle) and new vs returning customers, computed once opened. */
function CohortPanel({ datasetId, analysis }: { datasetId: string; analysis: KpiAnalysis }) {
  const [grain, setGrain] = useState<Grain>("auto");
  const body: CohortRequest = { customer: analysis.customer ?? "", date: analysis.date ?? "", grain };
  const cohort = useKpiCohort(datasetId, body);
  const result = cohort.data;
  const unit = result ? t(`kpi.cohort.unit.${result.grain}`) : "";
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs text-ink-2">{t("kpi.cohort.grain")}</span>
        <SegmentedControl
          label={t("kpi.cohort.grain")}
          value={grain}
          onChange={setGrain}
          options={(["auto", "month", "week"] as const).map((value) => ({ value, label: t(`kpi.cohort.grain.${value}`) }))}
        />
      </div>
      {cohort.error ? (
        <Failure error={cohort.error} />
      ) : !result ? (
        <Loading label={t("kpi.cohort.loading")} className="h-64" />
      ) : (
        <div className={`space-y-4 ${cohort.isFetching ? "opacity-60 transition-opacity" : ""}`}>
          <p className="text-[15px] font-medium text-ink-1">{result.summary}</p>
          <Notes lines={result.warnings.map((w) => splitReason(w).text)} />
          <figure className="space-y-1">
            <figcaption className="text-xs text-ink-2">{t("kpi.cohort.matrix", { unit })}</figcaption>
            <Plot figure={cohortFigure(result)} height={Math.min(520, 60 + 28 * result.cohorts.length)} label={t("kpi.cohort.matrix", { unit })} />
          </figure>
          <figure className="space-y-1">
            <figcaption className="text-xs text-ink-2">{t("kpi.cohort.newReturning", { unit })}</figcaption>
            <Plot figure={newReturningFigure(result)} height={200} label={t("kpi.cohort.newReturning", { unit })} />
          </figure>
          <Why lines={[analysis.reason, result.grain_reason, result.reason]} />
        </div>
      )}
    </div>
  );
}

/** 060: how far an ordered status goes, and where it drops. */
function FunnelPanel({ datasetId, analysis }: { datasetId: string; analysis: KpiAnalysis }) {
  const body: FunnelRequest = { column: analysis.column ?? "", stages: analysis.stages, key: analysis.key ?? null };
  const funnel = useKpiFunnel(datasetId, body);
  const result = funnel.data;
  if (funnel.error) return <Failure error={funnel.error} />;
  if (!result) return <Loading label={t("kpi.funnel.loading")} className="h-48" />;
  return (
    <div className="space-y-3">
      <p className="text-[15px] font-medium text-ink-1">{splitReason(result.summary).text}</p>
      <Notes lines={result.warnings} />
      <Plot figure={funnelFigure(result)} height={60 + 36 * result.stages.length} label={analysis.name} />
      <div>
        <p className="text-xs text-ink-2">{t("kpi.funnel.conversions")}</p>
        <ul className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[13px]">
          {conversions(result).map((step) => (
            <li key={step.to} className="tabular-nums text-ink-1">
              {step.from} → {step.to}: {formatShare(step.share)}
            </li>
          ))}
        </ul>
      </div>
      <Why lines={[analysis.reason, result.reason]} />
    </div>
  );
}

function AnalysisChip({ analysis, open, onToggle }: { analysis: KpiAnalysis; open: boolean; onToggle: () => void }) {
  const id = useId();
  const Icon = analysis.kind === "cohort" ? Users : Filter;
  return (
    <span className="group relative inline-flex">
      <button
        type="button"
        aria-expanded={open}
        aria-describedby={id}
        onClick={onToggle}
        className={`inline-flex items-center gap-1 border-b-2 px-0.5 pb-0.5 text-[13px] ${open ? "border-accent font-medium text-ink-1" : "border-transparent text-ink-2 hover:border-line hover:text-ink-1"}`}
      >
        <Icon size={12} strokeWidth={1.75} aria-hidden />
        {analysis.name}
      </button>
      <TipBox id={id} placement="below">
        {splitReason(analysis.reason).text}
      </TipBox>
    </span>
  );
}

/** "Also on this data:" cohorts and funnel the rules found; nothing runs before a click (onion). */
export function Analyses({ datasetId, analyses }: { datasetId: string; analyses: KpiAnalysis[] }) {
  const [open, setOpen] = useState<string | null>(null);
  if (analyses.length === 0) return null;
  const shown = analyses.find((a) => a.kind === open);
  return (
    <section className="space-y-3" aria-label={t("kpi.analyses")}>
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1" role="group" aria-label={t("kpi.analyses")}>
        <span className="pr-1 text-xs text-ink-3">{t("kpi.analyses")}</span>
        {analyses.map((analysis) => (
          <AnalysisChip key={analysis.kind} analysis={analysis} open={open === analysis.kind} onToggle={() => setOpen(open === analysis.kind ? null : analysis.kind)} />
        ))}
      </div>
      {shown && (
        <div className="space-y-3 border-y border-line py-4">
          <div className="flex items-center justify-between gap-2">
            <h2 className="text-[15px] font-semibold text-ink-1">{shown.name}</h2>
            <button type="button" onClick={() => setOpen(null)} aria-label={t("kpi.analysis.close")} title={t("kpi.analysis.close")} className="inline-flex size-8 items-center justify-center rounded text-ink-3 hover:text-ink-1">
              <X size={14} aria-hidden />
            </button>
          </div>
          {shown.kind === "cohort" ? <CohortPanel datasetId={datasetId} analysis={shown} /> : <FunnelPanel datasetId={datasetId} analysis={shown} />}
        </div>
      )}
    </section>
  );
}
