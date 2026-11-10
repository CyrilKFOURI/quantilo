import { Link, useNavigate, useParams } from "@tanstack/react-router";
import { ArrowLeft, ChevronLeft, ChevronRight, ScatterChart } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { readDefaults } from "../../lib/defaults";

import type { AnalysisType, Column, VariableProfile } from "../../api/client";
import { useActiveFilters } from "../../api/filters";
import { useChart, useColumns, useDataset, useUpdateColumns, useVariable } from "../../api/hooks";
import { NoRows } from "../../features/filters/NoRows";
import { formatNumber } from "../../charts/figures";
import { TypeBadge } from "../../features/columns/TypeBadge";
import { CategoryTable } from "../../features/variable/CategoryTable";
import { FitPanel } from "../../features/variable/FitPanel";
import { Metrics, type Metric } from "../../features/variable/Metrics";
import { isNormal, type Alpha } from "../../features/variable/normality";
import { NormalityPanel } from "../../features/variable/NormalityPanel";
import { NumericChart } from "../../features/variable/NumericChart";
import { OutliersPanel } from "../../features/variable/OutliersPanel";
import { RobustBlock } from "../../features/variable/RobustBlock";
import { TailPanel } from "../../features/variable/TailPanel";
import {
  CategoricalPanel,
  DatePanel,
  IdentifierPanel,
  TextPanel,
} from "../../features/variable/profilePanels";
import { GROUPS, StatTable, statRow } from "../../features/variable/StatTable";
import { ChartView } from "../../features/variable-grid/ChartView";
import { ViewSwitch } from "../../features/variable-grid/ViewSwitch";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { useStoredState } from "../../lib/useStoredState";
import { Button } from "../../ui/Button";
import { ErrorNote } from "../../ui/fields";
import { Tabs } from "../../ui/Tabs";
import { Loading } from "../../ui/Loading";

type NumericTab = "summary" | "tests" | "distribution" | "outliers" | "tail" | "categories";

export function VariablePage() {
  const { datasetId, column: name } = useParams({ from: "/d/$datasetId/variable/$column" });
  const dataset = useDataset(datasetId);
  const ready = dataset.data?.status === "ready";
  const columns = useColumns(datasetId, ready);
  const update = useUpdateColumns(datasetId);
  const navigate = useNavigate();

  const used = useMemo(
    () => (columns.data ?? []).filter((c) => !c.excluded).sort((a, b) => a.position - b.position),
    [columns.data],
  );
  const column = columns.data?.find((c) => c.name === name);
  const index = used.findIndex((c) => c.name === name);
  const previous = index > 0 ? used[index - 1] : undefined;
  const next = index >= 0 && index < used.length - 1 ? used[index + 1] : undefined;
  const analysable = Boolean(column && !column.excluded && column.type !== "ignored");
  const profile = useVariable(datasetId, name, ready && analysable);

  useEffect(() => {
    const go = (target: Column | undefined) =>
      target && void navigate({ to: "/d/$datasetId/variable/$column", params: { datasetId, column: target.name } });
    const onKey = (event: KeyboardEvent) => {
      const typing = event.target instanceof HTMLElement && event.target.closest("input, select, textarea");
      if (typing || event.altKey || event.ctrlKey || event.metaKey) return;
      if (event.key === "ArrowLeft" && event.shiftKey) go(previous);
      if (event.key === "ArrowRight" && event.shiftKey) go(next);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navigate, datasetId, previous, next]);

  if (dataset.isError || columns.isError) {
    return (
      <div className="px-(--gutter) py-8">
        <ErrorNote>{(dataset.error ?? columns.error)?.message ?? ""}</ErrorNote>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5 px-(--gutter) py-6">
      <header className="space-y-2">
        <Link
          to="/d/$datasetId"
          params={{ datasetId }}
          className="inline-flex items-center gap-1 text-xs text-ink-3 hover:text-ink-1"
        >
          <ArrowLeft size={13} strokeWidth={1.5} aria-hidden />
          {t("variable.back")}
        </Link>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <h1 className="truncate font-mono text-xl font-semibold tracking-tight text-ink-1" title={name}>
              {name}
            </h1>
            {column && (
              <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-3">
                <div className="-ml-1.5">
                  <TypeBadge
                    type={column.type}
                    detectedType={column.detected_type}
                    reasons={column.reasons}
                    role={column.role}
                    onChange={(type: AnalysisType) => update.mutate([{ name: column.name, type }])}
                  />
                </div>
                {column.role_label && <span>{column.role_label}</span>}
              </div>
            )}
          </div>
          <div className="flex items-center gap-1.5">
            <Link
              to="/d/$datasetId/compare"
              params={{ datasetId }}
              search={(previous) => ({ ...previous, x: name, y: undefined, z: undefined, color: undefined })}
              className="inline-flex h-(--control-h) items-center gap-1.5 rounded-md border border-line bg-surface-1 px-3 text-[13px] text-ink-1 hover:bg-surface-2"
            >
              <ScatterChart size={15} strokeWidth={1.5} aria-hidden />
              {t("variable.compare")}
            </Link>
            <NavButton datasetId={datasetId} target={previous} label={t("variable.previous")} direction="previous" />
            {index >= 0 && (
              <span className="px-1 text-xs text-ink-3 tabular-nums">
                {t("variable.position", { index: String(index + 1), total: String(used.length) })}
              </span>
            )}
            <NavButton datasetId={datasetId} target={next} label={t("variable.next")} direction="next" />
          </div>
        </div>
      </header>

      {columns.data && !column ? (
        <p className="text-ink-2">{t("variable.notFound")}</p>
      ) : column?.excluded ? (
        <p className="text-ink-2">{t("variable.excluded")}</p>
      ) : column?.type === "ignored" ? (
        <p className="text-ink-2">{t("variable.ignored")}</p>
      ) : profile.isError ? (
        <ErrorNote>{profile.error.message}</ErrorNote>
      ) : !profile.data || !column ? (
        <Loading label={t("variable.loading", { name })} className="h-80 rounded-lg" />
      ) : (
        <VariableBody key={name} datasetId={datasetId} column={column} profile={profile.data} />
      )}
    </div>
  );
}

function NavButton({
  datasetId,
  target,
  label,
  direction,
}: {
  datasetId: string;
  target: Column | undefined;
  label: string;
  direction: "previous" | "next";
}) {
  const Icon = direction === "previous" ? ChevronLeft : ChevronRight;
  const hint = target ? `${label}: ${target.name} (Shift + ${direction === "previous" ? "←" : "→"})` : label;
  if (!target) return <Button variant="ghost" icon={Icon} iconOnly aria-label={label} title={label} disabled />;
  return (
    <Link
      to="/d/$datasetId/variable/$column"
      params={{ datasetId, column: target.name }}
      aria-label={hint}
      title={hint}
      className="inline-flex size-(--control-h) items-center justify-center rounded-md text-ink-2 hover:bg-surface-2 hover:text-ink-1"
    >
      <Icon size={15} strokeWidth={1.5} aria-hidden />
    </Link>
  );
}

function VariableBody({ datasetId, column, profile }: { datasetId: string; column: Column; profile: VariableProfile }) {
  const filtered = useActiveFilters(datasetId).length > 0;
  const rows = rowsOf(profile);
  if (filtered && rows === 0) return <NoRows datasetId={datasetId} />;
  return (
    <div className="space-y-3">
      {filtered && rows !== null && (
        <p className="text-xs text-accent">{t("filters.filteredNote", { rows: formatCount(rows) })}</p>
      )}
      <VariableSections datasetId={datasetId} column={column} profile={profile} />
    </div>
  );
}

function rowsOf(profile: VariableProfile): number | null {
  const section = profile.numeric?.describe ?? profile.categorical ?? profile.date ?? profile.text ?? profile.identifier;
  return section ? section.rows : null;
}

function VariableSections({ datasetId, column, profile }: { datasetId: string; column: Column; profile: VariableProfile }) {
  if (profile.numeric) return <NumericBody datasetId={datasetId} column={column} profile={profile} />;
  return (
    <div className="space-y-5">
      {column.type !== "identifier" && <ChartBlock datasetId={datasetId} column={column} />}
      {profile.categorical && <CategoricalPanel profile={profile.categorical} />}
      {profile.categorical && <FullCategories datasetId={datasetId} column={column} />}
      {profile.date && <DatePanel profile={profile.date} />}
      {profile.text && <TextPanel profile={profile.text} />}
      {profile.identifier && <IdentifierPanel profile={profile.identifier} />}
    </div>
  );
}

function NumericBody({ datasetId, column, profile }: { datasetId: string; column: Column; profile: VariableProfile }) {
  const numeric = profile.numeric;
  const [tab, setTab] = useState<NumericTab>("summary");
  const [alpha, setAlpha] = useStoredState<Alpha>("de.variable.alpha", readDefaults().alpha);
  if (!numeric) return null;
  const summary = numeric.describe;
  const normal = isNormal(numeric.normality, Number(alpha));
  const metrics: Metric[] = [
    { label: t("vstat.count"), value: formatCount(summary.count) },
    {
      label: t("vstat.missing"),
      value: formatCount(summary.missing),
      note: summary.missing && summary.rows ? `${((100 * summary.missing) / summary.rows).toFixed(1)}%` : undefined,
    },
    { label: t("vstat.mean"), value: formatNumber(summary.mean) },
    { label: t("vstat.median"), value: formatNumber(summary.median) },
    { label: t("vstat.std"), value: formatNumber(summary.std) },
    { label: t("vstat.skewness"), value: formatNumber(summary.skewness), explain: t("explain.skewness") },
    {
      label: t("variable.tab.tests"),
      value: normal === null ? "—" : normal ? t("normality.shortAccept") : t("normality.shortReject"),
      tone: normal === null ? "text-ink-3" : normal ? "text-ok" : "text-warn",
    },
  ];
  const tabs = [
    { value: "summary" as const, label: t("variable.tab.summary") },
    { value: "tests" as const, label: t("variable.tab.tests") },
    { value: "distribution" as const, label: t("variable.tab.distribution") },
    { value: "outliers" as const, label: t("variable.tab.outliers") },
    // 056, R-TAIL-1: the Tail tab only when the profile's rule fires.
    ...(numeric.tail_rule?.fires ? [{ value: "tail" as const, label: t("variable.tab.tail") }] : []),
    ...(profile.categorical ? [{ value: "categories" as const, label: t("variable.tab.categories") }] : []),
  ];
  const robust = numeric.robust?.fires ? numeric.robust : null;

  return (
    <div className="space-y-5">
      <Metrics items={metrics} />
      <NumericChart datasetId={datasetId} column={column.name} type={column.type} profile={numeric} />
      <Tabs label={t("variable.tabs")} value={tab} items={tabs} onChange={setTab}>
        {tab === "summary" && (
          <div className="space-y-5">
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              {GROUPS.map((group) => (
                <StatTable key={group.title} title={t(group.title)} rows={group.keys.map((key) => statRow(summary, key))} />
              ))}
            </div>
            {robust && <RobustBlock robust={robust} />}
          </div>
        )}
        {tab === "tail" && <TailPanel datasetId={datasetId} column={column.name} />}
        {tab === "tests" && <NormalityPanel normality={numeric.normality} alpha={alpha} onAlpha={setAlpha} />}
        {tab === "distribution" && <FitPanel datasetId={datasetId} column={column.name} profile={numeric} />}
        {tab === "outliers" && (
          <div className="space-y-3">
            <OutliersPanel datasetId={datasetId} column={column.name} />
            <Link
              to="/d/$datasetId/segments"
              params={{ datasetId }}
              search={{ section: "unusual" }}
              className="inline-block text-[13px] text-accent hover:underline"
            >
              {t("variable.unusualRows")}
            </Link>
          </div>
        )}
        {tab === "categories" && profile.categorical && (
          <div className="space-y-4">
            <CategoricalPanel profile={profile.categorical} />
            <FullCategories datasetId={datasetId} column={column} />
          </div>
        )}
      </Tabs>
    </div>
  );
}

/** The column's own chart with its views (timeline / month / day of week, bars / pie...). */
function ChartBlock({ datasetId, column }: { datasetId: string; column: Column }) {
  const [view, setView] = useState<string | null>(null);
  const [step, setStep] = useState<string | null>(null);
  const chart = useChart(datasetId, column.name, { view, step, full: true }, true);
  const data = chart.data;
  const bars = data?.kind === "bars" && data.view === "bars" && !data.ordered;
  const height = bars ? Math.min(Math.max(300, (data.bars.length + 2) * 22), 2200) : 320;
  return (
    <section className="rounded-lg border border-line bg-surface-1 p-3">
      {data && data.views.length > 1 && (
        <div className="mb-2">
          <ViewSwitch chart={data} onView={setView} onStep={setStep} />
        </div>
      )}
      {data ? (
        <div className="max-h-[440px] overflow-y-auto">
          <ChartView chart={data} type={column.type} name={column.name} height={height} datasetId={datasetId} />
        </div>
      ) : (
        <Loading label={t("loading.chart")} className="h-80 rounded-md" />
      )}
    </section>
  );
}

function FullCategories({ datasetId, column }: { datasetId: string; column: Column }) {
  const chart = useChart(datasetId, column.name, { full: true }, true);
  if (!chart.data || chart.data.kind !== "bars") return null;
  return <CategoryTable chart={chart.data} />;
}
