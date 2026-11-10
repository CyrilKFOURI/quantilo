import {
  Link,
  useNavigate,
  useParams,
  useSearch,
} from "@tanstack/react-router";
import { ArrowLeftRight } from "lucide-react";
import { useMemo, useState } from "react";

import type { AnalysisType, Column, Comparison } from "../../api/client";
import { useColumns, useCompare, useDataset } from "../../api/hooks";
import { Plot } from "../../charts/Plot";
import {
  contingencyFigure,
  dateCountsFigure,
  densityFigure,
  groupsFigure,
  scatter3dFigure,
  scatterFigure,
  timelineFigure,
  type GroupView,
  type TableView,
} from "../../features/compare/figures";
import { TestsPanel } from "../../features/compare/TestsPanel";
import { GroupIntervals, PairwiseDifferences } from "../../features/compare/GroupInference";
import {
  shownLayers,
  TimeOptions,
  TimeTable,
  timeSummary,
  validateTimeSearch,
  type TimeSearch,
} from "../../features/compare/TimeOptions";
import { useExperimentCandidates } from "../../features/decisions/api";
import { NoRows } from "../../features/filters/NoRows";
import { t, type MessageKey } from "../../lib/i18n";
import { PHONE, useMedia } from "../../lib/useMedia";
import { Button } from "../../ui/Button";
import { Checkbox, ErrorNote, Field, Select } from "../../ui/fields";
import { ParamPanel } from "../../ui/ParamPanel";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { Loading } from "../../ui/Loading";
import { InfoTip } from "../../ui/InfoTip";

export type CompareSearch = {
  x?: string;
  y?: string;
  z?: string;
  color?: string;
  /** Whole-number columns kept as numbers (R-CMP-02), comma-separated. */
  numbers?: string;
} & TimeSearch;

const text = (value: unknown) =>
  typeof value === "string" && value ? value : undefined;

export function validateCompareSearch(
  search: Record<string, unknown>,
): CompareSearch {
  return {
    x: text(search.x),
    y: text(search.y),
    z: text(search.z),
    color: text(search.color),
    numbers: text(search.numbers),
    ...validateTimeSearch(search),
  };
}

type Role = "num" | "cat" | "date";
const ROLE: Partial<Record<AnalysisType, Role>> = {
  numeric_continuous: "num",
  numeric_discrete: "num",
  categorical_nominal: "cat",
  categorical_ordinal: "cat",
  boolean: "cat",
  datetime: "date",
};
const HEIGHT = 460;
const MAX_DISCRETE_CATEGORIES = 12; // R-CMP-01

const distinctOf = (column: Column) => {
  const value = (column.summary as { distinct_count?: unknown } | null)?.distinct_count;
  return typeof value === "number" ? value : null;
};

/** A whole number with at most 12 values reads as categories, unless kept as numbers. */
const readsAsLevels = (column: Column) =>
  column.type === "numeric_discrete" && (distinctOf(column) ?? Infinity) <= MAX_DISCRETE_CATEGORIES;

const LEVEL_TYPES = new Set<AnalysisType>(["categorical_nominal", "categorical_ordinal", "boolean", "numeric_discrete"]);
const METRIC_TYPES = new Set<AnalysisType>(["numeric_continuous", "numeric_discrete", "boolean"]);

/** A column of 2 to 5 values against a number or yes/no: it reads as an experiment (034). */
export function experimentOf(x: Column, y: Column): { variant: string; metric: string } | null {
  const levels = (c: Column) => LEVEL_TYPES.has(c.type) && (distinctOf(c) ?? 0) >= 2 && (distinctOf(c) ?? 0) <= 5;
  const pairs = [
    [x, y],
    [y, x],
  ].filter(([variant, metric]) => variant && metric && levels(variant) && METRIC_TYPES.has(metric.type));
  // a yes/no column is more likely the metric than the variant
  pairs.sort(([a], [b]) => Number(a?.type === "boolean") - Number(b?.type === "boolean"));
  const [variant, metric] = pairs[0] ?? [];
  return variant && metric ? { variant: variant.name, metric: metric.name } : null;
}

function roleOf(column: Column, numbers: string[]): Role | undefined {
  if (readsAsLevels(column) && !numbers.includes(column.name)) return "cat";
  return ROLE[column.type];
}

export function ComparePage() {
  const phone = useMedia(PHONE);
  const { datasetId } = useParams({ from: "/d/$datasetId/compare" });
  const search = useSearch({ from: "/d/$datasetId/compare" });
  const navigate = useNavigate({ from: "/d/$datasetId/compare" });
  const dataset = useDataset(datasetId);
  const columns = useColumns(datasetId, dataset.data?.status === "ready");

  const usable = useMemo(
    () =>
      (columns.data ?? [])
        .filter((c) => !c.excluded && ROLE[c.type])
        .sort((a, b) => a.position - b.position),
    [columns.data],
  );
  const byName = new Map(usable.map((c) => [c.name, c]));
  const numbers = (search.numbers ?? "").split(",").filter(Boolean);
  const role = (column: Column) => roleOf(column, numbers);
  const x = search.x && byName.get(search.x);
  const y = search.y && byName.get(search.y);
  const bothNumeric = Boolean(x && y && role(x) === "num" && role(y) === "num");
  const grouped = Boolean(
    x && y && [role(x), role(y)].includes("num") && [role(x), role(y)].includes("cat"),
  );
  const z = bothNumeric && search.z ? byName.get(search.z) : undefined;
  const colorAllowed = Boolean(
    x && y && (bothNumeric || grouped || role(x) === "date" || role(y) === "date"),
  );
  // with a number and two categories, the server lays them out (R-CMP-04/05): any colour works
  const color = search.color ? byName.get(search.color) : undefined;
  const toggleNumbers = (name: string) => {
    const next = numbers.includes(name) ? numbers.filter((n) => n !== name) : [...numbers, name];
    set({ numbers: next.length ? next.join(",") : undefined });
  };
  const params =
    x && y && x.name !== y.name
      ? {
          x: x.name,
          y: y.name,
          z: z?.name ?? null,
          color: colorAllowed ? (color?.name ?? null) : null,
          period: search.period,
          agg: search.agg,
          window: search.window,
          align: search.align,
          numbers: numbers.filter((n) => n === x.name || n === y.name || n === color?.name),
        }
      : null;
  const comparison = useCompare(datasetId, params);
  const pair = x && y && x.name !== y.name ? experimentOf(x, y) : null;
  // R-098-06: offered only when the group column is a variant by R-EXP-01, with the rule's reason
  const candidates = useExperimentCandidates(datasetId, Boolean(pair) && dataset.data?.status === "ready");
  const rule = pair ? candidates.data?.variants.find((v) => v.column === pair.variant && v.planned) : undefined;
  const experiment = pair && rule ? { ...pair, why: rule.why } : null;

  const set = (patch: CompareSearch) =>
    void navigate({
      search: (previous) => ({ ...previous, ...patch }),
      replace: true,
    });

  return (
    <div className="flex flex-col gap-5 px-(--gutter) py-6">
      <header className="max-w-prose">
        <h1 className="text-xl font-semibold tracking-tight">
          {t("compare.title")}
        </h1>
        <p className="mt-1 text-ink-2">{t("compare.description")}</p>
      </header>

      {phone ? (
        // 041: on a phone the four pickers fold into one line ("units × revenue"), open while
        // nothing is chosen.
        <ParamPanel
          label={t("compare.settings")}
          summary={[x && y ? `${x.name} × ${y.name}` : t("compare.pickShort"), z?.name, colorAllowed && color?.name]}
          defaultOpen={!x || !y}
          footer={
            <Button
              variant="ghost"
              icon={ArrowLeftRight}
              disabled={!x || !y}
              onClick={() => set({ x: search.y, y: search.x })}
            >
              {t("compare.swap")}
            </Button>
          }
        >
          <div className="relative">
            <ColumnPicker
              label="X"
              value={x ? x.name : ""}
              columns={usable}
              onChange={(name) => set({ x: name })}
            />
            {x && readsAsLevels(x) && (
              <ReadAs column={x} asNumbers={numbers.includes(x.name)} onToggle={() => toggleNumbers(x.name)} />
            )}
          </div>
          <div className="relative">
            <ColumnPicker
              label="Y"
              value={y ? y.name : ""}
              columns={usable}
              onChange={(name) => set({ y: name })}
            />
            {y && readsAsLevels(y) && (
              <ReadAs column={y} asNumbers={numbers.includes(y.name)} onToggle={() => toggleNumbers(y.name)} />
            )}
          </div>
          <ColumnPicker
            label={t("compare.z")}
            value={z?.name ?? ""}
            columns={usable.filter((c) => role(c) === "num")}
            optional
            disabled={!bothNumeric}
            hint={bothNumeric ? undefined : t("compare.zHint")}
            onChange={(name) => set({ z: name || undefined })}
          />
          <ColumnPicker
            label={t("compare.color")}
            value={colorAllowed ? (color?.name ?? "") : ""}
            columns={usable.filter((c) => role(c) === "cat")}
            optional
            disabled={!colorAllowed}
            hint={colorAllowed ? undefined : t("compare.colorHint")}
            onChange={(name) => set({ color: name || undefined })}
          />
        </ParamPanel>
      ) : (
        <div className="flex flex-wrap items-end gap-x-3 gap-y-6 pb-4">
          <div className="relative">
            <ColumnPicker
              label="X"
              value={x ? x.name : ""}
              columns={usable}
              onChange={(name) => set({ x: name })}
            />
            {x && readsAsLevels(x) && (
              <ReadAs column={x} asNumbers={numbers.includes(x.name)} onToggle={() => toggleNumbers(x.name)} />
            )}
          </div>
          <Button
            variant="ghost"
            icon={ArrowLeftRight}
            iconOnly
            aria-label={t("compare.swap")}
            title={t("compare.swap")}
            disabled={!x || !y}
            onClick={() => set({ x: search.y, y: search.x })}
          />
          <div className="relative">
            <ColumnPicker
              label="Y"
              value={y ? y.name : ""}
              columns={usable}
              onChange={(name) => set({ y: name })}
            />
            {y && readsAsLevels(y) && (
              <ReadAs column={y} asNumbers={numbers.includes(y.name)} onToggle={() => toggleNumbers(y.name)} />
            )}
          </div>
          <ColumnPicker
            label={t("compare.z")}
            value={z?.name ?? ""}
            columns={usable.filter((c) => role(c) === "num")}
            optional
            disabled={!bothNumeric}
            hint={bothNumeric ? undefined : t("compare.zHint")}
            onChange={(name) => set({ z: name || undefined })}
          />
          <ColumnPicker
            label={t("compare.color")}
            value={colorAllowed ? (color?.name ?? "") : ""}
            columns={usable.filter((c) => role(c) === "cat")}
            optional
            disabled={!colorAllowed}
            hint={colorAllowed ? undefined : t("compare.colorHint")}
            onChange={(name) => set({ color: name || undefined })}
          />
        </div>

      )}

      {columns.isPending ? (
        <Loading label={t("loading.columns")} className="h-[460px] rounded-lg" />
      ) : !params ? (
        <p className="rounded-lg border border-dashed border-line px-4 py-10 text-center text-ink-2">
          {x && y ? t("compare.differentColumns") : t("compare.pick")}
        </p>
      ) : comparison.isError ? (
        <ErrorNote>{comparison.error.message}</ErrorNote>
      ) : !comparison.data ? (
        <Loading label={t("loading.comparison")} className="h-[460px] rounded-lg" />
      ) : comparison.data.total === 0 ? (
        <NoRows datasetId={datasetId} />
      ) : (
        <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_320px]">
          {comparison.data.rows < 3 ? (
            <p className="rounded-lg border border-line bg-surface-1 px-4 py-10 text-center text-ink-2">
              {comparison.data.message}
            </p>
          ) : (
            <ChartArea
              key={`${params.x}|${params.y}|${params.z}`}
              result={comparison.data}
              time={search}
              onTimeChange={(patch) => set(patch)}
              experiment={experiment}
            />
          )}
          <TestsPanel result={comparison.data} />
        </div>
      )}
    </div>
  );
}

/** "Pclass: 3 values, read as categories · As numbers" under a picker (R-CMP-01/02). */
function ReadAs({ column, asNumbers, onToggle }: { column: Column; asNumbers: boolean; onToggle: () => void }) {
  return (
    <p className="absolute top-full left-0 mt-1 text-[11px] whitespace-nowrap text-ink-3 max-sm:static max-sm:whitespace-normal">
      {asNumbers
        ? t("compare.readAsNumbers")
        : t("compare.readAsLevels", { count: String(distinctOf(column)) })}{" "}
      <button type="button" onClick={onToggle} className="text-accent hover:underline">
        {asNumbers ? t("compare.asCategories") : t("compare.asNumbers")}
      </button>
    </p>
  );
}

function ColumnPicker({
  label,
  value,
  columns,
  onChange,
  optional = false,
  disabled = false,
  hint,
  unavailable,
}: {
  label: string;
  value: string;
  columns: Column[];
  onChange: (name: string) => void;
  optional?: boolean;
  disabled?: boolean;
  hint?: string;
  /** Why a column cannot be picked here (null: it can). */
  unavailable?: (column: Column) => string | null;
}) {
  return (
    <Field label={label}>
      <Select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
        title={hint}
        className="w-48"
      >
        <option value="">
          {optional ? t("compare.none") : t("compare.choose")}
        </option>
        {columns.map((column) => {
          const reason = unavailable?.(column) ?? null;
          return (
            <option key={column.name} value={column.name} disabled={reason !== null} title={reason ?? undefined}>
              {column.name}
              {reason ? ` (${reason})` : ""}
            </option>
          );
        })}
      </Select>
    </Field>
  );
}

type View = "points" | "density" | "3d" | GroupView | TableView | "series";

function viewsFor(result: Comparison): View[] {
  if (result.kind === "scatter")
    return result.z ? ["3d", "points", "density"] : ["points", "density"];
  if (result.kind === "groups") return ["box", "violin", "means"];
  if (result.kind === "contingency") return ["heatmap", "stacked"];
  return ["series"];
}

function pickView(result: Comparison, chosen: View | null): View {
  const views = viewsFor(result);
  return chosen && views.includes(chosen) ? chosen : (views[0] ?? "series");
}

function figureFor(
  result: Comparison,
  view: View,
  regression: boolean,
  time: TimeSearch,
  lowess = false,
) {
  if (view === "3d") return scatter3dFigure(result);
  if (view === "points") return scatterFigure(result, regression, lowess);
  if (view === "density") return densityFigure(result);
  if (view === "box" || view === "violin" || view === "means")
    return groupsFigure(result, view);
  if (view === "heatmap" || view === "stacked")
    return contingencyFigure(result, view);
  return result.kind === "date_counts"
    ? dateCountsFigure(result)
    : timelineFigure(result, shownLayers(time.show));
}

function ChartArea({
  result,
  time,
  onTimeChange,
  experiment,
}: {
  result: Comparison;
  time: TimeSearch;
  onTimeChange: (patch: TimeSearch) => void;
  experiment: { variant: string; metric: string; why: string } | null;
}) {
  const [chosen, setChosen] = useState<View | null>(null);
  const [regression, setRegression] = useState(true);
  // LOWESS (064, R-SM-12): on by the rule until the user ticks or unticks it for this pair.
  const pair = `${result.x}|${result.y}`;
  const [lowessChoice, setLowessChoice] = useState<{ pair: string; on: boolean } | null>(null);
  const lowess =
    lowessChoice?.pair === pair ? lowessChoice.on : Boolean(result.lowess?.suggested);
  const view = pickView(result, chosen);
  const figure = useMemo(
    () => figureFor(result, view, regression, time, lowess),
    [result, view, regression, time, lowess],
  );
  const views = viewsFor(result);

  return (
    <section className="min-w-0 rounded-lg border border-line bg-surface-1 p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        {views.length > 1 ? (
          <SegmentedControl
            label={t("compare.view")}
            value={view}
            options={views.map((value) => ({
              value,
              label: t(`compare.view.${value}`),
            }))}
            onChange={setChosen}
          />
        ) : (
          <span className="text-xs text-ink-3">
            {result.time
              ? timeSummary(result)
              : result.step
                ? t("compare.step", { step: result.step })
                : ""}
          </span>
        )}
        {result.time && <AnalyzeLink result={result} time={time} />}
        {experiment && <ExperimentLink {...experiment} />}
        {result.kind === "scatter" && !result.z && <ControlsLink x={result.x} y={result.y} />}
        {view === "points" && result.regression && (
          <Checkbox
            label={t("compare.regression", {
              r2:
                result.regression.r2 === null
                  ? "—"
                  : result.regression.r2.toFixed(3),
            })}
            checked={regression}
            onChange={(event) => setRegression(event.target.checked)}
            className="text-xs text-ink-2"
          />
        )}
        {view === "points" && result.lowess && (
          <span className="inline-flex items-center gap-1">
            <Checkbox
              label={t("compare.lowess")}
              checked={lowess}
              onChange={(event) => setLowessChoice({ pair, on: event.target.checked })}
              className="text-xs text-ink-2"
            />
            <InfoTip text={result.lowess.reason} label={t("compare.lowessWhy")} />
          </span>
        )}
      </div>
      {result.time && (
        <div className="mb-2">
          <TimeOptions result={result} search={time} onChange={onTimeChange} />
          {(result.time.first_partial || result.time.last_partial) &&
            (result.time.aggregate === "sum" ||
              result.time.aggregate === "count") && (
              <p className="mt-1 text-xs text-ink-3">
                {t(
                  result.time.first_partial && result.time.last_partial
                    ? "time.partial.both"
                    : result.time.first_partial
                      ? "time.partial.first"
                      : "time.partial.last",
                  {
                    period: t(
                      `time.period.${result.time.period}` as MessageKey,
                    ),
                  },
                )}
              </p>
            )}
          {result.time.coarsened && (
            <p className="mt-1 text-xs text-ink-3">
              {t("time.coarsened", {
                period: t(`time.period.${result.time.period}` as MessageKey),
              })}
            </p>
          )}
        </div>
      )}
      <Plot
        figure={figure}
        height={HEIGHT}
        label={t("compare.chartLabel", { x: result.x, y: result.y })}
      />
      {result.time && <TimeTable result={result} />}
      {result.kind === "groups" && (
        <>
          <GroupIntervals result={result} />
          <PairwiseDifferences result={result} />
        </>
      )}
      {view === "density" && (
        <p className="mt-1.5 text-xs text-ink-3">{t("compare.densityNote")}</p>
      )}
      {(view === "points" || view === "3d") && result.sample && (
        <p className="mt-1.5 text-xs text-ink-3">
          {t("compare.sampleNote", {
            sample: result.sample.toLocaleString("en-US"),
            rows: result.rows.toLocaleString("en-US"),
          })}
        </p>
      )}
    </section>
  );
}

/** Opens the Time series page on the same date, value, period, aggregate and group. */
function AnalyzeLink({
  result,
  time,
}: {
  result: Comparison;
  time: TimeSearch;
}) {
  const { datasetId } = useParams({ from: "/d/$datasetId/compare" });
  const date = result.roles[result.x] === "date" ? result.x : result.y;
  const value = date === result.x ? result.y : result.x;
  return (
    <Link
      to={"/d/$datasetId/timeseries" as never}
      params={{ datasetId } as never}
      search={
        {
          time: date,
          value,
          group: result.color ?? undefined,
          period: time.period,
          agg: time.agg,
          window: time.window,
          align: time.align,
        } as never
      }
      className="text-xs text-accent hover:underline"
    >
      {t("time.analyze")}
    </Link>
  );
}

/** 068: Y on X in the Regression tab, where the strongest links of Y can join as controls. */
function ControlsLink({ x, y }: { x: string; y: string }) {
  const { datasetId } = useParams({ from: "/d/$datasetId/compare" });
  return (
    <Link
      to={"/d/$datasetId/regression" as never}
      params={{ datasetId } as never}
      search={{ target: y, columns: x, focus: x } as never}
      className="text-xs text-accent hover:underline"
    >
      {t("compare.addControls")}
    </Link>
  );
}

/** Opens the Experiments module with this column as the variant and the other as the metric;
 *  the info says why the column reads as a variant (R-EXP-01). */
function ExperimentLink({ variant, metric, why }: { variant: string; metric: string; why: string }) {
  const { datasetId } = useParams({ from: "/d/$datasetId/compare" });
  return (
    <span className="inline-flex items-center gap-1">
      <Link
        to={"/d/$datasetId/experiments" as never}
        params={{ datasetId } as never}
        search={{ variant, metric } as never}
        className="text-xs text-accent hover:underline"
      >
        {t("compare.analyzeExperiment")}
      </Link>
      {why && <InfoTip text={`R-EXP-01 · ${why}`} label={t("compare.experimentWhy")} />}
    </span>
  );
}
