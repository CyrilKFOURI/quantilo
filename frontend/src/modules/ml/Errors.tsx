import { Link } from "@tanstack/react-router";
import { ArrowRight } from "lucide-react";
import { useState } from "react";

import type { MlErrorPrediction, MlErrors, MlFairness, MlRun, MlSlice } from "../../api/client";
import type { Condition } from "../../api/filters";
import { useMlErrors, useMlFairness } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { Plot } from "../../charts/Plot";
import { formatP } from "../../features/variable/normality";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote, Select } from "../../ui/fields";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { RowExplanation } from "./Explain";
import { errorByColumnFigure } from "./errorFigures";
import { Panel } from "./ResultParts";
import { Loading } from "../../ui/Loading";

const SMALL_MULTIPLES = 6;
const pct = (value: number | null | undefined, digits = 0) =>
  value === null || value === undefined ? "—" : `${(value * 100).toFixed(digits)} %`;

/** The error as the analysis measures it: a share of wrong predictions, or a number. */
function errorValue(value: number, measure: string): string {
  return measure === "misclassified" ? pct(value) : formatNumber(value);
}

function Section({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <section className="space-y-2" aria-label={title}>
      <h3 className="text-[13px] font-medium text-ink-1">{title}</h3>
      {hint && <p className="max-w-prose text-xs text-ink-2">{hint}</p>}
      {children}
    </section>
  );
}

/** Where the model is wrong (036): subgroups, hints, error by column, worst predictions, groups
 *  of a sensitive column. Computed on the run's held-out predictions only. */
export function ErrorsTab({ run, datasetId }: { run: MlRun; datasetId: string }) {
  const [measure, setMeasure] = useState<string | null>(null);
  const found = useMlErrors(datasetId, run.id, measure);
  if (found.isError) return <ErrorNote>{found.error.message}</ErrorNote>;
  if (!found.data)
    return <Loading label={t("ml.errors.computing")} className="h-40 rounded-md" />;
  const errors = found.data;
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <p className="min-w-0 flex-1 text-xs text-ink-3">
          {t("ml.errors.summary", {
            rows: errors.rows.toLocaleString("en-US"),
            source: errors.source,
            label: errors.label,
            overall: errorValue(errors.overall, errors.measure),
            tested: errors.tested.toLocaleString("en-US"),
            columns: String(errors.columns),
          })}
        </p>
        {errors.measures.length > 1 && (
          <SegmentedControl
            label={t("ml.errors.measure")}
            value={errors.measure}
            options={errors.measures.map((value) => ({ value, label: t(`ml.errors.measure.${value}` as MessageKey) }))}
            onChange={setMeasure}
          />
        )}
      </div>
      <Slices errors={errors} datasetId={datasetId} />
      {errors.hints.length > 0 && (
        <Section title={t("ml.errors.hints")} hint={t("ml.errors.hintsHint")}>
          <ul className="space-y-1 text-[13px] text-ink-1">
            {errors.hints.map((h) => (
              <li key={h.text} className="border-l-2 border-line pl-2">
                {h.text}
              </li>
            ))}
          </ul>
        </Section>
      )}
      <ByColumn errors={errors} />
      <Worst errors={errors} run={run} datasetId={datasetId} />
      <Groups run={run} datasetId={datasetId} />
    </div>
  );
}

function Slices({ errors, datasetId }: { errors: MlErrors; datasetId: string }) {
  const [open, setOpen] = useState<number | null>(null);
  return (
    <Section title={t("ml.errors.slices")} hint={t("ml.errors.slicesHint")}>
      {errors.slices.length === 0 ? (
        <p className="text-[13px] text-ink-2">{errors.message}</p>
      ) : (
        <ol className="divide-y divide-line border-y border-line">
          {errors.slices.map((slice, i) => (
            <li key={slice.where}>
              <button
                type="button"
                onClick={() => setOpen(open === i ? null : i)}
                aria-expanded={open === i}
                className="flex w-full items-baseline gap-3 py-2 text-left hover:bg-surface-1"
              >
                <span className="w-5 shrink-0 text-xs text-ink-3 tabular-nums">{i + 1}</span>
                <span className="min-w-0 flex-1 text-[13px] text-ink-1">{slice.sentence}</span>
                <span className="shrink-0 text-xs text-ink-2 tabular-nums" title={t("ml.errors.ratioHint")}>
                  ×{slice.ratio.toFixed(1)}
                </span>
              </button>
              {open === i && <SliceDetail slice={slice} errors={errors} datasetId={datasetId} />}
            </li>
          ))}
        </ol>
      )}
    </Section>
  );
}

function SliceDetail({ slice, errors, datasetId }: { slice: MlSlice; errors: MlErrors; datasetId: string }) {
  // the run's own filters, then the subgroup's conditions: the same rows on the other pages
  const conditions: Condition[] = [...errors.conditions, ...slice.conditions];
  const search = { filters: { dataset: datasetId, conditions } };
  const regression = errors.task === "regression";
  return (
    <div className="space-y-2 pb-3 pl-8">
      <p className="text-xs text-ink-3">
        {t("ml.errors.sliceFacts", {
          rows: slice.rows.toLocaleString("en-US"),
          q: formatP(slice.q_value).startsWith("<") ? `q ${formatP(slice.q_value)}` : `q = ${formatP(slice.q_value)}`,
          by: t(`ml.errors.foundBy.${slice.found_by}` as MessageKey),
        })}
      </p>
      <div className="flex flex-wrap items-center gap-4 text-xs">
        <Link to="/d/$datasetId/data" params={{ datasetId }} search={search} className="inline-flex items-center gap-1 text-accent hover:underline">
          {t("ml.errors.openData")}
          <ArrowRight size={12} strokeWidth={1.75} aria-hidden />
        </Link>
        <Link to="/d/$datasetId/compare" params={{ datasetId }} search={search} className="inline-flex items-center gap-1 text-accent hover:underline">
          {t("ml.errors.openCompare")}
          <ArrowRight size={12} strokeWidth={1.75} aria-hidden />
        </Link>
        <span className="text-ink-3">{t("ml.errors.allRows")}</span>
      </div>
      <PredictionTable rows={slice.examples} regression={regression} label={t("ml.errors.examples")} />
    </div>
  );
}

function PredictionTable({
  rows,
  regression,
  label,
  selected,
  onPick,
}: {
  rows: MlErrorPrediction[];
  regression: boolean;
  label: string;
  selected?: number | null;
  onPick?: (row: number) => void;
}) {
  return (
    <div className="max-h-80 overflow-auto rounded-md border border-line">
      <table className="w-full border-collapse text-[13px] tabular-nums" aria-label={label}>
        <thead className="sticky top-0 bg-surface-2 text-left text-xs text-ink-3">
          <tr>
            <th className="px-3 py-1.5 font-normal">{t("ml.row")}</th>
            <th className="px-3 py-1.5 font-normal">{t("ml.actual")}</th>
            <th className="px-3 py-1.5 font-normal">{t("ml.predicted")}</th>
            <th className="px-3 py-1.5 text-right font-normal">{regression ? t("ml.errors.absoluteError") : t("ml.errors.actualProbability")}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((p) => (
            <tr
              key={p.row}
              onClick={onPick ? () => onPick(p.row) : undefined}
              aria-selected={onPick ? selected === p.row : undefined}
              className={`border-t border-line ${onPick ? "cursor-pointer hover:bg-surface-2" : ""} ${selected === p.row ? "bg-accent-soft" : ""}`}
            >
              <td className="px-3 py-1.5 text-ink-3">
                {onPick ? (
                  <button type="button" className="underline-offset-2 hover:underline" aria-label={t("ml.explain.explainRow", { row: String(p.row + 1) })}>
                    {p.row + 1}
                  </button>
                ) : (
                  p.row + 1
                )}
              </td>
              <td className="px-3 py-1.5">{p.actual}</td>
              <td className="px-3 py-1.5">{p.predicted}</td>
              <td className="px-3 py-1.5 text-right">{regression ? formatNumber(p.error) : pct(p.probability, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ByColumn({ errors }: { errors: MlErrors }) {
  const [all, setAll] = useState(false);
  const columns = all ? errors.by_column : errors.by_column.slice(0, SMALL_MULTIPLES);
  if (!errors.by_column.length) return null;
  return (
    <Section title={t("ml.errors.byColumn")} hint={t("ml.errors.byColumnHint", { label: errors.label })}>
      <div className="grid gap-x-5 gap-y-4 @2xl:grid-cols-2 @6xl:grid-cols-3">
        {columns.map((column) => {
          const title = `${column.column}${column.excluded ? ` · ${t("ml.errors.excluded")}` : ""}`;
          return (
            <Panel key={column.column} title={title}>
              <Plot figure={errorByColumnFigure(column, errors.overall, errors.label)} height={200} label={t("ml.errors.byColumnOf", { column: column.column })} />
            </Panel>
          );
        })}
      </div>
      {errors.by_column.length > SMALL_MULTIPLES && (
        <Button variant="ghost" onClick={() => setAll(!all)}>
          {all ? t("ml.errors.fewerColumns") : t("ml.errors.allColumns", { count: String(errors.by_column.length) })}
        </Button>
      )}
    </Section>
  );
}

function Worst({ errors, run, datasetId }: { errors: MlErrors; run: MlRun; datasetId: string }) {
  const [row, setRow] = useState<number | null>(null);
  return (
    <Section title={t("ml.errors.worst")} hint={t("ml.errors.worstHint", { count: String(errors.worst.length) })}>
      {row !== null && <RowExplanation datasetId={datasetId} run={run} row={row} />}
      <PredictionTable rows={errors.worst} regression={errors.task === "regression"} label={t("ml.errors.worst")} selected={row} onPick={setRow} />
    </Section>
  );
}

function Groups({ run, datasetId }: { run: MlRun; datasetId: string }) {
  const [column, setColumn] = useState<string | null>(null);
  const found = useMlFairness(datasetId, run.id, column);
  // an identifier or free text cannot be cut into groups
  const columns = run.preparation.decisions.filter((d) => d.column !== run.target && d.type !== "identifier" && d.type !== "text").map((d) => d.column);
  return (
    <Section title={t("ml.errors.groups")} hint={t("ml.errors.groupsHint")}>
      <label className="flex items-center gap-2 text-xs text-ink-2">
        {t("ml.errors.sensitive")}
        <Select aria-label={t("ml.errors.sensitive")} value={column ?? ""} onChange={(e) => setColumn(e.target.value || null)} className="h-7 text-xs">
          <option value="">{t("ml.errors.pickColumn")}</option>
          {columns.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </Select>
      </label>
      {found.isError && <ErrorNote>{found.error.message}</ErrorNote>}
      {found.data && <GroupTable groups={found.data} />}
    </Section>
  );
}

function GroupTable({ groups }: { groups: MlFairness }) {
  const shares = groups.metrics.some((m) => m === "error_rate" || m === "positive_rate" || m === "recall");
  const show = (value: number | null | undefined) => (value === null || value === undefined ? "—" : shares ? pct(value, 1) : formatNumber(value));
  return (
    <div className="space-y-1">
      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full border-collapse text-[13px] tabular-nums" aria-label={t("ml.errors.groups")}>
          <thead className="bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th className="px-3 py-1.5 font-normal">{groups.column}</th>
              <th className="px-3 py-1.5 text-right font-normal">{t("ml.errors.rows")}</th>
              {groups.metrics.map((m) => (
                <th key={m} className="px-3 py-1.5 text-right font-normal">
                  {groups.labels[m]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {groups.groups.map((g) => (
              <tr key={g.group} className={`border-t border-line ${g.few ? "text-ink-3" : ""}`}>
                <td className="px-3 py-1.5">{g.group}</td>
                <td className="px-3 py-1.5 text-right text-ink-2">{g.rows.toLocaleString("en-US")}</td>
                {groups.metrics.map((m) => (
                  <td key={m} className="px-3 py-1.5 text-right">
                    {show(g.values[m])}
                  </td>
                ))}
              </tr>
            ))}
            <tr className="border-t border-line font-medium text-ink-1">
              <td className="px-3 py-1.5">{t("ml.errors.largestGap")}</td>
              <td />
              {groups.metrics.map((m) => (
                <td key={m} className="px-3 py-1.5 text-right">
                  {show(groups.gaps[m])}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
      <p className="text-xs text-ink-3">
        {groups.positive ? `${t("ml.errors.positiveClass", { positive: groups.positive })} · ` : ""}
        {t("ml.errors.displayOnly")}
      </p>
    </div>
  );
}
