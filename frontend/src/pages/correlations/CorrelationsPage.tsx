import { Link, useNavigate, useParams } from "@tanstack/react-router";
import { Columns3 } from "lucide-react";
import { useMemo, useState } from "react";

import type { AnalysisType, AssociationMatrix, Column, CorrelationMethod, Correction } from "../../api/client";
import { useColumns, useCorrelations, useDataset, useDerivedColumns } from "../../api/hooks";
import { derivedNote } from "../../features/derived/derived";
import { Plot, type PointClick } from "../../charts/Plot";
import { EvidenceBadge, evidenceRank } from "../../features/correlations/evidence";
import { FocusPanel } from "../../features/correlations/FocusPanel";
import { matrixFigure, METRIC_LABELS } from "../../features/correlations/figures";
import { TYPE_META } from "../../features/columns/columnTypes";
import { InsightsList } from "../../features/correlations/InsightsList";
import { StructureSection } from "../../features/correlations/StructureSection";
import { MultipleTestingNote, NoLongerHolds, noLongerHolds } from "../../features/inference/MultipleTesting";
import { formatP } from "../../features/variable/normality";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { useStoredState } from "../../lib/useStoredState";
import { Checkbox, ErrorNote, Select } from "../../ui/fields";
import { Popover } from "../../ui/Popover";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { Loading } from "../../ui/Loading";

const ELIGIBLE = new Set(["numeric_continuous", "numeric_discrete", "categorical_nominal", "categorical_ordinal", "boolean"]);
const PAIR_ROWS = 15;
const CORRECTIONS: { value: Correction; label: string }[] = [
  { value: "auto", label: "Auto" },
  { value: "bh", label: "Benjamini–Hochberg" },
  { value: "holm", label: "Holm" },
  { value: "bonferroni", label: "Bonferroni" },
];

export function CorrelationsPage() {
  const { datasetId } = useParams({ from: "/d/$datasetId/correlations" });
  const navigate = useNavigate();
  const dataset = useDataset(datasetId);
  const ready = dataset.data?.status === "ready";
  const columns = useColumns(datasetId, ready);
  // 058: derived columns can be ticked in the chooser, never in the matrix by default (R-DER-07)
  const derived = useDerivedColumns(datasetId, ready);
  const key = (name: string) => `de.correlations.${datasetId}.${name}`;
  const [method, setMethod] = useStoredState<CorrelationMethod>(key("method"), "pearson");
  const [clustered, setClustered] = useStoredState(key("clustered"), true);
  const [chosen, setChosen] = useStoredState<string[] | null>(key("columns"), null);
  const [correction, setCorrection] = useStoredState<Correction>(key("correction"), "auto");
  const [focus, setFocus] = useState("");
  const correlations = useCorrelations(datasetId, method, chosen, ready, correction);
  const matrix = correlations.data?.matrix;

  const eligible = useMemo(
    () => (columns.data ?? []).filter((c) => !c.excluded && ELIGIBLE.has(c.type)).map((c) => c.name),
    [columns.data],
  );
  const order = useMemo(
    () => (matrix ? (clustered ? matrix.clustered : matrix.columns.map((_, i) => i)) : []),
    [matrix, clustered],
  );
  const figure = useMemo(() => (matrix ? matrixFigure(matrix, order) : null), [matrix, order]);
  const derivedEligible = useMemo(
    () => (derived.data ?? []).filter((c) => ELIGIBLE.has(c.type)),
    [derived.data],
  );
  const types = useMemo(
    () => new Map([...(columns.data ?? []), ...(derived.data ?? [])].map((c) => [c.name, c.type])),
    [columns.data, derived.data],
  );
  const typeOf = (name: string): AnalysisType | undefined => types.get(name);

  const openPair = (point: PointClick) => {
    const x = String(point.x);
    const y = String(point.y);
    if (x === y) return;
    void navigate({
      to: "/d/$datasetId/compare",
      params: { datasetId },
      search: (previous: Record<string, unknown>) => ({ ...previous, x, y, z: undefined, color: undefined }),
    });
  };

  return (
    <div className="flex flex-col gap-5 px-(--gutter) py-6">
      <header className="max-w-prose">
        <h1 className="text-xl font-semibold tracking-tight">{t("correlations.title")}</h1>
        <p className="mt-1 text-ink-2">{t("correlations.description")}</p>
      </header>

      {correlations.isError && <ErrorNote>{correlations.error.message}</ErrorNote>}
      {matrix?.multiple && <MultipleTestingNote multiple={matrix.multiple} className="max-w-prose" />}
      {correlations.data && <InsightsList datasetId={datasetId} insights={correlations.data.insights} typeOf={typeOf} />}

      <div className="flex flex-wrap items-center gap-3">
        <SegmentedControl
          label={t("correlations.method")}
          value={method}
          options={[
            { value: "pearson", label: "Pearson" },
            { value: "spearman", label: "Spearman" },
          ]}
          onChange={setMethod}
        />
        <label className="inline-flex items-center gap-2 text-[13px] text-ink-2">
          {t("correlations.correction")}
          <Select
            value={correction}
            onChange={(event) => setCorrection(event.target.value as Correction)}
            title={t("correlations.correctionHint")}
            className="w-44"
          >
            {CORRECTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.value === "auto" ? t("correlations.correctionAuto") : option.label}
              </option>
            ))}
          </Select>
        </label>
        <Checkbox label={t("correlations.cluster")} checked={clustered} onChange={(event) => setClustered(event.target.checked)} className="text-[13px] text-ink-2" />
        <ColumnChooser eligible={eligible} derived={derivedEligible} chosen={chosen} onChange={setChosen} />
        {matrix && (
          <span className="text-xs text-ink-3">
            {matrix.sampled
              ? t("correlations.sampled", { rows: formatCount(matrix.rows), total: formatCount(matrix.total) })
              : t("correlations.rows", { rows: formatCount(matrix.rows) })}
          </span>
        )}
      </div>

      {!matrix ? (
        <Loading label={t("loading.correlations")} className="h-[520px] rounded-lg" />
      ) : matrix.columns.length < 2 ? (
        <p className="rounded-lg border border-dashed border-line px-4 py-10 text-center text-ink-2">{t("correlations.tooFew")}</p>
      ) : (
        <>
          <section className="rounded-lg border border-line bg-surface-1 p-3">
            {figure && (
              <Plot
                figure={figure}
                height={Math.min(Math.max(420, matrix.columns.length * 30 + 140), 1400)}
                label={t("correlations.matrix")}
                onPointClick={openPair}
              />
            )}
            <p className="mt-1.5 text-xs text-ink-3">{t("correlations.legend")}</p>
            {(matrix.excluded.length > 0 || matrix.truncated > 0) && <Excluded matrix={matrix} />}
          </section>
          <FocusPanel datasetId={datasetId} matrix={matrix} focus={focus} onFocus={setFocus} typeOf={typeOf} />
        </>
      )}

      {matrix && matrix.pairs.length > 0 && <StrongestPairs datasetId={datasetId} matrix={matrix} typeOf={typeOf} />}
      {correlations.data?.structure?.proposed && (
        <StructureSection datasetId={datasetId} rule={correlations.data.structure} columns={chosen} />
      )}
    </div>
  );
}

function ColumnChooser({
  eligible,
  derived,
  chosen,
  onChange,
}: {
  eligible: string[];
  /** 058: derived columns, unticked until the user ticks them (R-DER-07). */
  derived: Column[];
  chosen: string[] | null;
  onChange: (columns: string[] | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const selected = chosen ?? eligible;
  const toggle = (name: string) => {
    const next = selected.includes(name) ? selected.filter((item) => item !== name) : [...selected, name];
    const stored = eligible.filter((item) => next.includes(item));
    const extra = derived.map((c) => c.name).filter((item) => next.includes(item));
    onChange(stored.length === eligible.length && !extra.length ? null : [...stored, ...extra]);
  };
  return (
    <Popover
      open={open}
      onOpenChange={setOpen}
      label={t("correlations.columns")}
      align="start"
      trigger={
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          className="inline-flex h-(--control-h) items-center gap-1.5 rounded-md border border-line bg-surface-1 px-2.5 text-[13px] text-ink-2 hover:bg-surface-2 hover:text-ink-1"
        >
          <Columns3 size={14} strokeWidth={1.5} aria-hidden />
          {t("correlations.columnsCount", { count: String(selected.length), total: String(eligible.length) })}
        </button>
      }
    >
      <div className="max-h-80 overflow-y-auto p-2">
        <div className="mb-1 flex gap-3 px-1 text-xs">
          <button type="button" className="text-ink-2 hover:text-ink-1" onClick={() => onChange(null)}>
            {t("filters.all")}
          </button>
          <button type="button" className="text-ink-2 hover:text-ink-1" onClick={() => onChange([])}>
            {t("filters.none")}
          </button>
        </div>
        {eligible.map((name) => (
          <label key={name} className="flex cursor-pointer items-center gap-2 rounded px-1 py-1 text-[13px] hover:bg-surface-2">
            <input type="checkbox" className="size-3.5 accent-(--accent)" checked={selected.includes(name)} onChange={() => toggle(name)} />
            <span className="truncate font-mono text-xs">{name}</span>
          </label>
        ))}
        {derived.length > 0 && (
          <>
            <p className="mt-2 border-t border-line px-1 pt-2 text-xs text-ink-3">{t("correlations.derived")}</p>
            {derived.map((column) => (
              <label
                key={column.name}
                title={derivedNote(column) ?? undefined}
                className="flex cursor-pointer items-center gap-2 rounded px-1 py-1 text-[13px] hover:bg-surface-2"
              >
                <input
                  type="checkbox"
                  className="size-3.5 accent-(--accent)"
                  checked={selected.includes(column.name)}
                  onChange={() => toggle(column.name)}
                />
                <span className="truncate font-mono text-xs">{column.name}</span>
              </label>
            ))}
          </>
        )}
      </div>
    </Popover>
  );
}

function Excluded({ matrix }: { matrix: AssociationMatrix }) {
  return (
    <details className="mt-2 text-xs text-ink-3">
      <summary className="cursor-pointer hover:text-ink-1">
        {t("correlations.excluded", { count: String(matrix.excluded.length + matrix.truncated) })}
      </summary>
      <ul className="mt-1 space-y-0.5">
        {matrix.excluded.map((item) => (
          <li key={item.name}>
            <span className="font-mono text-ink-2">{item.name}</span> · {item.reason}
          </li>
        ))}
        {matrix.truncated > 0 && <li>{t("correlations.truncated", { count: String(matrix.truncated) })}</li>}
      </ul>
    </details>
  );
}

function StrongestPairs({
  datasetId,
  matrix,
  typeOf,
}: {
  datasetId: string;
  matrix: AssociationMatrix;
  typeOf: (name: string) => AnalysisType | undefined;
}) {
  // Best evidence first: a strong-looking value on few rows per group can be chance.
  const pairs = [...matrix.pairs]
    .sort((a, b) => evidenceRank(b.evidence) - evidenceRank(a.evidence) || Math.abs(b.value) - Math.abs(a.value))
    .slice(0, PAIR_ROWS);
  const tone = (name: string) => {
    const type = typeOf(name);
    return type ? TYPE_META[type].tone : "text-ink-1";
  };
  return (
    <section>
      <div className="mb-2 flex flex-wrap items-baseline gap-x-3">
        <h2 className="text-sm font-semibold">{t("correlations.pairs")}</h2>
        <span className="text-xs text-ink-3">{t("correlations.pairsHint")}</span>
      </div>
      <div className="overflow-x-auto rounded-md border border-line">
        {/* 041: a phone keeps the pair, the value, adj. p, the evidence and View (measure, p and n hidden). */}
        <table className="w-full border-collapse text-[13px] max-sm:[&_tr>:nth-child(3)]:hidden max-sm:[&_tr>:nth-child(5)]:hidden max-sm:[&_tr>:nth-child(8)]:hidden">
          <thead className="bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th scope="col" className="px-3 py-1.5 font-normal">A</th>
              <th scope="col" className="px-3 py-1.5 font-normal">B</th>
              <th scope="col" className="px-3 py-1.5 font-normal">{t("correlations.metric")}</th>
              <th scope="col" className="px-3 py-1.5 text-right font-normal">{t("correlations.value")}</th>
              <th scope="col" className="px-3 py-1.5 text-right font-normal">p</th>
              <th scope="col" className="px-3 py-1.5 text-right font-normal">adj. p</th>
              <th scope="col" className="px-3 py-1.5 font-normal">{t("evidence.title")}</th>
              <th scope="col" className="px-3 py-1.5 text-right font-normal">n</th>
              <th scope="col" className="px-3 py-1.5 font-normal"><span className="sr-only">{t("insights.view")}</span></th>
            </tr>
          </thead>
          <tbody>
            {pairs.map((pair) => {
              return (
                <tr key={`${pair.a}|${pair.b}`} className={`border-t border-line ${pair.evidence === "none" ? "opacity-55" : ""}`}>
                  <td className={`px-3 py-1.5 font-mono text-xs font-medium ${tone(pair.a)}`}>{pair.a}</td>
                  <td className={`px-3 py-1.5 font-mono text-xs font-medium ${tone(pair.b)}`}>{pair.b}</td>
                  <td className="px-3 py-1.5 text-ink-2">{METRIC_LABELS[pair.metric]}</td>
                  <td className={`px-3 py-1.5 text-right tabular-nums ${pair.value < 0 ? "text-bad" : "text-ink-1"}`}>{pair.value.toFixed(3)}</td>
                  <td className="px-3 py-1.5 text-right text-ink-3 tabular-nums">{formatP(pair.p_value)}</td>
                  <td className="px-3 py-1.5 text-right text-ink-2 tabular-nums">
                    <span className="whitespace-nowrap">{formatP(pair.q_value ?? null)}</span>
                    {noLongerHolds(pair.p_value, pair.q_value) && <NoLongerHolds />}
                  </td>
                  <td className="px-3 py-1.5">
                    <EvidenceBadge level={pair.evidence} />
                  </td>
                  <td className="px-3 py-1.5 text-right text-ink-3 tabular-nums">{formatCount(pair.n)}</td>
                  <td className="px-3 py-1.5 text-right">
                    <Link
                      to="/d/$datasetId/compare"
                      params={{ datasetId }}
                      search={(previous) => ({ ...previous, x: pair.a, y: pair.b, z: undefined, color: undefined })}
                      className="text-xs text-accent hover:underline"
                    >
                      {t("insights.view")}
                    </Link>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
