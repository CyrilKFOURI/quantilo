import { ChevronRight } from "lucide-react";
import { useMemo, useState } from "react";

import type { Factors, Structure, StructureRule } from "../../api/client";
import { useFactors, useStructure } from "../../api/hooks";
import { Plot, type Figure } from "../../charts/Plot";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Loading } from "../../ui/Loading";
import { formatP } from "../variable/normality";
import { Metrics, type Metric } from "../variable/Metrics";
import { biplotFigure, loadingsFigure, screeFigure } from "./structureFigures";

const STRONG = 0.5;
/** Two decimals, without a "-0.00". */
const fixed = (value: number) => (Math.abs(value) < 0.005 ? 0 : value).toFixed(2);

/** Structure (078), Advanced: shown when the rule proposes it, folded, and computed only once
 * opened. Its More fold (factor analysis, KMO, Bartlett) is computed only once opened too. */
export function StructureSection({ datasetId, rule, columns }: { datasetId: string; rule: StructureRule; columns: string[] | null }) {
  const [open, setOpen] = useState(false);
  return (
    <details className="group rounded-lg border border-line bg-surface-1" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary className="flex min-h-11 cursor-pointer list-none items-start gap-2 px-3 py-2.5 select-none hover:bg-surface-2 [&::-webkit-details-marker]:hidden">
        <ChevronRight size={16} strokeWidth={1.5} aria-hidden className="mt-0.5 shrink-0 text-ink-3 transition-transform group-open:rotate-90" />
        <span className="min-w-0">
          <span className="flex flex-wrap items-baseline gap-x-2">
            <span className="text-sm font-semibold text-ink-1">{t("correlations.structure.title")}</span>
            <span className="text-xs text-ink-3">{t("correlations.structure.level")}</span>
          </span>
          <span className="mt-0.5 block text-xs text-ink-2">{rule.reason}</span>
        </span>
      </summary>
      {open && (
        <div className="border-t border-line px-3 py-3">
          <StructureBody datasetId={datasetId} columns={columns} />
        </div>
      )}
    </details>
  );
}

function StructureBody({ datasetId, columns }: { datasetId: string; columns: string[] | null }) {
  const structure = useStructure(datasetId, columns, true);
  if (structure.isError) return <p className="text-bad">{structure.error.message}</p>;
  if (!structure.data) return <Loading label={t("correlations.structure.loading")} className="h-72 rounded-md" />;
  if (structure.data.reason) return <p className="text-ink-2">{structure.data.reason}</p>;
  return <StructureResults datasetId={datasetId} columns={columns} structure={structure.data} />;
}

export function StructureResults({ datasetId, columns, structure }: { datasetId: string; columns: string[] | null; structure: Structure }) {
  const scree = useMemo(() => screeFigure(structure), [structure]);
  const loadings = useMemo(() => loadingsFigure(structure), [structure]);
  const biplot = useMemo(() => biplotFigure(structure), [structure]);
  const [more, setMore] = useState(false);
  const kept = structure.retained;
  const rows = structure.sampled
    ? t("correlations.structure.sampled", { rows: formatCount(structure.rows), total: formatCount(structure.total) })
    : t("correlations.structure.rows", { rows: formatCount(structure.rows) });
  return (
    <div className="space-y-4">
      <div className="max-w-prose space-y-1">
        <p className="font-medium text-ink-1">{structure.conclusion}</p>
        <p className="text-xs text-ink-3">
          {rows} · {t("correlations.structure.parallel", { permutations: formatCount(structure.permutations), rows: formatCount(structure.parallel_rows) })}
        </p>
        {structure.notes.map((note) => (
          <p key={note} className="text-xs text-ink-3">
            {note}
          </p>
        ))}
        {structure.left_out.length > 0 && (
          <p className="text-xs text-ink-3">
            {t("correlations.structure.leftOut")}{" "}
            {structure.left_out.map((item, i) => (
              <span key={item.name}>
                {i > 0 && "; "}
                <span className="font-mono text-ink-2">{item.name}</span> ({item.reason})
              </span>
            ))}
          </p>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <StructureChart
          caption={
            kept > 0
              ? t("correlations.structure.screeCaption", { count: String(kept), share: `${Math.round((structure.components[kept - 1]?.cumulative ?? 0) * 100)}%` })
              : t("correlations.structure.screeNone")
          }
          figure={scree}
          height={280}
          label={t("correlations.structure.scree")}
        />
        <StructureChart
          caption={t("correlations.structure.loadingsCaption")}
          figure={loadings}
          height={Math.min(Math.max(280, structure.columns.length * 22 + 90), 1000)}
          label={t("correlations.structure.loadings")}
        />
        {structure.biplot && (
          <StructureChart
            caption={t("correlations.structure.biplotCaption", { points: formatCount(structure.biplot.points), of: formatCount(structure.biplot.of) })}
            figure={biplot}
            height={280}
            label={t("correlations.structure.biplot")}
          />
        )}
      </div>

      <details className="border-t border-line pt-3" onToggle={(event) => setMore(event.currentTarget.open)}>
        <summary className="inline-flex min-h-11 cursor-pointer items-center text-[13px] text-accent select-none hover:underline sm:min-h-0">
          {t("correlations.structure.more")}
        </summary>
        {more && (
          <div className="mt-2">
            <FactorsBody datasetId={datasetId} columns={columns} />
          </div>
        )}
      </details>
    </div>
  );
}

function StructureChart({ caption, figure, height, label }: { caption: string; figure: Figure; height: number; label: string }) {
  return (
    <figure className="min-w-0 rounded-md border border-line p-2">
      <figcaption className="mb-1 px-1 text-xs text-ink-2">{caption}</figcaption>
      <Plot figure={figure} height={height} label={label} />
    </figure>
  );
}

function FactorsBody({ datasetId, columns }: { datasetId: string; columns: string[] | null }) {
  const factors = useFactors(datasetId, columns, true);
  if (factors.isError) return <p className="text-bad">{factors.error.message}</p>;
  if (!factors.data) return <Loading label={t("correlations.factors.loading")} className="h-48 rounded-md" />;
  if (factors.data.reason) return <p className="text-ink-2">{factors.data.reason}</p>;
  return <FactorsResults factors={factors.data} />;
}

export function FactorsResults({ factors }: { factors: Factors }) {
  const names = Array.from({ length: factors.k }, (_, i) => `F${i + 1}`);
  const metrics: Metric[] = [
    { label: "KMO", value: factors.kmo === null ? "—" : factors.kmo.toFixed(2), note: factors.kmo_label, explain: t("correlations.factors.kmoHint") },
    {
      label: t("correlations.factors.bartlett"),
      value: factors.bartlett_chi2 === null ? "—" : formatCount(Math.round(factors.bartlett_chi2)),
      note: factors.bartlett_p === null ? undefined : `df ${formatCount(factors.bartlett_df)} · ${formatP(factors.bartlett_p)}`,
      explain: t("correlations.factors.bartlettHint"),
    },
    {
      label: t("correlations.factors.count"),
      value: String(factors.k),
      note: factors.rotation ?? undefined,
      explain: t("correlations.factors.countHint"),
    },
  ];
  return (
    <div className="space-y-3">
      <div className="max-w-prose space-y-1">
        <p className="font-medium text-ink-1">{factors.conclusion}</p>
        {factors.notes.map((note) => (
          <p key={note} className="text-xs text-ink-3">
            {note}
          </p>
        ))}
        <p className="text-xs text-ink-3">
          {factors.sampled
            ? t("correlations.factors.sampled", { rows: formatCount(factors.rows) })
            : t("correlations.structure.rows", { rows: formatCount(factors.rows) })}
        </p>
      </div>
      <Metrics items={metrics} />
      <div className="overflow-x-auto rounded-md border border-line">
        {/* 041: the numbers keep their line, the column name gives way; a phone hides ψ (= 1 − h²)
            and tightens the cells. */}
        <table className="w-full border-collapse text-[13px] max-sm:[&_td]:px-2 max-sm:[&_th]:px-2" aria-label={t("correlations.factors.table")}>
          <thead className="bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th scope="col" className="px-3 py-1.5 font-normal">{t("correlations.factors.column")}</th>
              {names.map((name, i) => (
                <th key={name} scope="col" data-keep-line className="px-3 py-1.5 text-right font-normal" title={factors.labels[i]}>
                  {name}
                </th>
              ))}
              <th scope="col" data-keep-line className="px-3 py-1.5 text-right font-normal" title={t("correlations.factors.communalityHint")}>
                h²
              </th>
              <th scope="col" data-keep-line className="px-3 py-1.5 text-right font-normal max-sm:hidden" title={t("correlations.factors.uniquenessHint")}>
                ψ
              </th>
              <th scope="col" data-keep-line className="px-3 py-1.5 text-right font-normal" title={t("correlations.factors.msaHint")}>
                KMO
              </th>
            </tr>
          </thead>
          <tbody>
            {factors.columns.map((column, row) => {
              const msa = factors.msa[row] ?? null;
              return (
                <tr key={column} className="border-t border-line">
                  <td className="px-3 py-1.5 font-mono text-xs text-ink-1">{column}</td>
                  {names.map((name, j) => {
                    const value = factors.loadings[row]?.[j] ?? 0;
                    return (
                      <td key={name} data-keep-line className={`px-3 py-1.5 text-right tabular-nums ${Math.abs(value) >= STRONG ? "font-medium text-ink-1" : "text-ink-3"}`}>
                        {fixed(value)}
                      </td>
                    );
                  })}
                  <td data-keep-line className="px-3 py-1.5 text-right text-ink-2 tabular-nums">
                    {(factors.communalities[row] ?? 0).toFixed(2)}
                  </td>
                  <td data-keep-line className="px-3 py-1.5 text-right text-ink-2 tabular-nums max-sm:hidden">
                    {(factors.uniqueness[row] ?? 0).toFixed(2)}
                  </td>
                  <td data-keep-line className={`px-3 py-1.5 text-right tabular-nums ${msa !== null && msa < STRONG ? "text-bad" : "text-ink-2"}`}>
                    {msa === null ? "—" : msa.toFixed(2)}
                  </td>
                </tr>
              );
            })}
          </tbody>
          <tfoot className="border-t border-line text-xs text-ink-3">
            <tr>
              <th scope="row" className="px-3 py-1.5 text-left font-normal">{t("correlations.factors.variance")}</th>
              {names.map((name, j) => (
                <td key={name} data-keep-line className="px-3 py-1.5 text-right tabular-nums">
                  {Math.round((factors.variance[j] ?? 0) * 100)}%
                </td>
              ))}
              <td />
              <td className="max-sm:hidden" />
              <td />
            </tr>
          </tfoot>
        </table>
      </div>
    </div>
  );
}
