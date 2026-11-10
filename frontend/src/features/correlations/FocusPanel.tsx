import { Link } from "@tanstack/react-router";
import { ArrowRight, Crosshair } from "lucide-react";
import { useMemo } from "react";

import type { AnalysisType, AssociationMatrix } from "../../api/client";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Select } from "../../ui/fields";
import { TYPE_META } from "../columns/columnTypes";
import { EvidenceBadge, evidenceRank, type Evidence } from "./evidence";
import { METRIC_LABELS } from "./figures";

const SHOWN = 15;
/** Targets the Regression tab can model: numbers and yes/no (068). */
const MODELLED = new Set(["numeric_continuous", "numeric_discrete", "boolean"]);

type FocusPanelProps = {
  datasetId: string;
  matrix: AssociationMatrix;
  focus: string;
  onFocus: (name: string) => void;
  typeOf: (name: string) => AnalysisType | undefined;
};

/** Pick one variable: the others ranked by how strongly they relate to it. */
export function FocusPanel({ datasetId, matrix, focus, onFocus, typeOf }: FocusPanelProps) {
  const index = matrix.columns.findIndex((c) => c.name === focus);
  const levels = useMemo(() => {
    const map = new Map<string, Evidence>();
    for (const pair of matrix.pairs) {
      if (pair.a === focus) map.set(pair.b, pair.evidence);
      if (pair.b === focus) map.set(pair.a, pair.evidence);
    }
    return map;
  }, [matrix.pairs, focus]);
  const ranked = useMemo(() => {
    if (index < 0) return [];
    return matrix.columns
      .map((column, j) => ({
        name: column.name,
        value: matrix.values[index]?.[j] ?? null,
        metric: matrix.metrics[index]?.[j] ?? "",
        n: matrix.n[index]?.[j] ?? 0,
        evidence: levels.get(column.name) ?? "none",
      }))
      .filter((item, j) => j !== index && item.value !== null)
      .map((item) => ({ ...item, value: item.value as number }))
      .sort((a, b) => evidenceRank(b.evidence) - evidenceRank(a.evidence) || Math.abs(b.value) - Math.abs(a.value));
  }, [matrix, index, levels]);

  return (
    <section className="rounded-lg border border-line bg-surface-1">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3">
        <div className="flex items-center gap-2">
          <Crosshair size={15} strokeWidth={1.5} className="text-ink-3" aria-hidden />
          <h2 className="text-sm font-semibold">{t("correlations.focus")}</h2>
          <span className="text-xs text-ink-3">{t("correlations.focusHint")}</span>
        </div>
        <Select aria-label={t("correlations.focusVariable")} value={focus} onChange={(event) => onFocus(event.target.value)} className="w-56">
          <option value="">{t("correlations.focusChoose")}</option>
          {matrix.columns.map((column) => (
            <option key={column.name} value={column.name}>
              {column.name}
            </option>
          ))}
        </Select>
      </header>

      {index >= 0 && MODELLED.has(matrix.columns[index]?.type ?? "") && (
        <div className="border-b border-line px-4 py-2">
          {/* 068: the regression of the focus on its strongest links (R-REG-21 picks them) */}
          <Link
            to={"/d/$datasetId/regression" as never}
            params={{ datasetId } as never}
            search={{ target: focus } as never}
            className="inline-flex items-center gap-1 text-xs text-accent hover:underline"
          >
            {t("correlations.focusModel", { name: focus })}
            <ArrowRight size={12} strokeWidth={1.75} aria-hidden />
          </Link>
        </div>
      )}

      {index < 0 ? (
        <div className="flex flex-wrap gap-1.5 px-4 py-4">
          {matrix.columns.map((column) => {
            const type = typeOf(column.name);
            return (
              <button
                key={column.name}
                type="button"
                onClick={() => onFocus(column.name)}
                className={`rounded-md border border-line bg-surface-2 px-2 py-1 font-mono text-xs hover:border-accent ${type ? TYPE_META[type].tone : "text-ink-1"}`}
              >
                {column.name}
              </button>
            );
          })}
        </div>
      ) : (
        <ol className="divide-y divide-line" aria-label={t("correlations.focusList", { name: focus })}>
          {ranked.slice(0, SHOWN).map((item, rank) => {
            const type = typeOf(item.name);
            const signed = item.metric === "pearson" || item.metric === "spearman";
            const strength = Math.min(Math.abs(item.value), 1);
            return (
              <li
                key={item.name}
                className={`grid grid-cols-[1.5rem_minmax(0,12rem)_minmax(0,1fr)_3.5rem_5.5rem_auto] items-center gap-3 px-4 py-2 ${item.evidence === "none" ? "opacity-55" : ""}`}
              >
                <span className="text-xs text-ink-3 tabular-nums">{rank + 1}</span>
                <span className={`truncate font-mono text-[13px] font-medium ${type ? TYPE_META[type].tone : "text-ink-1"}`} title={item.name}>
                  {item.name}
                </span>
                <div className="flex min-w-0 items-center gap-2">
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-2">
                    <div
                      className={`h-full rounded-full ${signed && item.value < 0 ? "bg-(--chart-8)" : "bg-(--chart-1)"}`}
                      style={{ width: `${strength * 100}%` }}
                    />
                  </div>
                  <span className="hidden w-28 truncate text-xs text-ink-3 sm:inline">{METRIC_LABELS[item.metric]}</span>
                </div>
                <span className="text-right text-[13px] text-ink-1 tabular-nums" title={`n = ${formatCount(item.n)}`}>
                  {item.value.toFixed(2)}
                </span>
                <EvidenceBadge level={item.evidence} />
                <Link
                  to="/d/$datasetId/compare"
                  params={{ datasetId }}
                  search={(previous) => ({ ...previous, x: focus, y: item.name, z: undefined, color: undefined })}
                  className="inline-flex items-center gap-1 text-xs text-ink-3 hover:text-accent"
                  aria-label={t("correlations.compareWith", { a: focus, b: item.name })}
                >
                  {t("insights.view")}
                  <ArrowRight size={12} strokeWidth={1.75} aria-hidden />
                </Link>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
