import { useQueries } from "@tanstack/react-query";

import { api, type MlRun, type MlRunSummary } from "../../api/client";
import { formatNumber } from "../../charts/figures";
import { Plot } from "../../charts/Plot";
import { t, type MessageKey } from "../../lib/i18n";
import { ErrorsCompared } from "./ErrorsCompare";
import { overlayPredictedFigure, overlayRocFigure } from "./figures";
import { Loading } from "../../ui/Loading";

/** Why two runs cannot be compared, or null: same target, classes and validation are needed. */
export function incomparable(first: MlRunSummary, other: MlRunSummary): string | null {
  if (first.target !== other.target) return t("ml.runCompare.otherTarget");
  if (JSON.stringify(first.classes ? { e: first.classes.edges, n: first.classes.names } : null) !== JSON.stringify(other.classes ? { e: other.classes.edges, n: other.classes.names } : null))
    return t("ml.runCompare.otherClasses");
  const key = (r: MlRunSummary) => {
    const v = r.validation;
    // a panel run is also judged on its period, embargo, window and criterion (035)
    return JSON.stringify([v.method, v.test_size, v.folds, v.repeats, v.group, v.time_column, v.seed, v.max_rows, v.period_column, v.embargo, v.window, v.criterion]);
  };
  if (key(first) !== key(other)) return t("ml.runCompare.otherValidation");
  return null;
}

/** Side by side: every metric, the settings that differ, the predictions or ROC curves overlaid. */
export function RunCompare({ datasetId, summaries }: { datasetId: string; summaries: MlRunSummary[] }) {
  const results = useQueries({
    queries: summaries.map((s) => ({
      queryKey: ["datasets", datasetId, "ml", "run", s.id],
      queryFn: () => api.mlRun(datasetId, s.id),
      staleTime: Infinity,
    })),
  });
  const runs = results.map((r) => r.data).filter((r): r is MlRun => Boolean(r));
  if (runs.length < summaries.length) return <Loading label={t("loading.runs")} className="h-40 rounded-md" />;
  const name = (run: MlRun, index: number) => `#${index + 1} ${summaries[index]?.label || run.model}`;
  const metrics = [...new Set(runs.flatMap((r) => Object.keys(r.scores)))].filter((m) => runs.some((r) => r.scores[m] !== null));
  const keys = [...new Set(runs.flatMap((r) => Object.keys(r.settings)))];
  const differs = (key: string) => new Set(runs.map((r) => JSON.stringify(r.settings[key] ?? null))).size > 1;
  const regression = runs.every((r) => r.regression);
  const binary = runs.every((r) => r.classification?.roc);
  return (
    <section className="space-y-3" aria-label={t("ml.runCompare.title")}>
      <h3 className="text-[13px] font-medium">{t("ml.runCompare.title")}</h3>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-[13px] tabular-nums">
          <thead className="text-left text-xs text-ink-3">
            <tr>
              <th className="py-1 pr-3 font-normal" />
              {runs.map((r, i) => (
                <th key={r.id} className="py-1 pr-3 text-right font-normal text-ink-1">
                  {name(r, i)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metrics.map((m) => (
              <tr key={m} className="border-t border-line">
                <td className="py-1 pr-3 text-ink-2">{t(`ml.metric.${m}` as MessageKey)}</td>
                {runs.map((r) => (
                  <td key={r.id} className="py-1 pr-3 text-right">
                    {formatNumber(r.scores[m])}
                  </td>
                ))}
              </tr>
            ))}
            {keys.filter(differs).map((key) => (
              <tr key={key} className="border-t border-line bg-accent-soft/60">
                <td className="py-1 pr-3 font-mono text-xs text-ink-1">{key}</td>
                {runs.map((r) => (
                  <td key={r.id} className="py-1 pr-3 text-right text-ink-1">
                    {r.settings[key] === undefined ? "—" : String(r.settings[key])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {keys.some(differs) && <p className="text-xs text-ink-3">{t("ml.runCompare.differ")}</p>}
      {regression && (
        <figure className="rounded-md border border-line bg-surface-1 p-2">
          <Plot figure={overlayPredictedFigure(runs.map((r, i) => ({ name: name(r, i), run: r })))} height={300} label={t("ml.predictedVsActual")} />
        </figure>
      )}
      {binary && (
        <figure className="rounded-md border border-line bg-surface-1 p-2">
          <Plot figure={overlayRocFigure(runs.map((r, i) => ({ name: name(r, i), run: r })))} height={300} label={t("ml.roc")} />
        </figure>
      )}
      {runs.length === 2 && runs[0] && runs[1] && (
        <ErrorsCompared datasetId={datasetId} first={runs[0].id} second={runs[1].id} names={[name(runs[0], 0), name(runs[1], 1)]} />
      )}
    </section>
  );
}
