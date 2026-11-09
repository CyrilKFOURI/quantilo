import { useState } from "react";

import { api, type MlPanel, type MlPeriodSummary, type MlRun } from "../../api/client";
import { useMlNeutralize } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { Plot } from "../../charts/Plot";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Checkbox, ErrorNote } from "../../ui/fields";
import { cumulativeFigure, foldsFigure, periodScoresFigure } from "./figures";
import { Panel, ScoreBand } from "./ResultParts";
import { metricName } from "./SettingsDrawer";

const HEIGHT = 240;
const pct = (value: number | null) => (value === null ? "—" : `${Math.round(value * 100)} %`);
const NUMBERS = new Set(["numeric_continuous", "numeric_discrete"]);

/** A panel run in one sentence: the mean score per period, its spread and the worst period. */
export function periodVerdict(panel: MlPanel): string {
  const s = panel.summary;
  return t("ml.periods.verdict", {
    metric: metricName(s.metric),
    period: panel.period_column,
    mean: formatNumber(s.mean),
    count: String(s.periods),
    std: formatNumber(s.std),
    worst: formatNumber(s.worst),
    worstPeriod: s.worst_period ?? "—",
  });
}

function summaryItems(summary: MlPeriodSummary) {
  const threshold = summary.threshold === null ? null : formatNumber(summary.threshold);
  return [
    { label: t("ml.periods.mean"), value: formatNumber(summary.mean), explain: t("ml.periods.explain.mean") },
    { label: t("ml.periods.std"), value: formatNumber(summary.std) },
    { label: t("ml.periods.sharpe"), value: formatNumber(summary.sharpe), explain: t("ml.periods.explain.sharpe") },
    ...(summary.share_good !== null && threshold !== null
      ? [
          {
            label: t("ml.periods.good"),
            value: pct(summary.share_good),
            note: t(summary.lower_is_better ? "ml.periods.goodNoteLower" : "ml.periods.goodNote", { threshold }),
            explain: t("ml.periods.explain.good"),
          },
        ]
      : []),
    { label: t("ml.periods.worst"), value: formatNumber(summary.worst), note: summary.worst_period ?? undefined },
    ...(summary.longest_bad_streak !== null
      ? [{ label: t("ml.periods.streak"), value: t("ml.periods.streakValue", { count: String(summary.longest_bad_streak) }) }]
      : []),
  ];
}

/** Scores period by period (035): summary, panel checks, per-period and cumulative charts, the
 *  folds on the time axis, the dependence on single columns and its neutralization. */
export function PeriodsTab({ run, datasetId }: { run: MlRun; datasetId: string }) {
  const panel = run.panel;
  if (!panel) return null;
  const metric = metricName(panel.summary.metric);
  return (
    <div className="space-y-4">
      <ScoreBand items={summaryItems(panel.summary)} />
      <p className="flex flex-wrap items-center justify-between gap-2 text-xs text-ink-3">
        <span>{t("ml.periods.criterion", { criterion: t(`ml.validation.criterion.${panel.criterion}` as MessageKey).toLowerCase() })}</span>
        <a href={api.mlPeriodsUrl(datasetId, run.id)} className="text-accent hover:underline">
          {t("ml.periods.export")}
        </a>
      </p>
      <section aria-label={t("ml.periods.alerts")}>
        {panel.alerts.length > 0 ? (
          <ul className="space-y-1 rounded-md border border-warn/40 bg-warn/8 px-3 py-2 text-xs text-ink-1">
            {panel.alerts.map((a) => (
              <li key={a.message}>{a.message}</li>
            ))}
          </ul>
        ) : (
          <p className="text-xs text-ink-3">{t("ml.periods.noAlert")}</p>
        )}
      </section>
      <div className="grid gap-x-5 gap-y-4 @2xl:grid-cols-2 @6xl:grid-cols-3">
        <Panel title={t("ml.periods.scores", { metric, period: panel.period_column })} wide>
          <Plot figure={periodScoresFigure(panel)} height={HEIGHT} label={t("ml.periods.scores", { metric, period: panel.period_column })} />
        </Panel>
        <Panel title={t("ml.periods.cumulative", { metric })}>
          <Plot figure={cumulativeFigure(panel)} height={HEIGHT} label={t("ml.periods.cumulative", { metric })} />
        </Panel>
        <Panel title={t("ml.periods.folds")}>
          <Plot figure={foldsFigure(panel)} height={Math.max(HEIGHT, 60 + 40 * panel.folds.length)} label={t("ml.periods.folds")} />
          <p className="mt-1 text-[11px] text-ink-3">
            {t("ml.periods.foldsHint", { window: panel.window, embargo: String(panel.embargo), step: String(panel.step) })}
          </p>
        </Panel>
      </div>
      <Exposure run={run} panel={panel} datasetId={datasetId} />
    </div>
  );
}

function Exposure({ run, panel, datasetId }: { run: MlRun; panel: MlPanel; datasetId: string }) {
  const numbers = run.preparation.decisions.filter((d) => d.use && NUMBERS.has(d.type)).map((d) => d.column);
  const followed = panel.exposures.map((e) => e.column);
  const columns = [...followed, ...numbers.filter((c) => !followed.includes(c))];
  const [picked, setPicked] = useState<string[]>(() => followed.slice(0, 1));
  const [proportion, setProportion] = useState(100);
  const neutralize = useMlNeutralize(datasetId);
  const found = neutralize.data;
  const metric = metricName(panel.summary.metric);
  if (!panel.exposures.length) return <p className="text-xs text-ink-3">{t("ml.exposure.none")}</p>;
  return (
    <section className="space-y-3 border-t border-line pt-3" aria-label={t("ml.exposure.title")}>
      <h3 className="text-sm font-medium text-ink-1">{t("ml.exposure.title")}</h3>
      <p className="max-w-prose text-[13px] text-ink-2">{t("ml.exposure.hint")}</p>
      <p className="text-xs text-ink-2">{t("ml.exposure.max", { value: formatNumber(panel.max_exposure), count: String(panel.exposure_columns) })}</p>
      <table className="w-full max-w-md border-collapse text-[13px] tabular-nums">
        <thead className="text-left text-xs text-ink-3">
          <tr>
            <th className="py-1 font-normal">{t("ml.exposure.column")}</th>
            <th className="py-1 text-right font-normal">{t("ml.exposure.mean")}</th>
          </tr>
        </thead>
        <tbody>
          {panel.exposures.map((e) => (
            <tr key={e.column} className="border-t border-line">
              <td className="py-1 font-mono text-xs text-ink-1">{e.column}</td>
              <td className="py-1 text-right">{formatNumber(e.mean)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="space-y-2">
        <h4 className="text-xs font-medium text-ink-2">{t("ml.neutralize.title")}</h4>
        <p className="max-w-prose text-xs text-ink-3">{t("ml.neutralize.hint")}</p>
        <fieldset className="max-h-40 overflow-y-auto rounded-md border border-line px-3 py-2">
          <legend className="sr-only">{t("ml.neutralize.columns")}</legend>
          <div className="grid gap-x-4 gap-y-1 sm:grid-cols-2">
            {columns.map((c) => (
              <Checkbox
                key={c}
                className="font-mono text-xs text-ink-1"
                label={c}
                checked={picked.includes(c)}
                onChange={(e) => setPicked(e.target.checked ? [...picked, c] : picked.filter((p) => p !== c))}
              />
            ))}
          </div>
        </fieldset>
        <div className="flex flex-wrap items-center gap-3 text-xs text-ink-2">
          <label className="flex items-center gap-2">
            {t("ml.neutralize.proportion")}
            <input
              type="range"
              min={0}
              max={100}
              step={10}
              value={proportion}
              onChange={(e) => setProportion(Number(e.target.value))}
              aria-label={t("ml.neutralize.proportion")}
              className="accent-(--accent)"
            />
            <span className="w-10 text-ink-1 tabular-nums">{proportion} %</span>
          </label>
          <Button
            variant="primary"
            disabled={picked.length === 0 || neutralize.isPending}
            title={picked.length === 0 ? t("ml.neutralize.pick") : undefined}
            onClick={() => neutralize.mutate({ runId: run.id, columns: picked, proportion: proportion / 100 })}
          >
            {neutralize.isPending ? t("ml.neutralize.running") : t("ml.neutralize.apply")}
          </Button>
        </div>
        {neutralize.isError && <ErrorNote>{neutralize.error.message}</ErrorNote>}
      </div>
      {found && (
        <div className="space-y-3" aria-live="polite">
          <p className="text-[13px] text-ink-1">
            {t("ml.neutralize.result", {
              metric,
              mean: formatNumber(found.summary.mean),
              before: formatNumber(panel.summary.mean),
              sharpe: formatNumber(found.summary.sharpe),
              sharpeBefore: formatNumber(panel.summary.sharpe),
            })}
          </p>
          <table className="w-full max-w-xl border-collapse text-[13px] tabular-nums">
            <thead className="text-left text-xs text-ink-3">
              <tr>
                <th className="py-1 font-normal" />
                <th className="py-1 text-right font-normal">{t("ml.neutralize.before")}</th>
                <th className="py-1 text-right font-normal">{t("ml.neutralize.after", { proportion: `${Math.round(found.proportion * 100)} %` })}</th>
              </tr>
            </thead>
            <tbody>
              {(
                [
                  [t("ml.periods.mean"), panel.summary.mean, found.summary.mean],
                  [t("ml.periods.std"), panel.summary.std, found.summary.std],
                  [t("ml.periods.sharpe"), panel.summary.sharpe, found.summary.sharpe],
                  [t("ml.periods.worst"), panel.summary.worst, found.summary.worst],
                ] as const
              ).map(([label, before, after]) => (
                <tr key={label} className="border-t border-line">
                  <td className="py-1 text-ink-2">{label}</td>
                  <td className="py-1 text-right">{formatNumber(before)}</td>
                  <td className="py-1 text-right text-ink-1">{formatNumber(after)}</td>
                </tr>
              ))}
              <tr className="border-t border-line">
                <td className="py-1 text-ink-2">{t("ml.periods.good")}</td>
                <td className="py-1 text-right">{pct(panel.summary.share_good)}</td>
                <td className="py-1 text-right text-ink-1">{pct(found.summary.share_good)}</td>
              </tr>
              {found.before.map((e, i) => (
                <tr key={e.column} className="border-t border-line">
                  <td className="py-1 text-ink-2">{t("ml.neutralize.exposure", { column: e.column })}</td>
                  <td className="py-1 text-right">{formatNumber(e.mean)}</td>
                  <td className="py-1 text-right text-ink-1">{formatNumber(found.after[i]?.mean ?? null)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <Panel title={`${t("ml.periods.scores", { metric, period: panel.period_column })} · ${t("ml.neutralize.after", { proportion: `${Math.round(found.proportion * 100)} %` })}`}>
            <Plot
              figure={periodScoresFigure({ periods: found.periods, summary: found.summary, period_column: panel.period_column })}
              height={HEIGHT}
              label={t("ml.periods.scores", { metric, period: panel.period_column })}
            />
          </Panel>
        </div>
      )}
    </section>
  );
}
