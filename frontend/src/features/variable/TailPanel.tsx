import { useMemo } from "react";

import type { Tail } from "../../api/client";
import { useTail } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { Plot, type Figure } from "../../charts/Plot";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Loading } from "../../ui/Loading";
import { Metrics, type Metric } from "./Metrics";
import { hillAt, hillFigure, meanExcessFigure, returnLevelFigure } from "./tailFigures";

/** The right tail (056): GPD above a threshold, VaR / ES, return levels and three charts. */
export function TailPanel({ datasetId, column }: { datasetId: string; column: string }) {
  const tail = useTail(datasetId, column, true);
  if (tail.isError) return <p className="text-bad">{tail.error.message}</p>;
  if (!tail.data) return <Loading label={t("variable.tail.loading")} className="h-72 rounded-md" />;
  if (tail.data.reason || tail.data.xi === null) {
    return <p className="text-ink-2">{t("variable.tail.none", { reason: tail.data.reason ?? "" })}</p>;
  }
  return <TailResults tail={tail.data} column={column} />;
}

function percent(level: number): string {
  return `${(100 * level).toLocaleString("en-US", { maximumFractionDigits: 1 })}%`;
}

function interval(low: number | null, high: number | null): string {
  if (low === null && high === null) return "—";
  return `${formatNumber(low)} – ${high === null ? t("variable.tail.unbounded") : formatNumber(high)}`;
}

export function TailResults({ tail, column }: { tail: Tail; column: string }) {
  const returns = useMemo(() => returnLevelFigure(tail), [tail]);
  const excess = useMemo(() => meanExcessFigure(tail), [tail]);
  const hill = useMemo(() => hillFigure(tail), [tail]);
  const threshold = tail.threshold;
  const hillPoint = hillAt(tail);
  const thousand = tail.return_levels.find((r) => r.period === 1000);

  const metrics: Metric[] = [
    {
      label: t("variable.tail.xi"),
      value: formatNumber(tail.xi),
      note: tail.xi_low !== null || tail.xi_high !== null ? interval(tail.xi_low, tail.xi_high) : undefined,
      explain: t("variable.tail.explainXi"),
    },
    ...(threshold
      ? [
          {
            label: t("variable.tail.threshold"),
            value: formatNumber(threshold.value),
            note: t("variable.tail.thresholdNote", { level: percent(threshold.level), count: formatCount(threshold.exceedances) }),
            explain: t("variable.tail.explainThreshold"),
          },
        ]
      : []),
    ...tail.risk.flatMap((risk) => [
      {
        label: t("variable.tail.var", { level: percent(risk.level) }),
        value: formatNumber(risk.var),
        explain: t("variable.tail.explainVar", { level: percent(risk.level), rest: percent(1 - risk.level) }),
      },
      {
        label: t("variable.tail.es", { level: percent(risk.level) }),
        value: formatNumber(risk.es),
        explain: t("variable.tail.explainEs", { rest: percent(1 - risk.level) }),
      },
    ]),
  ];

  return (
    <div className="space-y-5">
      <div className="space-y-1">
        <p className="font-medium text-ink-1">{tail.conclusion}</p>
        {tail.notes.map((note) => (
          <p key={note} className="text-xs text-ink-3">
            {note}
          </p>
        ))}
      </div>

      <Metrics items={metrics} />

      <section className="min-w-0">
        <h3 className="mb-1.5 text-xs font-medium text-ink-3">{t("variable.tail.levels")}</h3>
        <div className="overflow-x-auto rounded-md border border-line">
          <table className="w-full border-collapse text-[13px]">
            <thead className="bg-surface-2 text-left text-xs text-ink-3">
              <tr>
                <th scope="col" className="px-3 py-1.5 font-normal">{t("variable.tail.once")}</th>
                <th scope="col" className="px-3 py-1.5 text-right font-normal">{t("variable.tail.value")}</th>
                <th scope="col" className="px-3 py-1.5 text-right font-normal">{t("variable.tail.interval")}</th>
                <th scope="col" className="px-3 py-1.5 text-right font-normal">{t("variable.tail.observed")}</th>
              </tr>
            </thead>
            <tbody>
              {tail.return_levels.map((level) => (
                <tr key={level.period} className="border-t border-line">
                  <td className="px-3 py-1.5 whitespace-nowrap text-ink-1">
                    {t("variable.tail.oneIn", { period: formatCount(level.period) })}
                  </td>
                  <td className="px-3 py-1.5 text-right font-medium tabular-nums">{formatNumber(level.value)}</td>
                  <td className="px-3 py-1.5 text-right whitespace-nowrap text-ink-2 tabular-nums">{interval(level.low, level.high)}</td>
                  <td className="px-3 py-1.5 text-right whitespace-nowrap text-ink-2 tabular-nums">
                    {level.beyond_data ? t("variable.tail.beyond") : formatNumber(level.empirical)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {tail.stability.length > 1 && (
          <p className="mt-1.5 text-xs text-ink-3">
            {t("variable.tail.stability", {
              values: tail.stability.map((s) => `${percent(s.level)}: ${formatNumber(s.xi)}`).join(" · "),
            })}
          </p>
        )}
      </section>

      <div className="grid gap-4 lg:grid-cols-3">
        <TailChart
          caption={thousand ? t("variable.tail.returnCaption", { value: formatNumber(thousand.value) }) : t("variable.tail.returnTitle")}
          figure={returns}
          label={`${t("variable.tail.returnTitle")}: ${column}`}
        />
        <TailChart caption={threshold?.why ?? t("variable.tail.excessTitle")} figure={excess} label={`${t("variable.tail.excessTitle")}: ${column}`} />
        <TailChart
          caption={
            hillPoint
              ? t("variable.tail.hillCaption", { k: formatCount(hillPoint.k), hill: formatNumber(hillPoint.xi), xi: formatNumber(tail.xi) })
              : t("variable.tail.hillTitle")
          }
          figure={hill}
          label={`${t("variable.tail.hillTitle")}: ${column}`}
        />
      </div>

      <p className="text-xs text-ink-3">{tail.rule.reason}</p>
    </div>
  );
}

function TailChart({ caption, figure, label }: { caption: string; figure: Figure; label: string }) {
  return (
    <figure className="min-w-0 rounded-md border border-line p-2">
      <figcaption className="mb-1 px-1 text-xs text-ink-2">{caption}</figcaption>
      <Plot figure={figure} height={260} label={label} />
    </figure>
  );
}
