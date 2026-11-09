import { CircleCheck, CircleHelp, TriangleAlert } from "lucide-react";
import { useId, useMemo } from "react";

import { Plot } from "../../charts/Plot";
import { t, type MessageKey } from "../../lib/i18n";
import type { ExperimentComparison, ExperimentCumulative, ExperimentMetricResult, ExperimentResult, ExperimentSegments, MetricKind } from "./api";
import { cumulativeFigure, intervalsFigure } from "./figures";
import { gap, interval, percent, pValue, value } from "./format";

/** The verdict in a word, with a status colour and an icon (colour is never alone). */
function Verdict({ comparison }: { comparison: ExperimentComparison }) {
  const { verdict } = comparison;
  if (verdict === "better" || verdict === "worse")
    return (
      <span className="inline-flex items-center gap-1 text-xs font-medium whitespace-nowrap text-ok">
        <CircleCheck size={13} strokeWidth={1.75} aria-hidden />
        {t(verdict === "better" ? "experiments.verdict.higher" : "experiments.verdict.lower")}
      </span>
    );
  if (verdict === "underpowered")
    return (
      <span className="inline-flex items-center gap-1 text-xs font-medium whitespace-nowrap text-warn">
        <TriangleAlert size={13} strokeWidth={1.75} aria-hidden />
        {t("experiments.verdict.underpowered")}
      </span>
    );
  return (
    <span className="inline-flex items-center gap-1 text-xs font-medium whitespace-nowrap text-ink-2">
      <CircleHelp size={13} strokeWidth={1.75} aria-hidden />
      {t(verdict === "not_tested" ? "experiments.verdict.notTested" : "experiments.verdict.unclear")}
    </span>
  );
}

export function Sentence({ comparison }: { comparison: ExperimentComparison }) {
  return (
    <p className="flex flex-wrap items-baseline gap-x-2 text-[13px] leading-relaxed text-ink-1">
      <Verdict comparison={comparison} />
      <span className="min-w-0">{comparison.sentence}</span>
    </p>
  );
}

function ComparisonTable({ metric, variants }: { metric: ExperimentMetricResult; variants: ExperimentResult["variants"] }) {
  const kind = metric.kind as MetricKind;
  const control = variants[0];
  const head = "px-2 py-1.5 text-right font-normal text-ink-3 first:pl-0 first:text-left";
  const cell = "px-2 py-1.5 text-right whitespace-nowrap tabular-nums first:pl-0 first:text-left";
  return (
    <div className="overflow-x-auto">
      {/* 041: a phone keeps variant, value, lift and adjusted p. */}
      <table className="w-full min-w-[560px] text-xs max-sm:[&_tr>:nth-child(2)]:hidden max-sm:[&_tr>:nth-child(4)]:hidden max-sm:[&_tr>:nth-child(6)]:hidden max-sm:[&_tr>:nth-child(8)]:hidden max-sm:[&_tr>:nth-child(9)]:hidden" aria-label={t("experiments.tableLabel", { metric: metric.label })}>
        <thead>
          <tr className="border-b border-line">
            <th className={head}>{t("experiments.col.variant")}</th>
            <th className={head}>{t("experiments.col.units")}</th>
            <th className={head}>{t("experiments.col.value")}</th>
            <th className={head}>{t("experiments.col.difference")}</th>
            <th className={head}>{t("experiments.col.lift")}</th>
            <th className={head}>p</th>
            <th className={head}>{t("experiments.col.adjusted")}</th>
            <th className={head}>{t("experiments.col.power")}</th>
            <th className={head}>{t("experiments.col.detectable")}</th>
          </tr>
        </thead>
        <tbody>
          {control && (
            <tr className="border-b border-line/60">
              <td className={`${cell} text-ink-2`}>
                {control.label}
                {control.label.toLowerCase() !== "control" && <span className="text-ink-3"> ({t("experiments.controlTag")})</span>}
              </td>
              <td className={cell}>{(metric.comparisons[0]?.n_control ?? control.units).toLocaleString("en-US")}</td>
              <td className={cell}>{value(kind, metric.values[0] ?? null)}</td>
              <td className={`${cell} text-ink-3`} colSpan={6}>
                {t("experiments.reference")}
              </td>
            </tr>
          )}
          {metric.comparisons.map((c) => (
            <tr key={c.variant} className="border-b border-line/60">
              <td className={cell}>{c.variant}</td>
              <td className={cell}>{c.n_variant.toLocaleString("en-US")}</td>
              <td className={cell}>{value(kind, c.variant_value)}</td>
              <td className={cell}>
                {gap(kind, c.diff)} <span className="text-ink-3">{interval(gap(kind, c.diff_low), gap(kind, c.diff_high))}</span>
              </td>
              <td className={cell}>
                {percent(c.lift)} <span className="text-ink-3">{interval(percent(c.lift_low), percent(c.lift_high))}</span>
              </td>
              <td className={cell}>{pValue(c.p_value)}</td>
              <td className={cell}>{pValue(c.p_adjusted)}</td>
              <td className={cell}>{c.power === null ? "—" : `${Math.round(c.power * 100)} %`}</td>
              <td className={cell}>{c.mde_relative === null ? "—" : `±${(c.mde_relative * 100).toFixed(1)} %`}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** "Understand this test": the test, what it assumes, how to read the interval and the power. */
function Understand({ metric }: { metric: ExperimentMetricResult }) {
  const tests = [...new Set(metric.comparisons.map((c) => c.test).filter((test) => test !== "none"))];
  return (
    <details className="group text-xs text-ink-2">
      <summary className="cursor-pointer text-accent select-none hover:underline">{t("experiments.understand")}</summary>
      <div className="mt-2 max-w-prose space-y-1.5 leading-relaxed">
        {tests.map((test) => (
          <p key={test}>{t(`experiments.test.${test}` as MessageKey)}</p>
        ))}
        {metric.cuped && <p>{t("experiments.explain.cuped")}</p>}
        <p>{t("experiments.explain.interval")}</p>
        <p>{t("experiments.explain.power")}</p>
        <p>{t("experiments.explain.holm")}</p>
      </div>
    </details>
  );
}

export function MetricSection({ metric, result }: { metric: ExperimentMetricResult; result: ExperimentResult }) {
  const figure = useMemo(() => intervalsFigure(metric), [metric]);
  const id = `experiment-metric-${useId().replace(/:/g, "")}`;
  const drawn = metric.comparisons.filter((c) => c.diff !== null).length;
  const checks = metric.comparisons.flatMap((c) => c.checks.map((check) => ({ ...check, variant: c.variant })));
  return (
    <section aria-labelledby={id} className="space-y-3 border-t border-line pt-5">
      <div className="space-y-0.5">
        <div className="flex flex-wrap items-baseline gap-x-2">
          <h2 id={id} className="font-mono text-[15px] font-semibold text-ink-1">
            {metric.label}
          </h2>
          <span className="text-xs text-ink-3">
            {t(`experiments.kind.${metric.kind}` as MessageKey)} · {t(metric.primary ? "experiments.primary" : "experiments.secondary")}
          </span>
        </div>
        <p className="text-xs text-ink-3">
          {metric.reason}
          {metric.aggregation && ` ${t("experiments.aggregation", { how: metric.aggregation })}`}
        </p>
      </div>
      <div className="space-y-1.5">
        {metric.comparisons.map((c) => (
          <Sentence key={c.variant} comparison={c} />
        ))}
      </div>
      {metric.cuped && <p className="border-l-2 border-line pl-3 text-xs text-ink-2">{metric.cuped.sentence}</p>}
      {drawn > 0 && (
        <div className="max-w-4xl">
          <Plot figure={figure} height={90 + 56 * drawn} label={t("experiments.intervalsLabel", { metric: metric.label })} />
        </div>
      )}
      <ComparisonTable metric={metric} variants={result.variants} />
      {checks.length > 0 && (
        <ul className="space-y-0.5 text-xs text-ink-3">
          {checks.map((check) => (
            <li key={`${check.variant}-${check.key}`}>
              <span className="text-ink-2">{check.variant}: </span>
              {check.note}
              {check.p_value !== null && ` p = ${pValue(check.p_value)}.`}
              {check.low !== null && check.high !== null && ` ${interval(gap(metric.kind as MetricKind, check.low), gap(metric.kind as MetricKind, check.high))}.`}
            </li>
          ))}
        </ul>
      )}
      <Understand metric={metric} />
    </section>
  );
}

export function SegmentsSection({ segments }: { segments: ExperimentSegments }) {
  return (
    <section aria-labelledby="experiment-segments" className="space-y-3 border-t border-line pt-5">
      <h2 id="experiment-segments" className="text-[15px] font-semibold text-ink-1">
        {t("experiments.segmentsTitle", { metric: segments.metric, column: segments.column })}
      </h2>
      <p className="border-l-2 border-warn pl-3 text-xs text-ink-2">{segments.warning}</p>
      <dl className="space-y-3">
        {segments.levels.map((level) => (
          <div key={level.level} className="grid gap-1 sm:grid-cols-[160px_minmax(0,1fr)]">
            <dt className="text-[13px] text-ink-2">
              <span className="font-medium text-ink-1">{level.level}</span>{" "}
              <span className="text-xs text-ink-3 tabular-nums">{t("experiments.units", { count: level.units.toLocaleString("en-US") })}</span>
            </dt>
            <dd className="space-y-1">
              {level.comparisons.map((c) => (
                <Sentence key={c.variant} comparison={c} />
              ))}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

export function CumulativeSection({ cumulative }: { cumulative: ExperimentCumulative }) {
  const figure = useMemo(() => cumulativeFigure(cumulative), [cumulative]);
  return (
    <section aria-labelledby="experiment-cumulative" className="space-y-2 border-t border-line pt-5">
      <h2 id="experiment-cumulative" className="text-[15px] font-semibold text-ink-1">
        {t("experiments.cumulativeTitle", { metric: cumulative.metric, grain: t(`experiments.grain.${cumulative.grain}` as MessageKey) })}
      </h2>
      <p className="border-l-2 border-warn pl-3 text-xs text-ink-2">{cumulative.warning}</p>
      <Plot figure={figure} height={300} label={t("experiments.cumulativeLabel", { metric: cumulative.metric })} />
    </section>
  );
}
