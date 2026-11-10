import { Link } from "@tanstack/react-router";
import type { Data, Layout, Shape } from "plotly.js-dist-min";
import { useMemo, useState } from "react";

import type {
  SeriesBreaks,
  SeriesResiduals,
  SeriesVolatility,
} from "../../api/client";
import { useActiveFilters } from "../../api/filters";
import { Plot, type Figure } from "../../charts/Plot";
import { formatNumber, withAlpha } from "../../charts/figures";
import { FitResults } from "../../features/variable/FitPanel";
import { qqFigure } from "../../features/variable/figures";
import { Metrics } from "../../features/variable/Metrics";
import { NormalityPanel } from "../../features/variable/NormalityPanel";
import type { Alpha } from "../../features/variable/normality";
import { t, type MessageKey } from "../../lib/i18n";
import { Select } from "../../ui/fields";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { MethodsTable } from "./Methods";
import { WindowsPicker } from "./Windows";
import type { TimeSeriesSearch } from "./search";

const FEW_ANOMALIES = 5;

type Verdict = { text: string; tone: "ok" | "none"; detail?: string } | null;
type Change = (patch: Partial<TimeSeriesSearch>) => void;

// --- Volatility ----------------------------------------------------------------------------------

function volatilityFigure(data: SeriesVolatility): Figure {
  return (tokens) => {
    const layout: Partial<Layout> & Record<string, unknown> = {
      showlegend: false,
      hovermode: "x unified",
      xaxis: { type: "date", anchor: "y2" },
      yaxis: {
        domain: [0.45, 1],
        title: { text: "Residuals", font: { size: 11 } },
        zeroline: true,
        zerolinecolor: tokens.line,
      },
      yaxis2: {
        domain: [0, 0.38],
        title: { text: "Rolling σ", font: { size: 11 } },
        rangemode: "tozero",
      },
    };
    return {
      data: [
        {
          type: "bar",
          x: data.periods,
          y: data.residuals,
          marker: { color: withAlpha(tokens.ink3, 0.6) },
          hovertemplate: "Residual: %{y:,.4g}<extra></extra>",
        } as Data,
        {
          type: "scatter",
          mode: "lines",
          x: data.periods,
          y: data.rolling_std,
          yaxis: "y2",
          line: { color: tokens.palette[0] ?? tokens.numeric, width: 2 },
          hovertemplate: `${data.window}-period σ: %{y:,.4g}<extra></extra>`,
        } as Data,
      ],
      layout,
    };
  };
}

export function VolatilityBody({
  data,
  windows,
  onWindow,
}: {
  data: SeriesVolatility;
  windows: number[];
  onWindow: (window: number | undefined) => void;
}) {
  if (data.message) return <p className="text-ink-2">{data.message}</p>;
  return (
    <div className="space-y-3">
      <div className="text-xs text-ink-2">
        <WindowsPicker
          label={t("ts.volatility.window")}
          windows={windows}
          auto={windows.length ? null : data.window}
          maximum={data.periods.length}
          multiple={false}
          onChange={(next) => onWindow(next[0])}
        />
      </div>
      <div className="rounded-lg border border-line bg-surface-1 p-3">
        <p className="mb-1 text-xs text-ink-3">
          {t("ts.volatility.on", {
            how: data.residuals_from,
            window: String(data.window),
          })}
        </p>
        <Plot
          figure={volatilityFigure(data)}
          height={320}
          label={t("ts.section.volatility")}
        />
      </div>
      {data.log_helps && (
        <p className="text-[13px] text-ink-1">{t("ts.volatility.logHelps")}</p>
      )}
      <MethodsTable results={data.methods} />
    </div>
  );
}

export function volatilityVerdict(data: SeriesVolatility): Verdict {
  const run = data.methods.filter((m) => m.significant !== null);
  if (!run.length) return null;
  const changing = run.filter((m) => m.significant).length * 2 > run.length;
  if (!changing) return { text: t("ts.volatility.steady"), tone: "none" };
  const direction =
    data.rising === null
      ? ""
      : ` (${t(data.rising ? "ts.volatility.rising" : "ts.volatility.falling")})`;
  return { text: `${t("ts.volatility.changing")}${direction}`, tone: "ok" };
}

// --- Residuals -----------------------------------------------------------------------------------

const SOURCES = ["residuals", "series", "returns"] as const;

export function ResidualsBody({
  data,
  onChange,
  source,
}: {
  data: SeriesResiduals;
  onChange: Change;
  source: string;
}) {
  const [alpha, setAlpha] = useState<Alpha>("0.05");
  const picker = (
    <SegmentedControl
      label={t("ts.residuals.source")}
      value={source}
      options={SOURCES.map((value) => ({
        value,
        label: t(`ts.residuals.source.${value}` as MessageKey),
      }))}
      onChange={(value) =>
        onChange({
          rsource:
            value === "residuals"
              ? undefined
              : (value as TimeSeriesSearch["rsource"]),
        })
      }
    />
  );
  const shape = data.distribution;
  const qq = useMemo(
    () =>
      shape?.qq
        ? qqFigure(
            shape.qq.theoretical,
            shape.qq.sample,
            { slope: shape.qq.slope, intercept: shape.qq.intercept },
            "Normal quantiles",
          )
        : null,
    [shape],
  );
  if (!shape) {
    return (
      <div className="space-y-3">
        {picker}
        <p className="text-ink-2">{data.message}</p>
      </div>
    );
  }
  const d = shape.describe;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        {picker}
        <span className="text-xs text-ink-3">
          {shape.method
            ? t("ts.residuals.method", { how: shape.method })
            : t(`ts.residuals.about.${shape.source}` as MessageKey)}
        </span>
      </div>
      <Metrics
        items={[
          { label: t("vstat.count"), value: d.count.toLocaleString("en-US") },
          { label: t("vstat.mean"), value: formatNumber(d.mean) },
          { label: t("vstat.std"), value: formatNumber(d.std) },
          {
            label: t("vstat.skewness"),
            value: formatNumber(d.skewness),
            explain: t("ts.residuals.skewness"),
          },
          {
            label: t("vstat.kurtosis"),
            value: formatNumber(d.kurtosis),
            explain: t("ts.residuals.kurtosis"),
          },
          {
            label: `${t("vstat.min")} / ${t("vstat.max")}`,
            value: `${formatNumber(d.min)} / ${formatNumber(d.max)}`,
          },
        ]}
      />
      {qq && (
        <figure className="max-w-xl rounded-md border border-line p-2">
          <figcaption className="mb-1 px-1 text-xs text-ink-3">
            {t("ts.residuals.qq")}
          </figcaption>
          <Plot figure={qq} height={260} label={t("ts.residuals.qq")} />
        </figure>
      )}
      <NormalityPanel
        normality={shape.normality}
        alpha={alpha}
        onAlpha={setAlpha}
      />
      {shape.fit.laws.length > 0 && (
        <FitResults
          fit={shape.fit}
          count={d.count}
          column={t(`ts.residuals.source.${shape.source}` as MessageKey)}
          chart={{ bins: shape.bins }}
        />
      )}
    </div>
  );
}

export function residualsVerdict(data: SeriesResiduals): Verdict {
  const shape = data.distribution;
  if (!shape) return null;
  const best = shape.fit.laws[0];
  const rejected = shape.normality.tests.filter(
    (test) => test.p_value !== null && test.p_value < 0.05,
  ).length;
  const tested = shape.normality.tests.filter(
    (test) => test.p_value !== null,
  ).length;
  const normal = tested > 0 && rejected * 2 <= tested;
  const text = t(normal ? "ts.residuals.normal" : "ts.residuals.notNormal");
  return {
    text: best
      ? `${text} · ${t("ts.residuals.best", { law: best.label })}`
      : text,
    tone: normal ? "ok" : "none",
  };
}

// --- Breaks & anomalies --------------------------------------------------------------------------

function breaksFigure(data: SeriesBreaks): Figure {
  return (tokens) => {
    const segments = data.breaks?.segments ?? [];
    const trend = data.breaks?.cost === "trend";
    const means: (number | null)[] = data.periods.map(() => null);
    for (const segment of segments) {
      const length = Math.max(segment.end - segment.start - 1, 1);
      for (let i = segment.start; i < segment.end; i++) {
        means[i] = trend
          ? segment.first +
            ((segment.last - segment.first) * (i - segment.start)) / length
          : segment.mean;
      }
    }
    const shapes: Partial<Shape>[] = data.points.map((point) => ({
      type: "line",
      x0: point.period,
      x1: point.period,
      yref: "paper",
      y0: 0,
      y1: 1,
      line: {
        color: tokens.palette[1] ?? tokens.ink2,
        width: 1.5,
        dash: "dash",
      },
    }));
    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          name: data.deseasonalized ? "Series without seasonality" : "Series",
          x: data.periods,
          y: data.values,
          line: {
            color: withAlpha(tokens.palette[0] ?? tokens.date, 0.6),
            width: 1.25,
          },
          hovertemplate: "%{x}: %{y:,.4g}<extra></extra>",
        } as Data,
        {
          type: "scatter",
          mode: "lines",
          name: trend ? "Segment trend" : "Segment mean",
          x: data.periods,
          y: means,
          line: {
            color: tokens.palette[0] ?? tokens.date,
            width: 2.5,
            shape: "hv",
          },
          hovertemplate: "Segment mean: %{y:,.4g}<extra></extra>",
        } as Data,
        {
          type: "scatter",
          mode: "markers",
          name: "Anomalies",
          x: data.anomalies.map((a) => a.period),
          y: data.anomalies.map((a) => data.values[a.index]),
          marker: {
            color: tokens.palette[3] ?? tokens.ink1,
            size: 9,
            symbol: "circle-open",
            line: { width: 2 },
          },
          hovertemplate: "%{x}: anomaly<extra></extra>",
        } as Data,
      ],
      layout: {
        showlegend: true,
        legend: { orientation: "h", x: 0, y: 1.12, font: { size: 11 } },
        shapes,
        xaxis: { type: "date" },
      },
    };
  };
}

export function BreaksBody({
  data,
  datasetId,
  search,
  onChange,
}: {
  data: SeriesBreaks;
  datasetId: string;
  search: TimeSeriesSearch;
  onChange: Change;
}) {
  const filters = useActiveFilters(datasetId);
  const [everyAnomaly, setEveryAnomaly] = useState(false);
  if (data.message) return <p className="text-ink-2">{data.message}</p>;
  const time = data.setup.time;
  // the table lists the strongest anomalies first; every one stays marked on the chart
  const listed = everyAnomaly ? data.anomalies : data.anomalies.slice(0, FEW_ANOMALIES);
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <SegmentedControl
          label={t("ts.breaks.search")}
          value={search.segment ?? "pelt"}
          options={[
            { value: "pelt", label: "PELT" },
            { value: "binseg", label: t("ts.breaks.binseg") },
          ]}
          onChange={(value) =>
            onChange({
              segment:
                value === "pelt"
                  ? undefined
                  : (value as TimeSeriesSearch["segment"]),
            })
          }
        />
        <label className="flex items-center gap-1.5 text-ink-2">
          {t("ts.breaks.cost")}
          <Select
            value={search.cost ?? ""}
            onChange={(event) =>
              onChange({
                cost: (event.target.value ||
                  undefined) as TimeSeriesSearch["cost"],
              })
            }
            className="h-7 text-xs"
          >
            <option value="">
              {t("time.auto", {
                value: t(
                  `ts.breaks.cost.${data.breaks?.cost ?? "mean"}` as MessageKey,
                ),
              })}
            </option>
            <option value="mean">{t("ts.breaks.cost.mean")}</option>
            <option value="mean_var">{t("ts.breaks.cost.mean_var")}</option>
            <option value="trend">{t("ts.breaks.cost.trend")}</option>
          </Select>
        </label>
        <label className="flex items-center gap-1.5 text-ink-2">
          {t("ts.breaks.sensitivity")}
          <Select
            value={search.sens ?? "medium"}
            onChange={(event) =>
              onChange({
                sens:
                  event.target.value === "medium"
                    ? undefined
                    : (event.target.value as TimeSeriesSearch["sens"]),
              })
            }
            className="h-7 text-xs"
          >
            {(["low", "medium", "high"] as const).map((level) => (
              <option key={level} value={level}>
                {t(`ts.breaks.sensitivity.${level}` as MessageKey)}
              </option>
            ))}
          </Select>
        </label>
      </div>
      <div className="rounded-lg border border-line bg-surface-1 p-3">
        {data.deseasonalized && (
          <p className="mb-1 text-xs text-ink-3">
            {t("ts.breaks.deseasonalized")}
          </p>
        )}
        <Plot
          figure={breaksFigure(data)}
          height={340}
          label={t("ts.section.breaks")}
        />
      </div>
      {data.points.length > 0 && (
        <ul className="space-y-1 text-[13px] text-ink-1">
          {data.points.map((point) => (
            <li key={point.index}>
              •{" "}
              {data.breaks?.cost === "trend"
                ? t("ts.breaks.slope", {
                    before: formatNumber(point.slope_before),
                    after: formatNumber(point.slope_after),
                    period: point.period,
                    unit: t(`time.period.${data.setup.period}` as MessageKey),
                  })
                : t(
                    point.after >= point.before
                      ? "ts.breaks.jumpUp"
                      : "ts.breaks.jumpDown",
                    {
                      before: formatNumber(point.before),
                      after: formatNumber(point.after),
                      period: point.period,
                    },
                  )}
            </li>
          ))}
        </ul>
      )}
      <MethodsTable results={data.methods} />

      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <h3 className="text-[13px] font-medium">{t("ts.anomalies")}</h3>
          <Select
            aria-label={t("ts.anomalies.method")}
            value={data.anomaly_method}
            onChange={(event) =>
              onChange({
                anomaly:
                  event.target.value === "robust_z"
                    ? undefined
                    : (event.target.value as TimeSeriesSearch["anomaly"]),
              })
            }
            className="h-7 text-xs"
          >
            {(["robust_z", "iqr", "rolling_z", "hampel", "gesd", "isolation_forest", "lof", "autoencoder"] as const).map(
              (method) => (
                <option key={method} value={method}>
                  {t(`ts.anomalies.${method}` as MessageKey)}
                </option>
              ),
            )}
            {/* niche and slower (a model fit): one level deeper (064, onion rule) */}
            <optgroup label={t("ts.advancedMore")}>
              <option value="kalman">{t("ts.anomalies.kalman")}</option>
            </optgroup>
          </Select>
          <label className="flex items-center gap-1.5 text-ink-2">
            {t("ts.anomalies.threshold")}
            <Select
              value={String(data.threshold)}
              disabled={data.anomaly_method === "gesd"}
              title={data.anomaly_method === "gesd" ? t("ts.anomalies.gesdAlpha") : undefined}
              onChange={(event) =>
                onChange({
                  threshold:
                    event.target.value === "3.5"
                      ? undefined
                      : Number(event.target.value),
                })
              }
              className="h-7 text-xs"
            >
              {[3, 3.5, 4].map((value) => (
                <option key={value} value={String(value)}>
                  {value}
                </option>
              ))}
            </Select>
          </label>
          <span className="text-ink-3">
            {t("ts.anomalies.on", { how: data.residuals_from })}
          </span>
        </div>
        {data.anomalies.length === 0 ? (
          <p className="text-ink-2">{t("ts.anomalies.none")}</p>
        ) : (
          <div className="overflow-x-auto rounded-md border border-line">
            <table className="w-full border-collapse text-[13px] tabular-nums">
              <thead className="bg-surface-2 text-left text-xs text-ink-3">
                <tr>
                  <th className="px-3 py-1.5 font-normal">
                    {t("time.col.period")}
                  </th>
                  <th className="px-3 py-1.5 text-right font-normal">
                    {t("ts.anomalies.value")}
                  </th>
                  <th className="px-3 py-1.5 text-right font-normal">
                    {t("ts.anomalies.expected")}
                  </th>
                  <th className="px-3 py-1.5 text-right font-normal">
                    {t("ts.anomalies.score")}
                  </th>
                  <th className="px-3 py-1.5 font-normal" />
                </tr>
              </thead>
              <tbody>
                {listed.map((row) => (
                  <tr key={row.index} className="border-t border-line">
                    <td className="px-3 py-1.5">{row.period}</td>
                    <td className="px-3 py-1.5 text-right">
                      {formatNumber(row.value)}
                    </td>
                    <td className="px-3 py-1.5 text-right text-ink-2">
                      {formatNumber(row.expected)}
                    </td>
                    <td className="px-3 py-1.5 text-right">
                      {row.score.toFixed(1)}
                    </td>
                    <td className="px-3 py-1.5">
                      <Link
                        to="/d/$datasetId/data"
                        params={{ datasetId }}
                        search={{
                          filters: {
                            dataset: datasetId,
                            conditions: [
                              ...filters,
                              {
                                column: time,
                                op: "between",
                                low: row.start,
                                high: row.end,
                                negate: false,
                              },
                            ],
                          },
                        }}
                        className="text-xs text-accent hover:underline"
                      >
                        {t("ts.anomalies.rows")}
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {data.anomalies.length > FEW_ANOMALIES && (
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
            <span className="text-ink-3">
              {everyAnomaly
                ? null
                : t("ts.anomalies.strongest", { shown: String(FEW_ANOMALIES), count: String(data.anomalies.length) })}
            </span>
            <button type="button" onClick={() => setEveryAnomaly(!everyAnomaly)} aria-expanded={everyAnomaly} className="rounded-md px-2 py-1 text-accent hover:bg-accent-soft">
              {everyAnomaly ? t("ts.anomalies.showFewer") : t("ts.anomalies.showAll", { count: String(data.anomalies.length) })}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export function breaksVerdict(data: SeriesBreaks): Verdict {
  if (data.message) return null;
  const breaks = data.points.length;
  const anomalies = data.anomalies.length;
  return {
    text: t("ts.breaks.summary", {
      breaks: String(breaks),
      anomalies: String(anomalies),
    }),
    tone: breaks || anomalies ? "ok" : "none",
  };
}
