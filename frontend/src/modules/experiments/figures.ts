import type { Data, Layout } from "plotly.js-dist-min";

import { withAlpha } from "../../charts/figures";
import type { ChartTokens } from "../../charts/theme";
import type { BanditReplay, BanditSimulation, ExperimentCumulative, ExperimentMetricResult, MetricKind, SampleSize } from "./api";

/* The Experiments page's charts (twins: quantilo/plots/experiments.py). A variant
   keeps its colour everywhere: the i-th variant after the control takes the i-th palette colour;
   the control is the reference (zero), never a series. */

type Built = { data: Data[]; layout?: Partial<Layout> };

const scale = (kind: MetricKind) => (kind === "rate" ? 100 : 1);
const unit = (kind: MetricKind) => (kind === "rate" ? " pts" : "");
const axisUnit = (kind: MetricKind) => (kind === "rate" ? " (pts)" : "");

/** An axis range that always shows zero, the reference, with some air on both sides. */
export function withZero(values: number[]): [number, number] {
  const low = Math.min(0, ...values);
  const high = Math.max(0, ...values);
  const pad = (high - low || 1) * 0.08;
  return [low - pad, high + pad];
}
const axisTitle = (text: string) => ({ title: { text, font: { size: 11 } } });
const LEGEND: Partial<Layout> = {
  showlegend: true,
  legend: { orientation: "h", x: 0, y: 1.12, font: { size: 11 } },
};

export function signed(value: number, digits = 1): string {
  const text = Math.abs(value).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  return value >= 0 ? `+${text}` : `−${text}`;
}

export function variantColor(tokens: ChartTokens, index: number): string {
  return tokens.palette[index % tokens.palette.length] ?? tokens.ink2;
}

const short = (value: number) => value.toLocaleString("en-US", { maximumSignificantDigits: 3 });

/** Each variant's difference to the control with its interval, against a zero line. */
export function intervalsFigure(metric: ExperimentMetricResult): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const kind = metric.kind as MetricKind;
    const k = scale(kind);
    const data: Data[] = [];
    const labels: string[] = [];
    const ends: number[] = [];
    metric.comparisons.forEach((c, i) => {
      if (c.diff === null) return;
      const color = variantColor(tokens, i);
      labels.push(c.variant);
      const low = c.diff_low === null ? null : c.diff_low * k;
      const high = c.diff_high === null ? null : c.diff_high * k;
      const diff = c.diff * k;
      ends.push(diff, ...(low === null ? [] : [low]), ...(high === null ? [] : [high]));
      const lift = c.lift === null ? "" : ` (${signed(c.lift * 100)} %)`;
      const text = `${signed(diff, Math.abs(diff) < 10 ? 2 : 1)}${unit(kind)}${lift}`;
      if (low !== null && high !== null)
        data.push({
          type: "scatter",
          mode: "lines",
          x: [low, high],
          y: [c.variant, c.variant],
          line: { color, width: 2 },
          hoverinfo: "skip",
        } as Data);
      const p = c.p_adjusted ?? c.p_value;
      data.push({
        type: "scatter",
        mode: "markers+text",
        x: [diff],
        y: [c.variant],
        text: [text],
        textposition: "top center",
        textfont: { size: 11, color: tokens.ink2 },
        marker: { color, size: 10, line: { color: tokens.surface, width: 2 } },
        customdata: [[c.control, low === null ? "—" : short(low), high === null ? "—" : short(high), p === null ? "—" : p.toPrecision(3)]],
        hovertemplate: `${c.variant} vs %{customdata[0]}<br>difference %{x:,.3g}${unit(kind)}<br>CI [%{customdata[1]} ; %{customdata[2]}]<br>p = %{customdata[3]}<extra></extra>`,
      } as Data);
    });
    return {
      data,
      layout: {
        xaxis: { ...axisTitle(`Difference to the control${axisUnit(kind)}`), range: withZero(ends), showgrid: true, zeroline: true, zerolinecolor: tokens.ink3, zerolinewidth: 1 },
        yaxis: { type: "category", showgrid: false, autorange: "reversed", categoryorder: "array", categoryarray: labels },
        margin: { t: 12 },
      },
    };
  };
}

/** The difference on every unit seen so far, period by period, with its interval. */
export function cumulativeFigure(cumulative: ExperimentCumulative): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const kind = cumulative.kind as MetricKind;
    const k = scale(kind);
    const scaled = (values: (number | null)[]) => values.map((v) => (v === null ? null : v * k));
    const data: Data[] = [];
    cumulative.lines.forEach((line, i) => {
      const color = variantColor(tokens, i);
      data.push(
        { type: "scatter", mode: "lines", x: line.periods, y: scaled(line.low), line: { width: 0 }, hoverinfo: "skip", showlegend: false } as Data,
        {
          type: "scatter",
          mode: "lines",
          x: line.periods,
          y: scaled(line.high),
          fill: "tonexty",
          fillcolor: withAlpha(color, 0.15),
          line: { width: 0 },
          hoverinfo: "skip",
          showlegend: false,
        } as Data,
        {
          type: "scatter",
          mode: "lines",
          name: `${line.variant} − control`,
          x: line.periods,
          y: scaled(line.diff),
          line: { color, width: 2 },
          hovertemplate: `%{x}<br>${line.variant}: %{y:,.3g}${unit(kind)}<extra></extra>`,
        } as Data,
      );
    });
    const layout: Partial<Layout> = {
      xaxis: { type: "date", nticks: 6, tickangle: 0 },
      yaxis: { ...axisTitle(`Cumulative difference${axisUnit(kind)}`), zeroline: true, zerolinecolor: tokens.ink3, zerolinewidth: 1 },
      hovermode: "x unified",
    };
    return { data, layout: cumulative.lines.length > 1 ? { ...layout, ...LEGEND } : layout };
  };
}

/** Units per group (log scale) against the smallest lift they detect, the plan marked. */
export function sampleSizeFigure(plan: SampleSize): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const relative = plan.effect_relative !== null;
    const factor = relative ? 100 : 1;
    const points = plan.curve.filter((p) => p.lift !== null);
    const y = points.map((p) => Math.abs((relative ? p.lift : p.absolute) ?? 0) * factor);
    const suffix = relative ? " %" : "";
    const color = tokens.palette[0] ?? tokens.ink1;
    const target = Math.abs((relative ? plan.effect_relative : plan.effect_absolute) ?? 0) * factor;
    const hover = `%{x:,} units per group<br>detects ±%{y:.3g}${suffix}`;
    return {
      data: [
        { type: "scatter", mode: "lines", x: points.map((p) => p.per_group), y, line: { color, width: 2 }, hovertemplate: `${hover}<extra></extra>` } as Data,
        {
          type: "scatter",
          mode: "markers+text",
          x: [plan.control],
          y: [target],
          text: [`Your plan: ${plan.control.toLocaleString("en-US")} per group`],
          textposition: "top right",
          textfont: { size: 11, color: tokens.ink1 },
          marker: { color, size: 10, line: { color: tokens.surface, width: 2 } },
          hovertemplate: `${hover}<extra>plan</extra>`,
        } as Data,
      ],
      layout: {
        xaxis: { ...axisTitle("Units per group (control)"), type: "log", showgrid: true },
        yaxis: { ...axisTitle(`Smallest detectable lift${relative ? " (%)" : ""}`), rangemode: "tozero" },
        margin: { t: 12 },
      },
    };
  };
}

/* --- Bandit view (049) ------------------------------------------------------------------------ */

const regretAxis = (kind: string) => `Cumulative regret${kind === "rate" ? " (conversions)" : ""}`;

/** The share of each period's units a bandit would have sent to each variant: the control in the
 *  neutral ink (the reference), the other variants in their colour. */
export function banditSharesFigure(replay: BanditReplay): (tokens: ChartTokens) => Built {
  return (tokens) => ({
    data: replay.shares.map(
      (share, i) =>
        ({
          type: "scatter",
          mode: "lines",
          name: share.variant,
          x: replay.periods,
          y: share.shares.map((s) => s * 100),
          line: { color: i === 0 ? tokens.ink3 : variantColor(tokens, i - 1), width: 2, shape: "hv" },
          hovertemplate: `%{x}<br>${share.variant}: %{y:.0f} %<extra></extra>`,
        }) as Data,
    ),
    layout: {
      xaxis: { type: "date", nticks: 6, tickangle: 0 },
      yaxis: { ...axisTitle("Bandit's share of the units (%)"), range: [0, 100] },
      hovermode: "x unified",
      ...LEGEND,
    },
  });
}

function regretLines(x: (string | number)[], fixed: number[], bandit: number[], tokens: ChartTokens): Data[] {
  return [
    {
      type: "scatter",
      mode: "lines",
      name: "Fixed split",
      x,
      y: fixed,
      line: { color: tokens.ink3, width: 2, dash: "dot" },
      hovertemplate: "Fixed split: %{y:,.3g}<extra></extra>",
    } as Data,
    {
      type: "scatter",
      mode: "lines",
      name: "Bandit",
      x,
      y: bandit,
      line: { color: tokens.palette[0] ?? tokens.ink1, width: 2 },
      hovertemplate: "Bandit: %{y:,.3g}<extra></extra>",
    } as Data,
  ];
}

/** The cumulative regret of the fixed split and of the bandit against shipping the winner from
 *  the first period: the gap between the lines is what the bandit gained. */
export function banditRegretFigure(replay: BanditReplay, kind: string): (tokens: ChartTokens) => Built {
  return (tokens) => ({
    data: regretLines(replay.periods, replay.regret_fixed, replay.regret_bandit, tokens),
    layout: {
      xaxis: { type: "date", nticks: 6, tickangle: 0 },
      yaxis: { ...axisTitle(regretAxis(kind)), zeroline: true, zerolinecolor: tokens.ink3, zerolinewidth: 1 },
      hovermode: "x unified",
      ...LEGEND,
    },
  });
}

/** Plan's simulation: the fixed split's expected cumulative regret and the bandit's mean over the
 *  runs, with its 10–90 % band, against the units that entered the test. */
export function banditSimulationFigure(simulation: BanditSimulation): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const x = simulation.cumulative_units;
    const color = tokens.palette[0] ?? tokens.ink1;
    return {
      data: [
        { type: "scatter", mode: "lines", x, y: simulation.regret_bandit_low, line: { width: 0 }, hoverinfo: "skip", showlegend: false } as Data,
        {
          type: "scatter",
          mode: "lines",
          x,
          y: simulation.regret_bandit_high,
          fill: "tonexty",
          fillcolor: withAlpha(color, 0.15),
          line: { width: 0 },
          hoverinfo: "skip",
          showlegend: false,
        } as Data,
        ...regretLines(x, simulation.regret_fixed, simulation.regret_bandit, tokens),
      ],
      layout: {
        xaxis: { ...axisTitle("Units in the test"), showgrid: true },
        yaxis: { ...axisTitle(regretAxis(simulation.kind)), rangemode: "tozero" },
        hovermode: "x unified",
        ...LEGEND,
      },
    };
  };
}
