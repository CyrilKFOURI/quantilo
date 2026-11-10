import type { Data, Layout } from "plotly.js-dist-min";

import type { Comparison } from "../../api/client";
import { formatNumber, withAlpha } from "../../charts/figures";
import type { ChartTokens } from "../../charts/theme";

type Built = { data: Data[]; layout?: Partial<Layout> };
type Figure = (tokens: ChartTokens) => Built;

const OTHERS = "Others";
const legend: Partial<Layout> = {
  showlegend: true,
  legend: { orientation: "h", x: 0, y: 1.1, font: { size: 11 } },
};
const axisTitle = (text: string) => ({ title: { text, font: { size: 11 } } });

/** Categorical colors in fixed order; "Others" is always muted grey. */
function colorOf(labels: string[], tokens: ChartTokens): Map<string, string> {
  const named = labels.filter((label) => label !== OTHERS);
  const map = new Map(
    named.map((label, i) => [label, tokens.palette[i % 8] ?? tokens.ink3]),
  );
  map.set(OTHERS, tokens.ink3);
  return map;
}

function uniqueInOrder(values: string[]): string[] {
  const counts = new Map<string, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  return [...counts.keys()].sort((a, b) =>
    a === OTHERS
      ? 1
      : b === OTHERS
        ? -1
        : (counts.get(b) ?? 0) - (counts.get(a) ?? 0),
  );
}

// --- num × num ---------------------------------------------------------------------------------

export function scatterFigure(
  result: Comparison,
  regression: boolean,
  lowess = false,
): Figure {
  return (tokens) => {
    const points = result.points;
    if (!points) return { data: [] };
    const size = points.x.length > 5000 ? 4 : 7;
    const traces: Data[] = [];
    if (points.color) {
      const labels = uniqueInOrder(points.color);
      const colors = colorOf(labels, tokens);
      for (const label of labels) {
        const index = points.color.flatMap((value, i) =>
          value === label ? [i] : [],
        );
        traces.push({
          type: "scattergl",
          mode: "markers",
          name: label,
          x: index.map((i) => points.x[i]),
          y: index.map((i) => points.y[i]),
          marker: {
            color: colors.get(label),
            size,
            opacity: 0.7,
            line: { color: tokens.surface, width: 0.5 },
          },
          hovertemplate: `${label}<br>%{x:,.4g}, %{y:,.4g}<extra></extra>`,
        } as Data);
      }
    } else {
      traces.push({
        type: "scattergl",
        mode: "markers",
        name: "rows",
        x: points.x,
        y: points.y,
        marker: {
          color: tokens.numeric,
          size,
          opacity: 0.55,
          line: { color: tokens.surface, width: 0.5 },
        },
        hovertemplate: "%{x:,.4g}, %{y:,.4g}<extra></extra>",
      } as Data);
    }
    const line = result.regression;
    if (regression && line && points.x.length) {
      const low = Math.min(...points.x);
      const high = Math.max(...points.x);
      traces.push({
        type: "scatter",
        mode: "lines",
        name: "Linear fit",
        x: [low, high],
        y: [
          line.intercept + line.slope * low,
          line.intercept + line.slope * high,
        ],
        line: { color: tokens.ink1, width: 2, dash: "dash" },
        hovertemplate: `y = ${formatNumber(line.slope)}·x + ${formatNumber(line.intercept)}<extra></extra>`,
      } as Data);
    }
    const curve = result.lowess;
    if (lowess && curve) {
      traces.push({
        type: "scatter",
        mode: "lines",
        name: "LOWESS",
        x: curve.x,
        y: curve.y,
        line: { color: tokens.accent, width: 2.5 },
        hovertemplate: "LOWESS: %{x:,.4g}, %{y:,.4g}<extra></extra>",
      } as Data);
    }
    return {
      data: traces,
      layout: {
        ...(points.color ? legend : {}),
        xaxis: { ...axisTitle(result.x), showgrid: true },
        yaxis: axisTitle(result.y),
      },
    };
  };
}

export function densityFigure(result: Comparison): Figure {
  return (tokens) => {
    const density = result.density;
    if (!density) return { data: [] };
    const centers = (edges: number[]) =>
      edges.slice(0, -1).map((edge, i) => (edge + (edges[i + 1] ?? edge)) / 2);
    return {
      data: [
        {
          type: "heatmap",
          x: centers(density.x_edges),
          y: centers(density.y_edges),
          z: density.counts.map((row) =>
            row.map((count) => (count === 0 ? null : count)),
          ),
          colorscale: [
            [0, withAlpha(tokens.numeric, 0.12)],
            [1, tokens.numeric],
          ],
          showscale: true,
          colorbar: {
            thickness: 10,
            outlinewidth: 0,
            tickfont: { size: 10, color: tokens.ink3 },
          },
          hovertemplate: "%{x:,.3g}, %{y:,.3g}<br>%{z:,} rows<extra></extra>",
        } as Data,
      ],
      layout: {
        xaxis: { ...axisTitle(result.x), showgrid: false },
        yaxis: { ...axisTitle(result.y), showgrid: false },
      },
    };
  };
}

export function scatter3dFigure(result: Comparison): Figure {
  return (tokens) => {
    const points = result.points;
    if (!points?.z) return { data: [] };
    const axis = (title: string) => ({
      title: { text: title, font: { size: 11, color: tokens.ink2 } },
      color: tokens.ink3,
      gridcolor: tokens.line,
      zerolinecolor: tokens.line,
      backgroundcolor: "rgba(0,0,0,0)",
    });
    const labels = points.color ? uniqueInOrder(points.color) : [null];
    const colors = points.color ? colorOf(labels as string[], tokens) : null;
    const traces = labels.map((label) => {
      const index = points.x.flatMap((_, i) =>
        label === null || points.color?.[i] === label ? [i] : [],
      );
      return {
        type: "scatter3d",
        mode: "markers",
        name: label ?? "rows",
        x: index.map((i) => points.x[i]),
        y: index.map((i) => points.y[i]),
        z: index.map((i) => points.z?.[i]),
        marker: {
          size: 3,
          color: label === null ? tokens.numeric : colors?.get(label),
          opacity: 0.75,
        },
        hovertemplate: `${label ? `${label}<br>` : ""}%{x:,.3g}, %{y:,.3g}, %{z:,.3g}<extra></extra>`,
      } as Data;
    });
    return {
      data: traces,
      layout: {
        ...(points.color ? legend : {}),
        margin: { l: 0, r: 0, t: 0, b: 0 },
        scene: {
          xaxis: axis(result.x),
          yaxis: axis(result.y),
          zaxis: axis(result.z ?? ""),
          bgcolor: "rgba(0,0,0,0)",
        },
      } as Partial<Layout>,
    };
  };
}

// --- num × groups -------------------------------------------------------------------------------

export type GroupView = "box" | "violin" | "means";

/** What gave a group's mean interval, for the hover (051). */
const intervalMethod = (group: Comparison["groups"][number]) => (group.ci_method === "bootstrap" ? "bootstrap" : "Student t");

/** Groups split by a colour: one box (violin, mean) per colour level, side by side. */
function coloredGroups(result: Comparison, view: GroupView, tokens: ChartTokens, vertical: boolean): Data[] {
  const colors = colorOf(result.color_levels, tokens);
  return result.color_levels.map((level) => {
    const groups = result.groups.filter((g) => g.color === level);
    const labels = groups.map((g) => g.label);
    const color = colors.get(level) ?? tokens.ink3;
    const shared = { name: level, legendgroup: level, offsetgroup: level };
    if (view === "means") {
      return {
        type: "scatter",
        mode: "markers",
        ...shared,
        [vertical ? "x" : "y"]: labels,
        [vertical ? "y" : "x"]: groups.map((g) => g.mean),
        marker: { size: 10, color },
        [vertical ? "error_y" : "error_x"]: {
          type: "data",
          symmetric: false,
          array: groups.map((g) => (g.ci_high ?? 0) - (g.mean ?? 0)),
          arrayminus: groups.map((g) => (g.mean ?? 0) - (g.ci_low ?? 0)),
          color,
          thickness: 1.5,
          width: 6,
        },
        customdata: groups.map((g) => [g.n, formatNumber(g.ci_low), formatNumber(g.ci_high), intervalMethod(g)]),
        hovertemplate: `${level} · %{${vertical ? "x" : "y"}}<br>mean %{${vertical ? "y" : "x"}:,.4g}<br>95% CI %{customdata[1]} – %{customdata[2]} (%{customdata[3]})<br>n = %{customdata[0]:,}<extra></extra>`,
      } as Data;
    }
    if (view === "violin") {
      return {
        type: "violin",
        ...shared,
        [vertical ? "x" : "y"]: groups.flatMap((g) => g.sample.map(() => g.label)),
        [vertical ? "y" : "x"]: groups.flatMap((g) => g.sample),
        orientation: vertical ? "v" : "h",
        line: { color, width: 1.5 },
        fillcolor: withAlpha(color, 0.25),
        box: { visible: true, width: 0.15, fillcolor: tokens.surface },
        meanline: { visible: true },
        points: false,
        spanmode: "hard",
      } as Data;
    }
    return {
      type: "box",
      ...shared,
      [vertical ? "x" : "y"]: labels,
      q1: groups.map((g) => g.q1),
      median: groups.map((g) => g.median),
      q3: groups.map((g) => g.q3),
      lowerfence: groups.map((g) => g.low),
      upperfence: groups.map((g) => g.high),
      mean: groups.map((g) => g.mean),
      orientation: vertical ? "v" : "h",
      fillcolor: withAlpha(color, 0.25),
      line: { color, width: 1.5 },
    } as Data;
  });
}

export function groupsFigure(result: Comparison, view: GroupView): Figure {
  return (tokens) => {
    const vertical = result.group_axis === "x";
    const numeric = vertical ? result.y : result.x;
    // the server may put the colour's category on the axis (R-CMP-04)
    const category = result.layout?.axis ?? (vertical ? result.x : result.y);
    const colors = colorOf(
      result.groups.map((g) => g.label),
      tokens,
    );
    const place = (labels: string[], values: (number | null)[]) =>
      vertical ? { x: labels, y: values } : { x: values, y: labels };
    let data: Data[];
    if (result.color_levels.length > 0) {
      data = coloredGroups(result, view, tokens, vertical);
    } else if (view === "means") {
      data = [
        {
          type: "scatter",
          mode: "markers",
          ...place(
            result.groups.map((g) => g.label),
            result.groups.map((g) => g.mean),
          ),
          marker: {
            size: 10,
            color: result.groups.map((g) => colors.get(g.label) ?? tokens.ink3),
          },
          [vertical ? "error_y" : "error_x"]: {
            type: "data",
            symmetric: false,
            array: result.groups.map((g) => (g.ci_high ?? 0) - (g.mean ?? 0)),
            arrayminus: result.groups.map(
              (g) => (g.mean ?? 0) - (g.ci_low ?? 0),
            ),
            color: tokens.ink2,
            thickness: 1.5,
            width: 6,
          },
          customdata: result.groups.map((g) => [
            g.n,
            formatNumber(g.ci_low),
            formatNumber(g.ci_high),
            intervalMethod(g),
          ]),
          hovertemplate: `%{${vertical ? "x" : "y"}}<br>mean %{${vertical ? "y" : "x"}:,.4g}<br>95% CI %{customdata[1]} – %{customdata[2]} (%{customdata[3]})<br>n = %{customdata[0]:,}<extra></extra>`,
        } as Data,
      ];
    } else if (view === "violin") {
      data = result.groups.map(
        (group) =>
          ({
            type: "violin",
            name: group.label,
            [vertical ? "y" : "x"]: group.sample,
            [vertical ? "x0" : "y0"]: group.label,
            orientation: vertical ? "v" : "h",
            line: { color: colors.get(group.label), width: 1.5 },
            fillcolor: withAlpha(colors.get(group.label) ?? tokens.ink3, 0.25),
            box: { visible: true, width: 0.15, fillcolor: tokens.surface },
            meanline: { visible: true },
            points: false,
            spanmode: "hard",
          }) as Data,
      );
    } else {
      data = result.groups.map(
        (group) =>
          ({
            type: "box",
            name: group.label,
            [vertical ? "x" : "y"]: [group.label],
            q1: [group.q1],
            median: [group.median],
            q3: [group.q3],
            lowerfence: [group.low],
            upperfence: [group.high],
            mean: [group.mean],
            orientation: vertical ? "v" : "h",
            fillcolor: withAlpha(colors.get(group.label) ?? tokens.ink3, 0.25),
            line: { color: colors.get(group.label), width: 1.5 },
            hoverinfo: vertical ? "y" : "x",
          }) as Data,
      );
    }
    const categoryAxis = {
      ...axisTitle(category),
      type: "category" as const,
      showgrid: false,
      ...(vertical ? {} : { autorange: "reversed" as const }),
    };
    const valueAxis = { ...axisTitle(numeric), showgrid: true };
    return {
      data,
      layout: {
        showlegend: result.color_levels.length > 0,
        legend: { title: { text: result.layout?.color ?? result.color ?? "" }, orientation: "h", y: 1.08 },
        boxmode: "group",
        violinmode: "group",
        scattermode: "group",
        xaxis: vertical ? categoryAxis : valueAxis,
        yaxis: vertical ? valueAxis : categoryAxis,
      } as Partial<Layout>,
    };
  };
}

// --- cat × cat ----------------------------------------------------------------------------------

export type TableView = "heatmap" | "stacked";

export function contingencyFigure(result: Comparison, view: TableView): Figure {
  return (tokens) => {
    const table = result.contingency;
    if (!table) return { data: [] };
    const columnTotals = table.columns.map((_, j) =>
      table.counts.reduce((sum, row) => sum + (row[j] ?? 0), 0),
    );
    if (view === "heatmap") {
      return {
        data: [
          {
            type: "heatmap",
            x: table.columns,
            y: table.rows,
            z: table.counts,
            text: table.counts.map((row) =>
              row.map((count) => (count ? count.toLocaleString("en-US") : "")),
            ),
            texttemplate:
              table.columns.length * table.rows.length <= 120 ? "%{text}" : "",
            textfont: { size: 11, color: tokens.ink1 },
            colorscale: [
              [0, withAlpha(tokens.categorical, 0.06)],
              [1, withAlpha(tokens.categorical, 0.7)],
            ],
            showscale: false,
            xgap: 2,
            ygap: 2,
            hovertemplate: `${result.x}: %{x}<br>${result.y}: %{y}<br>%{z:,} rows<extra></extra>`,
          } as Data,
        ],
        layout: {
          xaxis: { ...axisTitle(result.x), type: "category", showgrid: false },
          yaxis: {
            ...axisTitle(result.y),
            type: "category",
            showgrid: false,
            autorange: "reversed",
          },
        },
      };
    }
    const colors = colorOf(table.rows, tokens);
    return {
      data: table.rows.map(
        (row, i) =>
          ({
            type: "bar",
            name: row,
            x: table.columns,
            y: table.columns.map(
              (_, j) =>
                ((table.counts[i]?.[j] ?? 0) / (columnTotals[j] || 1)) * 100,
            ),
            customdata: table.columns.map((_, j) => table.counts[i]?.[j] ?? 0),
            marker: {
              color: colors.get(row),
              line: { color: tokens.surface, width: 1 },
            },
            hovertemplate: `${row}<br>%{x}: %{y:.1f}% (%{customdata:,} rows)<extra></extra>`,
          }) as Data,
      ),
      layout: {
        ...legend,
        barmode: "stack",
        xaxis: { ...axisTitle(result.x), type: "category" },
        yaxis: {
          ...axisTitle(`${result.y} (% of each ${result.x})`),
          ticksuffix: "%",
          range: [0, 100],
        },
      },
    };
  };
}

// --- dates --------------------------------------------------------------------------------------

/** Aggregated values, rolling mean and (for a single series) a ± 1σ band. */
export function timelineFigure(
  result: Comparison,
  layers: Set<"values" | "mean" | "band"> = new Set(["values", "mean", "band"]),
): Figure {
  return (tokens) => {
    const numeric = result.roles[result.x] === "date" ? result.y : result.x;
    const single = result.series.length === 1;
    const colors = colorOf(
      result.series.map((s) => s.label ?? ""),
      tokens,
    );
    const time = result.time;
    const what = time ? `${time.aggregate} ${numeric}` : numeric;
    const rolled = time?.window
      ? `${time.window}-${time.period} rolling mean`
      : "rolling mean";
    const data: Data[] = [];
    for (const series of result.series) {
      const color = single
        ? (tokens.palette[0] ?? tokens.date)
        : (colors.get(series.label ?? "") ?? tokens.ink3);
      const name = series.label ?? what;
      const hasRolling = series.rolling_mean.length > 0;
      if (single && hasRolling && layers.has("band")) {
        const upper = series.rolling_mean.map((m, i) =>
          m === null || series.rolling_std[i] == null
            ? null
            : m + (series.rolling_std[i] ?? 0),
        );
        const lower = series.rolling_mean.map((m, i) =>
          m === null || series.rolling_std[i] == null
            ? null
            : m - (series.rolling_std[i] ?? 0),
        );
        data.push(
          {
            type: "scatter",
            mode: "lines",
            x: series.periods,
            y: lower,
            line: { width: 0 },
            hoverinfo: "skip",
            showlegend: false,
            name: "lower",
          } as Data,
          {
            type: "scatter",
            mode: "lines",
            x: series.periods,
            y: upper,
            fill: "tonexty",
            fillcolor: withAlpha(color, 0.15),
            line: { width: 0 },
            hoverinfo: "skip",
            name: "± 1σ band",
          } as Data,
        );
      }
      if (layers.has("values") || !hasRolling) {
        data.push({
          type: "scatter",
          mode: series.periods.length > 60 ? "lines" : "lines+markers",
          name: single ? what : name,
          legendgroup: name,
          x: series.periods,
          y: series.values,
          customdata: series.counts,
          connectgaps: false,
          line: {
            color:
              hasRolling && layers.has("mean") ? withAlpha(color, 0.55) : color,
            width: 1.25,
          },
          marker: { size: 4 },
          hovertemplate: `${series.label ? `${series.label}<br>` : ""}%{x}<br>${what}: %{y:,.4g}<br>%{customdata:,} rows<extra></extra>`,
        } as Data);
      }
      if (hasRolling && layers.has("mean")) {
        data.push({
          type: "scatter",
          mode: "lines",
          name: single ? rolled : `${name} · mean`,
          legendgroup: name,
          showlegend: single || !layers.has("values"),
          x: series.periods,
          y: series.rolling_mean,
          customdata: series.rolling_std,
          line: { color, width: 2.5 },
          hovertemplate: `${series.label ? `${series.label}<br>` : ""}%{x}<br>${rolled}: %{y:,.4g}<br>σ %{customdata:,.3g}<extra></extra>`,
        } as Data);
      }
    }
    return {
      data,
      layout: {
        ...legend,
        hovermode: "x unified",
        xaxis: { type: "date", nticks: 8, tickangle: 0 },
        yaxis: axisTitle(what),
      },
    };
  };
}

export function dateCountsFigure(result: Comparison): Figure {
  return (tokens) => {
    const colors = colorOf(
      result.series.map((s) => s.label ?? ""),
      tokens,
    );
    return {
      data: result.series.map(
        (series) =>
          ({
            type: "scatter",
            mode: "lines",
            stackgroup: "rows",
            name: series.label,
            x: series.periods,
            y: series.values,
            line: { color: colors.get(series.label ?? ""), width: 1 },
            hovertemplate: `${series.label}<br>%{x}: %{y:,} rows<extra></extra>`,
          }) as Data,
      ),
      layout: {
        ...legend,
        xaxis: { type: "date", nticks: 8, tickangle: 0 },
        yaxis: axisTitle("rows"),
      },
    };
  };
}
