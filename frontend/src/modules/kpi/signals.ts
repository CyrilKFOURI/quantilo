import type { Layout } from "plotly.js-dist-min";

import type { KpiResult } from "../../api/client";
import type { ChartTokens } from "../../charts/theme";

/** 054: what the engine signals, placed by its scope; never shown by colour alone. */
export type KpiSignal = KpiResult["signals"][number];
type Annotation = NonNullable<Layout["annotations"]>[number];
type Shape = NonNullable<Layout["shapes"]>[number];
export type SignalScope = KpiSignal["scope"];
export type CellFlag = { kind: KpiSignal["kind"]; severity: KpiSignal["severity"]; message: string };

/** The table marks these kinds on their cells (a level shift is the chart's business). */
const TABLE_KINDS = new Set<KpiSignal["kind"]>(["anomaly", "low_n", "thin_period"]);

export const byScope = (result: KpiResult, scope: SignalScope): KpiSignal[] =>
  (result.signals ?? []).filter((s) => s.scope === scope);

const key = (measure: string, period: string | null, g1: string | null, g2: string | null) =>
  `${measure}\u0001${period ?? ""}\u0001${g1 ?? "\u0000"}\u0001${g2 ?? "\u0000"}`;

/** One lookup for the table: a cell's flag by measure, period start (null: the whole range) and
 *  labels. A signal for every measure (a thin period) marks every cell of its period. */
export function cellFlags(result: KpiResult) {
  const exact = new Map<string, CellFlag>();
  const periods = new Map<string, CellFlag>();
  for (const signal of result.signals ?? []) {
    if (!TABLE_KINDS.has(signal.kind)) continue;
    for (const cell of signal.cells ?? []) {
      // a cell says its own count; the sentence about every cell stays under the tiles
      const n = typeof cell.value === "number" ? Math.round(cell.value).toLocaleString("en-US") : null;
      const message =
        signal.kind === "low_n" && n !== null
          ? `Low n: ${n} values, under 30 (${signal.rule})`
          : signal.kind === "thin_period" && n !== null
            ? `Thin period: ${n} rows, under 20 % of the median period (${signal.rule})`
            : signal.message;
      const flag: CellFlag = { kind: signal.kind, severity: signal.severity, message };
      if (signal.measure === null) {
        if (cell.period) periods.set(cell.period, flag);
      } else {
        exact.set(key(signal.measure, cell.period, cell.g1 ?? null, cell.g2 ?? null), flag);
      }
    }
  }
  return (measure: string, period: string | null, g1: string | null, g2: string | null): CellFlag | null =>
    exact.get(key(measure, period, g1, g2)) ?? (period ? periods.get(period) : undefined) ?? null;
}

/** Markers on a chart over time (P12): a ▲ above an unusual period, a dotted rule before the
 *  first period of a new level; the hover says the message. Neutral ink: the symbol and the
 *  words carry the meaning. */
export function signalMarks(
  result: KpiResult,
  measure: string,
  tokens: ChartTokens,
): Pick<Layout, "annotations" | "shapes"> | null {
  const series = byScope(result, "series").filter((s) => s.measure === measure);
  if (!series.length || !result.resolved.time) return null;
  const index = new Map(result.periods.map((p, i) => [p.start, i]));
  const annotations: Partial<Annotation>[] = [];
  const shapes: Partial<Shape>[] = [];
  for (const signal of series) {
    const at = index.get(signal.cells?.[0]?.period ?? "");
    if (at === undefined) continue;
    if (signal.kind === "anomaly") {
      annotations.push({
        x: at,
        xref: "x",
        y: 1,
        yref: "paper",
        yanchor: "top",
        text: "▲",
        showarrow: false,
        font: { size: 10, color: tokens.ink2 },
        hovertext: signal.message,
        captureevents: true,
      });
    } else if (signal.kind === "level_shift") {
      shapes.push({
        type: "line",
        x0: at - 0.5,
        x1: at - 0.5,
        xref: "x",
        y0: 0,
        y1: 1,
        yref: "paper",
        line: { color: tokens.ink3, width: 1, dash: "dot" },
      });
      annotations.push({
        x: at - 0.5,
        xref: "x",
        y: 1,
        yref: "paper",
        yanchor: "top",
        xanchor: "left",
        text: "level shift",
        showarrow: false,
        font: { size: 10, color: tokens.ink3 },
        hovertext: signal.message,
        captureevents: true,
      });
    }
  }
  return annotations.length || shapes.length ? { annotations: annotations as Annotation[], shapes: shapes as Shape[] } : null;
}
