import type { Chart } from "../../api/client";
import type { Condition } from "../../api/filters";
import type { PointClick } from "../../charts/Plot";
import { formatNumber } from "../../charts/figures";
import { t } from "../../lib/i18n";
import { parts } from "./conditions";

export type ChartAction = { label: string; condition: Condition };

const PART_VIEWS = new Set(["month", "day_of_week", "hour"]);

/** What a click on a chart can filter: "Keep only" / "Exclude" a bar, or keep a histogram range. */
export function actionsFor(column: string, chart: Chart, point: PointClick): ChartAction[] {
  if (chart.kind === "histogram" && chart.view !== "lengths") {
    const bin = chart.bins[point.index];
    if (!bin) return [];
    const range = { column, op: "between" as const, low: bin.start, high: bin.end };
    const labels = { low: formatNumber(bin.start), high: formatNumber(bin.end) };
    return [
      { label: t("filters.keepRange", labels), condition: { ...range, negate: false } },
      { label: t("filters.excludeRange", labels), condition: { ...range, negate: true } },
    ];
  }
  if (chart.kind !== "bars") return [];
  const custom = Array.isArray(point.customdata) ? point.customdata : [];
  const label = String(custom[0] ?? point.y ?? point.x ?? "");
  if (label === t("fig.chart.missing")) {
    return [
      { label: t("filters.keepOnly", { value: label }), condition: { column, op: "is_null", negate: false } },
      { label: t("filters.exclude", { value: label }), condition: { column, op: "is_null", negate: true } },
    ];
  }
  if (!label || chart.view === "pie" || label.startsWith(`${t("fig.chart.others")} (`)) return [];
  if (PART_VIEWS.has(chart.view)) {
    const value = parts()[chart.view]?.values.find((item) => item.label === label || item.english === label || item.value === label)?.value;
    if (value === undefined) return [];
    const base = { column, op: "part_in" as const, part: chart.view as Condition["part"], values: [value] };
    return [
      { label: t("filters.keepOnly", { value: label }), condition: { ...base, negate: false } },
      { label: t("filters.exclude", { value: label }), condition: { ...base, negate: true } },
    ];
  }
  const base = { column, op: "in" as const, values: [label] };
  return [
    { label: t("filters.keepOnly", { value: label }), condition: { ...base, negate: false } },
    { label: t("filters.exclude", { value: label }), condition: { ...base, negate: true } },
  ];
}
