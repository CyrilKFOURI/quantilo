import type { KpiResult, KpiSeries } from "../../api/client";
import type { ShowAs } from "./definition";
import { cellFlags, type CellFlag } from "./signals";
import { t } from "../../lib/i18n";

/** "Show as" modes that only mean something for a measure that adds up (spec R-TAB-11). */
export const ADDITIVE_MODES = new Set<ShowAs>(["share_total", "share_row", "share_column", "running"]);
/** Modes that read the previous period or last year: they need a time axis. */
export const TIME_MODES = new Set<ShowAs>(["change", "change_pct", "change_yoy", "running", "rolling", "trend"]);
/** 053: smoothed modes, drawn as a curve over the values (R-SHOW-01, R-SHOW-02). */
export const SMOOTH_MODES = new Set<ShowAs>(["rolling", "trend"]);
/** R-SHOW-01: the trailing window of a rolling mean, by grain (a short cycle). */
export const ROLLING_WINDOW: Record<string, number> = { day: 7, week: 4, month: 3, quarter: 4, year: 3 };

/** The rolling mean's window for this result (3 when the grain is unknown). */
export const rollingWindow = (result: KpiResult) => ROLLING_WINDOW[result.resolved.grain?.value ?? ""] ?? 3;

/** R-SHOW-01: the mean of the complete periods of the window ending at each period; empty until
 *  the window is full, and for an incomplete period. */
export function rollingMean(values: (number | null)[], partial: boolean[], window: number): (number | null)[] {
  return values.map((value, i) => {
    if (i < window - 1 || partial[i] || value === null) return null;
    const kept = values.slice(i - window + 1, i + 1).filter((v, j): v is number => v !== null && !partial[i - window + 1 + j]);
    return kept.length ? kept.reduce((a, b) => a + b, 0) / kept.length : null;
  });
}
export const DIVERGING_MODES = new Set<ShowAs>(["change", "change_pct", "change_yoy"]);
export const COLUMNS_MAX_PERIODS = 13;

type Key = string | null;
const keyOf = (measure: string, g1: Key, g2: Key) => `${measure}\u0001${g1 ?? "\u0000"}\u0001${g2 ?? "\u0000"}`;

/** Finds a series by measure and split labels (null = the total over that split). */
export function seriesIndex(result: KpiResult) {
  const map = new Map<string, KpiSeries>();
  for (const s of result.series) map.set(keyOf(s.measure, s.g1, s.g2), s);
  return (measure: string, g1: Key = null, g2: Key = null) => map.get(keyOf(measure, g1, g2));
}

const ratio = (a: number | null, b: number | null) =>
  a === null || b === null || b === 0 ? null : a / b;
const change = (a: number | null, b: number | null) =>
  a === null || b === null || b === 0 ? null : (a - b) / Math.abs(b);

/** The mode actually applied to a measure: a share or a running total of a mean falls back to
 *  the value (the option is greyed out with its reason in Customize). */
export function effectiveMode(showAs: ShowAs, additive: boolean, time: boolean): ShowAs {
  if (ADDITIVE_MODES.has(showAs) && !additive) return "value";
  if (TIME_MODES.has(showAs) && !time) return "value";
  return showAs;
}

/** One series' values per period as "Show as" presents them (share of row / column are the
 *  table's business: here they read as a share of the period's total). */
export function shownValues(result: KpiResult, s: KpiSeries, mode: ShowAs): (number | null)[] {
  const find = seriesIndex(result);
  if (mode === "share_total" || mode === "share_row" || mode === "share_column") {
    const total = find(s.measure)?.values ?? [];
    return s.values.map((v, i) => ratio(v, total[i] ?? null));
  }
  if (mode === "change") return s.values.map((v, i) => (v === null || s.previous[i] === null ? null : v - (s.previous[i] as number)));
  if (mode === "change_pct") return s.values.map((v, i) => change(v, s.previous[i] ?? null));
  if (mode === "change_yoy") return s.values.map((v, i) => change(v, s.last_year[i] ?? null));
  if (mode === "running") {
    let sum = 0;
    return s.values.map((v) => (sum += v ?? 0));
  }
  if (mode === "rolling") return rollingMean(s.values, result.periods.map((p) => p.partial), rollingWindow(result));
  if (mode === "trend") return s.trend ?? s.values.map(() => null);
  return s.values;
}

/** A series over the whole range: shares are of the grand total, changes have no total. */
export function shownOverall(result: KpiResult, s: KpiSeries, mode: ShowAs): number | null {
  if (mode === "share_total" || mode === "share_row" || mode === "share_column") {
    return ratio(s.overall, seriesIndex(result)(s.measure)?.overall ?? null);
  }
  if (DIVERGING_MODES.has(mode)) return null;
  return s.overall;
}

// --- Table ---------------------------------------------------------------------------------------

export type HeadCell = { label: string; span: number };
export type TableCell = {
  value: number | null;
  measure: string;
  mode: ShowAs;
  partial: boolean;
  total: boolean;
  shade: number | null; // 0…1 sequential, −1…1 diverging
  /** 054: a signal on this cell (low n, unusual period, thin period), with its sentence. */
  flag: CellFlag | null;
};
export type TableRow = {
  key: string;
  label: string;
  indent: number;
  kind: "data" | "subtotal" | "total" | "section";
  partial: boolean;
  cells: TableCell[];
};
export type Table = {
  corner: string;
  head: HeadCell[][];
  rows: TableRow[];
  note: string | null;
  sortable: boolean;
};

export type TableOptions = {
  layout: string; // the resolved layout: periods_rows · periods_columns · groups_rows · crosstab · single
  showAs: ShowAs;
  totals: boolean;
  shading: boolean;
  compare: ("previous" | "last_year")[];
  sort: string | null; // "<column index>:asc|desc"
  hideOther: boolean;
  labels: Record<string, string>; // display name per measure id
};

const OTHER = "Other";

export function buildTable(result: KpiResult, options: TableOptions): Table {
  const find = seriesIndex(result);
  const time = result.resolved.time;
  const measures = result.measures.map((m) => ({
    id: m.id,
    label: options.labels[m.id] ?? m.label,
    mode: effectiveMode(options.showAs, m.additive, time),
  }));
  const levels = result.splits.map((split) =>
    options.hideOther ? split.levels.filter((l) => l !== OTHER) : split.levels,
  );
  const first = levels[0] ?? [];
  const second = levels[1] ?? [];
  const periods = result.periods;
  const cache = new Map<string, (number | null)[]>();
  const flagOf = cellFlags(result);
  const values = (s: KpiSeries, mode: ShowAs) => {
    const key = `${mode}|${s.measure}|${s.g1}|${s.g2}`;
    if (!cache.has(key)) cache.set(key, shownValues(result, s, mode));
    return cache.get(key) as (number | null)[];
  };
  const cell = (measure: (typeof measures)[number], g1: Key, g2: Key, p: number | null, total = false): TableCell => {
    const s = find(measure.id, g1, g2);
    // shares of a row or a column start from the values, divided once the totals are laid out
    const base = measure.mode === "share_row" || measure.mode === "share_column" ? "value" : measure.mode;
    const value = s ? (p === null ? shownOverall(result, s, base) : values(s, base)[p] ?? null) : null;
    return {
      value,
      measure: measure.id,
      mode: measure.mode,
      partial: p !== null && Boolean(periods[p]?.partial),
      total: total || g1 === null || (result.splits.length > 1 && g2 === null) || p === null,
      shade: null,
      flag: flagOf(measure.id, p === null ? null : (periods[p]?.start ?? null), g1, g2),
    };
  };
  const compareCells = (measure: (typeof measures)[number], g1: Key, g2: Key, p: number): TableCell[] => {
    const s = find(measure.id, g1, g2);
    return options.compare.map((kind) => {
      const now = s?.values[p] ?? null;
      const before = (kind === "previous" ? s?.previous[p] : s?.last_year[p]) ?? null;
      const mode: ShowAs = kind === "previous" ? "change" : "change_yoy";
      const value = kind === "previous" ? (now === null || before === null ? null : now - before) : change(now, before);
      return { value, measure: measure.id, mode, partial: Boolean(periods[p]?.partial), total: false, shade: null, flag: null };
    });
  };
  const compareHead = options.compare.map((kind) => (kind === "previous" ? t("fig.pivot.vsPrevious") : t("fig.pivot.vsLastYear")));
  const multi = measures.length > 1;
  let table: Table;

  if (options.layout === "single") {
    table = {
      corner: t("fig.pivot.measure"),
      head: [[{ label: t("fig.pivot.value"), span: 1 }]],
      rows: measures.map((m) => ({ key: m.id, label: m.label, indent: 0, kind: "data", partial: false, cells: [cell(m, null, null, null, false)] })),
      note: null,
      sortable: false,
    };
  } else if (options.layout === "groups_rows") {
    const main = measures[0];
    const share = result.measures[0]?.additive;
    const head = [[...measures.map((m) => ({ label: m.label, span: 1 })), ...(share ? [{ label: t("fig.pivot.ofTotal"), span: 1 }] : [])]];
    const row = (g1: Key, kind: TableRow["kind"]): TableRow => ({
      key: g1 ?? "total",
      label: g1 ?? t("fig.pivot.total"),
      indent: 0,
      kind,
      partial: false,
      cells: [
        ...measures.map((m) => cell(m, g1, null, null, kind === "total")),
        ...(share && main ? [{ ...cell({ ...main, mode: "share_total" }, g1, null, null, kind === "total") }] : []),
      ],
    });
    table = {
      corner: result.splits[0]?.column ?? "",
      head,
      rows: [...first.map((g) => row(g, "data")), ...(options.totals ? [row(null, "total")] : [])],
      note: null,
      sortable: true,
    };
  } else if (options.layout === "crosstab") {
    const columns = [...second, ...(options.totals ? [null] : [])];
    const head: HeadCell[][] = [];
    if (multi) head.push(measures.map((m) => ({ label: m.label, span: columns.length })));
    head.push(measures.flatMap(() => columns.map((g2) => ({ label: g2 ?? t("fig.pivot.total"), span: 1 }))));
    const row = (g1: Key, kind: TableRow["kind"]): TableRow => ({
      key: g1 ?? "total",
      label: g1 ?? t("fig.pivot.total"),
      indent: 0,
      kind,
      partial: false,
      cells: measures.flatMap((m) => columns.map((g2) => cell(m, g1, g2, null, kind === "total" || g2 === null))),
    });
    table = {
      corner: `${result.splits[0]?.column} \\ ${result.splits[1]?.column}`,
      head,
      rows: [...first.map((g) => row(g, "data")), ...(options.totals ? [row(null, "total")] : [])],
      note: null,
      sortable: true,
    };
  } else if (options.layout === "periods_columns") {
    const limit = result.splits.length === 2 && periods.length > COLUMNS_MAX_PERIODS;
    const shown = periods.map((_, i) => i).slice(limit ? -COLUMNS_MAX_PERIODS : 0);
    const last = shown.findLast((i) => !periods[i]?.partial) ?? shown[shown.length - 1];
    const head = [[
      ...shown.map((i) => ({ label: (periods[i]?.label ?? "") + (periods[i]?.partial ? " †" : ""), span: 1 })),
      ...(options.totals ? [{ label: t("fig.pivot.total"), span: 1 }] : []),
      ...compareHead.map((label) => ({ label: `${label} (${last === undefined ? "" : periods[last]?.label ?? ""})`, span: 1 })),
    ]];
    const line = (m: (typeof measures)[number], g1: Key, g2: Key, label: string, indent: number, kind: TableRow["kind"]): TableRow => ({
      key: `${m.id}|${g1}|${g2}`,
      label,
      indent,
      kind,
      partial: false,
      cells: [
        ...shown.map((i) => cell(m, g1, g2, i, kind !== "data")),
        ...(options.totals ? [cell(m, g1, g2, null, true)] : []),
        ...(last !== undefined ? compareCells(m, g1, g2, last) : []),
      ],
    });
    const rows: TableRow[] = [];
    for (const m of measures) {
      if (multi) rows.push({ key: `section-${m.id}`, label: m.label, indent: 0, kind: "section", partial: false, cells: [] });
      if (result.splits.length === 0) rows.push(line(m, null, null, multi ? t("fig.pivot.value") : m.label, 0, "data"));
      else if (result.splits.length === 1) {
        for (const g1 of first) rows.push(line(m, g1, null, g1, 0, "data"));
        if (options.totals) rows.push(line(m, null, null, t("fig.pivot.total"), 0, "total"));
      } else {
        for (const g1 of first) {
          rows.push(line(m, g1, null, g1, 0, "subtotal"));
          for (const g2 of second) rows.push(line(m, g1, g2, g2, 1, "data"));
        }
        if (options.totals) rows.push(line(m, null, null, t("fig.pivot.total"), 0, "total"));
      }
    }
    table = {
      corner: result.splits.map((s) => s.column).join(" › ") || t("fig.pivot.measure"),
      head,
      rows,
      note: limit ? t("fig.pivot.limitNote", { shown: String(COLUMNS_MAX_PERIODS), total: String(periods.length) }) : null,
      sortable: !multi && result.splits.length === 1,
    };
  } else {
    // periods_rows
    const groups: [Key, Key][] =
      result.splits.length === 0
        ? [[null, null]]
        : result.splits.length === 1
          ? [...first.map((g): [Key, Key] => [g, null]), ...(options.totals ? [[null, null] as [Key, Key]] : [])]
          : [
              ...first.flatMap((g1) => [...second.map((g2): [Key, Key] => [g1, g2]), [g1, null] as [Key, Key]]),
              ...(options.totals ? [[null, null] as [Key, Key]] : []),
            ];
    const span = groups.length * (1 + options.compare.length);
    const head: HeadCell[][] = [];
    if (multi && result.splits.length) head.push(measures.map((m) => ({ label: m.label, span })));
    if (result.splits.length === 2) {
      head.push(measures.flatMap(() => [
        ...first.map((g1) => ({ label: g1, span: (second.length + 1) * (1 + options.compare.length) })),
        ...(options.totals ? [{ label: t("fig.pivot.total"), span: 1 + options.compare.length }] : []),
      ]));
    }
    head.push(
      measures.flatMap((m) =>
        groups.flatMap(([g1, g2]) => [
          {
            label: result.splits.length === 0 ? m.label : result.splits.length === 1 ? g1 ?? t("fig.pivot.total") : g2 ?? (g1 ? t("fig.pivot.groupTotal", { group: g1 }) : t("fig.pivot.total")),
            span: 1,
          },
          ...compareHead.map((label) => ({ label, span: 1 })),
        ]),
      ),
    );
    const rows: TableRow[] = periods.map((period, p) => ({
      key: period.start,
      label: period.label + (period.partial ? " †" : ""),
      indent: 0,
      kind: "data",
      partial: period.partial,
      cells: measures.flatMap((m) => groups.flatMap(([g1, g2]) => [cell(m, g1, g2, p), ...compareCells(m, g1, g2, p)])),
    }));
    if (options.totals) {
      rows.push({
        key: "total",
        label: t("fig.pivot.total"),
        indent: 0,
        kind: "total",
        partial: false,
        cells: measures.flatMap((m) =>
          groups.flatMap(([g1, g2]) => [cell(m, g1, g2, null, true), ...options.compare.map(() => ({ ...cell(m, g1, g2, null, true), value: null, flag: null }))]),
        ),
      });
    }
    table = { corner: t("fig.pivot.period"), head, rows, note: null, sortable: true };
  }

  if (options.showAs === "share_row" || options.showAs === "share_column") relativeToTotals(table, options.showAs);
  if (options.sort && table.sortable) sortRows(table, options.sort);
  if (options.shading) shade(table);
  return table;
}

/** Share of the row's or the column's total, read from the table's own total cells. */
function relativeToTotals(table: Table, mode: "share_row" | "share_column") {
  const raw = table.rows.map((row) => row.cells.map((c) => c.value));
  const totalRow = table.rows.findIndex((r) => r.kind === "total");
  table.rows.forEach((row, r) => {
    row.cells.forEach((c, i) => {
      if (c.mode !== mode || c.value === null) return;
      if (mode === "share_column") {
        const base = totalRow >= 0 ? raw[totalRow]?.[i] : null;
        c.value = base ? (raw[r]?.[i] ?? 0) / base : null;
      } else {
        const block = row.cells.filter((x) => x.measure === c.measure);
        const totalCell = block.findLast((x) => x.total);
        const base = totalCell ? raw[r]?.[row.cells.indexOf(totalCell)] : null;
        c.value = base ? (raw[r]?.[i] ?? 0) / base : null;
      }
    });
  });
}

function sortRows(table: Table, sort: string) {
  const [index, direction] = sort.split(":");
  const column = Number(index);
  if (!Number.isInteger(column)) return;
  const sign = direction === "asc" ? 1 : -1;
  const body = table.rows.filter((r) => r.kind === "data");
  const rest = table.rows.filter((r) => r.kind !== "data");
  const value = (row: TableRow) => (column < 0 ? null : row.cells[column]?.value ?? null);
  body.sort((a, b) => {
    if (column < 0) return sign * a.label.localeCompare(b.label);
    const x = value(a);
    const y = value(b);
    if (x === null) return 1;
    if (y === null) return -1;
    return sign * (x - y);
  });
  table.rows = [...body, ...rest];
}

/** Light to dark per measure block (totals left out); changes diverge around 0 (R-TAB-08/09). */
function shade(table: Table) {
  const groups = new Map<string, TableCell[]>();
  for (const row of table.rows) {
    if (row.kind !== "data") continue;
    for (const c of row.cells) {
      if (c.total || c.value === null) continue;
      const key = `${c.measure}|${c.mode}`;
      groups.set(key, [...(groups.get(key) ?? []), c]);
    }
  }
  for (const cells of groups.values()) {
    const vals = cells.map((c) => c.value as number);
    if (cells[0] && DIVERGING_MODES.has(cells[0].mode)) {
      const extent = Math.max(...vals.map(Math.abs));
      for (const c of cells) c.shade = extent ? (c.value as number) / extent : 0;
    } else {
      const low = Math.min(...vals);
      const high = Math.max(...vals);
      for (const c of cells) c.shade = high > low ? ((c.value as number) - low) / (high - low) : null;
    }
  }
}

/** The table as text: the header rows (spans repeated as blanks) then the raw values. */
export function toDelimited(table: Table, separator: "," | "\t"): string {
  const escape = (text: string) =>
    separator === "," && /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text.replace(/[\t\n]/g, " ");
  const lines = table.head.map((row, i) => [
    i === table.head.length - 1 ? table.corner : "",
    ...row.flatMap((h) => [h.label, ...Array.from({ length: h.span - 1 }, () => "")]),
  ]);
  for (const row of table.rows) {
    lines.push([`${"  ".repeat(row.indent)}${row.label}`, ...row.cells.map((c) => (c.value === null ? "" : String(c.value)))]);
  }
  return lines.map((cells) => cells.map(escape).join(separator)).join("\n");
}
