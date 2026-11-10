import type { Column, KpiDefinition } from "../../api/client";

export type Measure = KpiDefinition["measures"][number];
export type Agg = Measure["agg"];
export type View = NonNullable<KpiDefinition["view"]>;
export type TableView = NonNullable<View["table"]>;
export type ShowAs = NonNullable<View["show_as"]>;
export type Chart = NonNullable<View["chart"]>;
export type Format = NonNullable<Measure["format"]>;
export type Grain = "day" | "week" | "month" | "quarter" | "year";

export const AGGS: Agg[] = ["count", "sum", "mean", "median", "min", "max", "weighted_mean", "distinct", "true_share", "share"];
/** 053 (R-MES-07): a delay between two dates is averaged, never summed. */
export const DELAY_AGGS: Agg[] = ["mean", "median", "min", "max"];
export type RatioKind = "ratio" | "margin" | "difference";
export const RATIO_KINDS: RatioKind[] = ["ratio", "margin", "difference"];
export const GRAINS: Grain[] = ["day", "week", "month", "quarter", "year"];
export const ADDITIVE = new Set<Agg>(["count", "sum"]);
export const MAX_MEASURES = 4;
export const MAX_SPLITS = 2;
export const RATIO = "ratio";

const NUMBERS = new Set(["numeric_continuous", "numeric_discrete"]);
const SPLITS = new Set(["categorical_nominal", "categorical_ordinal", "boolean"]);
const KEYS = new Set(["identifier", "categorical_nominal"]);
/** 053 (R-MES-01, P18): averaged with a weight when the dataset has a quantity. */
const WEIGHTED_ROLES = new Set(["unit_price", "percentage"]);
const SUM_ROLES = new Set(["amount", "quantity", "cost", "target"]);
// 048 (P1, R-SUG-09): a price or a level is averaged, never summed; same words as the engine's
// kpi/suggest.py LEVEL_WORDS.
const LEVEL_WORDS = new Set(["price", "prices", "prix", "tarif", "unit", "salary", "salaries", "salaire", "wage", "wages", "avg", "average", "mean", "moyenne", "moyen", "rate", "balance"]);

/** "UnitPrice" / "unit_price" / "Prix moyen" → its words, as the engine reads column names. */
export function nameWords(name: string): string[] {
  const spaced = name.replace(/([a-z0-9])([A-Z])/g, "$1 $2");
  const plain = spaced.normalize("NFKD").replace(/[\u0300-\u036f]/g, "");
  return plain.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
}

/** A price, a salary, an average: an amount that is a level, not a flow. */
export function isLevel(column: Column): boolean {
  if (column.role === "unit_price") return true;
  return column.role === "amount" && nameWords(column.name).some((w) => LEVEL_WORDS.has(w));
}

/** Columns an aggregate can read (Rows reads none). */
export function columnsFor(agg: Agg, columns: Column[]): Column[] {
  const usable = columns.filter((c) => !c.excluded);
  if (agg === "count") return [];
  if (agg === "distinct") return usable.filter((c) => c.type !== "ignored");
  if (agg === "true_share") return usable.filter((c) => c.type === "boolean");
  if (agg === "share") return usable.filter((c) => SPLITS.has(c.type));
  return usable.filter((c) => NUMBERS.has(c.type));
}

export const dateColumns = (columns: Column[]) =>
  columns.filter((c) => !c.excluded && c.type === "datetime");
export const splitColumns = (columns: Column[]) =>
  columns.filter((c) => !c.excluded && SPLITS.has(c.type));

/** The default aggregate of a column (spec R-MES-01): sums for amounts, costs and volumes, means
 *  for prices and other numbers, "% true" for yes/no columns, distinct counts for the rest. */
export function defaultAgg(column: Column): Agg {
  if (column.type === "boolean") return "true_share";
  if (NUMBERS.has(column.type)) return column.role && SUM_ROLES.has(column.role) && !isLevel(column) ? "sum" : "mean";
  return "distinct";
}

/** The weights a weighted average can use: numbers, quantities first. */
export function weightColumns(columns: Column[], except: string | null | undefined): Column[] {
  const numbers = columnsFor("sum", columns).filter((c) => c.name !== except);
  return [...numbers.filter((c) => c.role === "quantity"), ...numbers.filter((c) => c.role !== "quantity")];
}

/** The keys a share can count distinct (customers, orders, ids), keys by role first. */
export function keyColumns(columns: Column[]): Column[] {
  const keys = columns.filter((c) => !c.excluded && KEYS.has(c.type));
  const named = (c: Column) => c.role === "order_key" || c.role === "customer_key";
  return [...keys.filter(named), ...keys.filter((c) => !named(c))];
}

/** The most frequent values of a column (its summary), for the values a share counts. */
export function levelsOf(column: Column | undefined): string[] {
  return column?.summary?.histogram.map((bar) => bar.label) ?? [];
}

/** The measure a column starts with (R-MES-01): its default aggregate, and since 053 a unit price,
 *  a level or a percentage weighted by the dataset's first quantity; a status counts its first value. */
export function defaultMeasure(column: Column, columns: Column[]): Partial<Measure> {
  const agg = defaultAgg(column);
  if (agg === "mean" && (WEIGHTED_ROLES.has(column.role ?? "") || isLevel(column))) {
    const weight = weightColumns(columns, column.name).find((c) => c.role === "quantity");
    if (weight) return { agg: "weighted_mean", weight: weight.name };
  }
  return { agg };
}

/** A delay between two dates (053, R-MES-07): `column` → `until`. */
export const isDelay = (measure: Measure) => Boolean(measure.until);

export function nextMeasureId(measures: Measure[]): string {
  const used = new Set(measures.map((m) => m.id));
  for (let i = 1; ; i++) if (!used.has(`m${i}`)) return `m${i}`;
}

/** The part of a definition that changes the numbers (everything but the display). */
export function dataPart(definition: KpiDefinition): string {
  const { view: _view, ...data } = definition;
  void _view;
  return JSON.stringify(data);
}

export function sameData(a: KpiDefinition | null, b: KpiDefinition | null): boolean {
  if (!a || !b) return a === b;
  return dataPart(computeDefinition(a)) === dataPart(computeDefinition(b));
}

export function viewOf(definition: KpiDefinition): Required<Omit<View, "table">> & { table: Required<TableView> } {
  const view = definition.view ?? {};
  const table = view.table ?? {};
  return {
    chart: view.chart ?? "auto",
    share: view.share ?? false,
    labels: view.labels ?? "auto",
    show_as: view.show_as ?? "value",
    line: view.line ?? false,
    spacing: view.spacing ?? "normal",
    palette: view.palette ?? null,
    table: {
      layout: table.layout ?? "auto",
      totals: table.totals ?? "auto",
      shading: table.shading ?? "auto",
      compare: table.compare ?? [],
      sort: table.sort ?? null,
    },
  };
}

export function withView(definition: KpiDefinition, patch: Partial<View>): KpiDefinition {
  return { ...definition, view: { ...definition.view, ...patch } };
}

export function withTable(definition: KpiDefinition, patch: Partial<TableView>): KpiDefinition {
  return withView(definition, { table: { ...definition.view?.table, ...patch } });
}

export function withFormat(definition: KpiDefinition, measureId: string, patch: Partial<Format>): KpiDefinition {
  if (measureId === RATIO && definition.ratio) {
    return { ...definition, ratio: { ...definition.ratio, format: { ...definition.ratio.format, ...patch } } };
  }
  return {
    ...definition,
    measures: definition.measures.map((m) => (m.id === measureId ? { ...m, format: { ...m.format, ...patch } } : m)),
  };
}

/** The display name and format of a measure are display settings too, stored on the measure. */
export function withLabel(definition: KpiDefinition, measureId: string, label: string | null): KpiDefinition {
  if (measureId === RATIO && definition.ratio) return { ...definition, ratio: { ...definition.ratio, label } };
  return { ...definition, measures: definition.measures.map((m) => (m.id === measureId ? { ...m, label } : m)) };
}

export function formatOf(definition: KpiDefinition, measureId: string): Format {
  const source = measureId === RATIO ? definition.ratio?.format : definition.measures.find((m) => m.id === measureId)?.format;
  return source ?? {};
}

/** Labels, formats and hiding "Other" change without recomputing: left out of what is sent. */
export function computeDefinition(definition: KpiDefinition): KpiDefinition {
  return {
    ...definition,
    other: "show",
    measures: definition.measures.map((m) => ({ ...m, label: null, format: undefined })),
    ratio: definition.ratio ? { ...definition.ratio, label: null, format: undefined } : definition.ratio,
  };
}

/** Number of display settings changed from "auto", for the Customize button's badge. */
export function customised(definition: KpiDefinition): number {
  const view = viewOf(definition);
  const formats = [...definition.measures.map((m) => m.format), definition.ratio?.format].filter(Boolean) as Format[];
  return (
    Number(view.chart !== "auto") +
    Number(view.share) +
    Number(view.labels !== "auto") +
    Number(view.show_as !== "value") +
    Number(view.line) +
    Number(view.spacing !== "normal") +
    Number(view.palette !== null) +
    Number(view.table.layout !== "auto") +
    Number(view.table.totals !== "auto") +
    Number(view.table.shading !== "auto") +
    Number(view.table.compare.length > 0) +
    Number((definition.top ?? 8) !== 8) +
    Number((definition.other ?? "show") !== "show") +
    Number((definition.group_order ?? "value") !== "value") +
    formats.filter((f) => (f.decimals ?? "auto") !== "auto" || (f.compact ?? "auto") !== "auto" || f.prefix || f.suffix).length +
    definition.measures.filter((m) => m.label).length +
    Number(Boolean(definition.ratio?.label))
  );
}

/** Everything in Customize back to automatic, keeping the measures, time and splits. */
export function resetView(definition: KpiDefinition): KpiDefinition {
  return {
    ...definition,
    view: undefined,
    top: 8,
    other: "show",
    group_order: "value",
    measures: definition.measures.map((m) => ({ ...m, label: null, format: undefined })),
    ratio: definition.ratio ? { ...definition.ratio, label: null, format: undefined } : definition.ratio,
  };
}
