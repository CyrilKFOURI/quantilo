import { keepPreviousData, useQuery } from "@tanstack/react-query";

import { request } from "../../api/client";
import type { components } from "../../api/schema";
import type { RowCondition, Step } from "./api";
import { nextId } from "./steps";

/** N40: a facet of a column of the draft (OpenRefine's facets), kept apart from api.ts. */

type Schemas = components["schemas"];
type Complete<T> = T extends (infer Item)[] ? Complete<Item>[] : T extends object ? { [K in keyof T]-?: Complete<T[K]> } : T;

export type Facet = Complete<Schemas["Facet"]>;
export type FacetKind = Facet["kind"];
export type FacetValue = Facet["values"][number];
export type FacetBin = Facet["bins"][number];
export type FacetSort = "count" | "name";

/** What is picked: values, the blank cells, the unreadable ones (any of them), or a range. */
export type Selection = { values: string[]; blank: boolean; unreadable: boolean; low: number | null; high: number | null };

export const NOTHING: Selection = { values: [], blank: false, unreadable: false, low: null, high: null };

export const isRange = (selection: Selection) => selection.low !== null || selection.high !== null;
export const isEmpty = (selection: Selection) => !selection.values.length && !selection.blank && !selection.unreadable && !isRange(selection);

/** The facet of `column` with `steps` applied; asked for only while the panel is open. */
export function useFacet(datasetId: string, steps: Step[], column: string | null, selection: Selection, sort: FacetSort, search: string) {
  const picked = isEmpty(selection) ? null : selection;
  return useQuery({
    queryKey: ["datasets", datasetId, "clean", "facet", steps, column, picked, sort, search],
    queryFn: () =>
      request<Facet>("POST", `/modules/clean/datasets/${datasetId}/facet`, {
        column,
        steps,
        selection: picked,
        sort,
        search: search.trim() || null,
      }),
    enabled: column !== null,
    placeholderData: keepPreviousData,
    staleTime: 60_000,
    retry: false,
  });
}

/** The steps a selection becomes (each written only when the user clicks Add step). */
export type Action = "merge" | "missing" | "remove" | "keep" | "outside";

/** The gestures a selection allows: values, blank and unreadable add up (the rows of any of them);
 *  a range stands alone. A merge needs values only; "read as missing" values or unreadable ones;
 *  "empty the values outside" a range of numbers. */
export function actionsOf(selection: Selection, kind: FacetKind): Action[] {
  if (isEmpty(selection)) return [];
  if (isRange(selection)) return kind === "number" ? ["remove", "keep", "outside"] : ["remove", "keep"];
  const found: Action[] = [];
  if (selection.values.length && !selection.blank && !selection.unreadable) found.push("merge");
  if (!selection.blank && (selection.values.length || selection.unreadable)) found.push("missing");
  found.push("remove", "keep");
  return found;
}

/** A bound of a range as a step reads it: a number as it is, a date as ISO text (DATE: the day). */
function bound(value: number, facet: Facet): string {
  if (facet.kind !== "date") return String(value);
  const iso = new Date(value * 1000).toISOString();
  return facet.dtype === "DATE" ? iso.slice(0, 10) : iso.slice(0, 19).replace("T", " ");
}

/** The values a selection names: those picked, and the unreadable ones when they are picked. */
function namedValues(selection: Selection, facet: Facet): string[] {
  const unreadable = selection.unreadable ? facet.unreadable_values.map((value) => value.value) : [];
  return [...new Set([...selection.values, ...unreadable])];
}

/** The conditions of the rows a selection picks, for a filter step (any of them; a range: all). */
function conditionsOf(selection: Selection, facet: Facet): { conditions: RowCondition[]; combine: "all" | "any" } {
  const column = facet.column;
  if (isRange(selection)) {
    const conditions: RowCondition[] = [];
    if (selection.low !== null) conditions.push({ column, op: "ge", value: bound(selection.low, facet), values: null });
    if (selection.high !== null) conditions.push({ column, op: "le", value: bound(selection.high, facet), values: null });
    return { conditions, combine: "all" };
  }
  const conditions: RowCondition[] = [];
  const values = namedValues(selection, facet);
  if (values.length) conditions.push({ column, op: "in", value: null, values });
  if (selection.blank) conditions.push({ column, op: "empty", value: null, values: null });
  return { conditions, combine: "any" };
}

/** The step of a gesture on a selection, as a step added by hand (origin "manual"). */
export function stepOf(action: Action, selection: Selection, facet: Facet, steps: Step[], target = ""): Step {
  const common = { id: nextId(steps), enabled: true, origin: "manual" as const, rule: null, suggestion: null };
  const column = facet.column;
  switch (action) {
    case "merge":
      return { ...common, op: "map_values", column, map: Object.fromEntries(selection.values.filter((value) => value !== target).map((value) => [value, target])) };
    case "missing":
      return { ...common, op: "null_values", column, values: namedValues(selection, facet), match: "exact" };
    case "outside":
      return { ...common, op: "null_outside", column, min: selection.low, max: selection.high };
    case "remove":
    case "keep":
      return { ...common, op: "filter_rows", ...conditionsOf(selection, facet), action: action === "keep" ? "keep" : "remove" };
  }
}

/** The rows a selection picks, from the facet's own counts (the server's `matched` when known). */
export function rowsOf(selection: Selection, facet: Facet): number {
  if (facet.matched !== null) return facet.matched;
  const counts = new Map(facet.values.map((value) => [value.value, value.rows]));
  return selection.values.reduce((sum, value) => sum + (counts.get(value) ?? 0), 0) + (selection.blank ? facet.blank : 0) + (selection.unreadable ? (facet.unreadable ?? 0) : 0);
}

/** Pick or drop a value. */
export function toggled(selection: Selection, value: string): Selection {
  const values = selection.values.includes(value) ? selection.values.filter((other) => other !== value) : [...selection.values, value];
  return { ...selection, values, low: null, high: null };
}

/** The range of the bins from `first` to `last` (any order). */
export function rangeOf(bins: FacetBin[], first: number, last: number): Selection {
  const [from, to] = first <= last ? [first, last] : [last, first];
  return { ...NOTHING, low: bins[from]?.low ?? null, high: bins[to]?.high ?? null };
}

/** A bin is in the range picked (its middle is: neighbours sharing an edge stay out). */
export function inRange(bin: FacetBin, selection: Selection): boolean {
  if (!isRange(selection)) return false;
  const middle = (bin.low + bin.high) / 2;
  return (selection.low === null || middle >= selection.low) && (selection.high === null || middle <= selection.high);
}
