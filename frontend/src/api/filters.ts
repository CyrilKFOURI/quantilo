import { useNavigate, useSearch } from "@tanstack/react-router";
import { useCallback } from "react";

import type { components } from "./schema";

type ConditionSchema = components["schemas"]["Condition"];
/** One filter condition, as the API takes it (see backend app/domain/filters); `keep` has a server default. */
export type Condition = Omit<ConditionSchema, "keep"> & { keep?: ConditionSchema["keep"] };

/** Filters live in the URL (`?filters=`), tagged with their dataset: another dataset ignores them. */
export type FilterSearch = { filters?: { dataset: string; conditions: Condition[] } };

const OPERATORS = new Set<Condition["op"]>([
  "eq",
  "gt",
  "gte",
  "lt",
  "lte",
  "between",
  "in",
  "contains",
  "starts_with",
  "is_null",
  "part_in",
  "last_days",
  "unique",
  "duplicated",
]);

function isCondition(value: unknown): value is Condition {
  if (!value || typeof value !== "object") return false;
  const item = value as Record<string, unknown>;
  return typeof item.column === "string" && OPERATORS.has(item.op as Condition["op"]);
}

/** Route search validation: keep a well-formed filter state, drop anything else. */
export function validateFilterSearch(search: Record<string, unknown>): FilterSearch {
  const raw = search.filters as { dataset?: unknown; conditions?: unknown } | undefined;
  if (!raw || typeof raw.dataset !== "string" || !Array.isArray(raw.conditions)) return {};
  const conditions = raw.conditions.filter(isCondition).map((c) => ({ ...c, negate: Boolean(c.negate) }));
  return conditions.length ? { filters: { dataset: raw.dataset, conditions } } : {};
}

const NONE: Condition[] = [];

/** The active conditions of a dataset (none when the URL's filters belong to another one). */
export function useActiveFilters(datasetId: string | undefined): Condition[] {
  const search = useSearch({ strict: false }) as FilterSearch;
  const filters = search.filters;
  return datasetId && filters?.dataset === datasetId ? filters.conditions : NONE;
}

/** Replace the active conditions of a dataset; [] clears them. */
export function useSetFilters(datasetId: string) {
  const navigate = useNavigate();
  return useCallback(
    (conditions: Condition[]) =>
      void navigate({
        to: ".",
        search: (previous: Record<string, unknown>) => ({
          ...previous,
          filters: conditions.length ? { dataset: datasetId, conditions } : undefined,
        }),
        replace: false,
      }),
    [navigate, datasetId],
  );
}

/** The `filters` query parameter, or nothing. */
export function filtersParam(conditions: Condition[] | undefined): Record<string, string> {
  return conditions?.length ? { filters: JSON.stringify(conditions) } : {};
}
