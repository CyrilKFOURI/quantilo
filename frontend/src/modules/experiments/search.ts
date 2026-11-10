import type { ExperimentDesign } from "./api";

export type ExperimentsTab = "analyze" | "plan" | "policies";

/** URL of the Experiments page: the tab, the experiment described (as edited), or simple seeds
 *  from another page (Compare, the assistant): a variant, a metric and a unit. */
export type ExperimentsSearch = {
  tab?: ExperimentsTab;
  design?: ExperimentDesign;
  variant?: string;
  metric?: string;
  unit?: string;
};

const text = (value: unknown) => (typeof value === "string" && value ? value : undefined);

function isDesign(value: unknown): value is ExperimentDesign {
  if (!value || typeof value !== "object") return false;
  const design = value as Record<string, unknown>;
  return (
    typeof design.variant === "string" &&
    Array.isArray(design.metrics) &&
    design.metrics.length > 0 &&
    design.metrics.every((m) => m && typeof m === "object" && typeof (m as { column?: unknown }).column === "string")
  );
}

export function validateExperimentsSearch(search: Record<string, unknown>): ExperimentsSearch {
  const result: ExperimentsSearch = {};
  if (search.tab === "plan" || search.tab === "analyze" || search.tab === "policies") result.tab = search.tab;
  if (isDesign(search.design)) result.design = search.design;
  const variant = text(search.variant);
  const metric = text(search.metric);
  const unit = text(search.unit);
  if (variant) result.variant = variant;
  if (metric) result.metric = metric;
  if (unit) result.unit = unit;
  return result;
}
