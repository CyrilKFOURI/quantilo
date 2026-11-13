export type CleanTab = "suggestions" | "steps" | "checks" | "versions";

/** URL of the Clean page: the tab (Suggestions by default). */
export type CleanSearch = { tab?: CleanTab };

export function validateCleanSearch(search: Record<string, unknown>): CleanSearch {
  return search.tab === "steps" || search.tab === "checks" || search.tab === "versions" ? { tab: search.tab } : {};
}
