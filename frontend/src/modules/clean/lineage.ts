import type { Dataset } from "../../api/client";
import type { LineageItem } from "./api";

/** A row of the home list (065): a dataset, indented under its original when it is one of its
 *  cleaned versions. */
export type FamilyRow = {
  dataset: Dataset;
  /** Set on a cleaned version. */
  version?: { seq: number; origin: string; latest: boolean; outOfDate: boolean; parentId: string };
  /** Cleaned versions of this original (to offer deleting them with it). */
  children: Dataset[];
};

/** The datasets in the server's order, each original followed by its versions (v1 first). A
 *  version whose original is gone is an ordinary dataset again (the link is removed with it). */
export function groupFamilies(datasets: Dataset[], lineage: LineageItem[]): FamilyRow[] {
  const byId = new Map(datasets.map((dataset) => [dataset.id, dataset]));
  const versions = new Map<string, LineageItem>();
  for (const item of lineage) if (byId.has(item.parent_id) && byId.has(item.dataset_id)) versions.set(item.dataset_id, item);
  const latest = new Map<string, number>();
  for (const item of versions.values()) latest.set(item.parent_id, Math.max(latest.get(item.parent_id) ?? 0, item.seq));
  const rows: FamilyRow[] = [];
  for (const dataset of datasets) {
    if (versions.has(dataset.id)) continue;
    const children = [...versions.values()].filter((item) => item.parent_id === dataset.id).sort((a, b) => a.seq - b.seq);
    rows.push({ dataset, children: children.map((item) => byId.get(item.dataset_id)!) });
    for (const item of children) {
      rows.push({
        dataset: byId.get(item.dataset_id)!,
        version: {
          seq: item.seq,
          origin: item.origin,
          latest: item.seq === latest.get(dataset.id),
          outOfDate: item.out_of_date,
          parentId: dataset.id,
        },
        children: [],
      });
    }
  }
  return rows;
}

/** The family of a dataset for the top bar: its original first, then the versions (v1 first);
 *  empty when it has none. */
export function familyOf(datasetId: string, lineage: LineageItem[]): { id: string; seq: number | null }[] {
  const parent = lineage.find((item) => item.dataset_id === datasetId)?.parent_id ?? datasetId;
  const versions = lineage.filter((item) => item.parent_id === parent).sort((a, b) => a.seq - b.seq);
  if (versions.length === 0) return [];
  return [{ id: parent, seq: null }, ...versions.map((item) => ({ id: item.dataset_id, seq: item.seq }))];
}

/** The same page of another dataset: `/d/old/columns?x` → `/d/new/columns`. */
export function samePageOf(pathname: string, datasetId: string): string {
  return pathname.replace(/^\/d\/[^/]+/, `/d/${encodeURIComponent(datasetId)}`);
}
