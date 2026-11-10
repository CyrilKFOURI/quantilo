import { useParams } from "@tanstack/react-router";
import { useEffect, useSyncExternalStore } from "react";

import { useDatasets } from "../../api/hooks";

/* 098: the dataset one works on stays open while browsing Home or Recipes: the sidebar's dataset
   entries lead back to it and the top bar keeps showing it. It changes only when another dataset
   is opened, or when it is closed (top bar, Ctrl+K). Kept in memory and in this browser. */

export const CURRENT_DATASET_KEY = "de.currentDataset";

let current: string | null | undefined; // undefined: not read from storage yet
const listeners = new Set<() => void>();

function read(): string | null {
  if (current === undefined) {
    try {
      current = localStorage.getItem(CURRENT_DATASET_KEY);
    } catch {
      current = null;
    }
  }
  return current;
}

function write(id: string | null) {
  if (read() === id) return;
  current = id;
  try {
    if (id) localStorage.setItem(CURRENT_DATASET_KEY, id);
    else localStorage.removeItem(CURRENT_DATASET_KEY);
  } catch {
    // Not persisted when storage is unavailable: it still holds for this session.
  }
  for (const listener of listeners) listener();
}

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

/** Closes the current dataset: Home and Recipes then show no dataset. */
export function closeCurrentDataset() {
  write(null);
}

/** For tests: forget what was read from storage. */
export function resetCurrentDataset() {
  current = undefined;
  for (const listener of listeners) listener();
}

/** The dataset of the address when there is one (`routed`), else the last one opened, as long as
 *  it still exists (a deleted dataset is forgotten). */
export function useCurrentDataset(): { datasetId: string | undefined; routed: boolean } {
  const { datasetId: fromRoute } = useParams({ strict: false }) as { datasetId?: string };
  const remembered = useSyncExternalStore(subscribe, read, read);
  const datasets = useDatasets();
  useEffect(() => {
    if (fromRoute) write(fromRoute);
  }, [fromRoute]);
  const gone = !fromRoute && remembered && datasets.data && !datasets.data.some((d) => d.id === remembered);
  useEffect(() => {
    if (gone) write(null);
  }, [gone]);
  if (fromRoute) return { datasetId: fromRoute, routed: true };
  return { datasetId: (!gone && remembered) || undefined, routed: false };
}
