import { useFamily, type Family, type FamilyVersion } from "./api";

/** C-62: the dataset an analysis should probably run on. Opened on an original that has a ready
 *  cleaned version, `latest` is that version (the highest seq); on a version, or without any,
 *  `latest` is null. Never a silent switch: pages say it and link to it (decision 6 of 097). */
export function useWorkingDataset(datasetId: string): { ready: boolean; family: Family | null; isOriginal: boolean; latest: FamilyVersion | null } {
  const { ready, family } = useFamily(datasetId);
  const isOriginal = !family || family.original_id === datasetId;
  return { ready, family, isOriginal, latest: isOriginal ? latestReady(family) : null };
}

export function latestReady(family: Family | null): FamilyVersion | null {
  if (!family) return null;
  return [...family.versions].filter((version) => version.status === "ready").sort((a, b) => b.seq - a.seq)[0] ?? null;
}

/** The same page on another dataset: `/d/<id>/kpi?x=1` → `/d/<other>/kpi?x=1`. */
export function samePageOn(pathname: string, from: string, to: string): string {
  const prefix = `/d/${from}`;
  return pathname.startsWith(prefix) ? `/d/${to}${pathname.slice(prefix.length)}` : `/d/${to}`;
}
