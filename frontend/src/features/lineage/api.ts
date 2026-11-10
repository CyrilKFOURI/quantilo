import { useQuery } from "@tanstack/react-query";

import { request } from "../../api/client";
import type { components } from "../../api/schema";

/* 095: the family of a dataset (its original and every cleaned version, 065), for the pages that
   compare versions (Changes). Read from the Clean module's route, as features/decisions reads
   Experiments': a page never imports a module. The module off (build or server) or an older
   server reads as "no family". */

type Complete<T> = T extends (infer Item)[] ? Complete<Item>[] : T extends object ? { [K in keyof T]-?: Complete<T[K]> } : T;
export type Family = Complete<components["schemas"]["CleanFamily"]>;
export type FamilyVersion = Family["versions"][number];

const ENABLED = import.meta.env.VITE_CLEAN_ENABLED !== "0";

export function fetchFamily(id: string) {
  return request<Family>("GET", `/modules/clean/datasets/${id}/versions`);
}

/** `ready` once the answer (or its absence) is known; `family` is null without cleaned versions
 *  to speak of. Same cache entry as the Clean page's family (refreshed with the datasets). */
export function useFamily(id: string): { ready: boolean; family: Family | null } {
  const query = useQuery({
    queryKey: ["datasets", id, "clean", "family"],
    queryFn: () => fetchFamily(id),
    enabled: ENABLED && Boolean(id),
    retry: false,
    staleTime: 10_000,
  });
  if (!ENABLED) return { ready: true, family: null };
  return { ready: !query.isPending, family: query.data ?? null };
}
