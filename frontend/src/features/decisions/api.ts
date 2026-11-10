import { useQuery } from "@tanstack/react-query";

import { request } from "../../api/client";
import { filtersParam, useActiveFilters } from "../../api/filters";
import type { components } from "../../api/schema";

/* 055: does a dataset look like logged decisions (an action, its reward, covariates)? Shared by
   the Experiments module (its Policies tab) and the ML page (a banner): a module never imports
   another one. */

type Complete<T> = T extends (infer Item)[] ? Complete<Item>[] : T extends object ? { [K in keyof T]-?: Complete<T[K]> } : T;
export type LoggedDecisions = Complete<components["schemas"]["LoggedDecisions"]>;

export function fetchLoggedDecisions(id: string, filters?: Parameters<typeof filtersParam>[0]) {
  const params = new URLSearchParams(filtersParam(filters)).toString();
  return request<LoggedDecisions>("GET", `/modules/experiments/datasets/${id}/decisions${params ? `?${params}` : ""}`);
}

export function useLoggedDecisions(id: string, enabled: boolean) {
  const filters = useActiveFilters(id);
  return useQuery({
    queryKey: ["datasets", id, "experiments", "decisions", filters],
    queryFn: () => fetchLoggedDecisions(id, filters),
    enabled,
    staleTime: 60_000,
  });
}
