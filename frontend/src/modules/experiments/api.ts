import { useQuery } from "@tanstack/react-query";

import { request } from "../../api/client";
import { filtersParam, useActiveFilters } from "../../api/filters";
import type { components } from "../../api/schema";

type Schemas = components["schemas"];
/** Every field present (the server always sends them), as the shared client does. */
type Complete<T> = T extends (infer Item)[] ? Complete<Item>[] : T extends object ? { [K in keyof T]-?: Complete<T[K]> } : T;

/** What the user describes: only the variant and the metrics are required. */
export type ExperimentDesign = Omit<Schemas["ExperimentDesign"], "alpha" | "metrics"> & {
  alpha?: number;
  metrics: ExperimentMetric[];
};
export type ExperimentMetric = Omit<Schemas["ExperimentMetric"], "kind"> & { kind?: Schemas["ExperimentMetric"]["kind"] };
export type MetricKind = Exclude<Schemas["ExperimentMetric"]["kind"], "auto">;
export type ExperimentCandidates = Complete<Schemas["ExperimentCandidates"]>;
export type ExperimentResult = Complete<Schemas["ExperimentResult"]>;
export type ExperimentMetricResult = ExperimentResult["metrics"][number];
export type ExperimentComparison = ExperimentMetricResult["comparisons"][number];
export type ExperimentCumulative = NonNullable<ExperimentResult["cumulative"]>;
export type ExperimentSegments = NonNullable<ExperimentResult["segments"]>;
export type MetricBaseline = Complete<Schemas["MetricBaseline"]>;
export type SampleSize = Complete<Schemas["SampleSize"]>;
export type SampleSizeRequest = Partial<Schemas["SampleSizeRequest"]> & Pick<Schemas["SampleSizeRequest"], "baseline" | "effect">;

const BASE = "/modules/experiments";
const query = (filters: Parameters<typeof filtersParam>[0], extra: Record<string, string> = {}) => {
  const params = new URLSearchParams({ ...extra, ...filtersParam(filters) }).toString();
  return params ? `?${params}` : "";
};

export const experimentsApi = {
  candidates: (id: string, filters?: Parameters<typeof filtersParam>[0]) =>
    request<ExperimentCandidates>("GET", `${BASE}/datasets/${id}/candidates${query(filters)}`),
  analyze: (id: string, design: ExperimentDesign, filters?: Parameters<typeof filtersParam>[0]) =>
    request<ExperimentResult>("POST", `${BASE}/datasets/${id}/analyze${query(filters)}`, design),
  baseline: (id: string, column: string, filters?: Parameters<typeof filtersParam>[0]) =>
    request<MetricBaseline>("GET", `${BASE}/datasets/${id}/baseline${query(filters, { column })}`),
  sampleSize: (plan: SampleSizeRequest) => request<SampleSize>("POST", `${BASE}/sample-size`, plan),
};

export function useExperimentCandidates(id: string, enabled: boolean) {
  const filters = useActiveFilters(id);
  return useQuery({
    queryKey: ["datasets", id, "experiments", "candidates", filters],
    queryFn: () => experimentsApi.candidates(id, filters),
    enabled,
    staleTime: 60_000,
  });
}

/** The analysis follows the page filters; the previous result stays shown while the next loads. */
export function useExperiment(id: string, design: ExperimentDesign | null) {
  const filters = useActiveFilters(id);
  return useQuery({
    queryKey: ["datasets", id, "experiments", "analyze", design, filters],
    queryFn: () => experimentsApi.analyze(id, design as ExperimentDesign, filters),
    enabled: Boolean(design),
    staleTime: 60_000,
    placeholderData: (previous) => previous,
    retry: false,
  });
}

export function useBaseline(id: string, column: string | null) {
  const filters = useActiveFilters(id);
  return useQuery({
    queryKey: ["datasets", id, "experiments", "baseline", column, filters],
    queryFn: () => experimentsApi.baseline(id, column as string, filters),
    enabled: Boolean(column),
    staleTime: 60_000,
  });
}

export function useSampleSize(plan: SampleSizeRequest | null) {
  return useQuery({
    queryKey: ["experiments", "sample-size", plan],
    queryFn: () => experimentsApi.sampleSize(plan as SampleSizeRequest),
    enabled: Boolean(plan),
    staleTime: Infinity,
    placeholderData: (previous) => previous,
    retry: false,
  });
}
