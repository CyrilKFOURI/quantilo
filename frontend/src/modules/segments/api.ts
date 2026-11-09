import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { request, type MlChoicesInput } from "../../api/client";
import { filtersParam, useActiveFilters, type Condition } from "../../api/filters";
import type { components } from "../../api/schema";

type Schemas = components["schemas"];
/** The server sends every field, defaults included: make them all required, deeply. */
type Complete<T> = T extends (infer Item)[] ? Complete<Item>[] : T extends object ? { [K in keyof T]-?: Complete<T[K]> } : T;

export type SegmentPlan = Complete<Schemas["SegmentPlan"]>;
export type SegmentDecision = SegmentPlan["preparation"]["decisions"][number];
export type SegmentJob = Complete<Schemas["SegmentJob"]>;
export type SegmentRun = Complete<Schemas["SegmentRun"]>;
export type Segmentation = SegmentRun["result"];
export type SegmentPortrait = Segmentation["segments"][number];
export type SegmentScore = Segmentation["scores"][number];
export type SegmentMap = Segmentation["map"];
export type SegmentProfileRow = Segmentation["profiles"][number];
export type SegmentDendrogram = NonNullable<Segmentation["dendrogram"]>;
export type SegmentRunSummary = Complete<Schemas["SegmentRunSummary"]>;
export type SegmentComparison = Complete<Schemas["SegmentComparison"]>;
export type SegmentMethod = SegmentPlan["method"];
export type UnusualRun = Complete<Schemas["UnusualRun"]>;
export type UnusualResult = UnusualRun["result"];
export type UnusualRow = UnusualResult["listed"][number];
export type UnusualRunSummary = Complete<Schemas["UnusualRunSummary"]>;
export type UnusualMark = Complete<Schemas["UnusualMark"]>;
export type RowValue = Complete<Schemas["RowValue"]>;
export type Mark = UnusualMark["mark"];

/** What the user set; everything left out follows the app's rules. */
export type SegmentRequest = {
  columns?: string[] | null;
  method?: "auto" | SegmentMethod;
  k?: number | null;
  preparation?: MlChoicesInput | null;
};
export type UnusualMethod = "auto" | "isolation_forest" | "mahalanobis" | "lof" | "segment";
export type UnusualRequest = {
  columns?: string[] | null;
  method?: UnusualMethod;
  preparation?: MlChoicesInput | null;
  segments_run?: string | null;
};

const BASE = "/modules/segments";
const query = (filters: Condition[]) => {
  const params = new URLSearchParams(filtersParam(filters)).toString();
  return params ? `?${params}` : "";
};

export const segmentsApi = {
  plan: (id: string, body: SegmentRequest, filters: Condition[]) =>
    request<SegmentPlan>("POST", `${BASE}/datasets/${id}/plan${query(filters)}`, body),
  start: (id: string, body: SegmentRequest, filters: Condition[]) =>
    request<SegmentJob>("POST", `${BASE}/datasets/${id}/runs${query(filters)}`, body),
  job: (jobId: string) => request<SegmentJob>("GET", `${BASE}/jobs/${jobId}`),
  cancel: (jobId: string) => request<SegmentJob>("DELETE", `${BASE}/jobs/${jobId}`),
  runs: (id: string) => request<SegmentRunSummary[]>("GET", `${BASE}/datasets/${id}/runs`),
  run: (id: string, runId: string) => request<SegmentRun>("GET", `${BASE}/datasets/${id}/runs/${runId}`),
  compare: (id: string, runIds: string[]) =>
    request<SegmentComparison>("GET", `${BASE}/datasets/${id}/runs/compare?runs=${runIds.map(encodeURIComponent).join(",")}`),
  rename: (id: string, runId: string, names: Record<string, string>) =>
    request<SegmentRun>("PATCH", `${BASE}/datasets/${id}/runs/${runId}/names`, { names }),
  tsne: (id: string, runId: string) => request<SegmentJob>("POST", `${BASE}/datasets/${id}/runs/${runId}/tsne`),
  save: (id: string, runId: string) => request<{ dataset_id: string }>("POST", `${BASE}/datasets/${id}/runs/${runId}/dataset`),
  profilesUrl: (id: string, runId: string) => `/api/v1${BASE}/datasets/${id}/runs/${runId}/profiles.csv`,
  findUnusual: (id: string, body: UnusualRequest, filters: Condition[]) =>
    request<SegmentJob>("POST", `${BASE}/datasets/${id}/unusual${query(filters)}`, body),
  unusualRuns: (id: string) => request<UnusualRunSummary[]>("GET", `${BASE}/datasets/${id}/unusual`),
  unusualRun: (id: string, runId: string) => request<UnusualRun>("GET", `${BASE}/datasets/${id}/unusual/${runId}`),
  row: (id: string, runId: string, row: number) => request<RowValue[]>("GET", `${BASE}/datasets/${id}/unusual/${runId}/rows/${row}`),
  exportUrl: (id: string, runId: string, threshold: number) =>
    `/api/v1${BASE}/datasets/${id}/unusual/${runId}/export.csv?threshold=${encodeURIComponent(String(threshold))}`,
  saveUnusual: (id: string, runId: string, threshold: number) =>
    request<{ dataset_id: string }>("POST", `${BASE}/datasets/${id}/unusual/${runId}/dataset`, { threshold }),
  mark: (id: string, key: string, mark: Mark | null) => request<UnusualMark[]>("PUT", `${BASE}/datasets/${id}/marks`, { key, mark }),
};

const key = (id: string) => ["datasets", id, "segments"];

/** The columns and method a segmentation would use, for the current choices and filters. */
export function useSegmentPlan(id: string, body: SegmentRequest, enabled: boolean) {
  const filters = useActiveFilters(id);
  return useQuery({
    queryKey: [...key(id), "plan", body, filters],
    queryFn: () => segmentsApi.plan(id, body, filters),
    enabled,
    staleTime: 60_000,
  });
}

export function useSegmentRuns(id: string, enabled: boolean) {
  return useQuery({ queryKey: [...key(id), "runs"], queryFn: () => segmentsApi.runs(id), enabled });
}

export function useSegmentRun(id: string, runId: string | null) {
  return useQuery({
    queryKey: [...key(id), "run", runId],
    queryFn: () => segmentsApi.run(id, runId ?? ""),
    enabled: Boolean(runId),
    retry: false,
  });
}

export function useSegmentCompare(id: string, runIds: string[]) {
  return useQuery({
    queryKey: [...key(id), "compare", runIds],
    queryFn: () => segmentsApi.compare(id, runIds),
    enabled: runIds.length >= 2,
  });
}

export function useStartSegments(id: string) {
  const filters = useActiveFilters(id);
  return useMutation({ mutationFn: (body: SegmentRequest) => segmentsApi.start(id, body, filters) });
}

export function useStartUnusual(id: string) {
  const filters = useActiveFilters(id);
  return useMutation({ mutationFn: (body: UnusualRequest) => segmentsApi.findUnusual(id, body, filters) });
}

/** Follows a job until it ends (also while the tab is hidden). */
export function useSegmentJob(jobId: string | null) {
  return useQuery({
    queryKey: ["segment-jobs", jobId],
    queryFn: () => segmentsApi.job(jobId ?? ""),
    enabled: Boolean(jobId),
    refetchInterval: (state) => (state.state.data?.status === "running" || !state.state.data ? 500 : false),
    refetchIntervalInBackground: true,
    retry: false,
  });
}

export function useCancelSegmentJob() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => segmentsApi.cancel(jobId),
    onSuccess: (job) => client.setQueryData(["segment-jobs", job.id], job),
  });
}

export function useSegmentActions(id: string) {
  const client = useQueryClient();
  const refresh = (runId: string) => {
    void client.invalidateQueries({ queryKey: [...key(id), "run", runId] });
    void client.invalidateQueries({ queryKey: [...key(id), "runs"] });
  };
  return {
    rename: useMutation({
      mutationFn: ({ runId, names }: { runId: string; names: Record<string, string> }) => segmentsApi.rename(id, runId, names),
      onSuccess: (run) => {
        client.setQueryData([...key(id), "run", run.id], run);
        void client.invalidateQueries({ queryKey: [...key(id), "runs"] });
      },
    }),
    save: useMutation({
      mutationFn: (runId: string) => segmentsApi.save(id, runId),
      onSuccess: (_created, runId) => {
        refresh(runId);
        void client.invalidateQueries({ queryKey: ["datasets"] });
      },
    }),
    tsne: useMutation({ mutationFn: (runId: string) => segmentsApi.tsne(id, runId) }),
    refresh,
  };
}

export function useUnusualRuns(id: string, enabled: boolean) {
  return useQuery({ queryKey: [...key(id), "unusual-runs"], queryFn: () => segmentsApi.unusualRuns(id), enabled });
}

export function useUnusualRun(id: string, runId: string | null) {
  return useQuery({
    queryKey: [...key(id), "unusual", runId],
    queryFn: () => segmentsApi.unusualRun(id, runId ?? ""),
    enabled: Boolean(runId),
    retry: false,
  });
}

export function useUnusualRow(id: string, runId: string, row: number | null) {
  return useQuery({
    queryKey: [...key(id), "unusual", runId, "row", row],
    queryFn: () => segmentsApi.row(id, runId, row ?? 0),
    enabled: row !== null,
  });
}

export function useUnusualActions(id: string, runId: string) {
  const client = useQueryClient();
  return {
    mark: useMutation({
      mutationFn: ({ key: rowKey, mark }: { key: string; mark: Mark | null }) => segmentsApi.mark(id, rowKey, mark),
      onSuccess: () => void client.invalidateQueries({ queryKey: [...key(id), "unusual", runId] }),
    }),
    save: useMutation({
      mutationFn: (threshold: number) => segmentsApi.saveUnusual(id, runId, threshold),
      onSuccess: () => void client.invalidateQueries({ queryKey: ["datasets"] }),
    }),
  };
}
