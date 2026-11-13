import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { request } from "../../api/client";
import type { components } from "../../api/schema";

/** Vision models (112, 113): the task rule, a training job, the kept runs, their predictions, the
 *  anomaly maps, and the weights of the deep models. Routes: backend/app/modules/vision/router.py. */

type Schemas = components["schemas"];
export type TaskOut = Schemas["TaskOut"];
export type Task = NonNullable<Schemas["TaskChoice"]["task"]>;
export type VisionJob = Schemas["VisionJob"];
export type VisionRun = Schemas["VisionRun"];
export type VisionResult = Schemas["VisionResult"];
export type VisionModel = Schemas["VisionModelResult"];
export type RunSummary = Schemas["VisionRunSummary"];
export type RunRequest = Schemas["VisionRunRequest"];
export type DetectorKey = NonNullable<RunRequest["detector"]>;
export type WeightOut = Schemas["WeightOut"];
export type DetectorOut = Schemas["DetectorOut"];
export type ClassificationReport = Schemas["ClassificationReport"];
export type AnomalyReport = Schemas["AnomalyReport"];
export type ThresholdChoice = Schemas["ThresholdChoice"];
/** One scored image: its table row, its fold, the truth, and each model's columns
 *  (`<key>_pred`, `<key>_proba`, `<key>_probs`, `<key>_score`, PatchCore's `<key>_box`...). */
export type Prediction = Record<string, unknown> & { row: number; image_id: string };

const base = "/modules/vision";
const set = (id: string) => `${base}/datasets/${encodeURIComponent(id)}`;

export const modelKeys = {
  task: (id: string, task: Task | undefined, target?: string) => ["vision", id, "task", task ?? "rule", target ?? null] as const,
  runs: (id: string) => ["vision", id, "runs"] as const,
  run: (id: string, run: string) => ["vision", id, "runs", run] as const,
  predictions: (id: string, run: string) => ["vision", id, "runs", run, "predictions"] as const,
  job: (job: string) => ["vision", "jobs", job] as const,
};

export const modelApi = {
  task: (id: string, task?: Task, target?: string) => {
    const params = new URLSearchParams();
    if (task) params.set("task", task);
    if (target) params.set("target", target);
    const query = params.toString();
    return request<TaskOut>("GET", `${set(id)}/task${query ? `?${query}` : ""}`);
  },
  start: (id: string, body: RunRequest) => request<VisionJob>("POST", `${set(id)}/runs`, body),
  job: (job: string) => request<VisionJob>("GET", `${base}/jobs/${encodeURIComponent(job)}`),
  cancel: (job: string) => request<VisionJob>("DELETE", `${base}/jobs/${encodeURIComponent(job)}`),
  runs: (id: string) => request<RunSummary[]>("GET", `${set(id)}/runs`),
  run: (id: string, run: string) => request<VisionRun>("GET", `${set(id)}/runs/${encodeURIComponent(run)}`),
  predictions: (id: string, run: string) =>
    request<Prediction[]>("GET", `${set(id)}/runs/${encodeURIComponent(run)}/predictions`),
  remove: (id: string, run: string) => request<void>("DELETE", `${set(id)}/runs/${encodeURIComponent(run)}`),
  download: (key: string) => request<WeightOut>("POST", `${base}/weights/${encodeURIComponent(key)}/download`),
};

/** The URL of a kept anomaly map (8-bit grey PNG) or of an autoencoder's reconstruction. */
export function runImageUrl(datasetId: string, runId: string, kind: "maps" | "reconstructions", model: string, row: number) {
  return `/api/v1${set(datasetId)}/runs/${encodeURIComponent(runId)}/${kind}/${encodeURIComponent(model)}/${row}`;
}

/** R-113-01: reading the rule is cheap (no model runs); a weights download in progress is followed. */
export function useTask(datasetId: string, task: Task | undefined, enabled: boolean, target?: string) {
  return useQuery({
    queryKey: modelKeys.task(datasetId, task, target),
    queryFn: () => modelApi.task(datasetId, task, target),
    enabled,
    retry: false,
    refetchInterval: (query) => (query.state.data?.weights.some((w) => w.download === "running") ? 1_500 : false),
  });
}

export function useRuns(datasetId: string, enabled: boolean) {
  return useQuery({ queryKey: modelKeys.runs(datasetId), queryFn: () => modelApi.runs(datasetId), enabled });
}

export function useRun(datasetId: string, runId: string | undefined) {
  return useQuery({
    queryKey: modelKeys.run(datasetId, runId ?? ""),
    queryFn: () => modelApi.run(datasetId, runId ?? ""),
    enabled: Boolean(runId),
    staleTime: Infinity,
  });
}

export function usePredictions(datasetId: string, runId: string | undefined) {
  return useQuery({
    queryKey: modelKeys.predictions(datasetId, runId ?? ""),
    queryFn: () => modelApi.predictions(datasetId, runId ?? ""),
    enabled: Boolean(runId),
    staleTime: Infinity,
  });
}

/** Follow a training job until it ends, also while the tab is in the background (CLAUDE.md). */
export function useVisionJob(jobId: string | null) {
  return useQuery({
    queryKey: modelKeys.job(jobId ?? ""),
    queryFn: () => modelApi.job(jobId ?? ""),
    enabled: Boolean(jobId),
    refetchInterval: (query) => (query.state.data?.status === "running" || !query.state.data ? 600 : false),
    refetchIntervalInBackground: true,
    retry: false,
  });
}

export function useStartRun(datasetId: string) {
  return useMutation({ mutationFn: (body: RunRequest) => modelApi.start(datasetId, body) });
}

export function useCancelRun() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => modelApi.cancel(jobId),
    onSuccess: (job) => client.setQueryData(modelKeys.job(job.id), job),
  });
}

export function useDeleteRun(datasetId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (runId: string) => modelApi.remove(datasetId, runId),
    onSuccess: () => void client.invalidateQueries({ queryKey: modelKeys.runs(datasetId) }),
  });
}

/** R-113-05: called only from the confirmation dialog, which showed the size and the licence. */
export function useDownloadWeights(datasetId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (key: string) => modelApi.download(key),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["vision", datasetId, "task"] }),
  });
}
