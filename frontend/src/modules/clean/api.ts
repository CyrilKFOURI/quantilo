import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { request } from "../../api/client";
import type { components } from "../../api/schema";
import { CLEAN_ENABLED } from "./flag";

/** Hooks of the Clean module (065), kept here rather than in api/hooks.ts (another branch edits
 *  it). Routes: /api/v1/modules/clean (backend/app/modules/clean/router.py). */

type Schemas = components["schemas"];
/** The server sends every field, defaults included: make them all required, deeply. */
type Complete<T> = T extends (infer Item)[] ? Complete<Item>[] : T extends object ? { [K in keyof T]-?: Complete<T[K]> } : T;

export type CleanDraft = Complete<Schemas["CleanDraft"]>;
export type Step = CleanDraft["steps"][number];
export type Op = Step["op"];
export type StepOf<O extends Op> = Extract<Step, { op: O }>;
export type RowCondition = Complete<Schemas["RowCondition"]>;
export type Suggestion = Complete<Schemas["CleanSuggestion"]>;
export type Suggestions = Complete<Schemas["CleanSuggestions"]>;
export type CleanPreview = Complete<Schemas["CleanPreview"]>;
export type PreviewRow = CleanPreview["sample"][number];
export type ColumnDiff = CleanPreview["columns"][number];
export type CleanJob = Complete<Schemas["CleanJob"]>;
export type CleanFamily = Complete<Schemas["CleanFamily"]>;
export type CleanVersion = CleanFamily["versions"][number];
export type CleanReport = Complete<Schemas["CleanReport"]>;
export type LineageItem = Complete<Schemas["CleanLineageItem"]>;
export type CleanMode = Complete<Schemas["CleanSettings"]>["mode"];

const BASE = "/modules/clean";

export const cleanApi = {
  settings: () => request<{ mode: CleanMode }>("GET", `${BASE}/settings`),
  updateSettings: (mode: CleanMode) => request<{ mode: CleanMode }>("PUT", `${BASE}/settings`, { mode }),
  lineage: () => request<LineageItem[]>("GET", `${BASE}/lineage`),
  suggestions: (id: string) => request<Suggestions>("GET", `${BASE}/datasets/${id}/suggestions`),
  draft: (id: string) => request<CleanDraft>("GET", `${BASE}/datasets/${id}/draft`),
  saveDraft: (id: string, steps: Step[]) => request<CleanDraft>("PUT", `${BASE}/datasets/${id}/draft`, { schema: 1, steps }),
  resetDraft: (id: string) => request<CleanDraft>("DELETE", `${BASE}/datasets/${id}/draft`),
  preview: (id: string, steps: Step[], stepId: string | null) =>
    request<CleanPreview>("POST", `${BASE}/datasets/${id}/preview`, { steps, step_id: stepId }),
  save: (id: string, steps: Step[]) => request<CleanJob>("POST", `${BASE}/datasets/${id}/versions`, { steps }),
  family: (id: string) => request<CleanFamily>("GET", `${BASE}/datasets/${id}/versions`),
  job: (jobId: string) => request<CleanJob>("GET", `${BASE}/jobs/${jobId}`),
  cancel: (jobId: string) => request<CleanJob>("DELETE", `${BASE}/jobs/${jobId}`),
};

export const cleanKeys = {
  settings: ["clean", "settings"] as const,
  lineage: ["clean", "lineage"] as const,
  suggestions: (id: string) => ["datasets", id, "clean", "suggestions"] as const,
  draft: (id: string) => ["datasets", id, "clean", "draft"] as const,
  family: (id: string) => ["datasets", id, "clean", "family"] as const,
};

export function useCleanSettings() {
  return useQuery({ queryKey: cleanKeys.settings, queryFn: cleanApi.settings, enabled: CLEAN_ENABLED });
}

export function useUpdateCleanSettings() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: cleanApi.updateSettings,
    onSuccess: (settings) => client.setQueryData(cleanKeys.settings, settings),
  });
}

/** Every cleaned version and its parent. An error (module off, an older server) reads as "no
 *  versions": the home page and the top bar then look as they did before 065. */
export function useLineage() {
  const query = useQuery({
    queryKey: cleanKeys.lineage,
    queryFn: cleanApi.lineage,
    enabled: CLEAN_ENABLED,
    retry: false,
    staleTime: 10_000,
  });
  return query.data ?? [];
}

export function useSuggestions(id: string, enabled: boolean) {
  return useQuery({ queryKey: cleanKeys.suggestions(id), queryFn: () => cleanApi.suggestions(id), enabled, staleTime: Infinity });
}

/** Read once: the page then edits its own copy (undo history), saved at every change. */
export function useDraft(id: string, enabled: boolean) {
  return useQuery({
    queryKey: cleanKeys.draft(id),
    queryFn: () => cleanApi.draft(id),
    enabled,
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  });
}

export function useSaveDraft(id: string) {
  return useMutation({ mutationFn: (steps: Step[]) => cleanApi.saveDraft(id, steps) });
}

export function useResetDraft(id: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: () => cleanApi.resetDraft(id),
    onSuccess: (draft) => client.setQueryData(cleanKeys.draft(id), draft),
  });
}

export function usePreview(id: string, steps: Step[] | undefined, stepId: string | null) {
  return useQuery({
    queryKey: ["datasets", id, "clean", "preview", steps, stepId],
    queryFn: () => cleanApi.preview(id, steps ?? [], stepId),
    enabled: steps !== undefined,
    placeholderData: keepPreviousData,
    staleTime: 60_000,
  });
}

export function useFamily(id: string, enabled = true) {
  return useQuery({ queryKey: cleanKeys.family(id), queryFn: () => cleanApi.family(id), enabled: enabled && CLEAN_ENABLED, retry: false });
}

/** Start a save and follow it (a job, then the new dataset's import). */
export function useSaveVersion(id: string) {
  return useMutation({ mutationFn: (steps: Step[]) => cleanApi.save(id, steps) });
}

export function useCleanJob(jobId: string | null) {
  const client = useQueryClient();
  return useQuery({
    queryKey: ["clean", "job", jobId],
    queryFn: async () => {
      const job = await cleanApi.job(jobId ?? "");
      if (job.status !== "running") {
        void client.invalidateQueries({ queryKey: ["datasets"] });
        void client.invalidateQueries({ queryKey: cleanKeys.lineage });
      }
      return job;
    },
    enabled: Boolean(jobId),
    refetchInterval: (query) => (query.state.data?.status === "running" || !query.state.data ? 500 : false),
    refetchIntervalInBackground: true,
  });
}

export function useCancelJob() {
  return useMutation({ mutationFn: cleanApi.cancel });
}
