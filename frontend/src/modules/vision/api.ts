import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { queryKeys } from "../../api/hooks";
import { request } from "../../api/client";
import type { components } from "../../api/schema";
import { imageSetApi } from "../../features/import/imageSetApi";

/** Vision (107, 108): the gallery's index, the quality report and catalogue, the user's decisions.
 *  Routes: backend/app/api/images.py. */

type Schemas = components["schemas"];
export type ImageIndex = Schemas["ImageIndexOut"];
export type ImageReport = Schemas["ImageReportOut"];
export type QualityOut = Schemas["QualityOut"];
export type QualityFinding = Schemas["QualityFindingOut"];
export type QualityGroup = Schemas["QualityGroupOut"];
export type QualityRule = Schemas["QualityRuleOut"];
export type Decisions = Schemas["ImageDecisionsOut"];
export type DecisionsApplied = Schemas["ImageDecisionsApplied"];
export type DecisionPatch = Record<string, "exclude" | "keep" | null>;

const base = (datasetId: string) => `/datasets/${encodeURIComponent(datasetId)}/images`;

export const visionKeys = {
  index: (id: string) => ["datasets", id, "images", "index"] as const,
  report: (id: string) => ["datasets", id, "images", "report"] as const,
  decisions: (id: string) => ["datasets", id, "images", "decisions"] as const,
  rules: ["image-quality-rules"] as const,
};

export const visionApi = {
  index: (id: string) => request<ImageIndex>("GET", `${base(id)}/index`),
  decisions: (id: string) => request<Decisions>("GET", `${base(id)}/decisions`),
  setDecisions: (id: string, decisions: DecisionPatch) =>
    request<Decisions>("PUT", `${base(id)}/decisions`, { decisions }),
  discard: (id: string) => request<Decisions>("DELETE", `${base(id)}/decisions`),
  apply: (id: string) => request<DecisionsApplied>("POST", `${base(id)}/decisions/apply`),
  rules: () => request<QualityRule[]>("GET", "/images/rules"),
};

export function useImageIndex(datasetId: string, enabled: boolean) {
  return useQuery({
    queryKey: visionKeys.index(datasetId),
    queryFn: () => visionApi.index(datasetId),
    enabled,
    staleTime: Infinity,
  });
}

export function useImageReport(datasetId: string, enabled: boolean) {
  return useQuery({
    queryKey: visionKeys.report(datasetId),
    queryFn: () => imageSetApi.report(datasetId),
    enabled,
    staleTime: Infinity,
  });
}

export function useQualityRules() {
  return useQuery({ queryKey: visionKeys.rules, queryFn: visionApi.rules, staleTime: Infinity });
}

export function useDecisions(datasetId: string, enabled: boolean) {
  return useQuery({
    queryKey: visionKeys.decisions(datasetId),
    queryFn: () => visionApi.decisions(datasetId),
    enabled,
  });
}

/** Every click is saved in the draft (R-108-04), shown at once (optimistic), rolled back on error. */
export function useSetDecisions(datasetId: string) {
  const client = useQueryClient();
  const key = visionKeys.decisions(datasetId);
  return useMutation({
    mutationFn: (patch: DecisionPatch) => visionApi.setDecisions(datasetId, patch),
    onMutate: async (patch) => {
      await client.cancelQueries({ queryKey: key });
      const previous = client.getQueryData<Decisions>(key);
      if (previous) {
        const draft = { ...previous.draft };
        for (const [image, decision] of Object.entries(patch)) {
          if (decision === null) delete draft[image];
          else draft[image] = decision;
        }
        const keys = new Set([...Object.keys(draft), ...Object.keys(previous.applied)]);
        const pending = [...keys].filter((k) => draft[k] !== previous.applied[k]).length;
        client.setQueryData<Decisions>(key, { ...previous, draft, pending });
      }
      return { previous };
    },
    onError: (_error, _patch, context) => {
      if (context?.previous) client.setQueryData(key, context.previous);
    },
    onSuccess: (data) => client.setQueryData(key, data),
  });
}

export function useDiscardDecisions(datasetId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: () => visionApi.discard(datasetId),
    onSuccess: (data) => client.setQueryData(visionKeys.decisions(datasetId), data),
  });
}

/** Saving writes a new version of the table: everything read from it is read again. */
export function useApplyDecisions(datasetId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: () => visionApi.apply(datasetId),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: queryKeys.dataset(datasetId) });
      void client.invalidateQueries({ queryKey: queryKeys.datasets, exact: true });
    },
  });
}
