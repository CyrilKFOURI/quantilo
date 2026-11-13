import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { request } from "../../api/client";
import type { components } from "../../api/schema";

/** 118: the set's concepts, and what explains the Advanced › More models and TCAV. Routes:
 *  backend/app/modules/vision/more_router.py. Kept evidence never changes, so it never goes
 *  stale; TCAV is asked for only when the user clicks (`enabled`). */

type Schemas = components["schemas"];
export type ConceptOut = Schemas["ConceptOut"];
export type MoreOption = Schemas["MoreOptionOut"];
export type MoreKey = MoreOption["key"];
export type WhyLooksLike = Schemas["WhyLooksLike"];
export type WhyLook = Schemas["WhyLook"];
export type WhyConcepts = Schemas["WhyConcepts"];
export type WhyConcept = Schemas["WhyConcept"];
export type WhyConceptHead = Schemas["WhyConceptHead"];
export type ProtoOverview = Schemas["ProtoOverview"];
export type TcavOptions = Schemas["TcavOptions"];
export type WhyTcav = Schemas["WhyTcav"];

const set = (id: string) => `/modules/vision/datasets/${encodeURIComponent(id)}`;
const runOf = (id: string, run: string) => `${set(id)}/runs/${encodeURIComponent(run)}`;

export const moreKeys = {
  concepts: (id: string) => ["vision", id, "concepts"] as const,
  looks: (id: string, run: string, image: string) => ["vision", id, "why", run, image, "looks"] as const,
  imageConcepts: (id: string, run: string, image: string) => ["vision", id, "why", run, image, "concepts"] as const,
  prototypes: (id: string, run: string) => ["vision", id, "explain", run, "protopnet"] as const,
  tcavOptions: (id: string, run: string) => ["vision", id, "explain", run, "tcav-options"] as const,
  tcav: (id: string, run: string, concept: string, layer: string | null) => ["vision", id, "explain", run, "tcav", concept, layer] as const,
};

export const moreApi = {
  concepts: (id: string) => request<ConceptOut[]>("GET", `${set(id)}/concepts`),
  save: (id: string, body: { name: string; image_ids: string[] }) => request<ConceptOut>("POST", `${set(id)}/concepts`, body),
  remove: (id: string, group: string) => request<void>("DELETE", `${set(id)}/concepts/${encodeURIComponent(group)}`),
  looks: (id: string, run: string, image: string) => request<WhyLooksLike>("GET", `${runOf(id, run)}/explain/images/${encodeURIComponent(image)}/looks`),
  imageConcepts: (id: string, run: string, image: string) =>
    request<WhyConcepts>("GET", `${runOf(id, run)}/explain/images/${encodeURIComponent(image)}/concepts`),
  prototypes: (id: string, run: string) => request<ProtoOverview | null>("GET", `${runOf(id, run)}/explain/prototypes`),
  tcavOptions: (id: string, run: string) => request<TcavOptions>("GET", `${runOf(id, run)}/explain/tcav/options`),
  tcav: (id: string, run: string, concept: string, layer: string | null) =>
    request<WhyTcav>("GET", `${runOf(id, run)}/explain/tcav?${new URLSearchParams(layer ? { concept, layer } : { concept }).toString()}`),
};

export function useConcepts(datasetId: string, enabled = true) {
  return useQuery({ queryKey: moreKeys.concepts(datasetId), queryFn: () => moreApi.concepts(datasetId), enabled });
}

/** Saving or deleting a concept changes what the Model tab offers: its rule is read again. */
function useRefreshConcepts(datasetId: string) {
  const client = useQueryClient();
  return () => {
    void client.invalidateQueries({ queryKey: moreKeys.concepts(datasetId) });
    void client.invalidateQueries({ queryKey: ["vision", datasetId, "task"] });
    void client.invalidateQueries({ queryKey: ["vision", datasetId, "explain"] });
  };
}

export function useSaveConcept(datasetId: string) {
  const refresh = useRefreshConcepts(datasetId);
  return useMutation({ mutationFn: (body: { name: string; image_ids: string[] }) => moreApi.save(datasetId, body), onSuccess: refresh });
}

export function useDeleteConcept(datasetId: string) {
  const refresh = useRefreshConcepts(datasetId);
  return useMutation({ mutationFn: (group: string) => moreApi.remove(datasetId, group), onSuccess: refresh });
}

export function useLooksLike(datasetId: string, run: string, image: string, enabled: boolean) {
  return useQuery({
    queryKey: moreKeys.looks(datasetId, run, image),
    queryFn: () => moreApi.looks(datasetId, run, image),
    enabled,
    staleTime: Infinity,
    retry: false,
  });
}

export function useImageConcepts(datasetId: string, run: string, image: string, enabled: boolean) {
  return useQuery({
    queryKey: moreKeys.imageConcepts(datasetId, run, image),
    queryFn: () => moreApi.imageConcepts(datasetId, run, image),
    enabled,
    staleTime: Infinity,
    retry: false,
  });
}

export function useProtoOverview(datasetId: string, run: string | null) {
  return useQuery({
    queryKey: moreKeys.prototypes(datasetId, run ?? ""),
    queryFn: () => moreApi.prototypes(datasetId, run ?? ""),
    enabled: run !== null,
    staleTime: Infinity,
    retry: false,
  });
}

export function useTcavOptions(datasetId: string, run: string | null) {
  return useQuery({
    queryKey: moreKeys.tcavOptions(datasetId, run ?? ""),
    queryFn: () => moreApi.tcavOptions(datasetId, run ?? ""),
    enabled: run !== null,
    retry: false,
  });
}

/** Computed only once the user asked for this concept and layer (`asked`). */
export function useTcav(datasetId: string, run: string | null, asked: { concept: string; layer: string | null } | null) {
  return useQuery({
    queryKey: moreKeys.tcav(datasetId, run ?? "", asked?.concept ?? "", asked?.layer ?? null),
    queryFn: () => moreApi.tcav(datasetId, run ?? "", asked?.concept ?? "", asked?.layer ?? null),
    enabled: run !== null && asked !== null,
    staleTime: Infinity,
    retry: false,
  });
}

/** R-VCB-05: the class probabilities from concept values, exactly as the bottleneck's head
 *  computes them (standardise, linear, softmax; one row of weights = a sigmoid). */
export function conceptProbabilities(head: WhyConceptHead, values: readonly number[]): number[] {
  const z = values.map((value, j) => (value - (head.mean[j] ?? 0)) / (head.scale[j] || 1));
  const logits = head.weight.map((row, k) => row.reduce((sum, w, j) => sum + w * (z[j] ?? 0), head.bias[k] ?? 0));
  if (head.weight.length === 1) {
    const p = 1 / (1 + Math.exp(-(logits[0] ?? 0)));
    return [1 - p, p];
  }
  const top = Math.max(...logits);
  const e = logits.map((value) => Math.exp(value - top));
  const total = e.reduce((a, b) => a + b, 0);
  return e.map((value) => value / total);
}
