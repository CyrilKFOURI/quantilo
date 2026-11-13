import { useQuery } from "@tanstack/react-query";

import { request } from "../../api/client";
import type { components } from "../../api/schema";
import { useRuns } from "./modelApi";

/** 117: the explanations of a kept vision run ("Why?", the Explain tab). Routes:
 *  backend/app/modules/vision/explain_router.py. Everything is computed on demand by the server
 *  and cached next to the run, so a query here never goes stale (`staleTime: Infinity`). */

type Schemas = components["schemas"];
export type RunSummary = Schemas["VisionRunSummary"];
export type WhyExplanation = Schemas["WhyExplanation"];
export type WhyImage = Schemas["WhyImage"];
export type WhyMethod = Schemas["WhyMethod"];
export type WhyMap = Schemas["WhyMap"];
export type WhySanity = Schemas["WhySanity"];
export type WhyModel = Schemas["WhyModel"];
export type ExplainOverview = Schemas["ExplainOverview"];

export type MapMethod =
  | "kept"
  | "occlusion"
  | "grad_cam"
  | "rollout"
  | "saliency"
  | "smoothgrad"
  | "integrated_gradients"
  | "patch_pca"
  | "patchcore";
export type SanityMethod = "grad_cam" | "saliency" | "integrated_gradients";

const base = (datasetId: string) => `/modules/vision/datasets/${encodeURIComponent(datasetId)}`;

function query(params: Record<string, string | null | undefined>): string {
  const kept = Object.entries(params).filter((entry): entry is [string, string] => typeof entry[1] === "string" && entry[1] !== "");
  return kept.length ? `?${new URLSearchParams(kept).toString()}` : "";
}

export const explainKeys = {
  overview: (datasetId: string, run: string, model: string | null) => ["vision", datasetId, "explain", run, model] as const,
  image: (datasetId: string, run: string, image: string, model: string | null) => ["vision", datasetId, "why", run, image, model] as const,
  map: (datasetId: string, run: string, image: string, model: string | null, method: MapMethod, target: string | null) =>
    ["vision", datasetId, "why", run, image, model, "map", method, target] as const,
  sanity: (datasetId: string, run: string, image: string, model: string | null, method: SanityMethod) =>
    ["vision", datasetId, "why", run, image, model, "sanity", method] as const,
};

export const explainApi = {
  overview: (datasetId: string, run: string, model: string | null) =>
    request<ExplainOverview>("GET", `${base(datasetId)}/runs/${run}/explain${query({ model })}`),
  image: (datasetId: string, run: string, image: string, model: string | null) =>
    request<WhyExplanation>("GET", `${base(datasetId)}/runs/${run}/explain/images/${encodeURIComponent(image)}${query({ model })}`),
  map: (datasetId: string, run: string, image: string, model: string | null, method: MapMethod, target: string | null) =>
    request<WhyMap>(
      "GET",
      `${base(datasetId)}/runs/${run}/explain/images/${encodeURIComponent(image)}/map${query({ method, model, target })}`,
    ),
  sanity: (datasetId: string, run: string, image: string, model: string | null, method: SanityMethod) =>
    request<WhySanity>(
      "GET",
      `${base(datasetId)}/runs/${run}/explain/images/${encodeURIComponent(image)}/sanity${query({ method, model })}`,
    ),
};

/** The kept runs, newest first: the same query as Results (113), refreshed when a run ends. */
export function useVisionRuns(datasetId: string, enabled = true) {
  return useRuns(datasetId, enabled);
}

export function useExplainOverview(datasetId: string, run: string | null, model: string | null) {
  return useQuery({
    queryKey: explainKeys.overview(datasetId, run ?? "", model),
    queryFn: () => explainApi.overview(datasetId, run ?? "", model),
    enabled: run !== null,
    staleTime: Infinity,
    retry: false,
  });
}

export function useWhy(datasetId: string, run: string | null, image: string, model: string | null) {
  return useQuery({
    queryKey: explainKeys.image(datasetId, run ?? "", image, model),
    queryFn: () => explainApi.image(datasetId, run ?? "", image, model),
    enabled: run !== null,
    staleTime: Infinity,
    retry: false,
  });
}

/** A map is asked for only once its method is chosen (Advanced and More are never computed
 *  before the user opens them, the onion rule). */
export function useWhyMap(
  datasetId: string,
  run: string | null,
  image: string,
  model: string | null,
  method: MapMethod | null,
  target: string | null,
) {
  return useQuery({
    queryKey: explainKeys.map(datasetId, run ?? "", image, model, method ?? "occlusion", target),
    queryFn: () => explainApi.map(datasetId, run ?? "", image, model, method ?? "occlusion", target),
    enabled: run !== null && method !== null,
    staleTime: Infinity,
    retry: false,
  });
}

export function useWhySanity(datasetId: string, run: string | null, image: string, model: string | null, method: SanityMethod | null) {
  return useQuery({
    queryKey: explainKeys.sanity(datasetId, run ?? "", image, model, method ?? "grad_cam"),
    queryFn: () => explainApi.sanity(datasetId, run ?? "", image, model, method ?? "grad_cam"),
    enabled: run !== null && method !== null,
    staleTime: Infinity,
    retry: false,
  });
}
