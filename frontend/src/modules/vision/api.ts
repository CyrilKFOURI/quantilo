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
/** c3: everything the draft holds (R-VED-01, R-VED-03, R-VQT-02); each part is optional. */
export type EditPatch = {
  decisions?: DecisionPatch;
  labels?: Record<string, string | null>;
  pins?: Record<string, boolean | null>;
  thresholds?: Record<string, number>;
};
export type QualityPreview = Schemas["QualityPreviewOut"];

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
  setDecisions: (id: string, patch: EditPatch) => request<Decisions>("PUT", `${base(id)}/decisions`, patch),
  previewQuality: (id: string, thresholds: Record<string, number>) =>
    request<QualityPreview>("POST", `${base(id)}/quality/preview`, { thresholds }),
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

function merged<T>(base: Readonly<Record<string, T>>, patch: Record<string, T | null> | undefined, drop: (value: T | null) => boolean) {
  const out: Record<string, T> = { ...base };
  for (const [key, value] of Object.entries(patch ?? {})) {
    if (value === null || drop(value)) delete out[key];
    else out[key] = value;
  }
  return out;
}

function changed(draft: Readonly<Record<string, unknown>>, applied: Readonly<Record<string, unknown>>) {
  const keys = new Set([...Object.keys(draft), ...Object.keys(applied)]);
  return [...keys].filter((k) => draft[k] !== applied[k]).length;
}

/** The draft after a patch, as the server will compute it (shown at once, R-108-04). */
export function patchedDecisions(previous: Decisions, patch: EditPatch): Decisions {
  const draft = merged(previous.draft, patch.decisions, () => false);
  const labels = merged(previous.labels_draft ?? {}, patch.labels, (value) => typeof value === "string" && value.trim() === "");
  const pins = merged(previous.pins_draft ?? {}, patch.pins, (value) => value === false);
  const thresholds = patch.thresholds ?? previous.thresholds_draft ?? {};
  const sameThresholds = JSON.stringify(Object.entries(thresholds).sort()) === JSON.stringify(Object.entries(previous.thresholds_applied ?? {}).sort());
  const pendingBy = {
    decisions: changed(draft, previous.applied),
    labels: changed(labels, previous.labels_applied ?? {}),
    pins: changed(pins, previous.pins_applied ?? {}),
    thresholds: sameThresholds ? 0 : 1,
  };
  return {
    ...previous,
    draft,
    labels_draft: labels,
    pins_draft: pins,
    thresholds_draft: thresholds,
    pending_by: pendingBy,
    pending: pendingBy.decisions + pendingBy.labels + pendingBy.pins + pendingBy.thresholds,
  };
}

/** Every click is saved in the draft (R-108-04), shown at once (optimistic), rolled back on error. */
export function useSetDecisions(datasetId: string) {
  const client = useQueryClient();
  const key = visionKeys.decisions(datasetId);
  return useMutation({
    mutationFn: (patch: EditPatch) => visionApi.setDecisions(datasetId, patch),
    onMutate: async (patch) => {
      await client.cancelQueries({ queryKey: key });
      const previous = client.getQueryData<Decisions>(key);
      if (previous) client.setQueryData<Decisions>(key, patchedDecisions(previous, patch));
      return { previous };
    },
    onError: (_error, _patch, context) => {
      if (context?.previous) client.setQueryData(key, context.previous);
    },
    onSuccess: (data) => client.setQueryData(key, data),
  });
}

/** R-VQT-01: what the rules would say with these thresholds (asked by a click, nothing written). */
export function usePreviewQuality(datasetId: string) {
  return useMutation({ mutationFn: (thresholds: Record<string, number>) => visionApi.previewQuality(datasetId, thresholds) });
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

// --- Map (109) and Prepare (110): backend/app/api/image_views.py --------------------------------

export type MapMethods = Schemas["ImageMapMethodsOut"];
export type MapMethod = Schemas["ImageMapMethodOut"];
export type MapProjection = Schemas["ImageMapProjectionOut"];
export type MapState = Schemas["ImageMapStateOut"];
export type ImageMap = Schemas["ImageMapOut"];
export type Projection = ImageMap["projection"];
export type Clustering = ImageMap["clustering"];
export type PrepareState = Schemas["ImagePrepOut"];
export type StepPreview = Schemas["ImagePrepPreviewOut"];
export type MapQuery = { method: string; projection: Projection; clustering: Clustering; k: number | null };

export const viewKeys = {
  methods: ["image-map-methods"] as const,
  map: (id: string, query: MapQuery) => ["datasets", id, "images", "map", query] as const,
  maps: (id: string) => ["datasets", id, "images", "map"] as const,
  prepare: (id: string) => ["datasets", id, "images", "prepare"] as const,
  preview: (id: string, imageId: string, step: string, recipe: string, seed: number) =>
    ["datasets", id, "images", "preview", imageId, step, recipe, seed] as const,
};

function mapPath(id: string, query: MapQuery): string {
  const params = new URLSearchParams({ method: query.method, projection: query.projection, clustering: query.clustering });
  if (query.k !== null) params.set("k", String(query.k));
  return `${base(id)}/map?${params.toString()}`;
}

export const viewsApi = {
  methods: () => request<MapMethods>("GET", "/images/map/methods"),
  download: (key: string) => request<MapMethods>("POST", `/images/weights/${encodeURIComponent(key)}/download`),
  map: (id: string, query: MapQuery) => request<MapState>("GET", mapPath(id, query)),
  build: (id: string, method: string, confirmLong = false) =>
    request<MapState>("POST", `${base(id)}/map`, { method, confirm_long: confirmLong }),
  prepare: (id: string) => request<PrepareState>("GET", `${base(id)}/prepare`),
  savePrepare: (id: string, recipe: Record<string, unknown>) => request<PrepareState>("PUT", `${base(id)}/prepare`, { recipe }),
  preview: (id: string, body: { image_id: string; step: string; recipe: Record<string, unknown>; seed: number }) =>
    request<StepPreview>("POST", `${base(id)}/prepare/preview`, body),
};

/** While a download runs, the list is read again every second (R-VDP-02: the user started it). */
export function useMapMethods() {
  return useQuery({
    queryKey: viewKeys.methods,
    queryFn: viewsApi.methods,
    staleTime: 60_000,
    refetchInterval: (query) => (Object.values(query.state.data?.downloads ?? {}).some((d) => d.state === "running") ? 1000 : false),
    refetchIntervalInBackground: true,
  });
}

/** The map, or where its build stands; polled while it is being built (R-VMAP-10). */
export function useImageMap(datasetId: string, query: MapQuery, enabled: boolean) {
  return useQuery({
    queryKey: viewKeys.map(datasetId, query),
    queryFn: () => viewsApi.map(datasetId, query),
    enabled,
    staleTime: Infinity,
    refetchInterval: (q) => (q.state.data?.state === "running" ? 800 : false),
    refetchIntervalInBackground: true,
  });
}

export function useBuildMap(datasetId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ method, confirmLong }: { method: string; confirmLong?: boolean }) => viewsApi.build(datasetId, method, confirmLong),
    onSuccess: () => void client.invalidateQueries({ queryKey: viewKeys.maps(datasetId) }),
  });
}

export function useDownloadWeights() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: viewsApi.download,
    onSuccess: (data) => client.setQueryData(viewKeys.methods, data),
  });
}

export function usePrepare(datasetId: string, enabled: boolean) {
  return useQuery({ queryKey: viewKeys.prepare(datasetId), queryFn: () => viewsApi.prepare(datasetId), enabled, staleTime: Infinity });
}

export function useSavePrepare(datasetId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (recipe: Record<string, unknown>) => viewsApi.savePrepare(datasetId, recipe),
    onSuccess: (data) => client.setQueryData(viewKeys.prepare(datasetId), data),
  });
}

/** One step on one image (the server keeps 64 of them; here they stay while the tab is open).
 *  While a new setting is computed, the previous picture of the same image stays shown. */
export function useStepPreview(datasetId: string, imageId: string, step: string, recipe: Record<string, unknown>, seed: number) {
  const key = JSON.stringify(recipe);
  return useQuery({
    queryKey: viewKeys.preview(datasetId, imageId, step, key, seed),
    queryFn: () => viewsApi.preview(datasetId, { image_id: imageId, step, recipe, seed }),
    staleTime: Infinity,
    gcTime: 5 * 60_000,
    placeholderData: (previous) => (previous?.image_id === imageId && previous.step === step ? previous : undefined),
  });
}

// --- b2: Quality > Advanced (R-VQ-16 from the latest run, R-VQ-17 on DINOv2 vectors) ----------

export type Semantic = Schemas["ImageSemanticOut"];
export type SemanticGroup = Schemas["ImageSemanticGroupOut"];
export type Mislabelled = Schemas["MislabelledOut"];

export const advancedKeys = {
  semantic: (id: string) => ["datasets", id, "images", "semantic"] as const,
  mislabelled: (id: string) => ["vision", id, "mislabelled"] as const,
};

export const advancedApi = {
  semantic: (id: string) => request<Semantic>("GET", `${base(id)}/semantic`),
  findSemantic: (id: string, confirmLong = false) =>
    request<Semantic>("POST", `${base(id)}/semantic`, { confirm_long: confirmLong }),
  mislabelled: (id: string) =>
    request<Mislabelled>("GET", `/modules/vision/datasets/${encodeURIComponent(id)}/mislabelled`),
};

/** Read only once Advanced is open (onion rule); polled while the vectors are computed. */
export function useSemantic(datasetId: string, enabled: boolean) {
  return useQuery({
    queryKey: advancedKeys.semantic(datasetId),
    queryFn: () => advancedApi.semantic(datasetId),
    enabled,
    refetchInterval: (query) => (query.state.data?.state === "running" ? 1000 : false),
    refetchIntervalInBackground: true,
  });
}

export function useFindSemantic(datasetId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (confirmLong: boolean) => advancedApi.findSemantic(datasetId, confirmLong),
    onSuccess: (data) => client.setQueryData(advancedKeys.semantic(datasetId), data),
  });
}

export function useMislabelled(datasetId: string, enabled: boolean) {
  return useQuery({
    queryKey: advancedKeys.mislabelled(datasetId),
    queryFn: () => advancedApi.mislabelled(datasetId),
    enabled,
  });
}
