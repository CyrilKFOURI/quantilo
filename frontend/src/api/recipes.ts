/** Recipes (030) and saved time series analyses: types, requests and hooks, kept apart from
 *  client.ts / hooks.ts so the feature stays in one place. */
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { request } from "./client";
import type { components } from "./schema";

type Schemas = components["schemas"];
type Complete<T> = T extends (infer Item)[]
  ? Complete<Item>[]
  : T extends object
    ? { [K in keyof T]-?: Complete<T[K]> }
    : T;

export type RecipeSummary = Complete<Schemas["RecipeSummary"]>;
export type RecipeDetail = Complete<Schemas["RecipeDetail"]>;
export type RecipeStepView = Complete<Schemas["RecipeStepView"]>;
export type RecipeDraft = Complete<Schemas["RecipeDraft"]>;
export type RecipePlan = Complete<Schemas["RecipePlan"]>;
export type ColumnMatch = RecipePlan["matching"]["columns"][number];
export type PlannedStep = RecipePlan["steps"][number];
export type RecipeRunSummary = Complete<Schemas["RecipeRunSummary"]>;
export type RecipeRunDetail = Complete<Schemas["RecipeRunDetail"]>;
export type StepResult = NonNullable<RecipeRunDetail["report"]>["steps"][number];
export type StepLink = NonNullable<StepResult["link"]>;
export type FigureChange = RecipeRunDetail["compared"][number];
export type FigureHistory = RecipeDetail["history"][number];
export type CompatibleRecipe = Complete<Schemas["CompatibleRecipe"]>;
export type SavedAnalysis = Complete<Schemas["SavedAnalysis"]>;
export type SeriesConfig = Partial<Schemas["SeriesConfig"]>;
export type Mapping = Record<string, string | null>;
export type CreateRecipe = {
  name: string;
  dataset_id: string;
  keep: string[];
  active_filter: string | null;
  report: boolean;
  csv: boolean;
};

const BASE = "/modules/recipes";

export const recipesApi = {
  list: () => request<RecipeSummary[]>("GET", `${BASE}/recipes`),
  detail: (id: string) => request<RecipeDetail>("GET", `${BASE}/recipes/${id}`),
  draft: (datasetId: string) => request<RecipeDraft>("GET", `${BASE}/datasets/${datasetId}/draft`),
  compatible: (datasetId: string) =>
    request<CompatibleRecipe[]>("GET", `${BASE}/datasets/${datasetId}/compatible`),
  create: (body: CreateRecipe) => request<RecipeSummary>("POST", `${BASE}/recipes`, body),
  importFile: (content: string) =>
    request<RecipeSummary>("POST", `${BASE}/recipes/import`, { content }),
  edit: (id: string, body: { name?: string; enabled?: Record<string, boolean>; remove?: string[] }) =>
    request<RecipeDetail>("PATCH", `${BASE}/recipes/${id}`, body),
  remove: (id: string) => request<void>("DELETE", `${BASE}/recipes/${id}`),
  plan: (id: string, datasetId: string, mapping: Mapping | null) =>
    request<RecipePlan>("POST", `${BASE}/recipes/${id}/plan`, { dataset_id: datasetId, mapping }),
  run: (id: string, datasetId: string, mapping: Mapping | null) =>
    request<RecipeRunSummary>("POST", `${BASE}/recipes/${id}/runs`, { dataset_id: datasetId, mapping }),
  runDetail: (runId: string) => request<RecipeRunDetail>("GET", `${BASE}/recipe-runs/${runId}`),
  cancel: (runId: string) => request<RecipeRunSummary>("DELETE", `${BASE}/recipe-runs/${runId}`),
  analyses: (datasetId: string) =>
    request<SavedAnalysis[]>("GET", `/modules/timeseries/datasets/${datasetId}/analyses`),
  saveAnalysis: (datasetId: string, name: string, config: SeriesConfig) =>
    request<SavedAnalysis>("POST", `/modules/timeseries/datasets/${datasetId}/analyses`, { name, config }),
  deleteAnalysis: (datasetId: string, analysisId: string) =>
    request<void>("DELETE", `/modules/timeseries/datasets/${datasetId}/analyses/${analysisId}`),
};

/** Download links: the browser saves the file the server names. */
export const recipeUrls = {
  file: (id: string) => `/api/v1${BASE}/recipes/${id}/export`,
  python: (id: string) => `/api/v1${BASE}/recipes/${id}/python`,
  history: (id: string) => `/api/v1${BASE}/recipes/${id}/history.csv`,
  output: (runId: string, name: string) =>
    `/api/v1${BASE}/recipe-runs/${runId}/files/${encodeURIComponent(name)}`,
};

export const recipeKeys = {
  all: ["recipes"] as const,
  detail: (id: string) => ["recipes", id] as const,
  run: (runId: string) => ["recipes", "runs", runId] as const,
  plan: (id: string, datasetId: string, mapping: Mapping | null) =>
    ["recipes", id, "plan", datasetId, mapping] as const,
  draft: (datasetId: string) => ["datasets", datasetId, "recipe-draft"] as const,
  compatible: (datasetId: string) => ["datasets", datasetId, "compatible-recipes"] as const,
  analyses: (datasetId: string) => ["datasets", datasetId, "ts-analyses"] as const,
};

export function useRecipes() {
  return useQuery({ queryKey: recipeKeys.all, queryFn: recipesApi.list });
}

/** A recipe; followed while one of its runs is going on (the baseline after creation). */
export function useRecipe(id: string) {
  return useQuery({
    queryKey: recipeKeys.detail(id),
    queryFn: () => recipesApi.detail(id),
    refetchInterval: (query) =>
      query.state.data?.runs.some((run) => run.status === "running") ? 1000 : false,
    refetchIntervalInBackground: true,
  });
}

export function useRecipeDraft(datasetId: string, enabled: boolean) {
  return useQuery({
    queryKey: recipeKeys.draft(datasetId),
    queryFn: () => recipesApi.draft(datasetId),
    enabled,
    staleTime: 0,
  });
}

export function useCompatibleRecipes(datasetId: string, enabled: boolean) {
  return useQuery({
    queryKey: recipeKeys.compatible(datasetId),
    queryFn: () => recipesApi.compatible(datasetId),
    enabled,
    retry: false,
  });
}

export function useRecipePlan(id: string, datasetId: string | null, mapping: Mapping | null) {
  return useQuery({
    queryKey: recipeKeys.plan(id, datasetId ?? "", mapping),
    queryFn: () => recipesApi.plan(id, datasetId ?? "", mapping),
    enabled: Boolean(datasetId),
    retry: false,
    placeholderData: keepPreviousData,
  });
}

/** A run: polled until it ends (progress step by step), even when the tab is hidden. */
export function useRecipeRun(runId: string | null) {
  return useQuery({
    queryKey: recipeKeys.run(runId ?? ""),
    queryFn: () => recipesApi.runDetail(runId ?? ""),
    enabled: Boolean(runId),
    refetchInterval: (query) => (query.state.data?.status === "running" ? 700 : false),
    refetchIntervalInBackground: true,
  });
}

export function useRecipeActions(id: string) {
  const client = useQueryClient();
  const refresh = () => client.invalidateQueries({ queryKey: recipeKeys.all });
  return {
    edit: useMutation({
      mutationFn: (body: Parameters<typeof recipesApi.edit>[1]) => recipesApi.edit(id, body),
      onSuccess: refresh,
    }),
    remove: useMutation({ mutationFn: () => recipesApi.remove(id), onSuccess: refresh }),
    run: useMutation({
      mutationFn: (input: { datasetId: string; mapping: Mapping | null }) =>
        recipesApi.run(id, input.datasetId, input.mapping),
      onSuccess: refresh,
    }),
    cancel: useMutation({ mutationFn: (runId: string) => recipesApi.cancel(runId) }),
  };
}

export function useCreateRecipe() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: recipesApi.create,
    onSuccess: () => client.invalidateQueries({ queryKey: recipeKeys.all }),
  });
}

export function useImportRecipe() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: recipesApi.importFile,
    onSuccess: () => client.invalidateQueries({ queryKey: recipeKeys.all }),
  });
}

export function useAnalyses(datasetId: string, enabled = true) {
  return useQuery({
    queryKey: recipeKeys.analyses(datasetId),
    queryFn: () => recipesApi.analyses(datasetId),
    enabled,
  });
}

export function useAnalysisActions(datasetId: string) {
  const client = useQueryClient();
  const refresh = () => client.invalidateQueries({ queryKey: recipeKeys.analyses(datasetId) });
  return {
    save: useMutation({
      mutationFn: (input: { name: string; config: SeriesConfig }) =>
        recipesApi.saveAnalysis(datasetId, input.name, input.config),
      onSuccess: refresh,
    }),
    remove: useMutation({
      mutationFn: (analysisId: string) => recipesApi.deleteAnalysis(datasetId, analysisId),
      onSuccess: refresh,
    }),
  };
}
