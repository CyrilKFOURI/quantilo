import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { request } from "../../api/client";
import { queryKeys } from "../../api/hooks";
import type { components } from "../../api/schema";
import { projectKeys } from "../projects/api";

/** The model of a project (088), kept here rather than in api/hooks.ts (other branches edit it).
 *  Routes: /api/v1/projects/{id}/model (backend/app/api/model.py). */

type Schemas = components["schemas"];
export type Model = Schemas["ModelOut"];
export type ModelTable = Schemas["ModelTableOut"];
export type ModelColumn = Schemas["ModelColumnOut"];
export type Relation = Schemas["RelationOut"];
export type RelationCheck = Schemas["CheckOut"];
export type Reason = Schemas["ReasonOut"];
export type JoinType = "left" | "inner" | "right" | "full";
export type BuildPreview = Schemas["PreviewOut"];
export type AnalysisBuild = Schemas["BuildOut"];
export type Position = { x: number; y: number };
export type Columns = { child_id: string; child_columns: string[]; parent_id: string; parent_columns: string[] };
/** C1 (R-REL-14): keys compared as written, or without case, accents and punctuation. */
export type Match = "exact" | "normalized";
/** C1 (R-BLD-07): a condition on a column of a table the build reaches. */
export type BuildFilter = Schemas["FilterIn"];

export const modelKeys = {
  model: (projectId: string) => ["model", projectId] as const,
  preview: (projectId: string, body: unknown) => ["model", projectId, "preview", body] as const,
};

const base = (projectId: string) => `/projects/${encodeURIComponent(projectId)}/model`;

export const modelApi = {
  get: (projectId: string) => request<Model>("GET", base(projectId)),
  save: (projectId: string, body: Columns & { join_type?: JoinType | null; status?: "accepted" | "rejected"; origin?: "detected" | "user"; match?: Match; active?: boolean }) =>
    request<Model>("POST", `${base(projectId)}/relations`, body),
  update: (projectId: string, id: string, body: { join_type?: JoinType; child_columns?: string[]; parent_columns?: string[]; match?: Match; active?: boolean }) =>
    request<Model>("PATCH", `${base(projectId)}/relations/${encodeURIComponent(id)}`, body),
  remove: (projectId: string, id: string) => request<Model>("DELETE", `${base(projectId)}/relations/${encodeURIComponent(id)}`),
  check: (projectId: string, body: Columns & { join_type?: JoinType; match?: Match }) => request<RelationCheck>("POST", `${base(projectId)}/check`, body),
  /** C1 (R-BLD-08): a date table, one row per day of a date column, related to its table. */
  dates: (projectId: string, body: { dataset_id: string; column: string; name?: string | null }) =>
    request<{ dataset_id: string }>("POST", `${base(projectId)}/dates`, body),
  layout: (projectId: string, body: { positions: Record<string, Position>; versions?: Record<string, string> }) =>
    request<Model>("PUT", `${base(projectId)}/layout`, body),
  preview: (projectId: string, body: BuildBody) => request<BuildPreview>("POST", `${base(projectId)}/preview`, body),
  build: (projectId: string, body: BuildBody) => request<{ dataset_id: string }>("POST", `${base(projectId)}/build`, body),
  rebuild: (projectId: string, datasetId: string) =>
    request<{ dataset_id: string }>("POST", `${base(projectId)}/builds/${encodeURIComponent(datasetId)}/rebuild`),
  aggregate: (projectId: string, body: { relation_id: string; measures?: { column: string; function: "sum" | "mean" | "min" | "max" }[] | null; name?: string | null }) =>
    request<{ dataset_id: string }>("POST", `${base(projectId)}/aggregate`, body),
};

export type BuildBody = { fact_id?: string | null; picks?: Record<string, string[]>; name?: string | null; filters?: BuildFilter[] };

/** The model: detected when it opens (bounded on the server, R-REL-10), then kept. */
export function useModel(projectId: string) {
  return useQuery({ queryKey: modelKeys.model(projectId), queryFn: () => modelApi.get(projectId), staleTime: 30_000 });
}

/** Every decision answers the whole model: written straight into the cache. */
function useModelMutation<T>(projectId: string, run: (input: T) => Promise<Model>) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: run,
    onSuccess: (model) => client.setQueryData(modelKeys.model(projectId), model),
  });
}

export function useSaveRelation(projectId: string) {
  return useModelMutation(projectId, (body: Parameters<typeof modelApi.save>[1]) => modelApi.save(projectId, body));
}

/** Accept several proposals one after the other (each answer is the model after it). */
export function useAcceptAll(projectId: string) {
  return useModelMutation(projectId, async (relations: Relation[]) => {
    let model: Model | undefined;
    for (const relation of relations) {
      model = await modelApi.save(projectId, { ...columnsOf(relation), status: "accepted" });
    }
    return model ?? modelApi.get(projectId);
  });
}

export function useUpdateRelation(projectId: string) {
  return useModelMutation(projectId, (input: { id: string; body: Parameters<typeof modelApi.update>[2] }) =>
    modelApi.update(projectId, input.id, input.body),
  );
}

export function useDeleteRelation(projectId: string) {
  return useModelMutation(projectId, (id: string) => modelApi.remove(projectId, id));
}

export function useSaveLayout(projectId: string) {
  return useModelMutation(projectId, (body: Parameters<typeof modelApi.layout>[1]) => modelApi.layout(projectId, body));
}

/** The analysis dataset counted before it is written (fast: the joins, no copy). */
export function useBuildPreview(projectId: string, body: BuildBody, enabled: boolean) {
  return useQuery({
    queryKey: modelKeys.preview(projectId, body),
    queryFn: () => modelApi.preview(projectId, body),
    enabled,
    staleTime: 10_000,
    retry: false,
  });
}

/** A new dataset in the project: the model, the projects and the dataset list change. */
function useCreating<T>(projectId: string, run: (input: T) => Promise<{ dataset_id: string }>) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: run,
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: modelKeys.model(projectId) });
      void client.invalidateQueries({ queryKey: projectKeys.all });
      void client.invalidateQueries({ queryKey: queryKeys.datasets });
    },
  });
}

export function useBuild(projectId: string) {
  return useCreating(projectId, (body: BuildBody) => modelApi.build(projectId, body));
}

export function useRebuild(projectId: string) {
  return useCreating(projectId, (datasetId: string) => modelApi.rebuild(projectId, datasetId));
}

export function useDateTable(projectId: string) {
  return useCreating(projectId, (body: Parameters<typeof modelApi.dates>[1]) => modelApi.dates(projectId, body));
}

export function useAggregate(projectId: string) {
  return useCreating(projectId, (body: Parameters<typeof modelApi.aggregate>[1]) => modelApi.aggregate(projectId, body));
}

export function columnsOf(relation: Columns): Columns {
  return {
    child_id: relation.child_id,
    child_columns: relation.child_columns,
    parent_id: relation.parent_id,
    parent_columns: relation.parent_columns,
  };
}
