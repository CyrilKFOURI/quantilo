import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { request } from "../../api/client";
import type { components } from "../../api/schema";

/** Projects (087), kept here rather than in api/hooks.ts (other branches edit it). Routes:
 *  /api/v1/projects (backend/app/api/projects.py). */

export type Project = components["schemas"]["ProjectOut"];
export type ProjectImages = components["schemas"]["ProjectImagesOut"];
export type ImageRef = components["schemas"]["ImageRefOut"];

export const projectKeys = { all: ["projects"] as const };

export const projectsApi = {
  list: () => request<Project[]>("GET", "/projects"),
  create: (name: string) => request<Project>("POST", "/projects", { name }),
  rename: (id: string, name: string) => request<Project>("PATCH", `/projects/${encodeURIComponent(id)}`, { name }),
  remove: (id: string) => request<void>("DELETE", `/projects/${encodeURIComponent(id)}`),
  /** `fromProjectId` (C1): out of that project only (a dataset may be in several). */
  move: (datasetIds: string[], projectId: string | null, fromProjectId: string | null = null) =>
    request<Project[]>("POST", "/projects/move", { dataset_ids: datasetIds, project_id: projectId, from_project_id: fromProjectId }),
  /** C1: also in this project, the others kept. */
  add: (datasetIds: string[], projectId: string) => request<Project[]>("POST", `/projects/${encodeURIComponent(projectId)}/add`, { dataset_ids: datasetIds }),
  get: (id: string) => request<Project>("GET", `/projects/${encodeURIComponent(id)}`),
  /** C1 (R-PRJ-11): its tables, its image sets, and the columns that name its images. */
  images: (id: string) => request<ProjectImages>("GET", `/projects/${encodeURIComponent(id)}/images`),
};

export function useProjectImages(id: string, enabled: boolean) {
  return useQuery({ queryKey: [...projectKeys.all, id, "images"], queryFn: () => projectsApi.images(id), enabled });
}

/** C1: add datasets to a project, their other projects kept. */
export function useAddToProject() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (input: { datasetIds: string[]; projectId: string }) => projectsApi.add(input.datasetIds, input.projectId),
    onSuccess: (projects) => client.setQueryData(projectKeys.all, projects),
  });
}

/** Every project with its datasets. A server without projects (or a failed call) reads as none:
 *  the home page is then the plain list it was before 087. */
export function useProjects(): Project[] {
  const query = useQuery({ queryKey: projectKeys.all, queryFn: projectsApi.list });
  return query.data ?? [];
}

export function useCreateProject() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: projectsApi.create,
    onSuccess: () => client.invalidateQueries({ queryKey: projectKeys.all }),
  });
}

export function useRenameProject() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (input: { id: string; name: string }) => projectsApi.rename(input.id, input.name),
    onSuccess: () => client.invalidateQueries({ queryKey: projectKeys.all }),
  });
}

export function useDeleteProject() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: projectsApi.remove,
    onSuccess: () => client.invalidateQueries({ queryKey: projectKeys.all }),
  });
}

/** Moves at once on screen (the answer is every project, written straight into the cache). */
export function useMoveDatasets() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (input: { datasetIds: string[]; projectId: string | null; fromProjectId?: string | null }) =>
      projectsApi.move(input.datasetIds, input.projectId, input.fromProjectId ?? null),
    onMutate: async ({ datasetIds, projectId, fromProjectId }) => {
      await client.cancelQueries({ queryKey: projectKeys.all });
      const before = client.getQueryData<Project[]>(projectKeys.all);
      if (before) client.setQueryData(projectKeys.all, moveLocally(before, datasetIds, projectId, fromProjectId ?? null));
      return { before };
    },
    onError: (_error, _input, context) => {
      if (context?.before) client.setQueryData(projectKeys.all, context.before);
    },
    onSuccess: (projects) => client.setQueryData(projectKeys.all, projects),
  });
}

/** The optimistic move: `ids` already include the family (the server finds it again anyway).
 *  `fromProjectId` (C1): out of that project only; null: out of every other project. */
export function moveLocally(projects: Project[], ids: string[], projectId: string | null, fromProjectId: string | null = null): Project[] {
  const moving = new Set(ids);
  return projects.map((project) => {
    const leaves = fromProjectId === null || project.id === fromProjectId || project.id === projectId;
    const kept = leaves ? project.dataset_ids.filter((id) => !moving.has(id)) : project.dataset_ids;
    return { ...project, dataset_ids: project.id === projectId ? [...kept, ...ids] : kept };
  });
}

/** The first project a dataset is in (its family's), or null. */
export function projectOf(datasetId: string, projects: Project[]): Project | null {
  return projects.find((project) => project.dataset_ids.includes(datasetId)) ?? null;
}

/** C1: every project a dataset is in (an item may be in several). */
export function projectsOf(datasetId: string, projects: Project[]): Project[] {
  return projects.filter((project) => project.dataset_ids.includes(datasetId));
}
