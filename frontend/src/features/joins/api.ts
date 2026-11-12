import { useMutation, useQueryClient } from "@tanstack/react-query";

import { request } from "../../api/client";
import { queryKeys } from "../../api/hooks";
import type { components } from "../../api/schema";
import { projectKeys } from "../projects/api";

/** Join the open dataset with another one (089), kept here rather than in api/hooks.ts.
 *  Routes: /api/v1/datasets/{id}/join (backend/app/api/joins.py). */

type Schemas = components["schemas"];
export type JoinSuggestion = Schemas["JoinSuggestOut"];
export type JoinKey = Schemas["JoinKeyOut"];
export type JoinPreview = Schemas["JoinPreviewOut"];
export type JoinBody = Schemas["JoinIn"];
export type JoinMode = JoinKey["mode"];
export type JoinType = "left" | "inner" | "right" | "full";

const base = (datasetId: string) => `/datasets/${encodeURIComponent(datasetId)}/join`;

export const joinKeys = {
  suggest: (datasetId: string, otherId: string) => ["join", datasetId, "suggest", otherId] as const,
  preview: (datasetId: string, body: JoinBody) => ["join", datasetId, "preview", body] as const,
};

export const joinApi = {
  suggest: (datasetId: string, otherId: string) =>
    request<JoinSuggestion>("POST", `${base(datasetId)}/suggest`, { other_id: otherId }),
  preview: (datasetId: string, body: JoinBody) => request<JoinPreview>("POST", `${base(datasetId)}/preview`, body),
  create: (datasetId: string, body: JoinBody) => request<{ dataset_id: string }>("POST", base(datasetId), body),
};

/** Creating the joined dataset adds a dataset (and maybe a project member): both lists refresh. */
export function useCreateJoin(datasetId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (body: JoinBody) => joinApi.create(datasetId, body),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: queryKeys.datasets });
      void client.invalidateQueries({ queryKey: projectKeys.all });
    },
  });
}
