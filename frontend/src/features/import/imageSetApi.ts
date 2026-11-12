import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { queryKeys } from "../../api/hooks";
import { request, type ImportResponse } from "../../api/client";
import type { components } from "../../api/schema";

/** Image sets (106), kept here rather than in api/hooks.ts (other branches edit it). Routes:
 *  /api/v1/images/* and /api/v1/datasets/{id}/images/* (backend/app/api/images.py). */

export type ImagePreview = components["schemas"]["ImagePreviewOut"];
export type ImageSetOptions = components["schemas"]["ImageSetOptions"];
export type ImageReport = components["schemas"]["ImageReportOut"];

export const imageSetApi = {
  preview: (path: string, options: ImageSetOptions) =>
    request<ImagePreview>("POST", "/images/preview", { path, options }),
  importSet: (path: string, name: string | null, options: ImageSetOptions, projectId: string | null) =>
    request<ImportResponse>("POST", "/images/import", { path, name, options, project_id: projectId }),
  report: (datasetId: string) =>
    request<ImageReport>("GET", `/datasets/${encodeURIComponent(datasetId)}/images/report`),
};

/** The URL of an image of a set (thumbnail, original or mask), for an <img src>. */
export function imageUrl(datasetId: string, imageId: string, kind: "thumb" | "file" | "mask" = "thumb") {
  return `/api/v1/datasets/${encodeURIComponent(datasetId)}/images/${encodeURIComponent(imageId)}/${kind}`;
}

export function useImagePreview(path: string, ignoreRules: string[]) {
  return useQuery({
    queryKey: ["image-preview", path, ignoreRules] as const,
    queryFn: () => imageSetApi.preview(path, { ignore_rules: ignoreRules }),
    retry: false,
    staleTime: Infinity,
    placeholderData: (previous) => previous,
  });
}

export function useImportImageSet() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (input: { path: string; name: string | null; options: ImageSetOptions; projectId: string | null }) =>
      imageSetApi.importSet(input.path, input.name, input.options, input.projectId),
    onSettled: (_data, _error, input) => {
      void client.invalidateQueries({ queryKey: queryKeys.datasets });
      if (input.projectId) void client.invalidateQueries({ queryKey: ["projects"] });
    },
  });
}
