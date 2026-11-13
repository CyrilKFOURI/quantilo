import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";

import { api, reportFileUrl } from "../../api/client";
import { useActiveFilters } from "../../api/filters";

const downloaded = new Set<string>();
/** 025, R-REP-17: the reports asked for the print preview are not downloaded when they are done. */
const forPreview = new Set<string>();

/** The dataset's HTML report (025): start it over the active filters, follow it, download it once
 * done (or keep it for the print preview, R-REP-17). The job id lives in the query cache, so the
 * Overview and Data pages share it. */
export function useReport(datasetId: string) {
  const client = useQueryClient();
  const filters = useActiveFilters(datasetId);
  const key = ["report", datasetId];
  const { data: jobId } = useQuery<string | null>({
    queryKey: key,
    queryFn: () => null,
    initialData: null,
    staleTime: Infinity,
  });
  const job = useQuery({
    queryKey: ["report-jobs", jobId],
    queryFn: () => api.reportJob(jobId ?? ""),
    enabled: Boolean(jobId),
    refetchInterval: (query) =>
      query.state.data?.status === "running" || !query.state.data ? 400 : false,
    refetchIntervalInBackground: true,
    retry: false,
  });
  const start = useMutation({
    // the flag only tells onSuccess what the report is for (download or preview)
    mutationFn: (preview: boolean) => (void preview, api.reportStart(datasetId, filters)),
    onSuccess: (started, preview) => {
      if (preview) forPreview.add(started.id);
      client.setQueryData(["report-jobs", started.id], started);
      client.setQueryData(key, started.id);
    },
  });
  const cancel = useMutation({
    mutationFn: (id: string) => api.reportCancel(id),
    onSuccess: (cancelled) =>
      client.setQueryData(["report-jobs", cancelled.id], cancelled),
  });

  const current = jobId ? job.data : undefined;
  useEffect(() => {
    if (current?.status !== "done" || downloaded.has(current.id) || forPreview.has(current.id)) return;
    downloaded.add(current.id);
    const link = document.createElement("a");
    link.href = reportFileUrl(current.id);
    link.download = current.file_name ?? "report.html";
    document.body.append(link);
    link.click();
    link.remove();
  }, [current]);

  return {
    job: current,
    running: start.isPending || current?.status === "running",
    error:
      start.error?.message ??
      (current?.status === "failed" ? current.error : null),
    start: () => start.mutate(false),
    /** R-REP-17: a report for the print preview (the finished one when its filters still hold). */
    startPreview: () => start.mutate(true),
    cancel: () => {
      if (current) cancel.mutate(current.id);
    },
  };
}

/** R-REP-17: the HTML of a finished report, read for the print preview. */
export function useReportHtml(jobId: string | null) {
  return useQuery({
    queryKey: ["report-html", jobId],
    queryFn: async () => {
      const response = await fetch(reportFileUrl(jobId ?? ""));
      if (!response.ok) throw new Error(`The report could not be read (${response.status}).`);
      return response.text();
    },
    enabled: Boolean(jobId),
    staleTime: Infinity,
    retry: false,
  });
}
