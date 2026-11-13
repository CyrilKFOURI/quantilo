import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { Merge } from "lucide-react";

import { ApiError } from "../../api/client";
import { queryKeys } from "../../api/hooks";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote } from "../../ui/fields";
import { joinApi, joinKeys } from "./api";

/** C1 (R-JN-06): on a dataset made by Join…, what it was joined from and, when a source was
 *  reimported or retyped since, "Out of date" with Rebuild (the same join on the sources as they
 *  are now; the previous version is kept, 031). Nothing for any other dataset. */
export function JoinLineage({ datasetId }: { datasetId: string }) {
  const client = useQueryClient();
  const lineage = useQuery({
    queryKey: joinKeys.lineage(datasetId),
    queryFn: () => joinApi.lineage(datasetId),
    retry: (count, error) => !(error instanceof ApiError && error.status === 404) && count < 1,
    staleTime: 10_000,
  });
  const rebuild = useMutation({
    mutationFn: () => joinApi.rebuild(datasetId),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: joinKeys.lineage(datasetId) });
      void client.invalidateQueries({ queryKey: queryKeys.datasets });
      void client.invalidateQueries({ queryKey: ["datasets", datasetId] });
    },
  });
  const data = lineage.data;
  if (!data) return null;
  const stale = data.out_of_date.length > 0;
  return (
    <div className={`flex max-w-prose flex-wrap items-center gap-x-2 gap-y-1 border-l-2 pl-3 text-[13px] text-ink-2 ${stale ? "border-warn" : "border-line"}`}>
      <Merge size={14} strokeWidth={1.75} className="shrink-0 text-ink-3" aria-hidden />
      <span>
        {t("data.join.lineage.from")}{" "}
        {data.sources.map((source, index) => (
          <span key={source.id}>
            {index > 0 && " + "}
            {source.exists ? (
              <Link to="/d/$datasetId/data" params={{ datasetId: source.id }} className="text-accent hover:underline">
                {source.name}
              </Link>
            ) : (
              <span className="line-through">{source.name}</span>
            )}
          </span>
        ))}
        {" · "}
        <span className="font-mono text-xs">
          {data.left_columns.join(" + ")} = {data.right_columns.join(" + ")}
        </span>
      </span>
      {stale ? (
        <>
          <span className="text-warn">{t("data.join.lineage.outOfDate", { why: data.out_of_date.join("; ") })}</span>
          {data.can_rebuild && (
            <Button disabled={rebuild.isPending} onClick={() => rebuild.mutate()}>
              {rebuild.isPending ? t("data.join.lineage.rebuilding") : t("data.join.lineage.rebuild")}
            </Button>
          )}
        </>
      ) : (
        <span className="text-ink-3">{t("data.join.lineage.current")}</span>
      )}
      {rebuild.isError && <ErrorNote>{rebuild.error.message}</ErrorNote>}
    </div>
  );
}
