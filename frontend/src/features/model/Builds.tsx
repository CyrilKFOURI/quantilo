import { Link } from "@tanstack/react-router";
import { RefreshCw } from "lucide-react";

import { formatCount, formatRelativeTime } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote } from "../../ui/fields";
import { useRebuild, type AnalysisBuild } from "./api";

const linkClass = "text-xs text-ink-2 underline decoration-line underline-offset-4 hover:text-accent hover:decoration-accent";

/** The analysis datasets built from this model (R-BLD-04): up to date or not (R-BLD-05), and the
 *  pages that use them first (KPI, ML). */
export function Builds({ projectId, builds }: { projectId: string; builds: AnalysisBuild[] }) {
  const rebuild = useRebuild(projectId);
  if (builds.length === 0) return null;
  return (
    <section className="space-y-3" aria-labelledby="model-builds">
      <div className="space-y-0.5">
        <h2 id="model-builds" className="text-sm font-semibold text-ink-1">
          {t("model.builds.title")}
        </h2>
        <p className="text-xs text-ink-3">{t("model.builds.body")}</p>
      </div>
      {rebuild.isError && <ErrorNote>{rebuild.error.message}</ErrorNote>}
      <ul className="divide-y divide-line rounded-md border border-line bg-surface-1">
        {builds.map((build) => {
          const stale = build.out_of_date.length > 0;
          const busy = rebuild.isPending && rebuild.variables === build.dataset_id;
          return (
            <li key={build.dataset_id} className="flex flex-wrap items-center gap-x-4 gap-y-1.5 px-3 py-2.5">
              <div className="min-w-0 flex-1 basis-60">
                <Link to="/d/$datasetId" params={{ datasetId: build.dataset_id }} className="block truncate text-[13px] font-medium text-ink-1 hover:text-accent">
                  {build.name}
                </Link>
                <p className="text-xs text-ink-3 tabular-nums" title={formatRelativeTime(build.updated_at)}>
                  {t(build.kind === "flat" ? "model.builds.flat" : "model.builds.aggregate")}
                  {" · "}
                  {build.status === "importing" ? t("model.builds.importing") : t("model.builds.rows", { rows: formatCount(build.rows) })}
                </p>
              </div>
              <p className={`min-w-0 basis-60 text-xs ${stale ? "text-warn" : "text-ok"}`}>
                {stale ? t("model.builds.outOfDate", { why: build.out_of_date.join("; ") }) : t("model.builds.current")}
              </p>
              <div className="flex items-center gap-3">
                <Link to="/d/$datasetId/kpi" params={{ datasetId: build.dataset_id }} className={linkClass}>
                  {t("model.builds.kpi")}
                </Link>
                <Link to="/d/$datasetId/ml" params={{ datasetId: build.dataset_id }} className={linkClass}>
                  {t("model.builds.ml")}
                </Link>
                {stale && build.can_rebuild && (
                  <Button icon={RefreshCw} disabled={rebuild.isPending} onClick={() => rebuild.mutate(build.dataset_id)}>
                    {busy ? t("model.builds.rebuilding") : t("model.builds.rebuild")}
                  </Button>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
