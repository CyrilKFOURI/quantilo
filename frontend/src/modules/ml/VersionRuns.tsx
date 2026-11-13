import { useQueries } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";

import { api } from "../../api/client";
import { useFamily } from "../../features/lineage/api";
import { t } from "../../lib/i18n";

/** C-62: the runs on the other versions of this dataset's family (original and cleaned versions),
 *  for the same target, grouped by version with a link: runs stay filed by dataset, the family is
 *  read so that a model trained on Cleaned v2 is not lost from sight on the original. Reads the
 *  runs lists only (already cached when visited), never trains. */
export function VersionRuns({ datasetId, target }: { datasetId: string; target: string | null }) {
  const { family } = useFamily(datasetId);
  const others = family
    ? [
        { id: family.original_id, name: t("clean.versions.original") },
        ...family.versions.filter((version) => version.status === "ready").map((version) => ({ id: version.dataset_id, name: t("clean.versions.name", { seq: String(version.seq) }) })),
      ].filter((member) => member.id !== datasetId)
    : [];
  const runs = useQueries({
    queries: others.map((member) => ({
      queryKey: ["datasets", member.id, "ml", "runs", target ?? ""],
      queryFn: () => api.mlRuns(member.id, target ?? ""),
      enabled: Boolean(target),
      retry: false,
    })),
  });
  const found = others.map((member, index) => ({ ...member, runs: runs[index]?.data ?? [] })).filter((member) => member.runs.length > 0);
  if (!target || found.length === 0) return null;
  return (
    <section aria-label={t("ml.versionRuns.title")} className="space-y-1 border-t border-line pt-3">
      <h3 className="text-xs text-ink-3">{t("ml.versionRuns.title")}</h3>
      <ul className="space-y-0.5 text-xs">
        {found.map((member) => {
          const best = member.runs.filter((run) => run.score !== null).sort((a, b) => (a.greater_is_better ? (b.score ?? 0) - (a.score ?? 0) : (a.score ?? 0) - (b.score ?? 0)))[0];
          return (
            <li key={member.id} className="flex min-w-0 items-baseline justify-between gap-2">
              <Link to="/d/$datasetId/ml" params={{ datasetId: member.id }} className="min-w-0 truncate text-accent hover:underline">
                {member.name}
              </Link>
              <span className="shrink-0 text-ink-3 tabular-nums">
                {t("ml.versionRuns.count", { count: String(member.runs.length) })}
                {best && best.score !== null ? ` · ${best.metric} ${best.score.toFixed(3)}` : ""}
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
