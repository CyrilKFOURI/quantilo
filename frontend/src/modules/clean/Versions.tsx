import { useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { GitCompareArrows, Trash2 } from "lucide-react";
import { useState } from "react";

import { useDeleteDataset } from "../../api/hooks";
import { formatCount, formatRelativeTime } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { ErrorNote } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { cleanKeys, type CleanFamily, type CleanVersion } from "./api";

/** The lineage of the original (Original → Cleaned v1 → v2), each version with its steps, what
 *  it changed, and Open, Compare with original, Delete. */
export function Versions({ family, originalName }: { family: CleanFamily | undefined; originalName: string }) {
  const [deleting, setDeleting] = useState<CleanVersion | null>(null);
  const remove = useDeleteDataset();
  const client = useQueryClient();
  if (!family) return <Loading label={t("loading.default")} className="h-32 rounded-lg" compact />;
  const latest = Math.max(0, ...family.versions.map((version) => version.seq));
  return (
    <section aria-label={t("clean.versions.label")} className="space-y-3">
      <ol className="divide-y divide-line rounded-lg border border-line bg-surface-1">
        <li className="flex flex-wrap items-baseline gap-x-3 gap-y-1 px-3 py-2.5">
          <Link to="/d/$datasetId" params={{ datasetId: family.original_id }} className="text-[13px] font-medium text-ink-1 hover:text-accent">
            {t("clean.versions.original")} · {originalName}
          </Link>
          <span className="text-xs text-ink-3">{t("clean.versions.originalHint")}</span>
        </li>
        {family.versions.map((version) => (
          <VersionRow key={version.dataset_id} version={version} latest={version.seq === latest} originalId={family.original_id} onDelete={() => setDeleting(version)} />
        ))}
      </ol>
      {family.versions.length === 0 && <p className="text-[13px] text-ink-2">{t("clean.versions.none")}</p>}
      {remove.error && <ErrorNote>{remove.error.message}</ErrorNote>}
      {deleting && (
        <Dialog
          open
          onOpenChange={(open) => !open && setDeleting(null)}
          title={t("clean.versions.deleteTitle", { name: t("clean.versions.name", { seq: String(deleting.seq) }) })}
          size="sm"
          footer={
            <>
              <Button variant="ghost" onClick={() => setDeleting(null)}>
                {t("import.cancel")}
              </Button>
              <Button
                variant="danger"
                disabled={remove.isPending}
                onClick={() =>
                  remove.mutate(deleting.dataset_id, {
                    onSuccess: () => {
                      setDeleting(null);
                      void client.invalidateQueries({ queryKey: cleanKeys.lineage });
                    },
                  })
                }
              >
                {t("clean.versions.delete")}
              </Button>
            </>
          }
        >
          <p className="text-ink-2">{t("clean.versions.deleteBody")}</p>
        </Dialog>
      )}
    </section>
  );
}

function VersionRow({ version, latest, originalId, onDelete }: { version: CleanVersion; latest: boolean; originalId: string; onDelete: () => void }) {
  const report = version.report;
  const facts = [
    t("clean.versions.steps", { count: String(version.steps.length) }),
    report && t("clean.banner.cells", { count: formatCount(report.cells_changed) }),
    report && report.rows_removed > 0 && t("clean.banner.rows", { count: formatCount(report.rows_removed) }),
    report && report.type_changes.length > 0 && t("clean.versions.typeChanges", { columns: report.type_changes.map((change) => change.column).join(", ") }),
  ].filter(Boolean);
  const ready = version.status === "ready";
  return (
    <li className="flex flex-wrap items-center gap-x-3 gap-y-1.5 py-2.5 pr-2 pl-6">
      <div className="min-w-0 flex-1 space-y-0.5">
        <p className="flex flex-wrap items-baseline gap-x-2 text-[13px]">
          {ready ? (
            <Link to="/d/$datasetId" params={{ datasetId: version.dataset_id }} className="font-medium text-ink-1 hover:text-accent">
              {t("clean.versions.name", { seq: String(version.seq) })}
            </Link>
          ) : (
            <span className="font-medium text-ink-2">{t("clean.versions.name", { seq: String(version.seq) })}</span>
          )}
          {version.origin === "auto" && <span className="text-xs text-ink-3">{t("clean.versions.auto")}</span>}
          {version.source_seq !== null && version.source_seq !== undefined && <span className="text-xs text-ink-3">{t("clean.versions.from", { seq: String(version.source_seq) })}</span>}
          {latest && <span className="text-xs text-accent">{t("clean.versions.latest")}</span>}
          {!ready && <span className="text-xs text-ink-3">{t("clean.versions.importing")}</span>}
          {version.out_of_date && (
            <span className="text-xs text-warn" title={t("clean.versions.outOfDateHint")}>
              {t("clean.versions.outOfDate")}
            </span>
          )}
        </p>
        <p className="text-xs text-ink-2 tabular-nums">{facts.join(" · ")}</p>
        <p className="text-[11px] text-ink-3">{t("clean.versions.made", { when: formatRelativeTime(version.created_at) })}</p>
      </div>
      <span className="flex shrink-0 items-center gap-1">
        {ready && (
          <Link
            to="/d/$datasetId/changes"
            params={{ datasetId: version.dataset_id }}
            search={{ vs: originalId }}
            className="inline-flex h-(--control-h) items-center gap-1.5 rounded-md px-2.5 text-[13px] text-ink-2 hover:bg-surface-2 hover:text-ink-1"
          >
            <GitCompareArrows size={15} strokeWidth={1.5} aria-hidden />
            {t("clean.versions.compare")}
          </Link>
        )}
        <Button variant="ghost" icon={Trash2} iconOnly aria-label={`${t("clean.versions.delete")}: ${t("clean.versions.name", { seq: String(version.seq) })}`} title={t("clean.versions.delete")} onClick={onDelete} />
      </span>
    </li>
  );
}
