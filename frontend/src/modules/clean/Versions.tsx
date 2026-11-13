import { useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { Braces, Database, Download, FileCode2, FileText, GitCompareArrows, RefreshCw, Repeat, ScrollText, Trash2 } from "lucide-react";
import { useState } from "react";

import { useDeleteDataset } from "../../api/hooks";
import { formatCount, formatRelativeTime } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { ErrorNote } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { Popover } from "../../ui/Popover";
import { cleanApi, cleanKeys, cleanUrls, type CleanFamily, type CleanVersion, type Step } from "./api";
import { DraftJournal, JournalDialog } from "./Journal";
import { RebuildDialog } from "./Rebuild";
import { ReplayDialog } from "./Replay";

/** 101 (C-39): a version's steps as a recipe file, to replay on another file. */
async function downloadRecipe(version: CleanVersion) {
  const recipe = await cleanApi.recipe(version.dataset_id);
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([JSON.stringify(recipe, null, 1)], { type: "application/json" }));
  link.download = `${version.name.replace(/[\\/:*?"<>|]+/g, "_")}.clean.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}

/** The lineage of the original (Original → Cleaned v1 → v2), each version with its steps, what
 *  it changed, and Open, Compare with original, Download recipe, Delete; and Replay a cleaning
 *  made elsewhere on this dataset (101). */
export function Versions({ datasetId, family, originalName, onReplay }: { datasetId: string; family: CleanFamily | undefined; originalName: string; onReplay: (steps: Step[]) => void }) {
  const [deleting, setDeleting] = useState<CleanVersion | null>(null);
  const [replaying, setReplaying] = useState(false);
  const [rebuilding, setRebuilding] = useState<CleanVersion | null>(null);
  const [recipeError, setRecipeError] = useState<string | null>(null);
  const [logging, setLogging] = useState<CleanVersion | null>(null);
  const remove = useDeleteDataset();
  const client = useQueryClient();
  if (!family) return <Loading label={t("loading.default")} className="h-32 rounded-lg" compact />;
  const latest = Math.max(0, ...family.versions.map((version) => version.seq));
  return (
    <section aria-label={t("clean.versions.label")} className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button icon={Repeat} onClick={() => setReplaying(true)}>
          {t("clean.replay.open")}
        </Button>
      </div>
      {replaying && <ReplayDialog datasetId={datasetId} onClose={() => setReplaying(false)} onUse={onReplay} />}
      {rebuilding && <RebuildDialog version={rebuilding} family={family} onClose={() => setRebuilding(null)} />}
      {recipeError && <ErrorNote>{recipeError}</ErrorNote>}
      <DraftJournal datasetId={datasetId} />
      {logging && <JournalDialog version={logging} onClose={() => setLogging(null)} />}
      <ol className="divide-y divide-line rounded-lg border border-line bg-surface-1">
        <li className="flex flex-wrap items-baseline gap-x-3 gap-y-1 px-3 py-2.5">
          <Link to="/d/$datasetId" params={{ datasetId: family.original_id }} className="text-[13px] font-medium text-ink-1 hover:text-accent">
            {t("clean.versions.original")} · {originalName}
          </Link>
          <span className="text-xs text-ink-3">{t("clean.versions.originalHint")}</span>
        </li>
        {family.versions.map((version) => (
          <VersionRow
            key={version.dataset_id}
            version={version}
            latest={version.seq === latest}
            originalId={family.original_id}
            onDelete={() => setDeleting(version)}
            onRebuild={() => setRebuilding(version)}
            onLog={() => setLogging(version)}
            onRecipe={() => {
              setRecipeError(null);
              downloadRecipe(version).catch((error: unknown) => setRecipeError(error instanceof Error ? error.message : String(error)));
            }}
          />
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

const item = "flex min-h-9 w-full items-start gap-2 rounded-md px-2 py-1.5 text-left text-[13px] text-ink-1 hover:bg-surface-2";

/** Download: the recipe to replay (101), and N45's code an auditor reads without the app: the
 *  DuckDB SQL of the steps and a Python script on the quantilo library. */
function DownloadMenu({ version, onRecipe }: { version: CleanVersion; onRecipe: () => void }) {
  const [open, setOpen] = useState(false);
  const name = t("clean.versions.name", { seq: String(version.seq) });
  return (
    <Popover
      open={open}
      onOpenChange={setOpen}
      label={`${t("clean.versions.download")}: ${name}`}
      trigger={<Button variant="ghost" icon={Download} iconOnly aria-label={`${t("clean.versions.download")}: ${name}`} aria-expanded={open} title={t("clean.versions.download")} onClick={() => setOpen(!open)} />}
    >
      <div className="w-72 space-y-0.5 p-1">
        <button
          type="button"
          className={item}
          onClick={() => {
            setOpen(false);
            onRecipe();
          }}
        >
          <Braces size={15} strokeWidth={1.5} className="mt-0.5 shrink-0 text-ink-3" aria-hidden />
          <span>
            {t("clean.versions.recipe")}
            <span className="block text-xs text-ink-3">{t("clean.versions.recipeHint")}</span>
          </span>
        </button>
        <a href={cleanUrls.sql(version.dataset_id)} download className={item} onClick={() => setOpen(false)}>
          <Database size={15} strokeWidth={1.5} className="mt-0.5 shrink-0 text-ink-3" aria-hidden />
          <span>
            {t("clean.versions.sql")}
            <span className="block text-xs text-ink-3">{t("clean.versions.sqlHint")}</span>
          </span>
        </a>
        <a href={cleanUrls.python(version.dataset_id)} download className={item} onClick={() => setOpen(false)}>
          <FileCode2 size={15} strokeWidth={1.5} className="mt-0.5 shrink-0 text-ink-3" aria-hidden />
          <span>
            {t("clean.versions.python")}
            <span className="block text-xs text-ink-3">{t("clean.versions.pythonHint")}</span>
          </span>
        </a>
        <a href={cleanUrls.journalMarkdown(version.dataset_id)} download className={item} onClick={() => setOpen(false)}>
          <FileText size={15} strokeWidth={1.5} className="mt-0.5 shrink-0 text-ink-3" aria-hidden />
          <span>
            {t("clean.journal.markdown")}
            <span className="block text-xs text-ink-3">{t("clean.journal.fileHint")}</span>
          </span>
        </a>
        <a href={cleanUrls.journalHtml(version.dataset_id)} download className={item} onClick={() => setOpen(false)}>
          <FileText size={15} strokeWidth={1.5} className="mt-0.5 shrink-0 text-ink-3" aria-hidden />
          <span>{t("clean.journal.html")}</span>
        </a>
      </div>
    </Popover>
  );
}

function VersionRow({
  version,
  latest,
  originalId,
  onDelete,
  onRecipe,
  onRebuild,
  onLog,
}: {
  version: CleanVersion;
  latest: boolean;
  originalId: string;
  onDelete: () => void;
  onRecipe: () => void;
  onRebuild: () => void;
  onLog: () => void;
}) {
  const report = version.report;
  const automatic = version.steps.filter((step) => step.enabled && step.origin === "auto").length;
  const facts = [
    t("clean.versions.steps", { count: String(version.steps.length) }),
    // N03: the reliable fixes applied by default, said apart from the user's own steps
    automatic > 0 && t("clean.versions.autoSteps", { count: String(automatic) }),
    report && t("clean.banner.cells", { count: formatCount(report.cells_changed) }),
    report && report.rows_removed > 0 && t("clean.banner.rows", { count: formatCount(report.rows_removed) }),
    report && report.type_changes.length > 0 && t("clean.versions.typeChanges", { columns: report.type_changes.map((change) => change.column).join(", ") }),
    // N31: the family's checks, as they ran on this version
    version.checks && (version.checks.failed ? t("clean.versions.checksFail", { count: formatCount(version.checks.failed), rows: formatCount(version.checks.failing_rows) }) : t("clean.versions.checksPass")),
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
        {ready && version.out_of_date && (
          <Button variant="ghost" icon={RefreshCw} title={t("clean.versions.rebuildHint")} onClick={onRebuild}>
            {t("clean.versions.rebuild")}
          </Button>
        )}
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
        <Button variant="ghost" icon={ScrollText} iconOnly aria-label={`${t("clean.journal.open")}: ${t("clean.versions.name", { seq: String(version.seq) })}`} title={t("clean.journal.open")} onClick={onLog} />
        <DownloadMenu version={version} onRecipe={onRecipe} />
        <Button variant="ghost" icon={Trash2} iconOnly aria-label={`${t("clean.versions.delete")}: ${t("clean.versions.name", { seq: String(version.seq) })}`} title={t("clean.versions.delete")} onClick={onDelete} />
      </span>
    </li>
  );
}
