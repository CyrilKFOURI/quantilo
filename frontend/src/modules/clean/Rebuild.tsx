import { useMutation, useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";

import { t } from "../../lib/i18n";
import { Button, buttonClass } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { ErrorNote } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { cleanApi, useCleanJob, type CleanFamily, type CleanVersion } from "./api";
import { Report } from "./Replay";

/** B1 (C-47): Rebuild from original. An out-of-date version's steps, checked on the dataset it
 *  was made from as it is now (the replay report, step by step), then saved as the next cleaned
 *  version; the old one stays as it is. */
export function RebuildDialog({ version, family, onClose }: { version: CleanVersion; family: CleanFamily; onClose: () => void }) {
  const madeFrom = version.source_id ?? family.original_id;
  const name = t("clean.versions.name", { seq: String(version.seq) });
  const next = t("clean.versions.name", { seq: String(Math.max(0, ...family.versions.map((item) => item.seq)) + 1) });
  const source = version.source_id && version.source_seq ? t("clean.versions.name", { seq: String(version.source_seq) }) : t("clean.rebuild.sourceOriginal");
  // asked for once the dialog is open (a click), never before
  const check = useQuery({
    queryKey: ["datasets", madeFrom, "clean", "rebuild", version.dataset_id],
    queryFn: () => cleanApi.replay(madeFrom, { version_id: version.dataset_id }),
    staleTime: 0,
    retry: false,
  });
  const start = useMutation({ mutationFn: () => cleanApi.rebuild(version.dataset_id) });
  const job = useCleanJob(start.data?.id ?? null);
  const running = start.isPending || job.data?.status === "running" || (Boolean(start.data) && !job.data);
  const done = job.data?.status === "done" ? job.data : null;
  const failed = start.error?.message ?? (job.data && ["failed", "cancelled"].includes(job.data.status) ? job.data.error : null);

  return (
    <Dialog
      open
      onOpenChange={(open) => !open && onClose()}
      title={t("clean.rebuild.title", { name })}
      size="lg"
      footer={
        done?.created_dataset_id ? (
          <>
            <Button variant="ghost" onClick={onClose}>
              {t("dialog.close")}
            </Button>
            <Link to="/d/$datasetId" params={{ datasetId: done.created_dataset_id }} className={buttonClass("primary")} onClick={onClose}>
              {t("clean.rebuild.open")}
            </Link>
          </>
        ) : (
          <>
            <Button variant="ghost" onClick={onClose}>
              {t("import.cancel")}
            </Button>
            <Button variant="primary" disabled={!check.data || running} onClick={() => start.mutate()}>
              {t("clean.rebuild.save", { next })}
            </Button>
          </>
        )
      }
    >
      <div className="space-y-4">
        <p className="max-w-prose text-[13px] text-ink-2">{t("clean.rebuild.intro", { source, next, name })}</p>
        {check.isPending && <Loading label={t("clean.rebuild.checking")} className="h-24 rounded-md" compact />}
        {check.error && <ErrorNote>{check.error.message}</ErrorNote>}
        {running && <Loading label={t("clean.rebuild.saving")} className="h-9 rounded-md" compact />}
        {done && <p className="text-[13px] text-ink-1">{t("clean.rebuild.done", { next })}</p>}
        {failed && <ErrorNote>{failed}</ErrorNote>}
        {check.data && <Report report={check.data} />}
      </div>
    </Dialog>
  );
}
