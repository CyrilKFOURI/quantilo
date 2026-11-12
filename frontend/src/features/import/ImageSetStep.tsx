import { useNavigate } from "@tanstack/react-router";
import { Info, Undo2 } from "lucide-react";
import { useEffect, useState } from "react";

import { useJob } from "../../api/hooks";
import { formatBytes, formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { Checkbox, ErrorNote, Field, ProgressBar, TextInput } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { useImagePreview, useImportImageSet, type ImagePreview } from "./imageSetApi";
import type { ImportStepProps } from "./stepTypes";

/** Image files and archives the file picker also offers (106); their content is checked later. */
export const IMAGE_ACCEPT = ".zip,.png,.jpg,.jpeg,.gif,.webp,.bmp,.tif,.tiff,.avif";

/** Reading rules the user can undo from the preview (they are passed back as `ignore_rules`). */
const UNDOABLE = new Set(["R-VIS-LAY-01", "R-VIS-LAY-02", "R-VIS-LAY-03", "R-VIS-LAY-04", "R-VIS-LAB-03"]);
const SHOWN_LABELS = 12;

/**
 * 106: a folder, a ZIP, an image or a Parquet file with an image column, imported as an image set
 * (a dataset with one row per image). The preview decodes nothing; every reading is a written rule
 * (R-VIS-*) shown with its reason and undoable here; the quality rules (R-VQ, 105) run during the
 * import job.
 */
export function ImageSetStep({ path, preview: tablePreview, onClose, onImported, project, onSimple }: ImportStepProps) {
  const navigate = useNavigate();
  const [ignored, setIgnored] = useState<string[]>([]);
  const [name, setName] = useState<string | null>(null);
  const [intoProject, setIntoProject] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const preview = useImagePreview(path, ignored);
  const importSet = useImportImageSet();
  const job = useJob(jobId);
  const data = preview.data;
  // R-VIS-SRC-03: a Parquet file is a table too; the one-table dialog stays one click away
  const asTable = tablePreview.detected?.format === "parquet";

  const openDataset = (datasetId: string) => {
    onClose();
    if (onImported) return onImported(datasetId);
    // R-107-02: an image set opens on its gallery
    void navigate({ to: "/d/$datasetId/vision", params: { datasetId } } as never);
  };
  const finished = job.data?.status === "done" ? job.data.dataset_id : null;
  useEffect(() => {
    if (finished) openDataset(finished);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finished]);
  const jobFailed = job.data?.status === "failed";
  const running = importSet.isPending || (jobId !== null && !jobFailed);
  const error = importError ?? (jobFailed ? (job.data?.message ?? t("import.failed")) : null);

  const start = () => {
    setImportError(null);
    importSet.mutate(
      {
        path,
        name: name ?? null,
        options: { ignore_rules: ignored },
        projectId: project && intoProject ? project.id : null,
      },
      {
        onSuccess: (response) => {
          if (response.reused || response.job?.status === "done") openDataset(response.dataset.id);
          else if (response.job?.status === "failed") setImportError(response.dataset.error ?? t("import.failed"));
          else if (response.job) setJobId(response.job.id);
        },
        onError: (failure) => setImportError(failure.message),
      },
    );
  };

  return (
    <Dialog
      open
      onOpenChange={(open) => !open && !running && onClose()}
      title={data?.suggested_name ?? tablePreview.suggested_name}
      description={data?.sentence ?? tablePreview.detected?.reason ?? path}
      size="lg"
      footer={
        running ? (
          <div className="flex w-full items-center gap-3">
            <span className="shrink-0 text-ink-2">
              {t("import.running")} · {job.data?.message ?? "…"}
            </span>
            <ProgressBar value={job.data?.progress ?? 0.02} label={t("import.running")} />
          </div>
        ) : (
          <>
            {asTable && (
              <Button variant="ghost" onClick={onSimple}>
                {t("vision.import.asTable")}
              </Button>
            )}
            <Button variant="ghost" onClick={onClose}>
              {t("import.cancel")}
            </Button>
            <Button variant="primary" disabled={!data || data.n_images === 0} onClick={start}>
              {data ? t("vision.import.start", { count: formatCount(data.n_images) }) : t("import.start")}
            </Button>
          </>
        )
      }
    >
      <div className="space-y-4">
        {error && <ErrorNote>{error}</ErrorNote>}
        {preview.isError && <ErrorNote>{preview.error.message}</ErrorNote>}
        {preview.isPending && <Loading label={t("vision.import.reading")} compact />}
        {data && (
          <>
            <fieldset disabled={running} className="flex flex-wrap items-end gap-x-6 gap-y-4">
              <Field label={t("preview.name")}>
                <TextInput
                  className="w-56"
                  value={name ?? data.suggested_name}
                  onChange={(event) => setName(event.target.value)}
                />
              </Field>
              {project && (
                <div className="flex h-(--control-h) items-center">
                  <Checkbox
                    checked={intoProject}
                    onChange={(event) => setIntoProject(event.target.checked)}
                    label={t("projects.import.add", { name: project.name })}
                  />
                </div>
              )}
            </fieldset>
            <Summary data={data} />
            <Rules data={data} ignored={ignored} onChange={setIgnored} disabled={running} />
            <LabelCounts data={data} />
            <Skipped data={data} />
            <p className="flex items-baseline gap-2 text-xs text-ink-3">
              <Info size={12} strokeWidth={1.5} aria-hidden className="shrink-0 translate-y-0.5" />
              {t("vision.import.quality")}
            </p>
          </>
        )}
      </div>
    </Dialog>
  );
}

function Summary({ data }: { data: ImagePreview }) {
  const formats = Object.entries(data.formats)
    .map(([format, count]) => `${format.toUpperCase()} ${formatCount(count)}`)
    .join(" · ");
  const items = [
    [t("vision.import.images"), formatCount(data.n_images)],
    [t("vision.import.size"), formatBytes(data.bytes)],
    [t("vision.import.formats"), formats || "—"],
    [t("vision.import.labels"), formatCount(Object.keys(data.labels).length)],
    [t("vision.import.masks"), formatCount(data.masks)],
  ] as const;
  return (
    <dl className="grid grid-cols-[repeat(auto-fit,minmax(120px,1fr))] gap-px overflow-hidden rounded-md border border-line bg-line text-xs">
      {items.map(([label, value]) => (
        <div key={label} className="min-w-0 bg-surface-1 px-3 py-2">
          <dt className="text-ink-3">{label}</dt>
          <dd className="truncate font-medium text-ink-1 tabular-nums">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function Rules({
  data,
  ignored,
  onChange,
  disabled,
}: {
  data: ImagePreview;
  ignored: string[];
  onChange: (codes: string[]) => void;
  disabled: boolean;
}) {
  if (data.rules.length === 0 && ignored.length === 0) return null;
  return (
    <section
      aria-label={t("preview.readByRule")}
      className="min-w-0 rounded-md border border-line bg-surface-1 px-3 py-2 text-xs text-ink-2"
    >
      <ul className="space-y-1.5">
        {data.rules.map((rule) => (
          <li key={`${rule.code}-${rule.message}`} className="flex min-w-0 items-start justify-between gap-3">
            <span className="min-w-0 break-words" title={rule.reason}>
              {rule.message}
            </span>
            {UNDOABLE.has(rule.code) && (
              <button
                type="button"
                disabled={disabled}
                onClick={() => onChange([...ignored, rule.code])}
                className="inline-flex min-h-6 shrink-0 items-center gap-1 text-accent hover:underline disabled:opacity-50 pointer-coarse:min-h-11"
              >
                <Undo2 size={12} strokeWidth={1.5} aria-hidden />
                {t("vision.import.undo")}
              </button>
            )}
          </li>
        ))}
        {ignored.map((code) => (
          <li key={code} className="flex min-w-0 items-start justify-between gap-3 text-ink-3">
            <span>{t("vision.import.undone", { code })}</span>
            <button
              type="button"
              disabled={disabled}
              onClick={() => onChange(ignored.filter((other) => other !== code))}
              className="inline-flex min-h-6 shrink-0 items-center text-accent hover:underline disabled:opacity-50 pointer-coarse:min-h-11"
            >
              {t("vision.import.redo")}
            </button>
          </li>
        ))}
      </ul>
      <p className="mt-1 font-medium text-ink-1">{t("preview.checkReading")}</p>
    </section>
  );
}

function LabelCounts({ data }: { data: ImagePreview }) {
  const labels = Object.entries(data.labels);
  const splits = Object.entries(data.splits);
  if (labels.length === 0 && splits.length === 0) return null;
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
      {labels.length > 0 && (
        <table className="w-full min-w-0 text-xs">
          <caption className="mb-1 text-left text-ink-2">{t("vision.import.byLabel")}</caption>
          <tbody>
            {labels.slice(0, SHOWN_LABELS).map(([label, count]) => (
              <tr key={label} className="border-t border-line">
                <td className="truncate py-1 pr-3 font-mono text-ink-1">{label}</td>
                <td className="py-1 text-right text-ink-2 tabular-nums">{formatCount(count)}</td>
              </tr>
            ))}
          </tbody>
          {labels.length > SHOWN_LABELS && (
            <tfoot>
              <tr className="border-t border-line text-ink-3">
                <td colSpan={2} className="py-1">
                  {t("vision.import.moreLabels", { count: formatCount(labels.length - SHOWN_LABELS) })}
                </td>
              </tr>
            </tfoot>
          )}
        </table>
      )}
      {splits.length > 0 && (
        <table className="w-full min-w-0 text-xs">
          <caption className="mb-1 text-left text-ink-2">{t("vision.import.bySplit")}</caption>
          <tbody>
            {splits.map(([split, count]) => (
              <tr key={split} className="border-t border-line">
                <td className="py-1 pr-3 font-mono text-ink-1">{split}</td>
                <td className="py-1 text-right text-ink-2 tabular-nums">{formatCount(count)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function Skipped({ data }: { data: ImagePreview }) {
  if (data.skipped_count === 0) return null;
  return (
    <details className="rounded-md border border-line px-3 py-2 text-xs text-ink-2">
      <summary className="min-h-6 cursor-pointer pointer-coarse:min-h-11">
        {t("vision.import.skipped", { count: formatCount(data.skipped_count) })}
      </summary>
      <ul className="mt-2 max-h-40 space-y-1 overflow-y-auto">
        {data.skipped.map((file) => (
          <li key={`${file.rel_path}-${file.code}`} className="min-w-0 break-words">
            <span className="font-mono text-ink-1">{file.rel_path}</span> · {file.reason} ({file.code})
          </li>
        ))}
      </ul>
    </details>
  );
}
