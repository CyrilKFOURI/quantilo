import { FolderOpen, Upload } from "lucide-react";
import { useRef, useState, type DragEvent } from "react";

import { useSamples, useUpload } from "../../api/hooks";
import { t } from "../../lib/i18n";
import { LITE } from "../../lite/flag";
import { LockBadge } from "../../lite/Locked";
import { Button } from "../../ui/Button";
import { ErrorNote } from "../../ui/fields";
import { BrowseDialog } from "./BrowseDialog";
import { IMAGE_ACCEPT } from "./ImageSetStep";
import { ImportDialog } from "./ImportDialog";

const ACCEPT = ".csv,.tsv,.parquet,.pq,.xlsx,.xlsm,.npy,.log"; // 090: text logs
// 106: images and ZIP archives of images (an image set)
const ACCEPT_ALL = `${ACCEPT},${IMAGE_ACCEPT}`;

/** Entry point of every import: drop zone, file picker, folder browser and samples. */
export function ImportPanel({
  compact = false,
  onImported,
  project = null,
}: {
  compact?: boolean;
  /** Called with the new dataset instead of opening it. */
  onImported?: (datasetId: string) => void;
  /** 087: the project open on the home page, offered in the import dialog (R-PRJ-05). */
  project?: { id: string; name: string } | null;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [browsing, setBrowsing] = useState(false);
  const [source, setSource] = useState<string | null>(null);
  const upload = useUpload();
  const samples = useSamples();

  const sendFile = (file: File | undefined) => {
    if (!file) return;
    upload.mutate(file, { onSuccess: (result) => setSource(result.path) });
  };

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    sendFile(event.dataTransfer.files[0]);
  };

  return (
    <section aria-label={t("import.drop")}>
      <div
        onDragOver={(event) => {
          // Files only: a dataset row dragged to a project (087) passes over without effect.
          if (!event.dataTransfer.types.includes("Files")) return;
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed text-center transition-colors ${
          compact ? "px-6 py-6" : "px-6 py-12"
        } ${dragging ? "border-accent bg-accent-soft" : "border-line bg-surface-1"}`}
      >
        <p className="text-sm font-medium text-ink-1">
          {upload.isPending
            ? t("import.uploading", { name: upload.variables.name })
            : dragging
              ? t("import.dropActive")
              : t("import.drop")}
        </p>
        <div className="flex flex-wrap items-center justify-center gap-2">
          <Button
            variant="primary"
            icon={Upload}
            disabled={upload.isPending}
            onClick={() => inputRef.current?.click()}
          >
            {t("import.chooseFile")}
          </Button>
          <span className="text-xs text-ink-3">{t("import.or")}</span>
          {LITE ? (
            // The browser cannot read folders on disk: the file picker is the way in.
            <span title={t("lite.locked")}>
              <Button icon={FolderOpen} disabled>
                {t("import.browse")}
                <LockBadge className="ml-1" />
              </Button>
            </span>
          ) : (
            <Button icon={FolderOpen} onClick={() => setBrowsing(true)}>
              {t("import.browse")}
            </Button>
          )}
        </div>
        <p className="text-xs text-ink-3">{t("import.local")}</p>
        <p className="text-xs text-ink-3">{t("import.clouds")}</p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_ALL}
          className="hidden"
          onChange={(event) => {
            sendFile(event.target.files?.[0]);
            event.target.value = "";
          }}
        />
      </div>

      {upload.isError && (
        <div className="mt-3">
          <ErrorNote>{upload.error.message}</ErrorNote>
        </div>
      )}

      {samples.data && samples.data.length > 0 && (
        <p className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-3">
          {t("samples.title")}
          {samples.data.map((sample) => (
            <button
              key={sample.path}
              type="button"
              onClick={() => setSource(sample.path)}
              className="font-mono text-ink-2 underline decoration-line underline-offset-4 hover:text-accent hover:decoration-accent"
            >
              {sample.name}
            </button>
          ))}
        </p>
      )}

      <BrowseDialog open={browsing} onOpenChange={setBrowsing} onSelect={setSource} />
      {source && (
        <ImportDialog
          key={source}
          path={source}
          onClose={() => setSource(null)}
          onImported={onImported}
          project={project}
        />
      )}
    </section>
  );
}
