import { useNavigate } from "@tanstack/react-router";
import { AlertTriangle } from "lucide-react";
import { useEffect, useState } from "react";

import type { CsvOptions, Preview, SourceOptions } from "../../api/client";
import { useImportDataset, useJob, usePreview } from "../../api/hooks";
import { baseName, formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { Checkbox, ErrorNote, Field, ProgressBar, Select, TextInput } from "../../ui/fields";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { PreviewTable } from "./PreviewTable";

const DELIMITERS = [
  { value: ",", label: "Comma  ," },
  { value: ";", label: "Semicolon  ;" },
  { value: "\t", label: "Tab" },
  { value: "|", label: "Pipe  |" },
];

type ImportDialogProps = {
  path: string;
  onClose: () => void;
  /** Called with the new dataset instead of opening it (e.g. a recipe run on a new file). */
  onImported?: (datasetId: string) => void;
};

export function ImportDialog({ path, onClose, onImported }: ImportDialogProps) {
  const navigate = useNavigate();
  const [csv, setCsv] = useState<CsvOptions | null>(null);
  const [sheets, setSheets] = useState<string[] | null>(null);
  const [files, setFiles] = useState<string[] | null>(null);
  const [folderMode, setFolderMode] = useState<"union" | "intersection">("union");
  const [excluded, setExcluded] = useState<Set<string>>(new Set());
  const [sampling, setSampling] = useState(false);
  const [samplePct, setSamplePct] = useState(10);
  const [name, setName] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [importError, setImportError] = useState<string | null>(null);

  const sourceOptions: SourceOptions = { csv, sheets, files, folder_mode: folderMode };
  const preview = usePreview(path, sourceOptions);
  const importDataset = useImportDataset();
  const job = useJob(jobId);
  const data = preview.data;

  /** A dataset whose types were never reviewed opens on the Columns page. */
  const openDataset = (datasetId: string, reviewed = false) => {
    onClose();
    if (onImported) return onImported(datasetId);
    void navigate({ to: reviewed ? "/d/$datasetId" : "/d/$datasetId/columns", params: { datasetId } });
  };

  const jobFailed = job.data?.status === "failed";
  const finishedDatasetId = job.data?.status === "done" ? job.data.dataset_id : null;
  useEffect(() => {
    if (finishedDatasetId) openDataset(finishedDatasetId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finishedDatasetId]);

  const running = importDataset.isPending || (jobId !== null && !jobFailed);
  const error = importError ?? (jobFailed ? (job.data?.message ?? t("import.failed")) : null);
  const columns = data?.columns ?? [];
  const selectedColumns = columns.filter((column) => !excluded.has(column.name));

  const startImport = () => {
    setImportError(null);
    setJobId(null);
    importDataset.mutate(
      {
        path,
        name: name ?? null,
        options: {
          ...sourceOptions,
          columns: excluded.size > 0 ? selectedColumns.map((column) => column.name) : null,
          sample_pct: sampling ? samplePct : 100,
        },
      },
      {
        onSuccess: (response) => {
          if (response.reused || response.job?.status === "done") {
            openDataset(response.dataset.id, Boolean(response.dataset.types_reviewed_at));
          } else if (response.job?.status === "failed") {
            setImportError(response.dataset.error ?? t("import.failed"));
          } else if (response.job) {
            setJobId(response.job.id);
          }
        },
        onError: (error) => setImportError(error.message),
      },
    );
  };

  const toggleColumn = (column: string) =>
    setExcluded((current) => {
      const next = new Set(current);
      if (next.has(column)) next.delete(column);
      else next.add(column);
      return next;
    });

  return (
    <Dialog
      open
      onOpenChange={(open) => !open && !running && onClose()}
      title={baseName(path)}
      description={
        data
          ? t("preview.summary", {
              format: data.format.toUpperCase(),
              rows: formatCount(data.row_estimate),
              columns: String(columns.length),
            })
          : path
      }
      size="lg"
      footer={
        running ? (
          <div className="flex w-full items-center gap-3">
            <span className="shrink-0 text-ink-2">
              {t("import.running")} · {job.data?.message ?? "…"}
            </span>
            <ProgressBar value={job.data?.progress ?? 0.05} label={t("import.running")} />
          </div>
        ) : (
          <>
            <Button variant="ghost" onClick={onClose}>
              {t("import.cancel")}
            </Button>
            <Button
              variant="primary"
              disabled={!data || selectedColumns.length === 0}
              onClick={startImport}
            >
              {t("import.start")}
            </Button>
          </>
        )
      }
    >
      {preview.isPending && <p className="py-10 text-center text-ink-3">{t("preview.loading")}</p>}

      <div className="space-y-4">
        {error && <ErrorNote>{error}</ErrorNote>}
        {preview.isError && <ErrorNote>{preview.error.message}</ErrorNote>}

        {(data || csv) && (
          <fieldset disabled={running} className="flex flex-wrap items-end gap-x-6 gap-y-4">
            <Field label={t("preview.name")}>
              <TextInput
                className="w-56"
                value={name ?? data?.suggested_name ?? ""}
                onChange={(event) => setName(event.target.value)}
              />
            </Field>

            {(csv ?? data?.csv) && (
              <CsvSettings value={(csv ?? data?.csv) as CsvOptions} onChange={setCsv} />
            )}

            {data?.sheets && data.sheets.length > 1 && (
              <SheetPicker
                sheets={data.sheets}
                selected={sheets ?? [data.sheets[0] ?? ""]}
                onChange={setSheets}
              />
            )}

            <Field label={t("preview.rows")}>
              <div className="flex h-(--control-h) items-center gap-3">
                <Checkbox
                  label={t("preview.sample")}
                  checked={sampling}
                  onChange={(event) => setSampling(event.target.checked)}
                />
                <TextInput
                  type="number"
                  min={1}
                  max={99}
                  aria-label={t("preview.samplePctLabel")}
                  disabled={!sampling}
                  value={samplePct}
                  onChange={(event) => setSamplePct(Number(event.target.value) || 1)}
                  className="w-16 tabular-nums"
                />
                <span className="text-ink-2">%</span>
              </div>
            </Field>
          </fieldset>
        )}

        {data?.kind === "folder" && (
          <FolderFiles
            data={data}
            selected={files ?? data.files.map((file) => file.name)}
            onChange={setFiles}
            mode={folderMode}
            onModeChange={setFolderMode}
            disabled={running}
          />
        )}

        {data && (
          <section>
            <div className="mb-2 flex items-baseline justify-between gap-4">
              <h3 className="text-xs text-ink-2">
                {t("preview.firstRows", { count: String(data.rows.length) })}
                {preview.isFetching && <span className="ml-2 text-ink-3">· {t("preview.updating")}</span>}
              </h3>
              <div className="flex items-baseline gap-3 text-xs">
                <span className="text-ink-2 tabular-nums">
                  {t("preview.columnsCount", {
                    selected: String(selectedColumns.length),
                    total: String(columns.length),
                  })}
                </span>
                {excluded.size > 0 && (
                  <button
                    type="button"
                    className="text-accent hover:underline"
                    onClick={() => setExcluded(new Set())}
                  >
                    {t("preview.selectAll")}
                  </button>
                )}
              </div>
            </div>
            <PreviewTable
              columns={columns}
              rows={data.rows}
              excluded={excluded}
              onToggleColumn={running ? undefined : toggleColumn}
            />
          </section>
        )}
      </div>
    </Dialog>
  );
}

function CsvSettings({ value, onChange }: { value: CsvOptions; onChange: (v: CsvOptions) => void }) {
  return (
    <>
      <Field label={t("preview.delimiter")}>
        <Select
          value={value.delimiter}
          onChange={(event) => onChange({ ...value, delimiter: event.target.value })}
        >
          {DELIMITERS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </Select>
      </Field>
      <Field label={t("preview.encoding")}>
        <Select
          value={value.encoding}
          onChange={(event) =>
            onChange({ ...value, encoding: event.target.value as CsvOptions["encoding"] })
          }
        >
          <option value="utf-8">UTF-8</option>
          <option value="latin-1">Latin-1</option>
          <option value="windows-1252">Windows-1252</option>
          <option value="utf-16">UTF-16</option>
        </Select>
      </Field>
      <Field label={t("preview.decimal")}>
        <Select
          value={value.decimal}
          onChange={(event) =>
            onChange({ ...value, decimal: event.target.value as CsvOptions["decimal"] })
          }
        >
          <option value=".">Dot  1.5</option>
          <option value=",">Comma  1,5</option>
        </Select>
      </Field>
      <div className="flex h-(--control-h) items-center">
        <Checkbox
          label={t("preview.header")}
          checked={value.header}
          onChange={(event) => onChange({ ...value, header: event.target.checked })}
        />
      </div>
    </>
  );
}

function SheetPicker({
  sheets,
  selected,
  onChange,
}: {
  sheets: string[];
  selected: string[];
  onChange: (sheets: string[]) => void;
}) {
  const toggle = (sheet: string) => {
    const next = selected.includes(sheet)
      ? selected.filter((name) => name !== sheet)
      : sheets.filter((name) => name === sheet || selected.includes(name));
    if (next.length > 0) onChange(next);
  };
  return (
    <Field label={t("preview.sheets")}>
      <div className="flex h-(--control-h) flex-wrap items-center gap-3" title={t("preview.sheetsHint")}>
        {sheets.map((sheet) => (
          <Checkbox
            key={sheet}
            label={sheet}
            checked={selected.includes(sheet)}
            onChange={() => toggle(sheet)}
          />
        ))}
      </div>
    </Field>
  );
}

function FolderFiles({
  data,
  selected,
  onChange,
  mode,
  onModeChange,
  disabled,
}: {
  data: Preview;
  selected: string[];
  onChange: (files: string[]) => void;
  mode: "union" | "intersection";
  onModeChange: (mode: "union" | "intersection") => void;
  disabled: boolean;
}) {
  const toggle = (file: string) => {
    const next = selected.includes(file)
      ? selected.filter((name) => name !== file)
      : data.files.map((f) => f.name).filter((name) => name === file || selected.includes(name));
    if (next.length > 0) onChange(next);
  };
  return (
    <fieldset disabled={disabled} className="rounded-md border border-line p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
        <legend className="text-xs text-ink-2">
          {t("preview.files")} · {selected.length}/{data.files.length}
          <span className="ml-2 text-ink-3">{t("preview.filesHint")}</span>
        </legend>
        <SegmentedControl
          label={t("preview.files")}
          value={mode}
          onChange={onModeChange}
          options={[
            { value: "union", label: t("preview.union") },
            { value: "intersection", label: t("preview.intersection") },
          ]}
        />
      </div>
      {data.schema_mismatch && (
        <p className="mb-2 flex items-center gap-1.5 text-xs text-warn">
          <AlertTriangle size={13} strokeWidth={1.5} aria-hidden />
          {t("preview.mismatch")}
        </p>
      )}
      <ul className="grid max-h-32 grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-x-4 gap-y-1 overflow-y-auto">
        {data.files.map((file) => (
          <li key={file.name}>
            <Checkbox
              label={<span className="truncate font-mono text-xs">{file.name}</span>}
              checked={selected.includes(file.name)}
              onChange={() => toggle(file.name)}
            />
          </li>
        ))}
      </ul>
    </fieldset>
  );
}
