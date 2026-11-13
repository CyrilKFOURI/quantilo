import { useNavigate } from "@tanstack/react-router";
import { useEffect, useState, type ReactNode } from "react";

import type { TableChoice } from "../../api/client";
import { useImportTables, useJob, useOriginal, usePreview, useSheetPreviews } from "../../api/hooks";
import { baseName, formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { PHONE, useMedia } from "../../lib/useMedia";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { Checkbox, ErrorNote, Field, ProgressBar, Select, TextInput } from "../../ui/fields";
import { InfoTip } from "../../ui/InfoTip";
import { Tabs } from "../../ui/Tabs";
import { ReadingNotes } from "../readings/ReadingNotes";
import { PreviewTable } from "./PreviewTable";
import { gridAreas, SheetGrid } from "./SheetGrid";
import { StackTables } from "./StackTables";
import { DOCUMENT_FORMATS, type ImportStepProps } from "./stepTypes";

const LEVEL_CLASS = { high: "text-ok", review: "text-warn", low: "text-ink-3" } as const;

/** A table the user marked by hand (U1): the detection runs on that range only. */
function manualChoice(sheet: string, range: string): TableChoice {
  const id = `${sheet}!${range}`;
  return {
    id,
    name: `${sheet} ${range}`,
    location: id,
    rows: 0,
    columns: 0,
    level: "review",
    confidence: 0,
    summary: t("preview.tables.manual"),
    selected: true,
    kind: "table",
    sheet,
    reasons: [],
    warnings: [],
    proposals: [],
  };
}

const RANGE = /^\$?[A-Za-z]{1,3}\$?\d+:\$?[A-Za-z]{1,3}\$?\d+$/;

/** 080's confidence levels: ticked from 0.7 (R-TBL-02), low under 0.4. */
export const LEVELS = { high: 0.7, review: 0.4 } as const;
export type Levels = { high: number; review: number };

/** b2 (Advanced › More): the level and the default tick of a table under the user's thresholds.
 *  A table marked by hand keeps its own (U1: what the user marked is never dropped). */
export function withLevels(table: TableChoice, levels: Levels, manual: boolean): TableChoice {
  if (manual || (levels.high === LEVELS.high && levels.review === LEVELS.review)) return table;
  const level = table.confidence >= levels.high ? "high" : table.confidence >= levels.review ? "review" : "low";
  return { ...table, level, selected: level === "high" };
}

/** b2 (T2u): "nil, none" -> ["nil", "none"]; null when empty (the API's default). */
export function tokenList(text: string): string[] | null {
  const tokens = [...new Set(text.split(",").map((token) => token.trim()).filter(Boolean))];
  return tokens.length > 0 ? tokens.slice(0, 50) : null;
}

function threshold(text: string, fallback: number): number {
  const value = Number(text);
  return text.trim() !== "" && Number.isFinite(value) && value >= 0 && value <= 1 ? value : fallback;
}

/**
 * 080b and 085: the tables of one file, the original on the left and the result on the right.
 * Everything is detected by written rules (080, 083); each table says its rules and its
 * confidence, nothing is saved before "Import selected".
 */
export function TablesStep({ path, preview, onClose, onImported, project, onSimple }: ImportStepProps) {
  const navigate = useNavigate();
  const phone = useMedia(PHONE);
  const document = DOCUMENT_FORMATS.includes(preview.format);
  const firstSheet = preview.tables?.[0]?.sheet ?? preview.sheets?.[0] ?? null;

  const [sheet, setSheet] = useState<string | null>(firstSheet);
  const [visited, setVisited] = useState<string[]>([]);
  const [ticks, setTicks] = useState<Record<string, boolean>>({});
  const [names, setNames] = useState<Record<string, string>>({});
  const [manual, setManual] = useState<TableChoice[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [reshape, setReshape] = useState(false);
  const [range, setRange] = useState("");
  const [pane, setPane] = useState<"original" | "result">("original");
  // b2: Advanced of a document (085) and Advanced › More of a workbook (080b)
  const [recordsPath, setRecordsPath] = useState("");
  const [oneTable, setOneTable] = useState(false);
  const [highText, setHighText] = useState(String(LEVELS.high));
  const [reviewText, setReviewText] = useState(String(LEVELS.review));
  const [emptyText, setEmptyText] = useState("");
  const [keepText, setKeepText] = useState("");
  const [intoProject, setIntoProject] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const [importError, setImportError] = useState<string | null>(null);

  const review = threshold(reviewText, LEVELS.review);
  const levels: Levels = { high: Math.max(threshold(highText, LEVELS.high), review), review };
  const records = recordsPath.trim() || null;
  const documentOptions = { records_path: records, one_table: oneTable || null };
  const reread = document && (records !== null || oneTable);
  // 085: another records path or "One table" reads the document again (its tables change)
  const documentPreview = usePreview(reread ? path : null, { ...documentOptions, sheets: null, folder_mode: "union" });
  const documentTables = reread ? documentPreview.data?.tables : preview.tables;
  const documentLinks = (reread ? documentPreview.data?.links : preview.links) ?? [];
  const tokens = { null_tokens: document ? null : tokenList(emptyText), text_tokens: document ? null : tokenList(keepText) };

  const others = useSheetPreviews(path, visited);
  const tablesOf = (name: string | null): TableChoice[] => {
    const found =
      document
        ? (documentTables ?? [])
        : name === firstSheet
          ? (preview.tables ?? [])
          : (others[visited.indexOf(name ?? "")]?.data?.tables ?? []);
    return [...found.map((table) => withLevels(table, levels, false)), ...manual.filter((table) => table.sheet === name)];
  };
  const known = document
    ? tablesOf(null)
    : [firstSheet, ...visited].flatMap((name) => tablesOf(name));
  const shownTables = document ? known : tablesOf(sheet);
  const sheetLoading = !document && sheet !== firstSheet && !others[visited.indexOf(sheet ?? "")]?.data;

  const isTicked = (table: TableChoice) => ticks[table.id] ?? table.selected;
  const chosen = known.filter(isTicked);
  const active = shownTables.find((table) => table.id === activeId) ?? shownTables[0] ?? null;

  const original = useOriginal(path, document ? null : sheet, document || Boolean(sheet));
  const result = usePreview(active ? path : null, {
    table: active?.id ?? null,
    reshape,
    sheets: null,
    folder_mode: "union",
    ...(document ? documentOptions : tokens),
  });
  const importTables = useImportTables();
  const job = useJob(jobId);

  const openDataset = (datasetId: string) => {
    onClose();
    if (onImported) return onImported(datasetId);
    void navigate({ to: "/d/$datasetId/columns", params: { datasetId } });
  };
  const finished = job.data?.status === "done" ? job.data.dataset_id : null;
  useEffect(() => {
    if (finished) openDataset(finished);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finished]);
  const jobFailed = job.data?.status === "failed";
  const running = importTables.isPending || (jobId !== null && !jobFailed);
  const error = importError ?? (jobFailed ? (job.data?.message ?? t("import.failed")) : null);

  const pickSheet = (name: string) => {
    setSheet(name);
    setActiveId(null);
    if (name !== firstSheet && !visited.includes(name)) setVisited([...visited, name]);
  };
  const addRange = () => {
    const clean = range.trim().toUpperCase().replaceAll("$", "");
    if (!sheet || !RANGE.test(clean)) return;
    const table = manualChoice(sheet, clean);
    if (!known.some((existing) => existing.id === table.id)) setManual([...manual, table]);
    setActiveId(table.id);
    setRange("");
  };

  /** `stacked` (S1, C1): these tables, which share their columns, as one dataset. */
  const start = (stacked: string[] | null = null) => {
    setImportError(null);
    importTables.mutate(
      {
        path,
        tables: stacked ? stacked.map((id) => ({ id, name: null })) : chosen.map((table) => ({ id: table.id, name: names[table.id] ?? table.name })),
        options: { reshape, folder_mode: "union", sample_pct: 100, ...(document ? documentOptions : tokens) },
        projectId: project && intoProject ? project.id : null,
        stack: stacked !== null,
      },
      {
        onSuccess: (response) => {
          const first = response.imports[0];
          if (!first) return;
          if (first.reused || first.job?.status === "done") openDataset(first.dataset.id);
          else if (first.job?.status === "failed") setImportError(first.dataset.error ?? t("import.failed"));
          else if (first.job) setJobId(first.job.id);
        },
        onError: (failure) => setImportError(failure.message),
      },
    );
  };

  const summary = t(document ? "preview.document.summary" : "preview.tables.summary", {
    count: String(known.length),
    name: baseName(path),
  });

  const originalPane = (
    <section className="min-w-0 space-y-2" aria-label={t("preview.tables.original")}>
      {!document && (preview.sheets?.length ?? 0) > 1 && (
        <Field label={t("preview.tables.sheet")}>
          <Select value={sheet ?? ""} onChange={(event) => pickSheet(event.target.value)} className="max-w-full">
            {preview.sheets?.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </Select>
        </Field>
      )}
      {original.isPending && <p className="py-6 text-center text-ink-3">{t("preview.tables.loading")}</p>}
      {original.isError && <ErrorNote>{original.error.message}</ErrorNote>}
      {original.data?.kind === "sheet" && (
        <>
          <SheetGrid
            cells={original.data.cells ?? []}
            areas={gridAreas(shownTables)}
            activeId={active?.id ?? null}
            onPick={setActiveId}
            label={t("preview.tables.original")}
          />
          <p className="text-xs text-ink-3">
            {t("preview.tables.legend")}
            {original.data.truncated && <> {t("preview.tables.truncated")}</>}
          </p>
        </>
      )}
      {original.data?.kind === "text" && (
        <pre className="max-h-[52dvh] overflow-auto rounded-md border border-line bg-surface-2 p-2 font-mono text-[11.5px] whitespace-pre-wrap break-all text-ink-2">
          {original.data.text}
        </pre>
      )}
    </section>
  );

  const resultPane = (
    <section className="min-w-0 space-y-3" aria-label={t("preview.tables.result")}>
      <fieldset disabled={running} className="min-w-0">
        <legend className="mb-1 text-xs text-ink-2">{t("preview.tables.list")}</legend>
        {sheetLoading && <p className="text-ink-3">{t("preview.tables.loading")}</p>}
        {!sheetLoading && shownTables.length === 0 && <p className="text-ink-3">{t("preview.tables.none")}</p>}
        <ul className="max-h-44 space-y-0.5 overflow-y-auto">
          {shownTables.map((table) => (
            <TableRow
              key={table.id}
              table={table}
              ticked={isTicked(table)}
              active={table.id === active?.id}
              onTick={(value) => setTicks({ ...ticks, [table.id]: value })}
              onShow={() => setActiveId(table.id)}
            />
          ))}
        </ul>
      </fieldset>

      {document && documentPreview.isError && <ErrorNote>{documentPreview.error.message}</ErrorNote>}
      {document && reread && documentPreview.isPending && <p className="text-ink-3">{t("preview.tables.loading")}</p>}
      {document && documentLinks.length > 0 && (
        <div className="text-xs text-ink-2">
          <span className="text-ink-3">{t("preview.tables.links")}: </span>
          {documentLinks
            .map((link) =>
              t("preview.tables.link", {
                parent: link.parent,
                child: link.child,
                parentKey: link.parent_key,
                childKey: link.child_key,
                rule: link.rule,
              }),
            )
            .join(" · ")}
        </div>
      )}

      {active ? (
        <div className="min-w-0 space-y-2">
          <div className="flex min-w-0 flex-wrap items-end gap-3">
            <Field label={t("preview.tables.name")}>
              <TextInput
                className="w-64 max-w-full"
                value={names[active.id] ?? active.name}
                onChange={(event) => setNames({ ...names, [active.id]: event.target.value })}
              />
            </Field>
          </div>
          <p className="text-ink-1">
            {active.summary}{" "}
            {result.data && (
              <span className="text-ink-3 tabular-nums">
                {t("preview.tables.size", {
                  rows: formatCount(result.data.row_estimate),
                  columns: String(result.data.columns.length),
                })}
              </span>
            )}
          </p>
          {(active.proposals ?? []).length > 0 && (
            <p className="text-xs text-ink-2">
              <span className="text-ink-3">{t("preview.tables.proposals")}: </span>
              {(active.proposals ?? []).join(" ")}
            </p>
          )}
          {result.isError && <ErrorNote>{result.error.message}</ErrorNote>}
          {result.isPending && <p className="py-4 text-center text-ink-3">{t("preview.loading")}</p>}
          {result.data && (
            <>
              <ReadingNotes
                notes={[...(active.warnings ?? []), ...(result.data.warnings ?? [])]}
                label={t("preview.readByRule")}
                hint={t("preview.checkReading")}
              />
              <PreviewTable columns={result.data.columns} rows={result.data.rows} />
            </>
          )}
        </div>
      ) : (
        <p className="text-ink-3">{t("preview.tables.pick")}</p>
      )}
    </section>
  );

  return (
    <Dialog
      open
      onOpenChange={(open) => !open && !running && onClose()}
      title={baseName(path)}
      description={summary}
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
            <Button variant="ghost" onClick={onSimple}>
              {t("preview.tables.simple")}
            </Button>
            <Button variant="ghost" onClick={onClose}>
              {t("import.cancel")}
            </Button>
            <Button variant="primary" disabled={chosen.length === 0} onClick={() => start()}>
              {t("preview.tables.import", { count: String(chosen.length) })}
            </Button>
          </>
        )
      }
    >
      <div className="space-y-4">
        {error && <ErrorNote>{error}</ErrorNote>}
        <ReadingNotes notes={detectionNotes(preview)} label={t("preview.detected.label")} />
        {project && (
          <Checkbox
            checked={intoProject}
            onChange={(event) => setIntoProject(event.target.checked)}
            label={t("projects.import.add", { name: project.name })}
          />
        )}

        {phone ? (
          <Tabs
            label={t("preview.tables.list")}
            value={pane}
            onChange={setPane}
            items={[
              { value: "original", label: t("preview.tables.original") },
              { value: "result", label: t("preview.tables.result") },
            ]}
          >
            {pane === "original" ? originalPane : resultPane}
          </Tabs>
        ) : (
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            {originalPane}
            {resultPane}
          </div>
        )}

        <details className="rounded-md border border-line px-3 py-2">
          <summary className="cursor-pointer text-xs text-ink-2">{t("preview.tables.advanced")}</summary>
          <div className="mt-3 flex flex-wrap items-end gap-x-6 gap-y-3">
            {!document && (
              <Field label={t("preview.tables.range")}>
                <div className="flex items-center gap-2">
                  <TextInput
                    className="w-32 font-mono"
                    placeholder="A4:L14"
                    value={range}
                    aria-describedby="range-hint"
                    onChange={(event) => setRange(event.target.value)}
                    onKeyDown={(event) => event.key === "Enter" && addRange()}
                  />
                  <Button onClick={addRange} disabled={!RANGE.test(range.trim())}>
                    {t("preview.tables.addRange")}
                  </Button>
                </div>
              </Field>
            )}
            {!document && (
              <div className="flex h-(--control-h) items-center">
                <Checkbox
                  checked={reshape}
                  onChange={(event) => setReshape(event.target.checked)}
                  label={t("preview.tables.reshape")}
                />
              </div>
            )}
            {document && (
              <Field label={t("preview.document.recordsPath")}>
                <TextInput
                  className="w-56 max-w-full font-mono"
                  placeholder="data.items"
                  value={recordsPath}
                  aria-describedby="document-hint"
                  onChange={(event) => setRecordsPath(event.target.value)}
                />
              </Field>
            )}
            {document && (
              <div className="flex h-(--control-h) items-center">
                <Checkbox checked={oneTable} onChange={(event) => setOneTable(event.target.checked)} label={t("preview.document.oneTable")} />
              </div>
            )}
          </div>
          {!document && (
            <p id="range-hint" className="mt-2 text-xs text-ink-3">
              {t("preview.tables.rangeHint")}
            </p>
          )}
          {document && (
            <p id="document-hint" className="mt-2 text-xs text-ink-3">
              {t("preview.document.advancedHint")}
            </p>
          )}
          {!document && (
            <details className="mt-3 border-t border-line pt-2">
              <summary className="cursor-pointer text-xs text-ink-2">{t("preview.tables.more")}</summary>
              <div className="params-grid mt-3 grid gap-x-6 gap-y-3 sm:grid-cols-2 lg:grid-cols-4">
                <Field label={t("preview.tables.highFrom")}>
                  <TextInput type="number" inputMode="decimal" min={0} max={1} step={0.05} className="w-24" value={highText} onChange={(event) => setHighText(event.target.value)} />
                </Field>
                <Field label={t("preview.tables.lowUnder")}>
                  <TextInput type="number" inputMode="decimal" min={0} max={1} step={0.05} className="w-24" value={reviewText} onChange={(event) => setReviewText(event.target.value)} />
                </Field>
                <Field label={t("preview.tables.emptyTokens")}>
                  <TextInput className="w-full font-mono" placeholder="nil, none, ?" value={emptyText} onChange={(event) => setEmptyText(event.target.value)} />
                </Field>
                <Field label={t("preview.tables.textTokens")}>
                  <TextInput className="w-full font-mono" placeholder="-, x" value={keepText} onChange={(event) => setKeepText(event.target.value)} />
                </Field>
              </div>
              <p className="mt-2 text-xs text-ink-3">{t("preview.tables.moreHint")}</p>
            </details>
          )}
          {!document && (preview.sheets?.length ?? 0) > 1 && <StackTables path={path} busy={running} onImport={(tables) => start(tables)} />}
        </details>
      </div>
    </Dialog>
  );
}

/** 082: the sentence when the name and the content of the file disagree (R-DET-01). */
function detectionNotes(preview: ImportStepProps["preview"]): string[] {
  return (preview.warnings ?? []).filter((note) => note.includes("R-DET-"));
}

function TableRow({
  table,
  ticked,
  active,
  onTick,
  onShow,
}: {
  table: TableChoice;
  ticked: boolean;
  active: boolean;
  onTick: (value: boolean) => void;
  onShow: () => void;
}) {
  const rules = (table.reasons ?? []).map((reason) => `${reason.rule}: ${reason.message}`).join("\n");
  const level: ReactNode = (
    <span className={`text-xs ${LEVEL_CLASS[table.level]}`}>{t(`preview.tables.level.${table.level}`)}</span>
  );
  return (
    <li
      className={`flex min-w-0 items-center gap-2 rounded px-1.5 py-1 pointer-coarse:min-h-11 ${active ? "bg-accent-soft" : "hover:bg-surface-2"}`}
    >
      <input
        type="checkbox"
        className="size-3.5 shrink-0 accent-(--accent)"
        checked={ticked}
        aria-label={table.name}
        onChange={(event) => onTick(event.target.checked)}
      />
      <button type="button" onClick={onShow} className="flex min-w-0 flex-1 items-baseline gap-2 text-left">
        <span className="truncate text-ink-1">{table.name}</span>
        <span className="shrink-0 font-mono text-xs text-ink-3">{table.location}</span>
      </button>
      {table.rows > 0 && <span className="shrink-0 text-xs text-ink-3 tabular-nums">{formatCount(table.rows)}</span>}
      {level}
      {rules && <InfoTip label={t("preview.tables.rules")} text={rules} />}
    </li>
  );
}
