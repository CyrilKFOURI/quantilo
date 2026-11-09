import { useNavigate, useParams, useSearch } from "@tanstack/react-router";
import { Columns3, Search } from "lucide-react";
import { useMemo, useState } from "react";

import type { Column, RowsMode, RowsView } from "../../api/client";
import { useActiveFilters } from "../../api/filters";
import { useColumnSummaries, useColumns, useDataset, useRowBlocks } from "../../api/hooks";
import { DataTable, type TableColumn } from "../../features/data/DataTable";
import { RecordList } from "../../features/data/RecordList";
import { ExportMenu } from "../../features/data/ExportMenu";
import { useTableLayout } from "../../features/data/useTableLayout";
import { ConditionEditor } from "../../features/filters/ConditionEditor";
import { NoRows } from "../../features/filters/NoRows";
import { useFilters } from "../../features/filters/useFilters";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { BELOW_LAPTOP, PHONE, useMedia } from "../../lib/useMedia";
import { Dialog } from "../../ui/Dialog";
import { ErrorNote, TextInput } from "../../ui/fields";
import { Popover } from "../../ui/Popover";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { Loading } from "../../ui/Loading";

export type DataSearch = { mode?: RowsMode; q?: string; sort?: string; desc?: boolean };

const MODES: RowsMode[] = ["first", "sample", "duplicates", "missing"];
const text = (value: unknown) => (typeof value === "string" && value ? value : undefined);

export function validateDataSearch(search: Record<string, unknown>): DataSearch {
  const mode = MODES.find((item) => item === search.mode);
  return {
    mode: mode === "first" ? undefined : mode,
    q: text(search.q),
    sort: text(search.sort),
    desc: search.desc === true || search.desc === "true" ? true : undefined,
  };
}

export function DataPage() {
  const { datasetId } = useParams({ from: "/d/$datasetId/data" });
  const search = useSearch({ from: "/d/$datasetId/data" });
  const navigate = useNavigate({ from: "/d/$datasetId/data" });
  const dataset = useDataset(datasetId);
  const ready = dataset.data?.status === "ready";
  const columns = useColumns(datasetId, ready);
  const summaries = useColumnSummaries(datasetId, ready);
  const filters = useActiveFilters(datasetId);
  const { add } = useFilters(datasetId);
  const layout = useTableLayout(datasetId);
  // 041: below a laptop the rows are cards (nothing scrolls sideways): 4 fields on a phone, 6 on a tablet.
  const phone = useMedia(PHONE);
  const belowLaptop = useMedia(BELOW_LAPTOP);
  const [draft, setDraft] = useState<{ term: string; for: string | undefined }>({ term: search.q ?? "", for: search.q });
  const [filtering, setFiltering] = useState<Column | null>(null);

  const view: RowsView = useMemo(
    () => ({ mode: search.mode ?? "first", search: search.q ?? "", sort: search.sort ?? null, desc: Boolean(search.desc) }),
    [search.mode, search.q, search.sort, search.desc],
  );
  const [first] = useRowBlocks(datasetId, view, [0], ready);
  const total = first?.data?.total ?? 0;
  // The URL changed from elsewhere (back button, a link): the field follows it.
  const term = draft.for === search.q ? draft.term : (search.q ?? "");

  const update = (next: Partial<DataSearch>) =>
    void navigate({ search: (previous) => ({ ...previous, ...next }), replace: true });

  const tableColumns = useMemo<TableColumn[]>(() => {
    const sent = first?.data?.columns.map((c) => c.name) ?? [];
    const position = new Map(sent.map((name, index) => [name, index]));
    const shown = (columns.data ?? []).filter((c) => !layout.hidden.includes(c.name) && position.has(c.name));
    const pinned = layout.pinned.flatMap((name) => shown.filter((c) => c.name === name));
    return [...pinned, ...shown.filter((c) => !layout.pinned.includes(c.name))].map((column) => ({
      column,
      index: position.get(column.name) ?? 0,
      bars: (summaries.data?.[column.name] ?? column.summary)?.histogram ?? [],
    }));
  }, [columns.data, first?.data?.columns, layout.hidden, layout.pinned, summaries.data]);

  const submitSearch = (value: string) => {
    setDraft({ term: value, for: value.trim() || undefined });
    update({ q: value.trim() || undefined });
  };

  return (
    <div className="flex h-full flex-col gap-4 px-(--gutter) py-6">
      <header className="max-w-prose">
        <h1 className="text-xl font-semibold tracking-tight">{t("data.title")}</h1>
        <p className="mt-1 text-ink-2">{t("data.description")}</p>
      </header>

      <div className="flex flex-wrap items-center gap-3">
        <SegmentedControl
          label={t("data.mode")}
          value={view.mode}
          options={MODES.map((mode) => ({ value: mode, label: t(`data.mode.${mode}`) }))}
          onChange={(mode) => update({ mode: mode === "first" ? undefined : mode })}
        />
        <form
          role="search"
          className="relative"
          onSubmit={(event) => {
            event.preventDefault();
            submitSearch(term);
          }}
        >
          <Search size={14} strokeWidth={1.5} className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-ink-3" aria-hidden />
          <TextInput
            type="search"
            aria-label={t("data.search")}
            placeholder={t("data.searchPlaceholder")}
            value={term}
            onChange={(event) => {
              setDraft({ term: event.target.value, for: search.q });
              if (!event.target.value) submitSearch("");
            }}
            onBlur={() => term.trim() !== (search.q ?? "") && submitSearch(term)}
            className="w-64 pl-8"
          />
        </form>
        <ColumnChooser columns={columns.data ?? []} hidden={layout.hidden} onChange={layout.setHidden} />
        <span className="text-xs text-ink-3 tabular-nums" aria-live="polite">
          {first?.data &&
            (view.mode === "sample"
              ? t("data.sampled", { rows: formatCount(total) })
              : t("data.rows", { rows: formatCount(total) }))}
        </span>
        <div className="ml-auto">
          <ExportMenu
            datasetId={datasetId}
            view={view}
            filters={filters}
            total={total}
            visible={tableColumns.map((item) => item.column.name)}
            allColumns={columns.data?.length ?? 0}
          />
        </div>
      </div>

      {first?.isError && <ErrorNote>{first.error.message}</ErrorNote>}
      {!first?.data ? (
        <Loading label={t("loading.rows")} className="min-h-80 flex-1 rounded-lg" />
      ) : total === 0 ? (
        <Empty datasetId={datasetId} view={view} filtered={filters.length > 0} />
      ) : (
        <div className="min-h-90 flex-1">
          {belowLaptop ? (
            <RecordList
              datasetId={datasetId}
              fields={phone ? 4 : 6}
              columns={tableColumns}
              total={total}
              view={view}
              onSort={(column, sort) => update({ sort: sort ? column : undefined, desc: sort === "desc" ? true : undefined })}
            />
          ) : (
            <DataTable
              datasetId={datasetId}
              columns={tableColumns}
              pinned={layout.pinned}
              total={total}
              view={view}
              onSort={(column, sort) => update({ sort: sort ? column : undefined, desc: sort === "desc" ? true : undefined })}
              onFilter={setFiltering}
              onHide={layout.hide}
              onPin={layout.togglePin}
            />
          )}
        </div>
      )}

      <Dialog
        open={filtering !== null}
        onOpenChange={(open) => !open && setFiltering(null)}
        title={t("data.filterTitle", { name: filtering?.name ?? "" })}
        size="sm"
      >
        {filtering && (
          <ConditionEditor
            datasetId={datasetId}
            columns={[filtering]}
            others={columns.data ?? []}
            onSave={(condition) => {
              add(condition);
              setFiltering(null);
            }}
            onCancel={() => setFiltering(null)}
          />
        )}
      </Dialog>
    </div>
  );
}

function Empty({ datasetId, view, filtered }: { datasetId: string; view: RowsView; filtered: boolean }) {
  if (filtered && !view.search && view.mode === "first") return <NoRows datasetId={datasetId} />;
  const message = view.search
    ? t("data.noMatch", { term: view.search })
    : view.mode === "duplicates"
      ? t("data.noDuplicates")
      : view.mode === "missing"
        ? t("data.noMissing")
        : t("data.noRows");
  return <p className="rounded-lg border border-dashed border-line px-4 py-10 text-center text-ink-2">{message}</p>;
}

function ColumnChooser({ columns, hidden, onChange }: { columns: Column[]; hidden: string[]; onChange: (hidden: string[]) => void }) {
  const [open, setOpen] = useState(false);
  const toggle = (name: string) => onChange(hidden.includes(name) ? hidden.filter((item) => item !== name) : [...hidden, name]);
  return (
    <Popover
      open={open}
      onOpenChange={setOpen}
      label={t("data.columns")}
      align="start"
      trigger={
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          className="inline-flex h-(--control-h) items-center gap-1.5 rounded-md border border-line bg-surface-1 px-2.5 text-[13px] text-ink-2 hover:bg-surface-2 hover:text-ink-1"
        >
          <Columns3 size={14} strokeWidth={1.5} aria-hidden />
          {t("correlations.columnsCount", { count: String(columns.length - hidden.length), total: String(columns.length) })}
        </button>
      }
    >
      <div className="max-h-80 w-64 overflow-y-auto p-2">
        <div className="mb-1 flex gap-3 px-1 text-xs">
          <button type="button" className="text-ink-2 hover:text-ink-1" onClick={() => onChange([])}>
            {t("filters.all")}
          </button>
          <button type="button" className="text-ink-2 hover:text-ink-1" onClick={() => onChange(columns.map((c) => c.name))}>
            {t("filters.none")}
          </button>
        </div>
        {columns.map((column) => (
          <label key={column.name} className="flex cursor-pointer items-center gap-2 rounded px-1 py-1 text-[13px] hover:bg-surface-2">
            <input
              type="checkbox"
              className="size-3.5 accent-(--accent)"
              checked={!hidden.includes(column.name)}
              onChange={() => toggle(column.name)}
            />
            <span className="truncate font-mono text-xs">{column.name}</span>
          </label>
        ))}
      </div>
    </Popover>
  );
}
