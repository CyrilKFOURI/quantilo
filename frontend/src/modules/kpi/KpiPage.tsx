import { useNavigate, useParams, useSearch } from "@tanstack/react-router";
import { Activity, ChevronDown, Copy, FolderOpen, Pencil, Plus, Trash2 } from "lucide-react";
import { useState } from "react";

import { ApiError, type KpiDefinition, type KpiResult, type SavedKpi } from "../../api/client";
import { useColumns, useDataset, useDerivedColumns, useKpiActions, useKpiCompute, useKpis, useKpiSuggestions } from "../../api/hooks";
import { withDerived } from "../../features/derived/derived";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote } from "../../ui/fields";
import { Popover } from "../../ui/Popover";
import { Builder } from "./Builder";
import { Customize } from "./Customize";
import {
  computeDefinition,
  dataPart,
  formatOf,
  RATIO,
  sameData,
  viewOf,
  withTable,
  type Format,
} from "./definition";
import { KpiCharts } from "./KpiCharts";
import type { TableOptions } from "./pivot";
import { PivotTable } from "./PivotTable";
import { HierarchyChip, SuggestionChips } from "./SuggestionChips";
import { ReadingNotes } from "../../features/readings/ReadingNotes";
import type { KpiSearch } from "./search";
import { Tiles } from "./Tiles";
import { Loading } from "../../ui/Loading";
import { SignalLines } from "./SignalLines";
import { byScope } from "./signals";

/** Saved KPIs, in a menu of the header: open, rename, duplicate, delete, or start a new one. */
function SavedMenu({
  kpis,
  current,
  onOpen,
  onNew,
  onRename,
  onDuplicate,
  onDelete,
}: {
  kpis: SavedKpi[];
  current: SavedKpi | undefined;
  onOpen: (id: string) => void;
  onNew: () => void;
  onRename: (kpi: SavedKpi) => void;
  onDuplicate: (id: string) => void;
  onDelete: (kpi: SavedKpi) => void;
}) {
  const [open, setOpen] = useState(false);
  const run = (action: () => void) => () => {
    setOpen(false);
    action();
  };
  return (
    <div className="flex items-center gap-2">
      <Popover
        open={open}
        onOpenChange={setOpen}
        label={t("kpi.saved")}
        trigger={
          <Button icon={FolderOpen} onClick={() => setOpen(!open)} aria-expanded={open}>
            <span className="max-w-56 truncate">{current ? current.name : t("kpi.saved")}</span>
            <span className="rounded-full bg-surface-2 px-1.5 text-[11px] text-ink-2">{kpis.length}</span>
            <ChevronDown size={14} aria-hidden />
          </Button>
        }
      >
        <div className="max-h-80 overflow-auto p-1.5 text-[13px]">
          {kpis.length === 0 && <p className="px-2 py-3 text-xs text-ink-3">{t("kpi.savedEmpty")}</p>}
          <ul>
            {kpis.map((kpi) => (
              <li key={kpi.id} className="group flex items-center rounded-md hover:bg-surface-2">
                <button
                  type="button"
                  onClick={run(() => onOpen(kpi.id))}
                  aria-current={kpi.id === current?.id ? "page" : undefined}
                  className={`min-w-0 flex-1 truncate px-2 py-1.5 text-left ${kpi.id === current?.id ? "font-medium text-ink-1" : "text-ink-2 group-hover:text-ink-1"}`}
                  title={kpi.name}
                >
                  {kpi.name}
                </button>
                {[
                  { icon: Pencil, label: t("kpi.rename"), action: () => onRename(kpi) },
                  { icon: Copy, label: t("kpi.duplicate"), action: () => onDuplicate(kpi.id) },
                  { icon: Trash2, label: t("kpi.delete"), action: () => onDelete(kpi) },
                ].map(({ icon: Icon, label, action }) => (
                  <button key={label} type="button" aria-label={`${label} ${kpi.name}`} title={label} onClick={run(action)} className="inline-flex size-7 shrink-0 items-center justify-center rounded text-ink-3 hover:text-ink-1">
                    <Icon size={13} aria-hidden />
                  </button>
                ))}
              </li>
            ))}
          </ul>
        </div>
      </Popover>
      <Button icon={Plus} variant="ghost" onClick={onNew}>
        {t("kpi.new")}
      </Button>
    </div>
  );
}

export function KpiPage() {
  const { datasetId } = useParams({ strict: false }) as { datasetId: string };
  const search = useSearch({ strict: false }) as KpiSearch;
  const navigate = useNavigate();
  const dataset = useDataset(datasetId);
  const ready = dataset.data?.status === "ready";
  const columns = useColumns(datasetId, ready);
  // 058: measures and splits can read a derived column ("review › Length", "skills › has SQL")
  const derived = useDerivedColumns(datasetId, ready);
  const suggestions = useKpiSuggestions(datasetId, ready);
  const saved = useKpis(datasetId, ready);
  const actions = useKpiActions(datasetId);
  const savedKpi = saved.data?.find((k) => k.id === search.kpi);

  // What is computed and shown: the URL's definition, else the saved KPI, else the first suggestion.
  const applied: KpiDefinition | null = search.def ?? savedKpi?.definition ?? suggestions.data?.kpis[0]?.definition ?? null;
  const [edit, setEdit] = useState<{ base: string; draft: KpiDefinition } | null>(null);
  const appliedKey = applied ? dataPart(applied) : "";
  // a draft belongs to the definition it was started from: another KPI opened drops it
  const draft = edit && edit.base === appliedKey ? edit.draft : applied;
  const dirty = Boolean(draft && applied && !sameData(draft, applied));
  const [panel, setPanel] = useState(false);
  const compute = useKpiCompute(datasetId, applied ? computeDefinition(applied) : null);
  const result: KpiResult | undefined = compute.data;
  const error = compute.error instanceof ApiError ? compute.error : null;

  const go = (next: KpiSearch) =>
    void navigate({ to: ".", search: (previous: Record<string, unknown>) => ({ ...previous, kpi: next.kpi, def: next.def }), replace: false });
  const apply = (next: KpiDefinition) => {
    go({ kpi: search.kpi, def: next });
    setEdit(null);
  };
  /** A display change: drawn at once, the builder's pending data changes kept. */
  const applyView = (next: KpiDefinition) => {
    go({ kpi: search.kpi, def: next });
    if (dirty && draft) setEdit({ base: dataPart(next), draft: { ...draft, view: next.view, other: next.other, measures: draft.measures.map((m) => ({ ...m, label: next.measures.find((x) => x.id === m.id)?.label ?? m.label, format: next.measures.find((x) => x.id === m.id)?.format ?? m.format })) } });
  };
  const setDraft = (next: KpiDefinition) => applied && setEdit({ base: appliedKey, draft: next });

  // display names and formats: the user's, else the automatic ones
  const labels: Record<string, string> = {};
  for (const m of result?.measures ?? []) labels[m.id] = m.label;
  for (const m of applied?.measures ?? []) if (m.label) labels[m.id] = m.label;
  if (applied?.ratio?.label) labels[RATIO] = applied.ratio.label;
  const formats: Record<string, Format> = {};
  if (applied) for (const m of result?.measures ?? []) formats[m.id] = formatOf(applied, m.id);

  const view = applied ? viewOf(applied) : null;
  const tableOptions: TableOptions | null = (() => {
    if (!result || !view) return null;
    const layout = view.table.layout !== "auto" && result.resolved.layout.options.some((o) => o.value === view.table.layout) ? view.table.layout : result.resolved.layout.value;
    return {
      layout,
      showAs: view.show_as,
      totals: view.table.totals !== "off",
      shading: view.table.shading === "auto" ? result.resolved.shading.value === "on" : view.table.shading === "on",
      compare: result.resolved.time ? view.table.compare : [],
      sort: view.table.sort,
      hideOther: applied?.other === "hide",
      labels,
    };
  })();

  const save = async () => {
    if (!applied || !result) return;
    const definition = dirty && draft ? draft : applied;
    if (savedKpi) {
      await actions.update.mutateAsync({ kpiId: savedKpi.id, definition });
      go({ kpi: savedKpi.id, def: undefined });
    } else {
      const name = window.prompt(t("kpi.savePrompt"), result.name);
      if (!name?.trim()) return;
      const created = await actions.create.mutateAsync({ name: name.trim(), definition });
      go({ kpi: created.id, def: undefined });
    }
    setEdit(null);
  };
  const changed = Boolean(savedKpi && search.def && JSON.stringify(search.def) !== JSON.stringify(savedKpi.definition));
  const swapLayout = () => {
    if (!applied || !tableOptions) return;
    const next = tableOptions.layout === "periods_rows" ? "periods_columns" : "periods_rows";
    applyView(withTable(applied, { layout: next === result?.resolved.layout.value ? "auto" : next }));
  };
  const splitsTaken = new Set((applied?.groups ?? []).map((g) => g.column));
  const splitChips = (suggestions.data?.splits ?? []).filter((s) => !splitsTaken.has(s));
  // 048: "Region › State" only on a KPI not split yet (both splits at once, R-SUG-18)
  const hierarchyChips = splitsTaken.size === 0 ? (suggestions.data?.hierarchies ?? []) : [];

  return (
    <div className="flex flex-col">
      <header className="flex flex-wrap items-start justify-between gap-4 px-(--gutter) pt-6 pb-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">{t("kpi.title")}</h1>
          <p className="mt-1 max-w-prose text-ink-2">{t("kpi.description")}</p>
        </div>
        <SavedMenu
          kpis={saved.data ?? []}
          current={savedKpi}
          onOpen={(id) => {
            go({ kpi: id, def: undefined });
            setEdit(null);
          }}
          onNew={() => {
            go({ kpi: undefined, def: undefined });
            setEdit(null);
          }}
          onRename={(kpi) => {
            const name = window.prompt(t("kpi.renamePrompt"), kpi.name);
            if (name?.trim()) actions.update.mutate({ kpiId: kpi.id, name: name.trim() });
          }}
          onDuplicate={(id) => actions.duplicate.mutate(id, { onSuccess: (copy) => go({ kpi: copy.id, def: undefined }) })}
          onDelete={(kpi) => {
            if (!window.confirm(t("kpi.deleteConfirm", { name: kpi.name }))) return;
            actions.remove.mutate(kpi.id, { onSuccess: () => search.kpi === kpi.id && go({ kpi: undefined, def: undefined }) });
          }}
        />
      </header>
      <div className="px-(--gutter) pb-10">
        <div className="min-w-0 space-y-3">
          {suggestions.data && suggestions.data.kpis.length > 0 && (
            <SuggestionChips
              kpis={suggestions.data.kpis}
              isActive={(definition) => Boolean(applied) && !search.kpi && sameData(applied, definition)}
              onPick={(definition) => {
                go({ kpi: undefined, def: definition });
                setEdit(null);
              }}
            />
          )}

          {suggestions.data && (suggestions.data.notes ?? []).length > 0 && (
            <ReadingNotes datasetId={datasetId} notes={suggestions.data.notes ?? []} label={t("kpi.notes")} />
          )}

          {columns.data && draft && (
            <Builder
              definition={draft}
              columns={withDerived(columns.data, derived.data)}
              result={result}
              invalid={typeof error?.details.field === "string" ? error.details.field : null}
              dirty={dirty}
              saving={actions.create.isPending || actions.update.isPending}
              saved={Boolean(savedKpi)}
              changed={changed}
              onChange={setDraft}
              onCalculate={() => draft && apply(draft)}
              onSave={() => void save()}
            />
          )}

          {error ? (
            <div className="space-y-2">
              <ErrorNote>{error.message}</ErrorNote>
              {typeof error.details.suggestion === "string" && applied?.time && (
                <Button onClick={() => apply({ ...applied, time: { ...applied.time!, grain: error.details.suggestion as never } })}>
                  {t("kpi.useGrain", { grain: t(`kpi.grain.${error.details.suggestion}` as never) })}
                </Button>
              )}
              {error.details.code === "no_rows" && applied?.time && (
                <Button onClick={() => apply({ ...applied, time: { ...applied.time!, period: { kind: "all" } } })}>{t("kpi.useAllDates")}</Button>
              )}
            </div>
          ) : !result || !view || !tableOptions || !applied ? (
            <Loading label={t("loading.kpis")} className="h-[420px]" />
          ) : (
            <div className="space-y-6">
              {draft && (
                // 045: the display settings sit above what they change, opened from their own line.
                <Customize
                  definition={applied}
                  result={result}
                  draft={draft}
                  open={panel}
                  onToggle={() => setPanel(!panel)}
                  onView={applyView}
                  onData={setDraft}
                />
              )}
              <div className={`min-w-0 space-y-6 ${compute.isFetching ? "opacity-60 transition-opacity" : ""}`} aria-busy={compute.isFetching}>
                {result.warnings.map((w) => (
                  <p key={w} className="border-l-2 border-accent pl-3 text-xs text-ink-2">
                    {w}
                  </p>
                ))}
                <div className="space-y-2">
                  <Tiles result={result} labels={labels} formats={formats} />
                  {/* 054: what the figures rest on (empty measure, repeated rows, low n, thin periods) */}
                  <SignalLines signals={[...byScope(result, "measure"), ...byScope(result, "cells")]} label={t("kpi.signal.quality")} />
                </div>
                <section aria-labelledby="kpi-chart" className="space-y-2">
                  <h2 id="kpi-chart" className="truncate text-[15px] font-semibold text-ink-1" title={result.name}>
                    {result.name}
                  </h2>
                  <KpiCharts
                    result={result}
                    chart={view.chart}
                    share={view.share}
                    showAs={view.show_as}
                    labelsMode={view.labels}
                    hideOther={applied.other === "hide"}
                    line={view.line}
                    palette={view.palette}
                    spacing={view.spacing}
                    labels={labels}
                    formats={formats}
                  />
                  {/* 054: the rate and mix effects of a rate split by a group, Simpson's paradox */}
                  <SignalLines signals={byScope(result, "change")} label={t("kpi.signal.change")} />
                  <p className="text-xs text-ink-3">
                    <span className="font-medium text-ink-2">{t("kpi.why")} </span>
                    {result.resolved.why.join(" · ")}
                    {byScope(result, "series").length > 0 && (
                      <>
                        {" · "}
                        <Activity size={12} strokeWidth={1.75} aria-hidden className="inline align-[-2px] text-ink-2" />{" "}
                        <span className="font-medium text-ink-2">{t("kpi.signal.series")} </span>
                        {byScope(result, "series")
                          .map((s) => s.message)
                          .join("; ")}
                      </>
                    )}
                  </p>
                  {splitChips.length + hierarchyChips.length > 0 && (applied.groups ?? []).length < 2 && (
                    <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label={t("kpi.splitSuggestions")}>
                      <span className="text-xs text-ink-3">{t("kpi.splitSuggestions")}</span>
                      {splitChips.map((column) => (
                        <button
                          key={column}
                          type="button"
                          onClick={() => apply({ ...applied, groups: [...(applied.groups ?? []), { column }] })}
                          className="px-0.5 text-[13px] text-accent hover:underline"
                        >
                          + {column}
                        </button>
                      ))}
                      {hierarchyChips.map((h) => (
                        <HierarchyChip
                          key={`${h.parent}>${h.child}`}
                          hierarchy={h}
                          onPick={() => apply({ ...applied, groups: [{ column: h.parent }, { column: h.child }] })}
                        />
                      ))}
                    </div>
                  )}
                </section>
                <PivotTable
                  result={result}
                  options={tableOptions}
                  formats={formats}
                  canSwap={result.resolved.layout.options.length > 1}
                  onSwap={swapLayout}
                  onSort={(sort) => applyView(withTable(applied, { sort }))}
                  fileName={result.name}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
