import { useNavigate, useParams } from "@tanstack/react-router";
import { Check, Info, RefreshCw, Search, TriangleAlert, Undo2 } from "lucide-react";
import { useMemo, useState } from "react";

import type { AnalysisType, Column, Role } from "../../api/client";
import { useColumnSummaries, useColumns, useDataset, useDetectColumns, useReviewTypes, useRoles } from "../../api/hooks";
import { ANALYSIS_TYPES, TYPE_META } from "../../features/columns/columnTypes";
import { OrderDialog } from "../../features/columns/OrderDialog";
import { RoleMenu } from "../../features/columns/RoleMenu";
import { Sparkline } from "../../features/columns/Sparkline";
import { TypeBadge } from "../../features/columns/TypeBadge";
import { useColumnHistory } from "../../features/columns/useColumnHistory";
import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Checkbox, ErrorNote, Select, TextInput } from "../../ui/fields";
import { SegmentedControl } from "../../ui/SegmentedControl";

type View = "all" | "review" | "excluded";

const CONFIDENCE_RANK = { low: 0, medium: 1, high: 2 } as const;

const needsReview = (column: Column) => column.confidence !== "high" && !column.overridden;

function percentOf(part: number, whole: number): string {
  if (whole === 0 || part === 0) return "0%";
  const value = (part / whole) * 100;
  return value < 1 ? "<1%" : `${Math.round(value)}%`;
}

export function ColumnsPage() {
  const { datasetId } = useParams({ from: "/d/$datasetId/columns" });
  const navigate = useNavigate();
  const dataset = useDataset(datasetId);
  const ready = dataset.data?.status === "ready";
  const columns = useColumns(datasetId, ready);
  const roles = useRoles();
  const detect = useDetectColumns(datasetId);
  const review = useReviewTypes(datasetId);
  const summaries = useColumnSummaries(datasetId, ready);
  const all = useMemo(() => {
    const filtered = summaries.data;
    if (!filtered) return columns.data ?? [];
    return (columns.data ?? []).map((column): Column => {
      const summary = filtered[column.name];
      return summary ? { ...column, summary } : column;
    });
  }, [columns.data, summaries.data]);
  const filteredRows = summaries.data ? Object.values(summaries.data)[0]?.rows : undefined;
  const history = useColumnHistory(datasetId, all);

  const [query, setQuery] = useState("");
  const [view, setView] = useState<View>("all");
  const [typeFilter, setTypeFilter] = useState<AnalysisType | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [ordering, setOrdering] = useState<Column | null>(null);

  const typeCounts = useMemo(() => {
    const counts = new Map<AnalysisType, number>();
    for (const column of all) counts.set(column.type, (counts.get(column.type) ?? 0) + 1);
    return counts;
  }, [all]);
  const reviewCount = all.filter(needsReview).length;
  const excludedCount = all.filter((column) => column.excluded).length;

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return all
      .filter((column) => (view === "review" ? needsReview(column) : true))
      .filter((column) => (view === "excluded" ? column.excluded : true))
      .filter((column) => (typeFilter ? column.type === typeFilter : true))
      .filter(
        (column) =>
          !needle ||
          column.name.toLowerCase().includes(needle) ||
          (column.role_label ?? "").toLowerCase().includes(needle),
      )
      .sort(
        (a, b) =>
          CONFIDENCE_RANK[a.confidence] - CONFIDENCE_RANK[b.confidence] || a.position - b.position,
      );
  }, [all, query, view, typeFilter]);

  const toggleSelected = (name: string) =>
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  const allVisibleSelected = visible.length > 0 && visible.every((c) => selected.has(c.name));
  const bulk = (change: { type?: AnalysisType; excluded?: boolean }) => {
    history.apply([...selected].map((name) => ({ name, ...change })));
    setSelected(new Set());
  };

  const validate = () =>
    review.mutate(undefined, {
      onSuccess: () => void navigate({ to: "/d/$datasetId", params: { datasetId } }),
    });

  if (dataset.isPending || (ready && columns.isPending)) {
    return <p className="px-(--gutter) py-8 text-ink-3">{t("columns.loading")}</p>;
  }
  const error = dataset.error ?? columns.error ?? history.error ?? detect.error ?? review.error;

  return (
    <div className="flex flex-col gap-5 px-(--gutter) py-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">{t("columns.title")}</h1>
          <p className="mt-1 text-ink-2">
            {t("columns.summary", {
              columns: String(all.length),
              review: String(reviewCount),
            })}
          </p>
          {filteredRows !== undefined && (
            <p className="mt-0.5 text-xs text-accent">{t("filters.columnsNote", { rows: formatCount(filteredRows) })}</p>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            variant="ghost"
            icon={Undo2}
            disabled={!history.canUndo || history.pending}
            onClick={history.undo}
            title={t("columns.undoHint")}
          >
            {t("columns.undo")}
          </Button>
          <Button icon={RefreshCw} disabled={detect.isPending} onClick={() => detect.mutate()}>
            {detect.isPending ? t("columns.detecting") : t("columns.redetect")}
          </Button>
          <Button variant="primary" icon={Check} disabled={review.isPending} onClick={validate}>
            {t("columns.validate")}
          </Button>
        </div>
      </header>

      {dataset.data && !dataset.data.types_reviewed_at && (
        <p className="flex items-start gap-2 rounded-md border border-line bg-surface-1 px-3 py-2.5 text-ink-2">
          <Info size={15} strokeWidth={1.5} className="mt-0.5 shrink-0 text-accent" aria-hidden />
          {t("columns.intro")}
        </p>
      )}
      {error && <ErrorNote>{error.message}</ErrorNote>}

      <div className="flex flex-wrap items-center gap-3">
        <div className="relative">
          <Search
            size={14}
            strokeWidth={1.5}
            className="pointer-events-none absolute top-1/2 left-2 -translate-y-1/2 text-ink-3"
            aria-hidden
          />
          <TextInput
            aria-label={t("columns.search")}
            placeholder={t("columns.search")}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            className="w-56 pl-7"
          />
        </div>
        <SegmentedControl
          label={t("columns.view")}
          value={view}
          onChange={setView}
          options={[
            { value: "all", label: t("columns.view.all", { count: String(all.length) }) },
            { value: "review", label: t("columns.view.review", { count: String(reviewCount) }) },
            { value: "excluded", label: t("columns.view.excluded", { count: String(excludedCount) }) },
          ]}
        />
        <div className="flex flex-wrap items-center gap-1" role="group" aria-label={t("columns.typeFilter")}>
          {ANALYSIS_TYPES.filter((type) => typeCounts.has(type)).map((type) => {
            const meta = TYPE_META[type];
            const Icon = meta.icon;
            const active = typeFilter === type;
            return (
              <button
                key={type}
                type="button"
                aria-pressed={active}
                onClick={() => setTypeFilter(active ? null : type)}
                className={`inline-flex h-7 items-center gap-1.5 rounded-md border px-2 text-xs transition-colors ${
                  active ? "border-accent bg-accent-soft text-ink-1" : "border-line text-ink-2 hover:bg-surface-1"
                }`}
              >
                <Icon size={13} strokeWidth={1.5} className={meta.tone} aria-hidden />
                {t(meta.label)}
                <span className="text-ink-3 tabular-nums">{typeCounts.get(type)}</span>
              </button>
            );
          })}
        </div>
      </div>

      {selected.size > 0 && (
        <div className="flex flex-wrap items-center gap-3 rounded-md border border-accent/40 bg-accent-soft px-3 py-2">
          <span className="font-medium">{t("columns.selected", { count: String(selected.size) })}</span>
          <Select
            aria-label={t("columns.setType")}
            value=""
            onChange={(event) => event.target.value && bulk({ type: event.target.value as AnalysisType })}
          >
            <option value="">{t("columns.setType")}</option>
            {ANALYSIS_TYPES.map((type) => (
              <option key={type} value={type}>
                {t(TYPE_META[type].label)}
              </option>
            ))}
          </Select>
          <Button onClick={() => bulk({ excluded: true })}>{t("columns.exclude")}</Button>
          <Button onClick={() => bulk({ excluded: false })}>{t("columns.include")}</Button>
          <Button variant="ghost" onClick={() => setSelected(new Set())}>
            {t("columns.clearSelection")}
          </Button>
        </div>
      )}

      <div className="overflow-x-auto rounded-lg border border-line bg-surface-1">
        {/* 041: below a laptop, detection and values are hidden; a phone also hides role and distinct. */}
        <table className="w-full border-collapse text-[13px] max-lg:[&_tr>:nth-child(5)]:hidden max-lg:[&_tr>:nth-child(6)]:hidden max-sm:[&_tr>:nth-child(4)]:hidden max-sm:[&_tr>:nth-child(8)]:hidden">
          <thead>
            <tr className="border-b border-line text-left text-xs text-ink-3">
              <th scope="col" className="w-8 px-3 py-2">
                <Checkbox
                  aria-label={t("columns.selectAll")}
                  checked={allVisibleSelected}
                  onChange={() =>
                    setSelected(allVisibleSelected ? new Set() : new Set(visible.map((c) => c.name)))
                  }
                />
              </th>
              <th scope="col" className="px-3 py-2 font-normal">{t("columns.col.column")}</th>
              <th scope="col" className="px-3 py-2 font-normal">{t("columns.col.type")}</th>
              <th scope="col" className="px-3 py-2 font-normal">{t("columns.col.role")}</th>
              <th scope="col" className="px-3 py-2 font-normal">{t("columns.col.why")}</th>
              <th scope="col" className="px-3 py-2 font-normal">{t("columns.col.values")}</th>
              <th scope="col" className="px-3 py-2 text-right font-normal">{t("columns.col.missing")}</th>
              <th scope="col" className="px-3 py-2 text-right font-normal">{t("columns.col.distinct")}</th>
              <th scope="col" className="px-3 py-2 text-center font-normal">{t("columns.col.include")}</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((column) => (
              <ColumnRow
                key={column.name}
                column={column}
                roles={roles.data ?? []}
                selected={selected.has(column.name)}
                onSelect={() => toggleSelected(column.name)}
                onChange={(change) => history.apply([{ name: column.name, ...change }])}
                onEditOrder={() => setOrdering(column)}
              />
            ))}
            {visible.length === 0 && (
              <tr>
                <td colSpan={9} className="px-3 py-10 text-center text-ink-3">
                  {t("columns.empty")}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {ordering && (
        <OrderDialog
          column={ordering.name}
          values={ordering.order ?? ordering.summary?.histogram.map((bar) => bar.label) ?? []}
          onSave={(order) => history.apply([{ name: ordering.name, order }])}
          onClose={() => setOrdering(null)}
        />
      )}
    </div>
  );
}

type RowChange = {
  type?: AnalysisType;
  role?: string;
  excluded?: boolean;
  reset?: boolean;
};

function ColumnRow({
  column,
  roles,
  selected,
  onSelect,
  onChange,
  onEditOrder,
}: {
  column: Column;
  roles: Role[];
  selected: boolean;
  onSelect: () => void;
  onChange: (change: RowChange) => void;
  onEditOrder: () => void;
}) {
  const summary = column.summary;
  const rows = summary?.rows ?? 0;
  const invalid = summary?.invalid_count ?? 0;

  return (
    <tr
      className={`border-b border-line align-top last:border-b-0 ${selected ? "bg-accent-soft/50" : "hover:bg-surface-0"} ${
        column.excluded ? "text-ink-3" : ""
      }`}
    >
      <td className="px-3 py-2.5">
        <Checkbox aria-label={t("columns.select", { name: column.name })} checked={selected} onChange={onSelect} />
      </td>
      <td className="max-w-56 px-3 py-2.5">
        <span className={`block truncate font-mono text-xs font-medium ${column.excluded ? "line-through" : "text-ink-1"}`} title={column.name}>
          {column.name}
        </span>
        <span className="text-[11px] text-ink-3 lowercase">{column.storage_dtype}</span>
        {column.overridden && <span className="ml-1.5 text-[11px] text-accent">{t("columns.edited")}</span>}
      </td>
      <td className="px-3 py-2">
        <TypeBadge
          type={column.type}
          detectedType={column.detected_type}
          reasons={column.reasons}
          onChange={(type) => onChange({ type })}
          onReset={column.overridden ? () => onChange({ reset: true }) : undefined}
        />
        {invalid > 0 && (
          <p className="mt-0.5 flex items-center gap-1 text-xs text-warn">
            <TriangleAlert size={12} strokeWidth={1.5} aria-hidden />
            {t("columns.invalid", { count: formatCount(invalid), pct: percentOf(invalid, rows) })}
          </p>
        )}
        {column.type === "datetime" && summary?.date && (
          <p className="mt-0.5 max-w-64 text-xs text-ink-2 max-sm:hidden" title={summary.date.parts.map(datePartLabel).join(", ")}>
            {summary.date.min?.slice(0, 10)} → {summary.date.max?.slice(0, 10)} · {summary.date.step}
            <span className="block truncate text-ink-3">
              {t("columns.dateParts", { parts: summary.date.parts.map(datePartLabel).join(", ") })}
            </span>
          </p>
        )}
        {column.type === "categorical_ordinal" && (
          <p className="mt-0.5 flex max-w-64 items-baseline gap-1.5 text-xs text-ink-2 max-sm:max-w-[30vw]">
            <span className="truncate" title={(column.order ?? []).join(" < ")}>
              {(column.order ?? []).join(" < ") || t("columns.noOrder")}
            </span>
            <button type="button" className="shrink-0 text-accent hover:underline" onClick={onEditOrder}>
              {t("columns.editOrder")}
            </button>
          </p>
        )}
      </td>
      <td className="px-3 py-2">
        <RoleMenu
          label={t("columns.roleOf", { name: column.name })}
          value={column.role ?? ""}
          roles={roles}
          onChange={(role) => onChange({ role })}
        />
      </td>
      <td className="max-w-80 px-3 py-2.5">
        <ConfidenceTag confidence={column.confidence} />
        <p className="mt-0.5 line-clamp-2 text-xs text-ink-2" title={column.reasons.join(" · ")}>
          {column.reasons.join(" · ")}
        </p>
      </td>
      <td className="max-w-64 px-3 py-2.5">
        <div className="flex items-center gap-2.5">
          <Sparkline bars={summary?.histogram ?? []} tone={column.excluded ? "fill-line" : "fill-ink-3"} />
          <span className="truncate text-xs text-ink-2" title={summary?.examples.join(", ")}>
            {summary?.examples.join(", ")}
          </span>
        </div>
      </td>
      <td className="px-3 py-2.5 text-right text-ink-2 tabular-nums">
        {percentOf(summary?.null_count ?? 0, rows)}
      </td>
      <td className="px-3 py-2.5 text-right text-ink-2 tabular-nums">
        {formatCount(summary?.distinct_count)}
      </td>
      <td className="px-3 py-2.5 text-center">
        <Checkbox
          aria-label={t("columns.includeColumn", { name: column.name })}
          checked={!column.excluded}
          onChange={(event) => onChange({ excluded: !event.target.checked })}
        />
      </td>
    </tr>
  );
}

const DATE_PART_KEYS: Record<string, MessageKey> = {
  year: "part.year",
  quarter: "part.quarter",
  month: "part.month",
  year_month: "part.year_month",
  week: "part.week",
  day_of_week: "part.day_of_week",
  day_of_month: "part.day_of_month",
  hour: "part.hour",
};

function datePartLabel(part: string): string {
  return DATE_PART_KEYS[part] ? t(DATE_PART_KEYS[part]) : part;
}

function ConfidenceTag({ confidence }: { confidence: Column["confidence"] }) {
  const tone = { high: "text-ok", medium: "text-warn", low: "text-bad" }[confidence];
  return (
    <span className={`inline-flex items-center gap-1.5 text-xs ${tone}`}>
      <span aria-hidden className="size-1.5 rounded-full bg-current" />
      {t(`confidence.${confidence}`)}
    </span>
  );
}
