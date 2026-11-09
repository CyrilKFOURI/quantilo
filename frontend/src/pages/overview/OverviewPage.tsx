import { Link, useNavigate, useParams } from "@tanstack/react-router";
import { ChevronLeft, ChevronRight, Columns3, Info, Search, TriangleAlert } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { PAGE_SIZES, readDefaults, TOP_NS } from "../../lib/defaults";

import type { Alert, AnalysisType, Column, Overview } from "../../api/client";
import { useColumns, useDataset, useDeleteDataset, useOpenDataset, useOverview, useUpdateColumns } from "../../api/hooks";
import { ANALYSIS_TYPES, TYPE_META } from "../../features/columns/columnTypes";
import { ChangesBanner } from "../../features/changes/ChangesBanner";
import { NoRows } from "../../features/filters/NoRows";
import { RecipeBanner, SaveRecipeButton } from "../../features/recipes/SaveRecipe";
import { ReportButton } from "../../features/report/ReportButton";
import { SummaryStrip } from "../../features/summary-strip/SummaryStrip";
import { SyntheticOrigin } from "../../features/synthetic-origin/SyntheticOrigin";
import { VariableCard } from "../../features/variable-grid/VariableCard";
import { t } from "../../lib/i18n";
import { useStoredState } from "../../lib/useStoredState";
import { Button } from "../../ui/Button";
import { ErrorNote, Select, TextInput } from "../../ui/fields";
import { Loading } from "../../ui/Loading";

type Sort = "file" | "name" | "missing" | "distinct" | "skewness";


function sortColumns(columns: Column[], sort: Sort, overview: Overview | undefined): Column[] {
  const stats = overview?.column_stats ?? {};
  const by = {
    file: (c: Column) => c.position,
    name: (c: Column) => c.name.toLowerCase(),
    missing: (c: Column) => -(stats[c.name]?.missing_pct ?? 0),
    distinct: (c: Column) => -(stats[c.name]?.distinct ?? 0),
    skewness: (c: Column) => -Math.abs(stats[c.name]?.skewness ?? 0),
  }[sort];
  return [...columns].sort((a, b) => {
    const left = by(a);
    const right = by(b);
    return left < right ? -1 : left > right ? 1 : a.position - b.position;
  });
}

/** While a dataset imports, it can be cancelled to get the app back at once. */
function ImportingNote({ datasetId }: { datasetId: string }) {
  const remove = useDeleteDataset();
  const navigate = useNavigate();
  return (
    <div className="flex flex-col items-start gap-3">
      <p className="text-ink-2">{t("dataset.importing")}</p>
      <Button
        variant="ghost"
        disabled={remove.isPending}
        onClick={() => remove.mutate(datasetId, { onSuccess: () => navigate({ to: "/" }) })}
      >
        {t("datasets.cancelImport")}
      </Button>
      {remove.error && <ErrorNote>{remove.error.message}</ErrorNote>}
    </div>
  );
}

export function OverviewPage() {
  const { datasetId } = useParams({ from: "/d/$datasetId" });
  const dataset = useDataset(datasetId);
  const ready = dataset.data?.status === "ready";
  const columns = useColumns(datasetId, ready);
  const overview = useOverview(datasetId, ready);
  const update = useUpdateColumns(datasetId);
  const markOpened = useOpenDataset();

  const key = (name: string) => `de.overview.${datasetId}.${name}`;
  const [query, setQuery] = useState("");
  const [typeFilter, setTypeFilter] = useStoredState<AnalysisType | null>(key("type"), null);
  const [sort, setSort] = useStoredState<Sort>(key("sort"), "file");
  const [pageSize, setPageSize] = useStoredState(key("pageSize"), readDefaults().pageSize);
  const [topN, setTopN] = useStoredState(key("topN"), readDefaults().topN);
  const [page, setPage] = useStoredState(key("page"), 0);

  useEffect(() => {
    markOpened.mutate(datasetId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId]);

  const used = useMemo(() => (columns.data ?? []).filter((c) => !c.excluded), [columns.data]);
  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const filtered = used
      .filter((c) => (typeFilter ? c.type === typeFilter : true))
      .filter((c) => !needle || c.name.toLowerCase().includes(needle));
    return sortColumns(filtered, sort, overview.data);
  }, [used, query, typeFilter, sort, overview.data]);
  const pages = Math.max(1, Math.ceil(visible.length / pageSize));
  const current = Math.min(page, pages - 1);
  const shown = visible.slice(current * pageSize, (current + 1) * pageSize);
  const typeCounts = useMemo(() => {
    const counts = new Map<AnalysisType, number>();
    for (const c of used) counts.set(c.type, (counts.get(c.type) ?? 0) + 1);
    return counts;
  }, [used]);

  const focusColumn = (name: string) => {
    setQuery(name);
    setTypeFilter(null);
    setPage(0);
  };

  if (dataset.isPending)
    return (
      <div className="px-(--gutter) py-8">
        <Loading label={t("dataset.loading")} className="h-36 rounded-lg" />
      </div>
    );
  if (dataset.isError) {
    return (
      <div className="px-(--gutter) py-8">
        <ErrorNote>{dataset.error.message}</ErrorNote>
      </div>
    );
  }
  const data = dataset.data;
  if (data.status !== "ready") {
    return (
      <div className="px-(--gutter) py-8">
        {data.status === "error" ? <ErrorNote>{data.error ?? t("dataset.failed")}</ErrorNote> : <ImportingNote datasetId={datasetId} />}
      </div>
    );
  }
  const error = columns.error ?? overview.error ?? update.error;

  return (
    <div className="flex flex-col gap-5 px-(--gutter) py-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 max-w-full">
          <h1 className="text-xl font-semibold tracking-tight">{t("overview.title")}</h1>
          <p className="mt-0.5 truncate font-mono text-[11px] text-ink-3" title={data.source_path}>
            {data.source_path}
          </p>
          <SyntheticOrigin datasetId={datasetId} />
        </div>
        <div className="flex flex-wrap items-start gap-2">
          <ReportButton datasetId={datasetId} />
          <SaveRecipeButton datasetId={datasetId} datasetName={data.name} />
          <Link
            to="/d/$datasetId/columns"
            params={{ datasetId }}
            className="inline-flex h-(--control-h) items-center gap-1.5 rounded-md border border-line bg-surface-1 px-3 text-[13px] text-ink-1 hover:bg-surface-2 max-sm:whitespace-nowrap"
          >
            <Columns3 size={15} strokeWidth={1.5} aria-hidden />
            {t("overview.reviewTypes")}
          </Link>
        </div>
      </header>

      {!data.types_reviewed_at && (
        <p className="flex items-start gap-2 rounded-md border border-line bg-surface-1 px-3 py-2.5 text-ink-2">
          <Info size={15} strokeWidth={1.5} className="mt-0.5 shrink-0 text-accent" aria-hidden />
          {t("overview.notReviewed")}
        </p>
      )}
      <RecipeBanner datasetId={datasetId} />
      <ChangesBanner dataset={data} />
      {error && <ErrorNote>{error.message}</ErrorNote>}

      {overview.data ? (
        <SummaryStrip overview={overview.data} />
      ) : (
        <Loading label={t("loading.summary")} className="h-36 rounded-lg" />
      )}
      {overview.data && overview.data.rows === 0 && overview.data.total_rows > 0 && <NoRows datasetId={datasetId} />}
      {overview.data && overview.data.alerts.length > 0 && (
        <Alerts alerts={overview.data.alerts} onColumn={focusColumn} />
      )}

      <div className="flex flex-wrap items-center gap-3">
        <div className="relative">
          <Search size={14} strokeWidth={1.5} className="pointer-events-none absolute top-1/2 left-2 -translate-y-1/2 text-ink-3" aria-hidden />
          <TextInput
            aria-label={t("overview.search")}
            placeholder={t("overview.search")}
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setPage(0);
            }}
            className="w-52 pl-7"
          />
        </div>
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
                onClick={() => {
                  setTypeFilter(active ? null : type);
                  setPage(0);
                }}
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
        <div className="ml-auto flex flex-wrap items-center gap-3 text-xs text-ink-2">
          <label className="flex items-center gap-1.5">
            {t("overview.sort")}
            <Select value={sort} onChange={(event) => setSort(event.target.value as Sort)}>
              <option value="file">{t("overview.sort.file")}</option>
              <option value="name">{t("overview.sort.name")}</option>
              <option value="missing">{t("overview.sort.missing")}</option>
              <option value="distinct">{t("overview.sort.distinct")}</option>
              <option value="skewness">{t("overview.sort.skewness")}</option>
            </Select>
          </label>
          <label className="flex items-center gap-1.5" title={t("overview.topNHint")}>
            {t("overview.topN")}
            <Select value={topN} onChange={(event) => setTopN(Number(event.target.value))}>
              {TOP_NS.map((value) => (
                <option key={value} value={value}>{value}</option>
              ))}
            </Select>
          </label>
          <label className="flex items-center gap-1.5">
            {t("overview.perPage")}
            <Select
              value={pageSize}
              onChange={(event) => {
                setPageSize(Number(event.target.value));
                setPage(0);
              }}
            >
              {PAGE_SIZES.map((value) => (
                <option key={value} value={value}>{value}</option>
              ))}
            </Select>
          </label>
        </div>
      </div>

      {columns.isPending ? (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-1 sm:gap-4 md:grid-cols-2 xl:grid-cols-3">
          <Loading label={t("loading.columns")} className="h-72 rounded-lg" compact />
          {Array.from({ length: 5 }, (_, index) => (
            <div key={index} aria-hidden className="loading-sheen relative h-72 overflow-hidden rounded-lg border border-line bg-surface-1" />
          ))}
        </div>
      ) : (
        <>
          <Pagination page={current} pages={pages} total={visible.length} pageSize={pageSize} onPage={setPage} />
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-1 sm:gap-4 md:grid-cols-2 xl:grid-cols-3">
            {shown.map((column) => (
              <VariableCard
                key={`${column.name}:${column.type}`}
                datasetId={datasetId}
                column={column}
                topN={topN}
                onChangeType={(type) => update.mutate([{ name: column.name, type }])}
              />
            ))}
          </div>
          {shown.length === 0 && <p className="py-10 text-center text-ink-3">{t("overview.empty")}</p>}
          {pages > 1 && <Pagination page={current} pages={pages} total={visible.length} pageSize={pageSize} onPage={setPage} />}
        </>
      )}
    </div>
  );
}

function Alerts({ alerts, onColumn }: { alerts: Alert[]; onColumn: (name: string) => void }) {
  const [open, setOpen] = useState(true);
  const warnings = alerts.filter((alert) => alert.severity === "warning").length;
  return (
    <section className="rounded-lg border border-line bg-surface-1">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 px-4 py-2.5 text-left"
      >
        <TriangleAlert size={15} strokeWidth={1.5} className={warnings ? "text-warn" : "text-ink-3"} aria-hidden />
        <span className="font-medium">{t("overview.alerts", { count: String(alerts.length) })}</span>
        <span className="ml-auto text-xs text-ink-3">{open ? t("overview.hide") : t("overview.show")}</span>
      </button>
      {open && (
        <ul className="divide-y divide-line border-t border-line">
          {alerts.map((alert) => (
            <li key={`${alert.kind}:${alert.message}`} className="flex flex-wrap items-baseline gap-x-3 gap-y-1 px-4 py-2">
              <span className={`text-xs ${alert.severity === "warning" ? "text-warn" : "text-ink-3"}`}>
                {alert.severity === "warning" ? t("overview.warning") : t("overview.info")}
              </span>
              <span className="text-ink-1">{alert.message}</span>
              <span className="flex flex-wrap gap-1">
                {alert.columns.slice(0, 8).map((name) => (
                  <button
                    key={name}
                    type="button"
                    onClick={() => onColumn(name)}
                    className="rounded border border-line px-1.5 font-mono text-[11px] text-ink-2 hover:border-accent hover:text-ink-1"
                  >
                    {name}
                  </button>
                ))}
                {alert.columns.length > 8 && <span className="text-xs text-ink-3">+{alert.columns.length - 8}</span>}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function Pagination({
  page,
  pages,
  total,
  pageSize,
  onPage,
}: {
  page: number;
  pages: number;
  total: number;
  pageSize: number;
  onPage: (page: number) => void;
}) {
  const first = total === 0 ? 0 : page * pageSize + 1;
  const last = Math.min(total, (page + 1) * pageSize);
  return (
    <nav aria-label={t("overview.pagination")} className="flex items-center justify-between text-xs text-ink-2">
      <span className="tabular-nums">{t("overview.range", { first: String(first), last: String(last), total: String(total) })}</span>
      <div className="flex items-center gap-1">
        <Button variant="ghost" icon={ChevronLeft} iconOnly aria-label={t("overview.previous")} disabled={page === 0} onClick={() => onPage(page - 1)} />
        <span className="tabular-nums">{t("overview.page", { page: String(page + 1), pages: String(pages) })}</span>
        <Button variant="ghost" icon={ChevronRight} iconOnly aria-label={t("overview.next")} disabled={page >= pages - 1} onClick={() => onPage(page + 1)} />
      </div>
    </nav>
  );
}
