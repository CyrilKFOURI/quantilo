import { Link, useNavigate, useParams, useSearch } from "@tanstack/react-router";
import { useMemo } from "react";

import type { Changes, Dataset } from "../../api/client";
import { ApiError } from "../../api/client";
import { useChanges, useColumns, useDataset, useDatasets, useVersions } from "../../api/hooks";
import { ChangesSummary } from "../../features/changes/ChangesBanner";
import { t, type MessageKey } from "../../lib/i18n";
import { ErrorNote, Select } from "../../ui/fields";
import { Tabs } from "../../ui/Tabs";
import { DistributionsTab } from "./DistributionsTab";
import { KpisTab, RelationsTab } from "./FollowTabs";
import { RowsTab } from "./RowsTab";
import { QualityTab, SchemaTab } from "./SchemaTabs";
import { Loading } from "../../ui/Loading";

export type ChangesTab = "schema" | "quality" | "distributions" | "rows" | "kpis" | "relations";
export type ChangesSearch = { vs?: string; tab?: ChangesTab; column?: string };

const TABS: ChangesTab[] = ["schema", "quality", "distributions", "rows", "kpis", "relations"];
const LABELS: Record<ChangesTab, MessageKey> = {
  schema: "changes.tab.schema",
  quality: "changes.tab.quality",
  distributions: "changes.tab.distributions",
  rows: "changes.tab.rows",
  kpis: "changes.tab.kpis",
  relations: "changes.tab.relations",
};
const ID = /^[\w-]{1,64}$/;
const text = (value: unknown) => (typeof value === "string" && value ? value : undefined);

export function validateChangesSearch(search: Record<string, unknown>): ChangesSearch {
  const vs = text(search.vs);
  return {
    vs: vs && ID.test(vs) ? vs : undefined,
    tab: TABS.find((tab) => tab === search.tab),
    column: text(search.column),
  };
}

/** Which tab opens first: the one that holds what matters most. */
function firstTab(changes: Changes): ChangesTab {
  if (changes.schema.some((change) => change.breaking)) return "schema";
  if (changes.counts.shifted > 0 || changes.counts.new_categories > 0) return "distributions";
  if (changes.schema.length > 0) return "schema";
  if (changes.quality.some((item) => item.notable) || changes.dates.some((item) => item.notable)) return "quality";
  return "distributions";
}

function badges(changes: Changes): Partial<Record<ChangesTab, string>> {
  const quality = changes.quality.filter((item) => item.notable).length + changes.dates.filter((item) => item.notable).length;
  return {
    schema: changes.schema.length ? String(changes.schema.length) : undefined,
    quality: quality ? String(quality) : undefined,
    distributions: changes.counts.shifted ? String(changes.counts.shifted) : undefined,
  };
}

export function ChangesPage() {
  const { datasetId } = useParams({ from: "/d/$datasetId/changes" });
  const search = useSearch({ from: "/d/$datasetId/changes" });
  const navigate = useNavigate({ from: "/d/$datasetId/changes" });
  const dataset = useDataset(datasetId);
  const ready = dataset.data?.status === "ready";
  const columns = useColumns(datasetId, ready);
  const vs = search.vs && search.vs !== datasetId ? search.vs : null;
  const versions = useVersions(dataset.data);
  const changes = useChanges(dataset.data, vs);
  const noPrevious = changes.error instanceof ApiError && changes.error.status === 404 && !vs;

  const update = (next: Partial<ChangesSearch>) =>
    void navigate({ search: (previous) => ({ ...previous, ...next }), replace: true });
  const types = useMemo(() => new Map((columns.data ?? []).map((c) => [c.name, c.type])), [columns.data]);
  const tab = search.tab ?? (changes.data ? firstTab(changes.data) : "distributions");

  return (
    <div className="flex flex-col gap-5 px-(--gutter) py-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="max-w-prose">
          <h1 className="text-xl font-semibold tracking-tight">{t("changes.title")}</h1>
          <p className="mt-1 text-ink-2">{t("changes.description")}</p>
        </div>
        {dataset.data && (
          <VersusPicker dataset={dataset.data} vs={vs} previous={!versions.data || versions.data.length > 1} onChange={(next) => update({ vs: next ?? undefined, column: undefined })} />
        )}
      </header>

      {noPrevious ? (
        <NoPrevious />
      ) : changes.isError ? (
        <ErrorNote>{changes.error.message}</ErrorNote>
      ) : !changes.data || !dataset.data ? (
        <Loading label={t("loading.changes")} className="h-24 rounded-lg" />
      ) : (
        <>
          <ChangesSummary changes={changes.data} />
          <Tabs
            label={t("changes.tabs")}
            value={tab}
            items={TABS.map((value) => ({ value, label: t(LABELS[value]), badge: badges(changes.data)[value] }))}
            onChange={(next) => update({ tab: next })}
          >
            {tab === "schema" && <SchemaTab changes={changes.data} />}
            {tab === "quality" && <QualityTab changes={changes.data} />}
            {tab === "distributions" && (
              <DistributionsTab
                changes={changes.data}
                types={types}
                column={search.column}
                onColumn={(column) => update({ column })}
              />
            )}
            {tab === "rows" && <RowsTab dataset={dataset.data} vs={vs} />}
            {tab === "kpis" && <KpisTab dataset={dataset.data} vs={vs} />}
            {tab === "relations" && <RelationsTab dataset={dataset.data} vs={vs} />}
          </Tabs>
        </>
      )}
    </div>
  );
}

const PREVIOUS = "__previous";

function VersusPicker({
  dataset,
  vs,
  previous,
  onChange,
}: {
  dataset: Dataset;
  vs: string | null;
  previous: boolean;
  onChange: (vs: string | null) => void;
}) {
  const datasets = useDatasets();
  const others = (datasets.data ?? []).filter((other) => other.id !== dataset.id && other.status === "ready");
  return (
    <label className="flex items-center gap-2 text-[13px] text-ink-2">
      {t("changes.comparedWith")}
      <Select value={vs ?? PREVIOUS} onChange={(event) => onChange(event.target.value === PREVIOUS ? null : event.target.value)} className="max-w-64">
        <option value={PREVIOUS}>{previous ? t("changes.previousVersion") : t("changes.noPreviousOption")}</option>
        {others.map((other) => (
          <option key={other.id} value={other.id}>
            {other.name}
          </option>
        ))}
      </Select>
    </label>
  );
}

function NoPrevious() {
  return (
    <div className="rounded-lg border border-dashed border-line px-6 py-10 text-center">
      <p className="font-medium text-ink-1">{t("changes.noPrevious")}</p>
      <p className="mx-auto mt-1 max-w-prose text-ink-2">{t("changes.noPreviousHint")}</p>
      <Link to="/" className="mt-3 inline-block text-[13px] text-accent hover:underline">
        {t("changes.goHome")}
      </Link>
    </div>
  );
}
