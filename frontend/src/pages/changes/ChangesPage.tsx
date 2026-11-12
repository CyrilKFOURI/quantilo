import { Link, useNavigate, useParams, useSearch } from "@tanstack/react-router";
import { useMemo } from "react";

import type { Changes } from "../../api/client";
import { api, ApiError } from "../../api/client";
import { useChanges, useColumns, useDataset, useDatasets, useVersions } from "../../api/hooks";
import { ChangesSummary } from "../../features/changes/ChangesBanner";
import { useFamily, type FamilyVersion } from "../../features/lineage/api";
import { describeStep } from "../../features/lineage/steps";
import { t, type MessageKey } from "../../lib/i18n";
import { ErrorNote, Select } from "../../ui/fields";
import { Tabs } from "../../ui/Tabs";
import { TextWords, wordColumns } from "../../features/text/TextWords";
import { DistributionsTab } from "./DistributionsTab";
import { KpisTab, RelationsTab } from "./FollowTabs";
import { RowsTab } from "./RowsTab";
import { QualityTab, SchemaTab } from "./SchemaTabs";
import { Loading } from "../../ui/Loading";
import { importOf, pickedValue, relationOf, targetOf, versionName, versusGroups, type Target, type VersusOption } from "./versus";

export type ChangesTab = "schema" | "quality" | "distributions" | "rows" | "kpis" | "relations";
/** `vs`: another dataset; `version`: an earlier import of this one (095). */
export type ChangesSearch = { vs?: string; version?: number; tab?: ChangesTab; column?: string };

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
  const version = Number(search.version);
  return {
    vs: vs && ID.test(vs) ? vs : undefined,
    version: Number.isInteger(version) && version > 0 ? version : undefined,
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
  const lineage = useFamily(datasetId);
  const versions = useVersions(dataset.data);
  const datasets = useDatasets();
  const imports = versions.data ?? [];
  const others = datasets.data ?? [];
  // The default (R-095-01) is only known with the family and the imports.
  const known = lineage.ready && !versions.isPending;
  const vs: Target = targetOf(datasetId, lineage.family, imports, search.vs, search.version);
  const changes = useChanges(dataset.data, vs, known);
  const noPrevious = changes.error instanceof ApiError && changes.error.status === 404 && !vs;
  const relation = known ? relationOf(datasetId, vs, lineage.family, imports, others) : null;

  const update = (next: Partial<ChangesSearch>) =>
    void navigate({ search: (previous) => ({ ...previous, ...next }), replace: true });
  const types = useMemo(() => new Map((columns.data ?? []).map((c) => [c.name, c.type])), [columns.data]);
  const tab = search.tab ?? (changes.data ? firstTab(changes.data) : "distributions");

  return (
    <div className="flex flex-col gap-5 px-(--gutter) py-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="max-w-prose min-w-0">
          <h1 className="text-xl font-semibold tracking-tight">{t("changes.title")}</h1>
          <p className="mt-1 text-ink-2">{t("changes.description")}</p>
        </div>
        {dataset.data && known && (
          <VersusPicker
            groups={versusGroups(datasetId, lineage.family, imports, others)}
            value={pickedValue(vs, imports)}
            onChange={(next) => {
              const version = importOf(next);
              update({ vs: version === null ? next : undefined, version: version ?? undefined, column: undefined });
            }}
          />
        )}
      </header>

      {relation && !noPrevious && <Relation sentence={relation.sentence} version={relation.version} />}

      {noPrevious ? (
        <NoPrevious />
      ) : changes.isError ? (
        <ErrorNote>{changes.error.message}</ErrorNote>
      ) : !changes.data || !dataset.data || !known ? (
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
              <div className="flex flex-col gap-5">
                <DistributionsTab
                  changes={changes.data}
                  types={types}
                  column={search.column}
                  onColumn={(column) => update({ column })}
                />
                <TextWords
                  columns={wordColumns(columns.data)}
                  queryKey={["datasets", datasetId, "changes", vs, "words"]}
                  fetch={async (column) => {
                    const answer = await api.changeWords(datasetId, vs, column);
                    if (!answer.available) throw new Error(answer.reason ?? "");
                    return answer.words;
                  }}
                  hint={t("changes.wordsHint")}
                />
              </div>
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

const NONE = "__none";

/** R-095-04: a native list, the file's own versions first, then the other datasets. */
function VersusPicker({
  groups,
  value,
  onChange,
}: {
  groups: { family: VersusOption[]; others: VersusOption[] };
  value: string | null;
  onChange: (target: string) => void;
}) {
  return (
    <label className="flex min-w-0 items-center gap-2 text-[13px] text-ink-2">
      {t("changes.comparedWith")}
      <Select
        value={value ?? NONE}
        onChange={(event) => {
          if (event.target.value !== NONE) onChange(event.target.value);
        }}
        className="max-w-64 min-w-0"
      >
        <optgroup label={t("changes.versus.family")}>
          {value === null && <option value={NONE}>{t("changes.noPreviousOption")}</option>}
          {groups.family.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </optgroup>
        {groups.others.length > 0 && (
          <optgroup label={t("changes.versus.others")}>
            {groups.others.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </optgroup>
        )}
      </Select>
    </label>
  );
}

const LISTED_STEPS = 4;

/** R-095-02/03: both sides named, and the cleaning steps that explain the changes. */
function Relation({ sentence, version }: { sentence: string; version: FamilyVersion | null }) {
  const active = (version?.steps ?? []).filter((step) => step.enabled);
  return (
    <section aria-label={t("changes.comparedWith")} className="-mt-2 max-w-prose">
      <p className="font-medium text-ink-1">{sentence}</p>
      {version && active.length > 0 && (
        <div className="mt-1.5">
          <p className="text-xs text-ink-3">{t("changes.versus.steps", { version: versionName(version.seq) })}</p>
          <ol className="mt-0.5 list-decimal space-y-0.5 pl-5 text-xs text-ink-2">
            {active.slice(0, LISTED_STEPS).map((step) => (
              <li key={step.id}>{describeStep(step)}</li>
            ))}
          </ol>
          {active.length > LISTED_STEPS && (
            <p className="mt-0.5 pl-5 text-xs text-ink-3">{t("changes.moreReasons", { count: String(active.length - LISTED_STEPS) })}</p>
          )}
        </div>
      )}
    </section>
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
