import { Link, useNavigate, useParams, useSearch } from "@tanstack/react-router";
import { ArrowRight } from "lucide-react";
import { useMemo, useState } from "react";

import { useDataset } from "../../api/hooks";
import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { PHONE, useMedia } from "../../lib/useMedia";
import { Button } from "../../ui/Button";
import { ErrorNote } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { PageHeader } from "../../ui/PageHeader";
import { Tabs, type TabItem } from "../../ui/Tabs";
import {
  useApplyDecisions,
  useDecisions,
  useDiscardDecisions,
  useImageIndex,
  useImageReport,
  useQualityRules,
  useSetDecisions,
  type DecisionPatch,
  type DecisionsApplied,
  type QualityRule,
} from "./api";
import { Gallery } from "./Gallery";
import { ImageSheet } from "./ImageSheet";
import { filterAndSort, groupsOf, MEASURES, sortedValues, toRows, type Decision, type Draft, type ImageRow, type Measure } from "./model";
import { Quality } from "./Quality";
import { CLEARED, filtersOf, READY_TABS, TABS, type VisionSearch, type VisionTab } from "./search";

const EMPTY_DRAFT: Draft = {};
const NO_RULES: QualityRule[] = [];

/**
 * 107: an image set's page. Gallery and Quality (108) work; Map, Prepare, Model, Results and Explain
 * are shown, disabled, with what they will do (R-107-12). Everything shown was computed at the
 * import: nothing slow runs when the page opens.
 */
export function VisionPage() {
  const { datasetId } = useParams({ strict: false }) as { datasetId: string };
  const search = useSearch({ strict: false }) as VisionSearch;
  const navigate = useNavigate();
  const phone = useMedia(PHONE);
  const set = (patch: VisionSearch) =>
    void navigate({ to: ".", search: (previous: Record<string, unknown>) => ({ ...previous, ...patch }), replace: true });
  const openImage = (key: string | undefined) =>
    void navigate({ to: ".", search: (previous: Record<string, unknown>) => ({ ...previous, image: key }), replace: search.image !== undefined && key !== undefined });

  const dataset = useDataset(datasetId);
  const isImages = dataset.data?.format === "images";
  const ready = dataset.data?.status === "ready" && isImages;
  const index = useImageIndex(datasetId, ready);
  const report = useImageReport(datasetId, ready);
  const rulesQuery = useQualityRules();
  const decisions = useDecisions(datasetId, ready);
  const saveDraft = useSetDecisions(datasetId);
  const discard = useDiscardDecisions(datasetId);
  const apply = useApplyDecisions(datasetId);
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const [saved, setSaved] = useState<DecisionsApplied | null>(null);

  const rows = useMemo(() => (index.data ? toRows(index.data) : []), [index.data]);
  const draft = decisions.data?.draft ?? EMPTY_DRAFT;
  const filters = useMemo(() => filtersOf(search), [search]);
  const shown = useMemo(
    () => filterAndSort(rows, filters, draft, search.sort ?? "file", Boolean(search.desc)),
    [rows, filters, draft, search.sort, search.desc],
  );
  const sorted = useMemo(
    () => Object.fromEntries(MEASURES.map((measure) => [measure, sortedValues(rows, measure)])) as Record<Measure, number[]>,
    [rows],
  );
  const groups = useMemo(() => groupsOf(rows), [rows]);
  const groupKind = useMemo(() => {
    const kinds = new Map<number, "exact" | "near">();
    for (const group of report.data?.quality?.groups ?? []) {
      const first = rows[group.rows[0] ?? -1];
      if (first?.group !== null && first?.group !== undefined) kinds.set(first.group, group.kind);
    }
    return kinds;
  }, [report.data, rows]);
  const rules = rulesQuery.data ?? NO_RULES;
  const rulesByCode = useMemo(() => new Map(rules.map((rule) => [rule.code, rule])), [rules]);

  const patch = (decisionsPatch: DecisionPatch) => {
    setSaved(null);
    saveDraft.mutate(decisionsPatch);
  };
  const decide = (keys: string[], decision: Decision | null) => patch(Object.fromEntries(keys.map((key) => [key, decision])));

  const tab: VisionTab = search.tab ?? "gallery";
  const findings = report.data?.quality?.findings.length ?? 0;
  const tabs: TabItem<VisionTab>[] = TABS.filter((value) => !phone || READY_TABS.has(value)).map((value) => ({
    value,
    label: t(`vision.tab.${value}` as MessageKey),
    badge: value === "quality" && findings > 0 ? formatCount(findings) : READY_TABS.has(value) ? undefined : t("vision.tab.coming"),
    disabled: !READY_TABS.has(value),
    hint: READY_TABS.has(value) ? undefined : t(`vision.tab.${value}Hint` as MessageKey),
  }));

  const open: ImageRow | undefined = search.image ? rows.find((row) => row.key === search.image) : undefined;

  if (dataset.data && !isImages) {
    return (
      <div className="px-(--gutter) py-6">
        <PageHeader title={t("vision.title")} description={t("vision.notImages")} />
        <Link to="/d/$datasetId" params={{ datasetId }} className="mt-3 inline-flex items-center gap-1 text-[13px] text-accent hover:underline">
          {t("vision.toOverview")} <ArrowRight size={13} strokeWidth={1.5} aria-hidden />
        </Link>
      </div>
    );
  }

  const pending = decisions.data?.pending ?? 0;
  const version = decisions.data?.version ?? null;

  return (
    <div className="flex min-h-full flex-col">
      <div className="flex flex-1 flex-col gap-5 px-(--gutter) pt-6 pb-8">
        <PageHeader title={t("vision.title")} description={t("vision.description")} />
        <Tabs label={t("vision.tabs")} value={tab} items={tabs} onChange={(next) => set({ tab: next === "gallery" ? undefined : next })}>
          {!ready || index.isPending ? (
            index.isError ? (
              <ErrorNote>{(index.error as Error).message}</ErrorNote>
            ) : (
              <Loading label={dataset.data?.status === "importing" ? t("vision.importing") : t("vision.loading")} className="h-72 rounded-lg" />
            )
          ) : index.isError ? (
            <ErrorNote>{(index.error as Error).message}</ErrorNote>
          ) : tab === "quality" ? (
            report.data ? (
              <Quality
                datasetId={datasetId}
                rows={rows}
                report={report.data}
                rules={rules}
                draft={draft}
                onDecide={decide}
                onPatch={patch}
                onOpen={(row) => openImage(row.key)}
                onShow={(quality) => set({ ...CLEARED, tab: undefined, quality })}
              />
            ) : report.isError ? (
              <ErrorNote>{(report.error as Error).message}</ErrorNote>
            ) : (
              <Loading label={t("vision.loadingReport")} className="h-48 rounded-lg" />
            )
          ) : (
            <Gallery
              datasetId={datasetId}
              rows={rows}
              shown={shown}
              draft={draft}
              rules={rulesByCode}
              search={search}
              set={set}
              selected={selected}
              onSelectedChange={setSelected}
              onOpen={(row) => openImage(row.key)}
              onDecide={decide}
            />
          )}
        </Tabs>
      </div>

      {(pending > 0 || saved || apply.isError || saveDraft.isError) && (
        <DecisionBar
          datasetId={datasetId}
          pending={pending}
          nextVersion={(version ?? 1) + 1}
          saving={apply.isPending}
          saved={saved}
          error={(apply.error ?? saveDraft.error) as Error | null}
          onDiscard={() => discard.mutate()}
          onSave={() => apply.mutate(undefined, { onSuccess: (result) => setSaved(result) })}
        />
      )}

      {open && (
        <ImageSheet
          datasetId={datasetId}
          row={open}
          shown={shown}
          siblings={open.group !== null ? (groups.get(open.group) ?? []).filter((other) => other.key !== open.key) : []}
          groupKind={open.group !== null ? (groupKind.get(open.group) ?? null) : null}
          sorted={sorted}
          rules={rulesByCode}
          draft={draft}
          onNavigate={(key) => openImage(key)}
          onClose={() => openImage(undefined)}
          onDecide={decide}
        />
      )}
    </div>
  );
}

/** 108: the decisions not saved yet, and saving them as a new version of the set (R-108-06). */
function DecisionBar({
  datasetId,
  pending,
  nextVersion,
  saving,
  saved,
  error,
  onDiscard,
  onSave,
}: {
  datasetId: string;
  pending: number;
  nextVersion: number;
  saving: boolean;
  saved: DecisionsApplied | null;
  error: Error | null;
  onDiscard: () => void;
  onSave: () => void;
}) {
  return (
    // the right padding keeps the actions clear of the assistant's floating button
    <div className="sticky bottom-0 z-20 border-t border-line bg-surface-1 py-2.5 pr-[216px] pl-(--gutter) max-sm:pr-16 pb-[max(10px,env(safe-area-inset-bottom))] shadow-[0_-8px_24px_-16px_rgb(0_0_0/0.25)]" role="region" aria-label={t("vision.decisions.label")}>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        {error ? (
          <p role="alert" className="min-w-0 flex-1 text-[13px] text-bad">
            {error.message}
          </p>
        ) : pending > 0 ? (
          <p className="min-w-0 flex-1 text-[13px] text-ink-1" aria-live="polite">
            {t(pending === 1 ? "vision.decisions.pendingOne" : "vision.decisions.pending", { count: formatCount(pending) })}
            <span className="ml-2 text-ink-2 max-sm:block max-sm:ml-0">{t("vision.decisions.explain")}</span>
          </p>
        ) : (
          saved && (
            <p className="min-w-0 flex-1 text-[13px] text-ink-1" aria-live="polite">
              {t("vision.decisions.saved", { version: String(saved.version ?? nextVersion - 1), excluded: formatCount(saved.excluded) })}{" "}
              <Link to="/d/$datasetId/changes" params={{ datasetId }} className="text-accent hover:underline">
                {t("vision.decisions.changes")}
              </Link>
            </p>
          )
        )}
        {pending > 0 && (
          <span className="flex gap-2 max-sm:w-full">
            <Button variant="ghost" onClick={onDiscard} disabled={saving} className="max-sm:flex-1 max-sm:justify-center">
              {t("vision.decisions.discard")}
            </Button>
            <Button variant="primary" onClick={onSave} disabled={saving} className="max-sm:flex-1 max-sm:justify-center">
              {saving ? t("vision.decisions.saving") : t("vision.decisions.save", { version: String(nextVersion) })}
            </Button>
          </span>
        )}
      </div>
    </div>
  );
}
