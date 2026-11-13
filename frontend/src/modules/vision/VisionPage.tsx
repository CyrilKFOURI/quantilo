import { Link, useNavigate, useParams, useSearch } from "@tanstack/react-router";
import { ArrowRight } from "lucide-react";
import { useMemo, useState } from "react";

import { useDataset } from "../../api/hooks";
import { LITE } from "../../lite/flag";
import { VisionBrowserNote } from "../../lite/VisionNote";
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
  type EditPatch,
  type QualityRule,
} from "./api";
import { Gallery } from "./Gallery";
import { Explain } from "./Explain";
import { WhyDialog } from "./ExplanationPanel";
import { ImageSheet } from "./ImageSheet";
import { MapTab, type MapPick } from "./Map";
import { Model } from "./ModelTab";
import { filterAndSort, groupsOf, labelChoices, MEASURES, sortedValues, toRows, withEdits, type Decision, type Draft, type Edits, type ImageRow, type Measure } from "./model";
import { Prepare } from "./Prepare";
import { Quality } from "./Quality";
import { Results } from "./Results";
import { useVisionJob } from "./modelApi";
import { CLEARED, filtersOf, READY_TABS, TABS, type VisionSearch, type VisionTab } from "./search";

const EMPTY_DRAFT: Draft = {};
const NO_RULES: QualityRule[] = [];

/**
 * 107: an image set's page. Gallery, Quality (108), Map (109), Prepare (110), Model and Results
 * (113) work; Explain is shown, disabled, with what it will do (R-107-12). Everything shown was
 * computed at the import (the map right after it): nothing slow runs when the page opens; a model
 * trains only when asked.
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
  // 109, R-VMAP-08: the images picked on the map (lasso, a group), shown in the gallery until cleared
  const [pick, setPick] = useState<MapPick | null>(null);
  // 113: the training job is followed here and kept in the URL, so it survives another tab, a
  // remount or a reload
  const jobId = search.job ?? null;
  const job = useVisionJob(jobId);
  const training = job.data?.status === "running";

  // c3: the labels and pins of the draft are shown everywhere at once (R-VED-01, R-VED-03)
  const labelsDraft = decisions.data?.labels_draft;
  const pinsDraft = decisions.data?.pins_draft;
  const edits = useMemo<Edits>(() => ({ labels: labelsDraft ?? {}, pins: pinsDraft ?? {} }), [labelsDraft, pinsDraft]);
  // the table holds the saved labels: the draft is read against the import's only once it is here
  const draftHere = decisions.data !== undefined;
  const rows = useMemo(() => {
    if (!index.data) return [];
    const read = toRows(index.data);
    return draftHere ? withEdits(read, edits) : read;
  }, [index.data, edits, draftHere]);
  const choices = useMemo(() => labelChoices(rows), [rows]);
  const draft = decisions.data?.draft ?? EMPTY_DRAFT;
  const filters = useMemo(() => filtersOf(search), [search]);
  const shown = useMemo(() => {
    const kept = filterAndSort(rows, filters, draft, search.sort ?? "file", Boolean(search.desc));
    if (!pick) return kept;
    const picked = new Set(pick.rows);
    return kept.filter((row) => picked.has(row.i));
  }, [rows, filters, draft, search.sort, search.desc, pick]);
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

  const edit = (editPatch: EditPatch) => {
    setSaved(null);
    saveDraft.mutate(editPatch);
  };
  const patch = (decisionsPatch: DecisionPatch) => edit({ decisions: decisionsPatch });
  const decide = (keys: string[], decision: Decision | null) => patch(Object.fromEntries(keys.map((key) => [key, decision])));

  const tab: VisionTab = search.tab ?? "gallery";
  const findings = report.data?.quality?.findings.length ?? 0;
  const tabs: TabItem<VisionTab>[] = TABS.filter((value) => !phone || READY_TABS.has(value)).map((value) => ({
    value,
    label: t(`vision.tab.${value}` as MessageKey),
    badge:
      value === "quality" && findings > 0
        ? formatCount(findings)
        : value === "model" && training
          ? t("vision.tab.training")
          : READY_TABS.has(value)
            ? undefined
            : t("vision.tab.coming"),
    disabled: !READY_TABS.has(value),
    hint: READY_TABS.has(value) ? undefined : t(`vision.tab.${value}Hint` as MessageKey),
  }));

  const open: ImageRow | undefined = search.image ? rows.find((row) => row.key === search.image) : undefined;
  // 117: "Why?" replaces the sheet (one dialog at a time); Results and Explain open it the same way
  const why: ImageRow | undefined = search.why ? rows.find((row) => row.key === search.why) : undefined;
  const openWhy = (key: string | undefined) =>
    void navigate({ to: ".", search: (previous: Record<string, unknown>) => ({ ...previous, why: key, image: undefined }), replace: search.why !== undefined && key !== undefined });

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
        {LITE && <VisionBrowserNote />}
        <Tabs label={t("vision.tabs")} value={tab} items={tabs} onChange={(next) => set({ tab: next === "gallery" ? undefined : next })}>
          {!ready || index.isPending ? (
            index.isError ? (
              <ErrorNote>{(index.error as Error).message}</ErrorNote>
            ) : (
              <Loading label={dataset.data?.status === "importing" ? t("vision.importing") : t("vision.loading")} className="h-72 rounded-lg" />
            )
          ) : index.isError ? (
            <ErrorNote>{(index.error as Error).message}</ErrorNote>
          ) : tab === "model" ? (
            <Model
              datasetId={datasetId}
              jobId={jobId}
              onJob={(id) => set({ job: id ?? undefined })}
              onOpenRun={(run) => set({ tab: "results", run, model: undefined, threshold: undefined, job: undefined })}
            />
          ) : tab === "results" ? (
            <Results
              datasetId={datasetId}
              rows={rows}
              runId={search.run}
              modelKey={search.model}
              threshold={search.threshold}
              set={set}
              onOpenImage={(row) => openImage(row.key)}
              onModelTab={() => set({ tab: "model" })}
              onWhy={(row) => openWhy(row.key)}
            />
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
                thresholds={decisions.data?.thresholds_draft ?? {}}
                onThresholds={(thresholds) => edit({ thresholds })}
              />
            ) : report.isError ? (
              <ErrorNote>{(report.error as Error).message}</ErrorNote>
            ) : (
              <Loading label={t("vision.loadingReport")} className="h-48 rounded-lg" />
            )
          ) : tab === "map" ? (
            <MapTab
              datasetId={datasetId}
              rows={rows}
              rules={rulesByCode}
              search={search}
              set={set}
              onOpen={(row) => openImage(row.key)}
              onShow={(next) => {
                setPick(next);
                // the images first: the set's statistics fold
                set({ ...CLEARED, tab: undefined, stats: false });
              }}
            />
          ) : tab === "prepare" ? (
            <Prepare datasetId={datasetId} rows={rows} search={search} set={set} />
          ) : tab === "explain" ? (
            <Explain
              datasetId={datasetId}
              rows={rows}
              run={search.run ?? null}
              model={search.model ?? null}
              onRun={(run) => set({ run, model: undefined })}
              onModel={(model) => set({ model })}
              onWhy={(row) => openWhy(row.key)}
            />
          ) : (
            <Gallery
              pick={pick ? { count: pick.rows.length, label: pick.label } : null}
              onClearPick={() => setPick(null)}
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
              labelChoices={choices}
              onEdit={edit}
            />
          )}
        </Tabs>
      </div>

      {(pending > 0 || saved || apply.isError || saveDraft.isError) && (
        <DecisionBar
          datasetId={datasetId}
          pending={pending}
          relabelled={(decisions.data?.pending_by?.labels ?? 0) + (decisions.data?.pending_by?.thresholds ?? 0)}
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
          onWhy={() => openWhy(open.key)}
        />
      )}

      {why && (
        <WhyDialog
          datasetId={datasetId}
          row={why}
          rows={rows}
          run={search.run ?? null}
          model={search.model ?? null}
          onRun={(run) => set({ run, model: undefined })}
          onModel={(model) => set({ model })}
          onWhy={(row) => openWhy(row.key)}
          onClose={() => openWhy(undefined)}
          onBack={() => void navigate({ to: ".", search: (previous: Record<string, unknown>) => ({ ...previous, why: undefined, image: why.key }), replace: true })}
        />
      )}
    </div>
  );
}

/** 108: the decisions not saved yet, and saving them as a new version of the set (R-108-06). */
function DecisionBar({
  datasetId,
  pending,
  relabelled,
  nextVersion,
  saving,
  saved,
  error,
  onDiscard,
  onSave,
}: {
  datasetId: string;
  pending: number;
  relabelled: number;
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
            {relabelled > 0 && <span className="ml-2 text-ink-2">{t("vision.edits.relabelledPending", { count: formatCount(relabelled) })}</span>}
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
