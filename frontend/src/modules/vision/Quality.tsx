import { Ban, Check, ChevronDown, Images, RotateCcw, ShieldAlert, TriangleAlert } from "lucide-react";
import { useMemo, useState } from "react";

import { imageUrl } from "../../features/import/imageSetApi";
import { formatBytes, formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import type { DecisionPatch, ImageReport, QualityFinding, QualityGroup, QualityRule } from "./api";
import { exclusionCause, isExcluded, type Decision, type Draft, type ImageRow, type QualityFilter } from "./model";
import { QualityAdvanced } from "./QualityAdvanced";
import { QualityThresholds } from "./QualityThresholds";
import { Thumbs } from "./QualityThumbs";

const GROUPS_STEP = 8;
const STRIP = 5;

type QualityProps = {
  datasetId: string;
  rows: readonly ImageRow[];
  report: ImageReport;
  rules: readonly QualityRule[];
  draft: Draft;
  onDecide: (keys: string[], decision: Decision | null) => void;
  /** Several decisions at once, in one request (keep one copy, exclude the others). */
  onPatch: (patch: DecisionPatch) => void;
  onOpen: (row: ImageRow) => void;
  onShow: (quality: QualityFilter) => void;
  /** c3, R-VQT-02: the draft's thresholds (those that differ from the defaults). */
  thresholds?: Readonly<Record<string, number>>;
  onThresholds?: (thresholds: Record<string, number>) => void;
};

const EFFECT: Record<QualityFinding["effect"], MessageKey> = {
  excluded: "vision.effect.excluded",
  flagged: "vision.effect.flagged",
  dataset: "vision.effect.dataset",
};

/**
 * 108: what the R-VQ rules found, one section per rule with its sentence, reason, effect and
 * thresholds (R-108-01, 02); the copies side by side (R-108-08); what was checked and found nothing
 * (R-108-03). Every decision goes into the draft, saved as a version from the decision bar.
 */
export function Quality({ datasetId, rows, report, rules, draft, onDecide, onPatch, onOpen, onShow, thresholds = {}, onThresholds }: QualityProps) {
  const quality = report.quality;
  const fired = useMemo(() => new Set(quality?.findings.map((finding) => finding.code) ?? []), [quality]);
  const excludedNow = rows.filter((row) => isExcluded(row, draft)).length;

  if (!quality) {
    return <p className="text-[13px] text-ink-2">{t("vision.quality.notRun")}</p>;
  }
  const findings = [...quality.findings].sort((a, b) => a.code.localeCompare(b.code));
  const leaks = findings.find((finding) => finding.code === "R-VQ-11");
  const sections = findings.filter((finding) => finding.code !== "R-VQ-11");
  const quiet = rules.filter((rule) => !fired.has(rule.code));

  return (
    <div className="max-w-5xl space-y-6">
      <header className="space-y-1.5">
        <p className="text-[15px] leading-snug font-medium text-pretty text-ink-1">{quality.sentence}</p>
        <p className="text-[13px] text-ink-2 tabular-nums">
          {t("vision.quality.totals", {
            flagged: formatCount(quality.flagged),
            excluded: formatCount(excludedNow),
          })}
          {quality.recommended_metric === "balanced_accuracy" && <> {t("vision.quality.metric")}</>}
        </p>
      </header>

      {leaks && (
        <div role="alert" className="flex gap-3 rounded-md border border-bad/40 bg-bad/8 px-3 py-2.5">
          <ShieldAlert size={18} strokeWidth={1.5} className="mt-0.5 shrink-0 text-bad" aria-hidden />
          <div className="min-w-0 space-y-1 text-[13px]">
            <p className="font-medium text-ink-1">
              {leaks.message} <span className="font-mono text-xs text-ink-2">{leaks.code}</span>
            </p>
            <p className="text-ink-2">{leaks.reason}</p>
            <Thumbs datasetId={datasetId} rows={rows} positions={leaks.rows} draft={draft} onOpen={onOpen} />
          </div>
        </div>
      )}

      {sections.map((finding) => (
        <FindingSection
          key={finding.code}
          datasetId={datasetId}
          finding={finding}
          rows={rows}
          draft={draft}
          onDecide={onDecide}
          onOpen={onOpen}
          onShow={onShow}
        />
      ))}

      {quality.groups.length > 0 && (
        <Duplicates datasetId={datasetId} groups={quality.groups} rows={rows} draft={draft} onDecide={onDecide} onPatch={onPatch} onOpen={onOpen} />
      )}

      {quality.notes.length > 0 && (
        <section className="space-y-1.5">
          <h2 className="text-[13px] font-semibold text-ink-1">{t("vision.quality.notes")}</h2>
          <ul className="list-disc space-y-1 pl-5 text-[13px] text-ink-2">
            {quality.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </section>
      )}

      {quiet.length > 0 && (
        <details className="group rounded-md border border-line bg-surface-1">
          <summary className="flex min-h-9 cursor-pointer list-none items-center gap-1.5 px-3 text-[13px] text-ink-1 pointer-coarse:min-h-11">
            <ChevronDown size={15} strokeWidth={1.5} aria-hidden className="text-ink-3 transition-transform duration-150 group-not-open:-rotate-90" />
            {t("vision.quality.quiet", { count: formatCount(quiet.length) })}
          </summary>
          <ul className="divide-y divide-line border-t border-line">
            {quiet.map((rule) => (
              <li key={rule.code} className="grid gap-x-3 gap-y-0.5 px-3 py-2 text-xs sm:grid-cols-[72px_minmax(0,14rem)_minmax(0,1fr)]">
                <span className="font-mono text-ink-3">{rule.code}</span>
                <span className="text-ink-1">{rule.title}</span>
                <span className="text-ink-2">{rule.reason}</span>
              </li>
            ))}
          </ul>
        </details>
      )}

      {onThresholds && <QualityThresholds datasetId={datasetId} rules={rules} draft={thresholds} onUse={onThresholds} />}

      <QualityAdvanced datasetId={datasetId} rows={rows} draft={draft} onDecide={onDecide} onOpen={onOpen} />

      {report.rules.length > 0 && (
        <details className="group rounded-md border border-line bg-surface-1">
          <summary className="flex min-h-9 cursor-pointer list-none items-center gap-1.5 px-3 text-[13px] text-ink-1 pointer-coarse:min-h-11">
            <ChevronDown size={15} strokeWidth={1.5} aria-hidden className="text-ink-3 transition-transform duration-150 group-not-open:-rotate-90" />
            {t("vision.quality.reading", { count: formatCount(report.rules.length) })}
          </summary>
          <ul className="space-y-1.5 border-t border-line px-3 py-2 text-xs">
            {report.rules.map((rule) => (
              <li key={`${rule.code}-${rule.message}`} className="text-ink-2">
                <span className="text-ink-1">{rule.message}</span> <span className="font-mono text-ink-3">{rule.code}</span>
                <span className="block">{rule.reason}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function FindingSection({
  datasetId,
  finding,
  rows,
  draft,
  onDecide,
  onOpen,
  onShow,
}: {
  datasetId: string;
  finding: QualityFinding;
  rows: readonly ImageRow[];
  draft: Draft;
  onDecide: (keys: string[], decision: Decision | null) => void;
  onOpen: (row: ImageRow) => void;
  onShow: (quality: QualityFilter) => void;
}) {
  const members = finding.rows.map((position) => rows[position]).filter((row): row is ImageRow => row !== undefined);
  const keys = members.map((row) => row.key);
  const decided = keys.filter((key) => draft[key] !== undefined);
  const parameters = Object.entries(finding.parameters);
  return (
    <section aria-labelledby={`finding-${finding.code}`} className="space-y-2 border-t border-line pt-4">
      <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
        <h2 id={`finding-${finding.code}`} className="flex min-w-0 flex-wrap items-baseline gap-x-2 text-[14px] font-semibold text-ink-1">
          <span className="font-mono text-xs font-normal text-ink-3">{finding.code}</span>
          {finding.title}
          <span className="text-[13px] font-normal text-ink-2 tabular-nums">
            {finding.count > 0 && finding.effect !== "dataset" ? t("vision.quality.count", { count: formatCount(finding.count) }) : null}
          </span>
          <Effect effect={finding.effect} />
        </h2>
        {members.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {finding.effect === "excluded" ? (
              <Button variant="ghost" icon={Check} onClick={() => onDecide(keys, "keep")}>
                {t("vision.quality.keepThese", { count: formatCount(keys.length) })}
              </Button>
            ) : (
              <Button variant="ghost" icon={Ban} onClick={() => onDecide(keys, "exclude")}>
                {t("vision.quality.excludeThese", { count: formatCount(keys.length) })}
              </Button>
            )}
            {decided.length > 0 && (
              <Button variant="ghost" icon={RotateCcw} onClick={() => onDecide(decided, null)}>
                {t("vision.decision.reset")}
              </Button>
            )}
            <Button variant="ghost" icon={Images} onClick={() => onShow(finding.code as QualityFilter)}>
              {t("vision.quality.showInGallery")}
            </Button>
          </div>
        )}
      </div>
      <p className="max-w-prose text-[13px] text-ink-1">{finding.message}</p>
      <p className="max-w-prose text-[13px] text-ink-2">{finding.reason}</p>
      {parameters.length > 0 && (
        <p className="text-xs text-ink-3">
          {t("vision.quality.thresholds")}{" "}
          {parameters.map(([name, value], k) => (
            <span key={name} className="font-mono">
              {k > 0 && ", "}
              {name} {Number(value.toPrecision(4))}
            </span>
          ))}
        </p>
      )}
      {members.length > 0 && <Thumbs datasetId={datasetId} rows={rows} positions={finding.rows} draft={draft} onOpen={onOpen} />}
    </section>
  );
}

function Effect({ effect }: { effect: QualityFinding["effect"] }) {
  const Icon = effect === "excluded" ? Ban : TriangleAlert;
  return (
    <span className="inline-flex items-center gap-1 self-center rounded-full border border-line px-2 text-[11px] leading-5 font-normal text-ink-2">
      <Icon size={11} strokeWidth={1.75} aria-hidden className={effect === "excluded" ? "text-bad" : "text-warn"} />
      {t(EFFECT[effect])}
    </span>
  );
}

function Duplicates({
  datasetId,
  groups,
  rows,
  draft,
  onDecide,
  onPatch,
  onOpen,
}: {
  datasetId: string;
  groups: readonly QualityGroup[];
  rows: readonly ImageRow[];
  draft: Draft;
  onDecide: (keys: string[], decision: Decision | null) => void;
  onPatch: (patch: DecisionPatch) => void;
  onOpen: (row: ImageRow) => void;
}) {
  const [shown, setShown] = useState(GROUPS_STEP);
  const exact = groups.filter((group) => group.kind === "exact").length;
  return (
    <section aria-labelledby="vision-duplicates" className="space-y-3 border-t border-line pt-4">
      <div className="space-y-1">
        <h2 id="vision-duplicates" className="text-[14px] font-semibold text-ink-1">
          {t("vision.quality.duplicates")}{" "}
          <span className="text-[13px] font-normal text-ink-2 tabular-nums">
            {t("vision.quality.groups", { count: formatCount(groups.length), exact: formatCount(exact), near: formatCount(groups.length - exact) })}
          </span>
        </h2>
        <p className="max-w-prose text-[13px] text-ink-2">{t("vision.quality.duplicatesHelp")}</p>
      </div>
      <ul className="space-y-3">
        {groups.slice(0, shown).map((group) => (
          <CompareStrip key={group.group} datasetId={datasetId} group={group} rows={rows} draft={draft} onDecide={onDecide} onPatch={onPatch} onOpen={onOpen} />
        ))}
      </ul>
      {groups.length > shown && (
        <Button onClick={() => setShown(shown + GROUPS_STEP)}>
          {t("vision.quality.moreGroups", { count: formatCount(Math.min(GROUPS_STEP, groups.length - shown)) })}
        </Button>
      )}
    </section>
  );
}

/** 108 (§ 9.2): two to five copies side by side, what tells them apart under each, and the choice. */
function CompareStrip({
  datasetId,
  group,
  rows,
  draft,
  onDecide,
  onPatch,
  onOpen,
}: {
  datasetId: string;
  group: QualityGroup;
  rows: readonly ImageRow[];
  draft: Draft;
  onDecide: (keys: string[], decision: Decision | null) => void;
  onPatch: (patch: DecisionPatch) => void;
  onOpen: (row: ImageRow) => void;
}) {
  const members = group.rows.map((position) => rows[position]).filter((row): row is ImageRow => row !== undefined);
  const keys = members.map((row) => row.key);
  const labels = new Set(members.map((row) => row.label));
  const splits = new Set(members.map((row) => row.split));
  const keepOnly = (key: string) => onPatch(Object.fromEntries(keys.map((other) => [other, other === key ? "keep" : "exclude"])));
  return (
    <li className="rounded-lg border border-line bg-surface-1 p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs">
        <span className="text-ink-2">
          <span className="font-medium text-ink-1">{group.kind === "exact" ? t("vision.quality.exact") : t("vision.quality.near")}</span>
          {" · "}
          {t("vision.quality.copies", { count: formatCount(members.length) })}
          {labels.size > 1 && <span className="ml-2 text-bad">{t("vision.quality.twoLabels")}</span>}
          {splits.size > 1 && <span className="ml-2 text-bad">{t("vision.quality.twoSplits")}</span>}
        </span>
        <span className="flex gap-1">
          <Button variant="ghost" icon={Check} onClick={() => onDecide(keys, "keep")}>
            {t("vision.quality.keepAll")}
          </Button>
          <Button variant="ghost" icon={RotateCcw} onClick={() => onDecide(keys, null)}>
            {t("vision.decision.reset")}
          </Button>
        </span>
      </div>
      <ul className="grid grid-cols-2 gap-3 sm:grid-cols-[repeat(auto-fill,minmax(150px,1fr))]">
        {members.slice(0, STRIP).map((row, k) => {
          const excluded = isExcluded(row, draft);
          const cause = exclusionCause(row, draft);
          const keptByRule = group.keep === group.rows[k] && cause.kind === "none" && draft[row.key] === undefined;
          return (
            <li key={row.key} className="min-w-0 space-y-1.5">
              <button type="button" onClick={() => onOpen(row)} title={row.key} className="group/copy relative block aspect-square w-full overflow-hidden rounded-md bg-surface-2">
                <img
                  src={imageUrl(datasetId, row.id, "thumb")}
                  alt={[row.name, row.label].filter(Boolean).join(", ")}
                  loading="lazy"
                  decoding="async"
                  className={`size-full object-contain transition-opacity ${excluded ? "opacity-35" : ""}`}
                />
                <span aria-hidden className="pointer-events-none absolute inset-0 rounded-md ring-1 ring-ink-1/8 ring-inset group-hover/copy:ring-2 group-hover/copy:ring-accent" />
              </button>
              <p className="truncate font-mono text-[11px] text-ink-1" title={row.key}>
                {row.name}
              </p>
              <p className="text-[11px] text-ink-2 tabular-nums">
                {row.width && row.height ? `${formatCount(row.width)} × ${formatCount(row.height)}` : "—"} · {formatBytes(row.bytes)}
                <br />
                {[row.label, row.split].filter(Boolean).join(" · ") || " "}
              </p>
              <p className={`flex items-center gap-1 text-[11px] ${excluded ? "text-ink-1" : "text-ink-2"}`}>
                {excluded ? <Ban size={11} strokeWidth={1.75} aria-hidden /> : <Check size={11} strokeWidth={2} aria-hidden className="text-ok" />}
                {excluded
                  ? cause.kind === "user"
                    ? t("vision.tile.excludedByYou")
                    : t("vision.tile.excludedBy", { code: cause.code ?? "" })
                  : keptByRule && group.kind === "exact"
                    ? t("vision.quality.keptByRule")
                    : cause.kind === "kept" || draft[row.key] === "keep"
                      ? t("vision.tile.kept")
                      : t("vision.quality.inUse")}
              </p>
              <Button variant="secondary" className="w-full justify-center" onClick={() => keepOnly(row.key)}>
                {t("vision.quality.keepThis")}
              </Button>
            </li>
          );
        })}
      </ul>
      {members.length > STRIP && <p className="mt-2 text-xs text-ink-3">{t("vision.quality.moreCopies", { count: formatCount(members.length - STRIP) })}</p>}
    </li>
  );
}
