import { ChevronLeft, ChevronRight, MapPinOff, ScanSearch, TriangleAlert } from "lucide-react";
import { useEffect, useRef } from "react";

import { imageUrl } from "../../features/import/imageSetApi";
import { formatBytes, formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { SegmentedControl } from "../../ui/SegmentedControl";
import type { QualityRule } from "./api";
import { ImageViewer } from "./ImageViewer";
import { exclusionCause, isExcluded, rankOf, type Decision, type Draft, type ImageRow, type Measure } from "./model";

/** Measures shown in the sheet, with the words of their rank (R-107-08). */
const MEASURE_ROWS: { measure: Measure; label: MessageKey; rank: MessageKey; digits: number }[] = [
  { measure: "brightness", label: "vision.measure.brightness", rank: "vision.rank.brightness", digits: 2 },
  { measure: "contrast", label: "vision.measure.contrast", rank: "vision.rank.contrast", digits: 3 },
  { measure: "sharpness", label: "vision.measure.sharpness", rank: "vision.rank.sharpness", digits: 0 },
  { measure: "saturation", label: "vision.measure.saturation", rank: "vision.rank.saturation", digits: 2 },
  { measure: "colourfulness", label: "vision.measure.colourfulness", rank: "vision.rank.colourfulness", digits: 1 },
  { measure: "entropy", label: "vision.measure.entropy", rank: "vision.rank.entropy", digits: 2 },
];

type SheetProps = {
  datasetId: string;
  row: ImageRow;
  /** The list the user is looking at (filtered and sorted): previous and next follow it. */
  shown: readonly ImageRow[];
  siblings: readonly ImageRow[];
  groupKind: "exact" | "near" | null;
  sorted: Record<Measure, number[]>;
  rules: ReadonlyMap<string, QualityRule>;
  draft: Draft;
  onNavigate: (key: string) => void;
  onClose: () => void;
  onDecide: (keys: string[], decision: Decision | null) => void;
  /** 117: open this image's "Why?" (what the kept model says about it, and its evidence). */
  onWhy?: () => void;
};

const number = (value: number | null, digits: number) =>
  value === null ? "—" : value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });

/** "500 × 500 JPEG, 48 KB" */
export function imageSentence(row: ImageRow): string {
  const size = row.width && row.height ? `${formatCount(row.width)} × ${formatCount(row.height)}` : null;
  return [size && row.format ? `${size} ${row.format.toUpperCase()}` : (size ?? row.format?.toUpperCase()), formatBytes(row.bytes)]
    .filter(Boolean)
    .join(", ");
}

/**
 * 107: one image. The original in the viewer (zoom, pan, pixel inspector, mask); its sentence,
 * label and split; the decision (108); its problems with their rules; its measures with their rank
 * among the set; its file (never a coordinate, R-IMG-05); its copies. ← → or a swipe go through
 * the list the gallery shows.
 */
export function ImageSheet({ datasetId, row, shown, siblings, groupKind, sorted, rules, draft, onNavigate, onClose, onDecide, onWhy }: SheetProps) {
  const position = shown.findIndex((other) => other.key === row.key);
  const previous = position > 0 ? shown[position - 1] : undefined;
  const next = position >= 0 && position < shown.length - 1 ? shown[position + 1] : undefined;
  const excluded = isExcluded(row, draft);
  const cause = exclusionCause(row, draft);
  const decision = draft[row.key] ?? "rules";

  const keys = useRef({ previous, next, row, onNavigate, onDecide });
  useEffect(() => {
    keys.current = { previous, next, row, onNavigate, onDecide };
  });
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target instanceof Element ? event.target : null;
      if (target?.closest("input, textarea, select, [contenteditable=true]") || event.ctrlKey || event.metaKey || event.altKey) return;
      const state = keys.current;
      if (event.key === "ArrowRight" && state.next) state.onNavigate(state.next.key);
      else if (event.key === "ArrowLeft" && state.previous) state.onNavigate(state.previous.key);
      else if (event.key === "e") state.onDecide([state.row.key], "exclude");
      else if (event.key === "k") state.onDecide([state.row.key], "keep");
      else if (event.key === "r") state.onDecide([state.row.key], null);
      else return;
      event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const status =
    cause.kind === "user"
      ? t("vision.sheet.excludedByYou")
      : cause.kind === "kept"
        ? t("vision.sheet.keptByYou")
        : excluded
          ? t("vision.sheet.excludedBy", { code: cause.code ?? "", title: rules.get(cause.code ?? "")?.title ?? "" })
          : t("vision.sheet.used");

  return (
    <Dialog
      open
      onOpenChange={(open) => !open && onClose()}
      title={row.name}
      description={row.key}
      size="lg"
      footer={
        <div className="flex w-full items-center justify-between gap-2">
          <Button variant="ghost" icon={ChevronLeft} disabled={!previous} onClick={() => previous && onNavigate(previous.key)}>
            <span className="max-sm:sr-only">{t("vision.sheet.previous")}</span>
          </Button>
          <span className="text-xs text-ink-3 tabular-nums">
            {position >= 0 ? t("vision.sheet.position", { at: formatCount(position + 1), of: formatCount(shown.length) }) : t("vision.sheet.notShown")}
          </span>
          <Button variant="ghost" onClick={() => next && onNavigate(next.key)} disabled={!next}>
            <span className="max-sm:sr-only">{t("vision.sheet.next")}</span>
            <ChevronRight size={15} strokeWidth={1.5} aria-hidden />
          </Button>
        </div>
      }
    >
      <div className="grid min-w-0 gap-5 lg:grid-cols-[minmax(0,1fr)_300px]">
        {row.decodeOk ? (
          <ImageViewer
            key={row.key}
            src={imageUrl(datasetId, row.id, "file")}
            maskSrc={row.hasMask ? imageUrl(datasetId, row.id, "mask") : null}
            alt={[row.name, row.label].filter(Boolean).join(", ")}
            onSwipe={(direction) => {
              const target = direction === 1 ? next : previous;
              if (target) onNavigate(target.key);
            }}
            className="h-[min(64dvh,640px)] max-sm:h-[50dvh]"
          />
        ) : (
          <div className="flex h-60 flex-col items-center justify-center gap-2 rounded-md bg-surface-2 p-6 text-center text-[13px] text-ink-2">
            <TriangleAlert size={18} strokeWidth={1.5} className="text-warn" aria-hidden />
            <p>{t("vision.sheet.notDecoded")}</p>
            {row.decodeError && <p className="font-mono text-xs text-ink-3">{row.decodeError}</p>}
          </div>
        )}

        <div className="min-w-0 space-y-5 text-[13px]">
          <section className="space-y-2">
            <div className="flex items-start justify-between gap-3">
              <p className="text-ink-1 tabular-nums">{imageSentence(row)}</p>
              {onWhy && row.decodeOk && (
                <Button variant="secondary" icon={ScanSearch} onClick={onWhy} title={t("vision.why.open")} className="shrink-0">
                  {t("vision.why.button")}
                </Button>
              )}
            </div>
            <dl className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
              <Pair label={t("vision.field.label")} value={row.label ?? t("vision.none")} mono />
              <Pair label={t("vision.field.split")} value={row.split ?? t("vision.none")} mono />
              {row.isNormal !== null && <Pair label={t("vision.field.normal")} value={row.isNormal ? t("vision.yes") : t("vision.no")} />}
            </dl>
          </section>

          <section className="space-y-2" aria-label={t("vision.sheet.decision")}>
            <p className={excluded ? "text-ink-1" : "text-ink-2"}>{status}</p>
            <SegmentedControl<Decision | "rules">
              label={t("vision.sheet.decision")}
              value={decision}
              onChange={(value) => onDecide([row.key], value === "rules" ? null : value)}
              options={[
                { value: "rules", label: t("vision.decision.rules") },
                { value: "exclude", label: t("vision.decision.exclude") },
                { value: "keep", label: t("vision.decision.keep") },
              ]}
            />
          </section>

          {row.issues.length > 0 && (
            <section className="space-y-2">
              <h3 className="font-semibold text-ink-1">{t("vision.sheet.issues")}</h3>
              <ul className="space-y-2">
                {row.issues.map((code) => {
                  const rule = rules.get(code);
                  return (
                    <li key={code} className="flex gap-2">
                      <TriangleAlert size={14} strokeWidth={1.5} className="mt-0.5 shrink-0 text-warn" aria-hidden />
                      <span className="min-w-0">
                        <span className="text-ink-1">
                          <span className="font-mono text-xs">{code}</span> {rule?.title}
                        </span>
                        {rule && <span className="block text-xs text-ink-2">{rule.reason}</span>}
                      </span>
                    </li>
                  );
                })}
              </ul>
            </section>
          )}

          <section className="space-y-1.5">
            <h3 className="font-semibold text-ink-1">{t("vision.sheet.measures")}</h3>
            <table className="w-full text-xs">
              <tbody>
                {MEASURE_ROWS.map(({ measure, label, rank, digits }) => {
                  const value = row[measure];
                  const share = rankOf(sorted[measure], value);
                  return (
                    <tr key={measure} className="border-t border-line first:border-t-0">
                      <th scope="row" className="py-1 pr-2 text-left font-normal text-ink-2">
                        {t(label)}
                      </th>
                      <td className="py-1 pr-2 text-right text-ink-1 tabular-nums">{number(value, digits)}</td>
                      <td className="py-1 text-right text-ink-3">
                        {share === null ? "" : t(rank, { share: `${Math.floor(share * 100)} %` })}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </section>

          <section className="space-y-1.5">
            <h3 className="font-semibold text-ink-1">{t("vision.sheet.file")}</h3>
            <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-xs">
              <Row label={t("vision.field.colourMode")} value={[row.colorMode, row.bitDepth ? `${row.bitDepth}-bit` : null].filter(Boolean).join(", ")} />
              <Row label={t("vision.field.megapixels")} value={row.megapixels === null ? null : number(row.megapixels, 2)} />
              <Row label={t("vision.field.aspect")} value={row.aspect === null ? null : number(row.aspect, 2)} />
              {row.frames !== null && row.frames > 1 && <Row label={t("vision.field.frames")} value={formatCount(row.frames)} />}
              <Row label={t("vision.field.camera")} value={row.camera} />
              <Row label={t("vision.field.dateTaken")} value={row.dateTaken?.replace("T", " ") ?? null} />
              {row.dominant && (
                <>
                  <dt className="text-ink-2">{t("vision.field.dominant")}</dt>
                  <dd className="flex items-center gap-1.5 font-mono text-ink-1">
                    {/* the colour is the image's data, not a theme colour */}
                    <span aria-hidden className="size-3 rounded-[3px] ring-1 ring-ink-1/15" style={{ backgroundColor: row.dominant }} />
                    {row.dominant}
                  </dd>
                </>
              )}
            </dl>
            {row.hasLocation && (
              <p className="flex items-start gap-1.5 pt-1 text-xs text-ink-2">
                <MapPinOff size={13} strokeWidth={1.5} className="mt-px shrink-0" aria-hidden />
                {t("vision.sheet.location")}
              </p>
            )}
          </section>

          {siblings.length > 0 && (
            <section className="space-y-2">
              <h3 className="font-semibold text-ink-1">
                {groupKind === "exact" ? t("vision.sheet.exactCopies") : t("vision.sheet.nearCopies")}
              </h3>
              <ul className="flex flex-wrap gap-2">
                {siblings.slice(0, 8).map((other) => (
                  <li key={other.key}>
                    <button
                      type="button"
                      onClick={() => onNavigate(other.key)}
                      title={other.key}
                      className="group/copy block w-[72px] text-left"
                    >
                      <span className="relative block size-[72px] overflow-hidden rounded-md bg-surface-2">
                        <img src={imageUrl(datasetId, other.id, "thumb")} alt={other.name} loading="lazy" decoding="async" className="size-full object-contain" />
                        <span aria-hidden className="pointer-events-none absolute inset-0 rounded-md ring-1 ring-ink-1/8 ring-inset group-hover/copy:ring-2 group-hover/copy:ring-accent" />
                      </span>
                      <span className="mt-1 block truncate font-mono text-[10px] text-ink-2">{other.split ?? other.name}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>
      </div>
    </Dialog>
  );
}

function Pair({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <dt className="text-ink-3">{label}</dt>
      <dd className={`text-ink-1 ${mono ? "font-mono" : ""}`}>{value}</dd>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string | null }) {
  if (!value) return null;
  return (
    <>
      <dt className="text-ink-2">{label}</dt>
      <dd className="min-w-0 truncate text-ink-1 tabular-nums">{value}</dd>
    </>
  );
}
