import { ArrowDownWideNarrow, ArrowUpNarrowWide, Ban, Check, RotateCcw, Search, X } from "lucide-react";
import { useMemo } from "react";

import { formatNumber } from "../../charts/figures";
import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { PHONE, useMedia } from "../../lib/useMedia";
import { Button } from "../../ui/Button";
import { Select, TextInput } from "../../ui/fields";
import { ParamPanel } from "../../ui/ParamPanel";
import { SegmentedControl } from "../../ui/SegmentedControl";
import type { QualityRule } from "./api";
import { DatasetStats, StatsStrip } from "./DatasetStats";
import { ImageGrid } from "./ImageGrid";
import { countBy, issueCounts, SORTS, type Decision, type Draft, type ImageRow, type QualityFilter, type SortKey } from "./model";
import { activeFilterCount, CLEARED, formatRange, parseRange, type Density, type VisionSearch } from "./search";

const SORT_LABEL: Record<SortKey, MessageKey> = {
  file: "vision.sort.file",
  name: "vision.sort.name",
  label: "vision.sort.label",
  brightness: "vision.measure.brightness",
  contrast: "vision.measure.contrast",
  sharpness: "vision.measure.sharpness",
  saturation: "vision.measure.saturation",
  megapixels: "vision.sort.megapixels",
  bytes: "vision.sort.bytes",
  aspect: "vision.sort.aspect",
};

type GalleryProps = {
  datasetId: string;
  rows: readonly ImageRow[];
  shown: readonly ImageRow[];
  draft: Draft;
  rules: ReadonlyMap<string, QualityRule>;
  search: VisionSearch;
  set: (patch: VisionSearch) => void;
  selected: ReadonlySet<string>;
  onSelectedChange: (next: Set<string>) => void;
  onOpen: (row: ImageRow) => void;
  onDecide: (keys: string[], decision: Decision | null) => void;
};

/**
 * 107: the set's figures and statistics, then a stuck toolbar (search, label, split, quality, size,
 * sort, density; the active filters as chips; the selection's actions), then every image.
 */
export function Gallery({ datasetId, rows, shown, draft, rules, search, set, selected, onSelectedChange, onOpen, onDecide }: GalleryProps) {
  const phone = useMedia(PHONE);
  const density: Density = search.density ?? "m";
  const labels = useMemo(() => [...countBy(rows, (row) => row.label).entries()].filter(([label]) => label !== null).sort((a, b) => String(a[0]).localeCompare(String(b[0]))), [rows]);
  const splits = useMemo(() => [...countBy(rows, (row) => row.split).entries()].filter(([split]) => split !== null).sort((a, b) => String(a[0]).localeCompare(String(b[0]))), [rows]);
  const issues = useMemo(() => issueCounts(rows), [rows]);
  const filters = activeFilterCount(search);
  const range = parseRange(search.range);

  // laptop and tablet: one dense row, the names read by screen readers; phone: labelled, in a ParamPanel
  const fields = (
    <Fields compact={!phone} search={search} set={set} labels={labels as [string, number][]} splits={splits as [string, number][]} issues={issues} rules={rules} />
  );

  return (
    <div className="space-y-4">
      <StatsStrip rows={rows} draft={draft} quality={search.quality} onQuality={(quality) => set({ quality })} />
      <DatasetStats
        rows={rows}
        open={search.stats !== false}
        onOpenChange={(open) => set({ stats: open ? undefined : false })}
        onRange={(next) => set({ range: formatRange(next) })}
        onLabel={(label) => set({ labels: [label] })}
      />

      <div className="sticky top-0 z-10 -mx-[var(--gutter)] space-y-2 border-b border-line bg-surface-0 px-(--gutter) py-2">
        {phone ? (
          <ParamPanel
            label={t("vision.gallery.filters")}
            summary={[
              search.labels?.join(", "),
              search.quality ? qualityName(search.quality, rules) : null,
              t(SORT_LABEL[search.sort ?? "file"]) + (search.desc ? " ↓" : ""),
              t("vision.gallery.count", { shown: formatCount(shown.length), total: formatCount(rows.length) }),
            ]}
            layout="stack"
          >
            {fields}
          </ParamPanel>
        ) : (
          <div className="flex flex-wrap items-center gap-x-2 gap-y-2">
            {fields}
            <div className="flex items-center">
              <SegmentedControl<Density>
                label={t("vision.gallery.density")}
                value={density}
                onChange={(value) => set({ density: value === "m" ? undefined : value })}
                options={[
                  { value: "s", label: t("vision.density.s") },
                  { value: "m", label: t("vision.density.m") },
                  { value: "l", label: t("vision.density.l") },
                ]}
              />
            </div>
            <span className="ml-auto text-xs text-ink-3 tabular-nums" aria-live="polite">
              {t("vision.gallery.count", { shown: formatCount(shown.length), total: formatCount(rows.length) })}
            </span>
          </div>
        )}

        {(filters > 0 || issues.length > 0) && (
          <div className="flex flex-wrap items-center gap-1.5">
            {range && (
              <Chip on onClick={() => set({ range: undefined })}>
                {t(`vision.measure.${range.measure}` as MessageKey)} {formatNumber(range.low)} {t("vision.to")} {formatNumber(range.high)}
              </Chip>
            )}
            {search.q && (
              <Chip on onClick={() => set({ q: undefined })}>
                “{search.q}”
              </Chip>
            )}
            {search.size && (
              <Chip on onClick={() => set({ size: undefined })}>
                {t(`vision.size.${search.size}` as MessageKey)}
              </Chip>
            )}
            {issues.map(([code, count]) => {
              const on = search.quality === code;
              return (
                <Chip key={code} on={on} onClick={() => set({ quality: on ? undefined : (code as QualityFilter) })} title={rules.get(code)?.reason}>
                  {rules.get(code)?.title ?? code} <span className="tabular-nums text-ink-3">{formatCount(count)}</span>
                </Chip>
              );
            })}
            {filters > 0 && (
              <button type="button" onClick={() => set(CLEARED)} className="ml-1 inline-flex min-h-6 items-center text-xs text-accent hover:underline pointer-coarse:min-h-11">
                {t("vision.gallery.clear")}
              </button>
            )}
          </div>
        )}

        {selected.size > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 rounded-md border border-line bg-surface-1 px-2 py-1" role="toolbar" aria-label={t("vision.gallery.selection")}>
            <span className="mr-1 text-xs font-medium text-ink-1 tabular-nums">{t("vision.gallery.selected", { count: formatCount(selected.size) })}</span>
            <Button variant="ghost" icon={Ban} onClick={() => onDecide([...selected], "exclude")}>
              {t("vision.decision.exclude")} <Kbd>e</Kbd>
            </Button>
            <Button variant="ghost" icon={Check} onClick={() => onDecide([...selected], "keep")}>
              {t("vision.decision.keep")} <Kbd>k</Kbd>
            </Button>
            <Button variant="ghost" icon={RotateCcw} onClick={() => onDecide([...selected], null)}>
              {t("vision.decision.reset")} <Kbd>r</Kbd>
            </Button>
            <Button variant="ghost" icon={X} className="ml-auto" onClick={() => onSelectedChange(new Set())}>
              {t("vision.gallery.unselect")}
            </Button>
          </div>
        )}
      </div>

      {shown.length === 0 ? (
        <div className="rounded-lg border border-dashed border-line px-6 py-12 text-center">
          <p className="text-[13px] text-ink-1">{t("vision.gallery.empty")}</p>
          <button type="button" onClick={() => set(CLEARED)} className="mt-2 text-[13px] text-accent hover:underline">
            {t("vision.gallery.clear")}
          </button>
        </div>
      ) : (
        <ImageGrid
          datasetId={datasetId}
          rows={shown}
          density={density}
          draft={draft}
          selected={selected}
          onSelectedChange={onSelectedChange}
          onOpen={onOpen}
          onDecide={onDecide}
        />
      )}
      <p className="text-xs text-ink-3 max-sm:hidden">{t("vision.gallery.keys")}</p>
    </div>
  );
}

export function qualityName(quality: QualityFilter, rules: ReadonlyMap<string, QualityRule>): string {
  if (quality.startsWith("R-VQ-")) return rules.get(quality)?.title ?? quality;
  return t(`vision.quality.${quality}` as MessageKey);
}

function Fields({
  compact,
  search,
  set,
  labels,
  splits,
  issues,
  rules,
}: {
  compact: boolean;
  search: VisionSearch;
  set: (patch: VisionSearch) => void;
  labels: [string, number][];
  splits: [string, number][];
  issues: [string, number][];
  rules: ReadonlyMap<string, QualityRule>;
}) {
  const sort = search.sort ?? "file";
  const field = compact ? "flex min-w-0 text-xs text-ink-2" : "flex min-w-0 flex-col gap-1 text-xs text-ink-2";
  const caption = compact ? "sr-only" : "";
  // a native list is as wide as its longest option: capped in the dense row
  const width = compact ? "max-w-44" : "w-full";
  return (
    <>
      <label className={`${field} ${compact ? "w-48" : "w-full"}`}>
        <span className={caption}>{t("vision.gallery.search")}</span>
        <span className="relative">
          <Search size={13} strokeWidth={1.5} aria-hidden className="pointer-events-none absolute top-1/2 left-2 -translate-y-1/2 text-ink-3" />
          <TextInput
            type="search"
            value={search.q ?? ""}
            placeholder={t("vision.gallery.searchPlaceholder")}
            onChange={(event) => set({ q: event.target.value || undefined })}
            className="w-full pl-7"
          />
        </span>
      </label>
      {labels.length > 0 && (
        <label className={field}>
          <span className={caption}>{t("vision.field.label")}</span>
          <Select className={width} value={search.labels?.[0] ?? ""} onChange={(event) => set({ labels: event.target.value ? [event.target.value] : undefined })}>
            <option value="">{t("vision.gallery.allLabels", { count: formatCount(labels.length) })}</option>
            {labels.map(([label, count]) => (
              <option key={label} value={label}>
                {label} ({formatCount(count)})
              </option>
            ))}
          </Select>
        </label>
      )}
      {splits.length > 1 && (
        <label className={field}>
          <span className={caption}>{t("vision.field.split")}</span>
          <Select className={width} value={search.splits?.[0] ?? ""} onChange={(event) => set({ splits: event.target.value ? [event.target.value] : undefined })}>
            <option value="">{t("vision.gallery.allSplits")}</option>
            {splits.map(([split, count]) => (
              <option key={split} value={split}>
                {split} ({formatCount(count)})
              </option>
            ))}
          </Select>
        </label>
      )}
      <label className={field}>
        <span className={caption}>{t("vision.gallery.quality")}</span>
        <Select className={width} value={search.quality ?? ""} onChange={(event) => set({ quality: (event.target.value || undefined) as QualityFilter | undefined })}>
          <option value="">{t("vision.quality.all")}</option>
          <option value="any">{t("vision.quality.any")}</option>
          <option value="none">{t("vision.quality.none")}</option>
          <option value="excluded">{t("vision.quality.excluded")}</option>
          <option value="decided">{t("vision.quality.decided")}</option>
          {issues.length > 0 && (
            <optgroup label={t("vision.gallery.byRule")}>
              {issues.map(([code, count]) => (
                <option key={code} value={code}>
                  {code} {rules.get(code)?.title ?? ""} ({formatCount(count)})
                </option>
              ))}
            </optgroup>
          )}
        </Select>
      </label>
      <label className={field}>
        <span className={caption}>{t("vision.gallery.size")}</span>
        <Select className={width} value={search.size ?? ""} onChange={(event) => set({ size: (event.target.value || undefined) as VisionSearch["size"] })}>
          <option value="">{t("vision.size.all")}</option>
          <option value="small">{t("vision.size.small")}</option>
          <option value="medium">{t("vision.size.medium")}</option>
          <option value="large">{t("vision.size.large")}</option>
        </Select>
      </label>
      <div className={compact ? "flex min-w-0 text-xs" : field}>
        <label htmlFor="vision-sort" className={compact ? "self-center pr-1.5 pl-1 text-ink-3" : ""}>
          {t("vision.gallery.sort")}
        </label>
        <span className="flex gap-1">
          <Select id="vision-sort" value={sort} onChange={(event) => set({ sort: event.target.value === "file" ? undefined : (event.target.value as SortKey) })}>
            {SORTS.map((key) => (
              <option key={key} value={key}>
                {t(SORT_LABEL[key])}
              </option>
            ))}
          </Select>
          <Button
            iconOnly
            icon={search.desc ? ArrowDownWideNarrow : ArrowUpNarrowWide}
            aria-label={search.desc ? t("vision.gallery.descending") : t("vision.gallery.ascending")}
            title={search.desc ? t("vision.gallery.descending") : t("vision.gallery.ascending")}
            onClick={() => set({ desc: search.desc ? undefined : true })}
          />
        </span>
      </div>
    </>
  );
}

function Chip({ on, onClick, title, children }: { on: boolean; onClick: () => void; title?: string; children: React.ReactNode }) {
  return (
    <button
      type="button"
      aria-pressed={on}
      title={title}
      onClick={onClick}
      className={`inline-flex h-6 items-center gap-1 rounded-full border px-2.5 text-xs transition-colors pointer-coarse:h-8 ${
        on ? "border-accent bg-accent-soft text-ink-1" : "border-line bg-surface-1 text-ink-2 hover:text-ink-1"
      }`}
    >
      {children}
      {on && <X size={11} strokeWidth={1.75} aria-hidden className="text-ink-3" />}
    </button>
  );
}

function Kbd({ children }: { children: React.ReactNode }) {
  return <kbd className="ml-0.5 rounded-[3px] border border-line px-1 font-mono text-[10px] leading-4 text-ink-3 max-sm:hidden">{children}</kbd>;
}
