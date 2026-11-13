import { ChevronRight, Download, GalleryHorizontalEnd, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { formatNumber } from "../../charts/figures";
import { Plot } from "../../charts/Plot";
import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { PHONE, useMedia } from "../../lib/useMedia";
import { LITE } from "../../lite/flag";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { ErrorNote, Field, ProgressBar, Select } from "../../ui/fields";
import { InfoTip } from "../../ui/InfoTip";
import { Loading } from "../../ui/Loading";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { readTokens } from "../../charts/theme";
import { rampGradient, sequentialStops } from "../../ui/colorRamp";
import { useBuildMap, useDownloadWeights, useImageMap, useMapMethods, type Clustering, type ImageMap, type MapMethod, type MapProjection, type Projection, type QualityRule } from "./api";
import { EmbeddingMap, legendName, MARK_FLOOR } from "./EmbeddingMap";
import { COLOUR_BY, colouring as colour, rowsOfEntry, type ColourBy } from "./mapModel";
import { MEASURES, type ImageRow, type Measure } from "./model";
import type { VisionSearch } from "./search";

export type MapPick = { rows: number[]; label: string };

type MapTabProps = {
  datasetId: string;
  rows: readonly ImageRow[];
  rules: ReadonlyMap<string, QualityRule>;
  search: VisionSearch;
  set: (patch: VisionSearch) => void;
  onOpen: (row: ImageRow) => void;
  /** Show these images in the gallery (R-VMAP-08). */
  onShow: (pick: MapPick) => void;
};

const COLOUR_LABEL: Record<ColourBy, MessageKey> = {
  label: "vision.map.by.label",
  cluster: "vision.map.by.cluster",
  quality: "vision.map.by.quality",
  split: "vision.map.by.split",
  brightness: "vision.measure.brightness",
  contrast: "vision.measure.contrast",
  sharpness: "vision.measure.sharpness",
  saturation: "vision.measure.saturation",
  colourfulness: "vision.measure.colourfulness",
  entropy: "vision.measure.entropy",
};

/**
 * 109: the images as a map (study § 4.4). The classic space was built right after the import
 * (R-VMAP-10); a deep one (114) is built when asked, its weights downloaded only after a
 * confirmation that shows their size (R-VMAP-09). The sentence above says what the map shows
 * (R-VMAP-05); the legend counts and selects; the lasso or a group opens the gallery on them.
 */
export function MapTab({ datasetId, rows, rules, search, set, onOpen, onShow }: MapTabProps) {
  const phone = useMedia(PHONE);
  const methods = useMapMethods();
  const method = search.method ?? "classic";
  const projection: Projection = search.proj ?? "pca";
  const clustering: Clustering = search.groups ?? "kmeans";
  const k = search.k ?? null;
  const query = useMemo(() => ({ method, projection, clustering, k }), [method, projection, clustering, k]);
  const state = useImageMap(datasetId, query, true);
  const build = useBuildMap(datasetId);
  const asked = useRef(new Set<string>());
  const hasLabels = useMemo(() => rows.some((row) => row.label !== null), [rows]);
  const by: ColourBy = search.color ?? (hasLabels ? "label" : "cluster");

  // R-VMAP-10: a set imported before 109 has no classic space; it is built once, here, with its
  // progress (the deep ones only when the user asks)
  const missing = state.data?.state === "missing";
  useEffect(() => {
    if (missing && method === "classic" && !asked.current.has(datasetId)) {
      asked.current.add(datasetId);
      build.mutate({ method });
    }
  }, [missing, method, datasetId, build]);

  const chosen = methods.data?.methods.find((m) => m.key === method);

  return (
    <div className="space-y-4">
      <Controls
        methods={methods.data?.methods ?? []}
        method={method}
        by={by}
        hasLabels={hasLabels}
        onMethod={(value) => set({ method: value === "classic" ? undefined : value })}
        onColour={(value) => set({ color: value === (hasLabels ? "label" : "cluster") ? undefined : value })}
        phone={phone}
      />

      {state.isError ? (
        <ErrorNote>{(state.error as Error).message}</ErrorNote>
      ) : !state.data || (state.data.state === "missing" && method === "classic") ? (
        <Loading label={t("vision.map.loading")} className="h-[420px] rounded-md" />
      ) : state.data.state === "running" ? (
        <Building done={state.data.done ?? 0} total={state.data.total ?? 0} message={state.data.message ?? ""} />
      ) : state.data.state === "failed" ? (
        <Failed
          error={state.data.error ?? ""}
          long={state.data.code === "LongRun"}
          onRetry={(confirmLong) => build.mutate({ method, confirmLong })}
        />
      ) : state.data.state === "missing" ? (
        <DeepStart method={chosen} onBuild={() => build.mutate({ method })} pending={build.isPending} error={build.error as Error | null} />
      ) : state.data.map ? (
        <MapView
          key={`${method}:${projection}:${clustering}:${k ?? "auto"}`}
          datasetId={datasetId}
          map={state.data.map}
          rows={rows}
          rules={rules}
          by={by}
          phone={phone}
          onOpen={onOpen}
          onShow={onShow}
        />
      ) : null}

      <Advanced search={search} set={set} map={state.data?.map ?? null} projections={methods.data?.projections ?? []} />
    </div>
  );
}

// --- controls ------------------------------------------------------------------------------------

function Controls({
  methods,
  method,
  by,
  hasLabels,
  onMethod,
  onColour,
  phone,
}: {
  methods: MapMethod[];
  method: string;
  by: ColourBy;
  hasLabels: boolean;
  onMethod: (value: string) => void;
  onColour: (value: ColourBy) => void;
  phone: boolean;
}) {
  const download = useDownloadWeights();
  const [confirm, setConfirm] = useState<MapMethod | null>(null);
  const first = methods.filter((m) => m.level === "first");
  const advanced = methods.filter((m) => m.level === "advanced");
  const current = methods.find((m) => m.key === method);
  const deepFirst = first.find((m) => m.key !== "classic");
  const downloads = useMapMethods().data?.downloads ?? {};
  const running = deepFirst ? downloads[deepFirst.key] : undefined;

  const option = (m: MapMethod) => (
    <option key={m.key} value={m.key} disabled={!m.available}>
      {m.title}
      {/* 119: in the browser version a deep method is desktop only, no download is offered (R-VIS-WEB-03) */}
      {m.available ? "" : LITE ? ` (${t("lite.desktopOnly")})` : m.downloaded ? ` (${t("vision.map.method.unavailable")})` : m.size ? ` (${t("vision.map.method.needsWeights", { size: m.size })})` : ""}
    </option>
  );

  return (
    <div className="space-y-2">
      <div className={phone ? "grid grid-cols-2 gap-2" : "flex flex-wrap items-end gap-x-3 gap-y-2"}>
        <Field label={t("vision.map.method")}>
          <Select value={method} onChange={(event) => onMethod(event.target.value)} className="min-w-0 sm:w-64">
            {first.map(option)}
            {advanced.length > 0 && <optgroup label={t("vision.map.method.advanced")}>{advanced.map(option)}</optgroup>}
          </Select>
        </Field>
        <Field label={t("vision.map.colourBy")}>
          <Select value={by} onChange={(event) => onColour(event.target.value as ColourBy)} className="min-w-0 sm:w-48">
            {COLOUR_BY.filter((value) => hasLabels || value !== "label").map((value) => (
              <option key={value} value={value}>
                {t(COLOUR_LABEL[value])}
              </option>
            ))}
          </Select>
        </Field>
        {current && <p className="text-xs text-ink-3 max-sm:col-span-2 sm:pb-2">{current.detail}</p>}
      </div>

      {deepFirst && !deepFirst.available && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 rounded-md border border-line bg-surface-1 px-3 py-2 text-xs text-ink-2">
          <span className="min-w-0 flex-1">
            <span className="font-medium text-ink-1">{deepFirst.title}</span>{" "}
            {running?.state === "running"
              ? t("vision.map.downloading", { done: formatNumber(running.received / 1e6), total: formatNumber(running.total / 1e6) })
              : running?.state === "failed"
                ? running.error
                : deepFirst.reason?.includes("R-VDP-02")
                  ? t("vision.map.deepOffer", { size: deepFirst.size ?? "", licence: deepFirst.licence ?? "" })
                  : deepFirst.reason}
          </span>
          {running?.state === "running" ? (
            <span className="w-40">
              <ProgressBar value={running.total ? running.received / running.total : 0} label={t("vision.map.downloadProgress")} />
            </span>
          ) : (
            deepFirst.size &&
            !deepFirst.downloaded &&
            !deepFirst.reason?.includes("R-VDP-01") && (
              <Button icon={Download} onClick={() => setConfirm(deepFirst)}>
                {t("vision.map.download", { size: deepFirst.size })}
              </Button>
            )
          )}
        </div>
      )}

      {confirm && (
        <Dialog
          open
          onOpenChange={(open) => !open && setConfirm(null)}
          title={t("vision.map.confirmTitle", { name: confirm.title })}
          size="sm"
          footer={
            <>
              <Button variant="ghost" onClick={() => setConfirm(null)}>
                {t("vision.map.confirmCancel")}
              </Button>
              <Button
                variant="primary"
                icon={Download}
                onClick={() => {
                  download.mutate(confirm.key);
                  setConfirm(null);
                }}
              >
                {t("vision.map.confirmDownload", { size: confirm.size ?? "" })}
              </Button>
            </>
          }
        >
          <div className="space-y-2 text-[13px] text-ink-2">
            <p>{t("vision.map.confirmBody", { name: confirm.title, size: confirm.size ?? "", licence: confirm.licence ?? "" })}</p>
            {confirm.note && <p className="text-xs">{confirm.note}</p>}
            <p className="text-xs text-ink-3">{t("vision.map.confirmSafety")}</p>
          </div>
        </Dialog>
      )}
    </div>
  );
}

function Building({ done, total, message }: { done: number; total: number; message: string }) {
  return (
    <div className="flex h-[420px] flex-col items-center justify-center gap-3 rounded-md border border-line bg-surface-1 px-6 text-center">
      <p className="text-[13px] text-ink-1">{t("vision.map.building")}</p>
      <div className="w-full max-w-sm">
        <ProgressBar value={total ? done / total : 0} label={t("vision.map.building")} />
      </div>
      <p className="text-xs text-ink-3 tabular-nums" aria-live="polite">
        {message}
      </p>
      <p className="max-w-md text-xs text-ink-3">{t("vision.map.buildingWhy")}</p>
    </div>
  );
}

function Failed({ error, long, onRetry }: { error: string; long: boolean; onRetry: (confirmLong: boolean) => void }) {
  return (
    <div className="space-y-2">
      <ErrorNote>{error}</ErrorNote>
      <Button onClick={() => onRetry(long)}>{long ? t("vision.map.runAnyway") : t("vision.map.retry")}</Button>
    </div>
  );
}

function DeepStart({ method, onBuild, pending, error }: { method: MapMethod | undefined; onBuild: () => void; pending: boolean; error: Error | null }) {
  return (
    <div className="flex h-[320px] flex-col items-start justify-center gap-2 rounded-md border border-line bg-surface-1 px-6">
      <p className="text-[13px] text-ink-1">{t("vision.map.deepStart", { name: method?.title ?? "" })}</p>
      <p className="max-w-lg text-xs text-ink-2">{t("vision.map.deepWhy")}</p>
      {error && <ErrorNote>{error.message}</ErrorNote>}
      <Button variant="primary" onClick={onBuild} disabled={pending || !method?.available}>
        {t("vision.map.deepBuild", { name: method?.title ?? "" })}
      </Button>
      {method && !method.available && <p className="text-xs text-ink-3">{method.reason}</p>}
    </div>
  );
}

// --- the map, its legend and the selection -------------------------------------------------------

function MapView({
  datasetId,
  map,
  rows,
  rules,
  by,
  phone,
  onOpen,
  onShow,
}: {
  datasetId: string;
  map: ImageMap;
  rows: readonly ImageRow[];
  rules: ReadonlyMap<string, QualityRule>;
  by: ColourBy;
  phone: boolean;
  onOpen: (row: ImageRow) => void;
  onShow: (pick: MapPick) => void;
}) {
  const colouring = useMemo(() => colour(rows, map, by), [rows, map, by]);
  const [focus, setFocus] = useState<string | null>(null);
  const [picked, setPicked] = useState<MapPick | null>(null);
  const selected = useMemo(() => (picked ? new Set(picked.rows) : null), [picked]);
  const measure = (MEASURES as readonly string[]).includes(by) ? (by as Measure) : null;
  const ruleTitle = (code: string) => rules.get(code)?.title;
  const name = (key: string) => legendName(key, by, ruleTitle);
  const keys = colouring.kind === "categorical" ? colouring.entries.map((e) => e.key) : [];
  const palette = readTokens().palette;

  const pickEntry = (key: string) => {
    const tableRows = rowsOfEntry(rows, map, by, key, keys);
    setPicked({ rows: tableRows, label: name(key) });
  };

  const axes =
    map.projection === "pca" && map.explained
      ? t("vision.map.axes", { a: formatNumber((map.explained[0] ?? 0) * 100), b: formatNumber((map.explained[1] ?? 0) * 100) })
      : t("vision.map.axesTsne");

  const describe = (row: ImageRow) => {
    if (!measure) return null;
    const value = row[measure];
    return value === null ? null : `${t(COLOUR_LABEL[measure])} ${formatNumber(value)}`;
  };

  return (
    <div className="space-y-3">
      <header className="space-y-1">
        <p className="text-[15px] leading-snug font-medium text-pretty text-ink-1">{map.sentence}</p>
        <p className="text-xs text-ink-2 tabular-nums">
          {t("vision.map.summary", { points: formatCount(map.points), of: formatCount(map.of), k: String(map.k) })} {axes}
        </p>
      </header>

      <div className={phone ? "space-y-3" : "grid grid-cols-[minmax(0,1fr)_248px] items-start gap-4"}>
        <EmbeddingMap
          datasetId={datasetId}
          map={map}
          rows={rows}
          colouring={colouring}
          focus={focus}
          selected={selected}
          onSelect={(tableRows) => setPicked(tableRows.length ? { rows: tableRows, label: t("vision.map.lassoLabel") } : null)}
          onOpen={onOpen}
          onSelectGroup={(cluster) => {
            const tableRows = map.clusters.flatMap((value, row) => (value === cluster ? [row] : []));
            setPicked({ rows: tableRows, label: t("vision.map.groupN", { n: String(cluster + 1) }) });
          }}
          label={t("vision.map.canvasLabel", { points: formatCount(map.points), by: t(COLOUR_LABEL[by]) })}
          height={phone ? 360 : 560}
          describe={describe}
        />

        <aside className="space-y-3" aria-label={t("vision.map.legend")}>
          {picked && (
            <div className="space-y-2 rounded-md border border-line bg-surface-1 p-3" aria-live="polite">
              <p className="text-[13px] text-ink-1">
                <span className="font-medium tabular-nums">{t("vision.map.selected", { count: formatCount(picked.rows.length) })}</span>
                <span className="block text-xs text-ink-2">{picked.label}</span>
              </p>
              <div className="flex flex-wrap gap-1.5">
                <Button variant="primary" icon={GalleryHorizontalEnd} onClick={() => onShow(picked)} disabled={picked.rows.length === 0}>
                  {t("vision.map.showInGallery")}
                </Button>
                <Button variant="ghost" icon={X} onClick={() => setPicked(null)}>
                  {t("vision.map.clear")}
                </Button>
              </div>
            </div>
          )}

          <div className="space-y-1.5">
            <p className="text-xs font-medium text-ink-2">{t(COLOUR_LABEL[by])}</p>
            {colouring.kind === "categorical" ? (
              <ul className="space-y-0.5">
                {colouring.entries.map((entry) => (
                  <LegendRow
                    key={entry.key}
                    name={name(entry.key)}
                    colour={entry.slot >= 0 ? (palette[entry.slot] ?? null) : null}
                    count={entry.count}
                    active={focus === entry.key}
                    faded={focus !== null && focus !== entry.key}
                    onFocus={() => setFocus(focus === entry.key ? null : entry.key)}
                    onSelect={() => pickEntry(entry.key)}
                  />
                ))}
              </ul>
            ) : (
              <SequentialLegend min={colouring.min} max={colouring.max} log={colouring.log} />
            )}
            {!picked && <p className="text-xs text-ink-3">{phone ? t("vision.map.howTouch") : t("vision.map.how")}</p>}
          </div>

          {map.notes.length > 0 && (
            <ul className="space-y-1 border-t border-line pt-2 text-xs text-ink-3">
              {map.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          )}
        </aside>
      </div>
    </div>
  );
}

function LegendRow({ name, colour, count, active, faded, onFocus, onSelect }: { name: string; colour: string | null; count: number; active: boolean; faded: boolean; onFocus: () => void; onSelect: () => void }) {
  return (
    <li className={`group flex items-center gap-1 rounded-md motion-safe:transition-opacity ${faded ? "opacity-45" : ""}`}>
      <button
        type="button"
        aria-pressed={active}
        onClick={onFocus}
        className="flex min-h-7 min-w-0 flex-1 items-center gap-2 rounded-md px-1.5 text-left text-[13px] text-ink-1 hover:bg-surface-2 aria-pressed:bg-surface-2 pointer-coarse:min-h-11"
      >
        {/* the canvas's colour for this entry (the palette chosen in Settings, 047); neutral: ink-3 */}
        <span className={`size-2.5 shrink-0 rounded-full ${colour ? "" : "bg-ink-3"}`} style={colour ? { background: colour } : undefined} aria-hidden />
        <span className="min-w-0 flex-1 truncate">{name}</span>
        <span className="text-xs text-ink-3 tabular-nums">{formatCount(count)}</span>
      </button>
      <button
        type="button"
        onClick={onSelect}
        aria-label={t("vision.map.selectEntry", { name })}
        title={t("vision.map.selectEntry", { name })}
        className="relative inline-flex size-7 shrink-0 items-center justify-center rounded-md text-ink-3 opacity-0 group-hover:opacity-100 hover:bg-surface-2 hover:text-ink-1 focus-visible:opacity-100 pointer-coarse:size-11 pointer-coarse:opacity-100"
      >
        <ChevronRight size={14} strokeWidth={1.5} aria-hidden />
      </button>
    </li>
  );
}

function SequentialLegend({ min, max, log }: { min: number; max: number; log: boolean }) {
  const gradient = useMemo(() => rampGradient(sequentialStops(readTokens(), MARK_FLOOR)), []);
  return (
    <div className="space-y-1">
      <span className="block h-2.5 w-full rounded-[2px] outline outline-1 -outline-offset-1 outline-line" style={{ background: gradient }} aria-hidden />
      <span className="flex justify-between text-xs text-ink-2 tabular-nums">
        <span>{formatNumber(min)}</span>
        {log && <span className="text-ink-3">{t("vision.map.logScale")}</span>}
        <span>{formatNumber(max)}</span>
      </span>
    </div>
  );
}

// --- Advanced (R-VMAP-03, R-VMAP-04): t-SNE, HDBSCAN, a forced k, the silhouette by k -------------

function Advanced({ search, set, map, projections }: { search: VisionSearch; set: (patch: VisionSearch) => void; map: ImageMap | null; projections: MapProjection[] }) {
  const [open, setOpen] = useState(Boolean(search.proj || search.groups || search.k));
  const scores = map?.scores ?? [];
  const extra = projections.find((projection) => projection.key === search.proj);
  return (
    <details open={open} className="group rounded-md border border-line bg-surface-1" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary className="flex min-h-11 cursor-pointer list-none items-center gap-2 px-3 py-2 select-none hover:bg-surface-2 [&::-webkit-details-marker]:hidden">
        <ChevronRight size={16} strokeWidth={1.5} aria-hidden className="shrink-0 text-ink-3 transition-transform group-open:rotate-90" />
        <span className="text-[13px] font-medium text-ink-1">{t("vision.map.advanced")}</span>
        <span className="text-xs text-ink-3">{t("vision.map.advancedHint")}</span>
      </summary>
      {open && (
        <div className="space-y-4 border-t border-line px-3 py-3">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <div className="space-y-1">
              <span className="flex items-center gap-1 text-xs text-ink-2">
                {t("vision.map.projection")}
                <InfoTip label={t("vision.map.projection")} text={t("vision.map.projectionTip")} />
              </span>
              <SegmentedControl<Projection>
                label={t("vision.map.projection")}
                value={search.proj ?? "pca"}
                onChange={(value) => set({ proj: value === "pca" ? undefined : value })}
                options={[
                  { value: "pca", label: t("vision.map.pca") },
                  { value: "tsne", label: t("vision.map.tsne") },
                  ...(extra ? [{ value: extra.key, label: extra.title }] : []),
                ]}
              />
            </div>
            <div className="space-y-1">
              <span className="flex items-center gap-1 text-xs text-ink-2">
                {t("vision.map.clustering")}
                <InfoTip label={t("vision.map.clustering")} text={t("vision.map.clusteringTip")} />
              </span>
              <SegmentedControl<Clustering>
                label={t("vision.map.clustering")}
                value={search.groups ?? "kmeans"}
                onChange={(value) => set({ groups: value === "kmeans" ? undefined : value, k: undefined })}
                options={[
                  { value: "kmeans", label: t("vision.map.kmeans") },
                  { value: "hdbscan", label: t("vision.map.hdbscan") },
                ]}
              />
            </div>
            <Field label={t("vision.map.k")}>
              <Select
                value={search.k ? String(search.k) : "auto"}
                disabled={search.groups === "hdbscan"}
                onChange={(event) => set({ k: event.target.value === "auto" ? undefined : Number(event.target.value) })}
              >
                <option value="auto">{t("vision.map.kAuto")}</option>
                {Array.from({ length: 11 }, (_, i) => i + 2).map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </Select>
            </Field>
          </div>
          <MoreProjections projections={projections} current={search.proj} onPick={(key) => set({ proj: key })} />
          {scores.length > 0 && map && (
            <div className="space-y-1">
              <p className="text-[13px] text-ink-1">{t("vision.map.silhouetteSentence", { k: String(map.k), s: formatNumber(map.silhouette ?? 0) })}</p>
              <Plot figure={silhouetteFigure(scores, map.k)} height={200} label={t("vision.map.silhouetteChart")} />
            </div>
          )}
        </div>
      )}
    </details>
  );
}

/** c3, R-VMAP-12: PaCMAP and TriMAP, in Advanced › More; computed only on a click, greyed with
 *  their reason and install command when the extra `interpret` cannot run here (R-XAI-10). */
function MoreProjections({ projections, current, onPick }: { projections: MapProjection[]; current: VisionSearch["proj"]; onPick: (key: MapProjection["key"]) => void }) {
  const [open, setOpen] = useState(current === "pacmap" || current === "trimap");
  if (projections.length === 0) return null;
  return (
    <details open={open} className="text-xs" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary className="w-fit cursor-pointer text-[13px] text-accent select-none hover:underline">{t("vision.map.more")}</summary>
      {open && (
        <div className="mt-2 space-y-3">
          <p className="max-w-prose text-xs text-ink-2">{t("vision.map.moreHint")}</p>
          {projections.map((projection) => (
            <div key={projection.key} className="space-y-1">
              <Button variant="ghost" disabled={!projection.available || current === projection.key} title={projection.reason ?? undefined} onClick={() => onPick(projection.key)}>
                {t("vision.map.draw", { method: projection.title })}
              </Button>
              <span className="ml-2 text-xs text-ink-3">{projection.licence}</span>
              {!projection.available && projection.reason && <p className="max-w-prose text-xs text-ink-2">{projection.reason}</p>}
              {!projection.available && projection.install && !LITE && (
                <p className="text-xs text-ink-3">
                  {t("vision.map.install")} <code className="font-mono text-ink-2">{projection.install}</code>
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </details>
  );
}

function silhouetteFigure(scores: ImageMap["scores"], chosen: number) {
  return (tokens: ReturnType<typeof readTokens>) => ({
    data: [
      {
        type: "bar" as const,
        x: scores.map((s) => s.k),
        y: scores.map((s) => s.silhouette ?? 0),
        marker: { color: scores.map((s) => (s.k === chosen ? tokens.palette[0] : tokens.line)) },
        hovertemplate: t("fig.vision.hoverSilhouette"),
      },
    ],
    layout: {
      bargap: 0.35,
      xaxis: { title: { text: "k" }, dtick: 1 },
      yaxis: { title: { text: t("vision.map.silhouette") }, rangemode: "tozero" as const },
      margin: { l: 48, r: 8, t: 8, b: 36 },
    },
  });
}
