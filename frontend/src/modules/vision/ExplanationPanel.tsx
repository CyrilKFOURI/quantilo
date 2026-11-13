import { ArrowLeft, Check, CircleAlert, ShieldAlert, ShieldCheck, X } from "lucide-react";
import { useMemo, useState } from "react";

import { Folded } from "../../features/text/Folded";
import { imageUrl } from "../../features/import/imageSetApi";
import { t } from "../../lib/i18n";
import { BeforeAfter } from "../../ui/BeforeAfter";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { ErrorNote, Select } from "../../ui/fields";
import { InfoTip } from "../../ui/InfoTip";
import { Loading } from "../../ui/Loading";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { CompareStrip, type StripPanel } from "./CompareStrip";
import {
  useVisionRuns,
  useWhy,
  useWhyMap,
  useWhySanity,
  type MapMethod,
  type RunSummary,
  type SanityMethod,
  type WhyExplanation,
  type WhyImage,
  type WhyMap,
  type WhyMethod,
} from "./explainApi";
import { formatValue } from "./heatmap";
import { DEFAULT_HEAT, HeatmapControls, HeatmapOverlay, useOpacityKeys, type HeatSettings } from "./HeatmapOverlay";
import type { ImageRow } from "./model";
import { ConceptsBlock, LooksLike } from "./MoreExplain";

/**
 * 117 (study §§ 8, 9.2, 9.3): "Why?" on one image. The model's own evidence comes first and is
 * computed when the panel opens: its prediction, the training images nearest in the space it
 * reads, the closest prototype of the class it chose; for anomalies, the nearest normal images
 * (the score itself) and the patch PCA's map. Advanced holds the counterfactual example,
 * PatchCore's nearest normal patch and the post-hoc maps (occlusion, Grad-CAM, attention
 * rollout), each with its written caveat; Advanced › More the costly ones (saliency, SmoothGrad,
 * Integrated Gradients) and the sanity test. Nothing folded is asked of the server before it is
 * opened and chosen (the onion rule).
 */

type PanelProps = {
  datasetId: string;
  row: ImageRow;
  /** Every image of the set, to name the evidence. */
  rows: readonly ImageRow[];
  run: string | null;
  model: string | null;
  onRun: (run: string | undefined) => void;
  onModel: (model: string | undefined) => void;
  /** Open another image's "Why?" (an example, a prototype). */
  onWhy: (row: ImageRow) => void;
};

const POST_HOC: readonly MapMethod[] = ["occlusion", "grad_cam", "rollout"];
const COSTLY: readonly MapMethod[] = ["saliency", "smoothgrad", "integrated_gradients"];

const pct = (value: number | null | undefined) =>
  value === null || value === undefined ? "-" : value >= 0.995 && value < 1 ? "> 0.99" : value.toFixed(2);

export function WhyDialog({ onClose, onBack, ...props }: PanelProps & { onClose: () => void; onBack: () => void }) {
  return (
    <Dialog
      open
      onOpenChange={(open) => !open && onClose()}
      title={t("vision.why.title", { name: props.row.name })}
      description={t("vision.why.subtitle")}
      size="lg"
      footer={
        <div className="flex w-full items-center justify-between gap-2">
          <Button variant="ghost" icon={ArrowLeft} onClick={onBack}>
            {t("vision.why.back")}
          </Button>
          <Button variant="secondary" onClick={onClose}>
            {t("vision.why.close")}
          </Button>
        </div>
      }
    >
      <ExplanationPanel {...props} />
    </Dialog>
  );
}

export function ExplanationPanel({ datasetId, row, rows, run, model, onRun, onModel, onWhy }: PanelProps) {
  const runs = useVisionRuns(datasetId);
  const explained = (runs.data ?? []).filter((summary) => summary.task !== "clusters");
  const chosen = explained.find((summary) => summary.id === run) ?? explained[0] ?? null;
  const why = useWhy(datasetId, chosen?.id ?? null, row.id, model);

  if (runs.isPending) return <Loading label={t("vision.why.loadingRuns")} className="h-48 rounded-lg" />;
  if (runs.isError) return <ErrorNote>{(runs.error as Error).message}</ErrorNote>;
  if (!chosen) return <NoRun />;

  return (
    <div className="min-w-0 space-y-6 text-[13px]">
      <RunBar runs={explained} chosen={chosen} why={why.data ?? null} onRun={onRun} onModel={onModel} />
      {why.isPending ? (
        <Loading label={t("vision.why.loading")} className="h-72 rounded-lg" />
      ) : why.isError ? (
        <ErrorNote>{(why.error as Error).message}</ErrorNote>
      ) : (
        <Explained key={`${why.data.run_id}-${why.data.model.key}-${row.id}`} datasetId={datasetId} row={row} rows={rows} why={why.data} onWhy={onWhy} />
      )}
    </div>
  );
}

function NoRun() {
  return (
    <div className="rounded-lg border border-dashed border-line px-5 py-8 text-center">
      <p className="font-medium text-ink-1">{t("vision.why.noRun")}</p>
      <p className="mx-auto mt-1 max-w-[52ch] text-ink-2">{t("vision.why.noRunHint")}</p>
    </div>
  );
}

function RunBar({
  runs,
  chosen,
  why,
  onRun,
  onModel,
}: {
  runs: readonly RunSummary[];
  chosen: RunSummary;
  why: WhyExplanation | null;
  onRun: (run: string | undefined) => void;
  onModel: (model: string | undefined) => void;
}) {
  return (
    <div className="flex min-w-0 flex-wrap items-end gap-x-4 gap-y-2">
      {runs.length > 1 && (
        <label className="flex min-w-0 flex-col gap-1">
          <span className="text-xs text-ink-2">{t("vision.why.run")}</span>
          <Select value={chosen.id} onChange={(event) => onRun(event.target.value === runs[0]?.id ? undefined : event.target.value)} className="max-w-full">
            {runs.map((summary) => (
              <option key={summary.id} value={summary.id}>
                {new Date(summary.created_at).toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" })} · {summary.default_title ?? summary.task}
              </option>
            ))}
          </Select>
        </label>
      )}
      {why && why.models.length > 1 && (
        <label className="flex min-w-0 flex-col gap-1">
          <span className="text-xs text-ink-2">{t("vision.why.model")}</span>
          <Select
            value={why.model.key}
            onChange={(event) => onModel(why.models.find((m) => m.key === event.target.value)?.default ? undefined : event.target.value)}
            className="max-w-full"
          >
            {why.models.map((option) => (
              <option key={option.key} value={option.key}>
                {option.title}
                {option.default ? ` (${t("vision.why.runDefault")})` : ""}
              </option>
            ))}
          </Select>
        </label>
      )}
      {why && why.models.length <= 1 && <p className="text-ink-2">{why.model.title}</p>}
    </div>
  );
}

// --- the explanation itself ------------------------------------------------------------------

type ExplainedProps = { datasetId: string; row: ImageRow; rows: readonly ImageRow[]; why: WhyExplanation; onWhy: (row: ImageRow) => void };

function Explained({ datasetId, row, rows, why, onWhy }: ExplainedProps) {
  const byId = useMemo(() => new Map(rows.map((r) => [r.id, r])), [rows]);
  const evidence = why.evidence;
  const pool: { ref: WhyImage; kind: "nearest" | "prototype" }[] = [
    ...evidence.nearest.map((ref) => ({ ref, kind: "nearest" as const })),
    ...(evidence.prototype ? [{ ref: evidence.prototype, kind: "prototype" as const }] : []),
  ];
  const [picked, setPicked] = useState(0);
  const current = pool[Math.min(picked, pool.length - 1)];
  const anomaly = why.task === "anomaly";
  // an anomaly map is low almost everywhere: start above its floor, so the image stays readable
  const [heat, setHeat] = useState<HeatSettings>(anomaly ? { ...DEFAULT_HEAT, threshold: 0.35 } : DEFAULT_HEAT);
  // the model's own map, first plane: the patch PCA's (refitted) or the one a detector kept (113)
  const own = useWhyMap(datasetId, why.run_id, row.id, why.model.key, anomaly ? ((why.own_map ?? null) as MapMethod | null) : null, null);
  const patch = why.patch ?? null;
  useOpacityKeys(heat, setHeat, anomaly && Boolean(own.data));

  const source = imageUrl(datasetId, row.id, "file");
  const nameOf = (ref: WhyImage) => byId.get(ref.image_id)?.name ?? ref.image_id.slice(0, 8);
  const openRef = (ref: WhyImage) => {
    const target = byId.get(ref.image_id);
    return target ? () => onWhy(target) : undefined;
  };

  const panels: StripPanel[] = [{ key: "image", label: t("vision.why.thisImage"), src: source, alt: row.name }];
  if (anomaly && own.data) {
    panels.push({
      key: "own",
      label: own.data.title,
      src: source,
      alt: row.name,
      heat: { map: own.data, settings: heat },
      boxes: patch ? [{ box: patch.box, label: t("vision.why.unusualArea") }] : [],
    });
  }
  if (patch) {
    panels.push({
      key: "normal-patch",
      label: t("vision.why.nearestNormal", { name: nameOf(patch.normal) }),
      src: imageUrl(datasetId, patch.normal.image_id, "file"),
      alt: nameOf(patch.normal),
      boxes: [{ box: patch.normal_box, label: t("vision.why.normalPatch") }],
      caption: <span className="tabular-nums">{t("vision.why.distance", { value: formatValue(patch.distance) })}</span>,
      onOpen: openRef(patch.normal),
    });
  }
  if (current) {
    panels.push({
      key: "evidence",
      label: current.kind === "prototype" ? t("vision.why.prototypeOf", { label: current.ref.label ?? "" }) : nameOf(current.ref),
      src: imageUrl(datasetId, current.ref.image_id, "file"),
      alt: nameOf(current.ref),
      caption: <RefCaption ref_={current.ref} anomaly={anomaly} />,
      onOpen: openRef(current.ref),
    });
  }

  return (
    <div className="min-w-0 space-y-6">
      <PredictionBlock why={why} />

      {/* 118: the More models' own evidence, first: it is how they computed the answer */}
      {why.model.key === "protopnet" && <LooksLike datasetId={datasetId} runId={why.run_id} row={row} nameOf={nameOf} openRef={openRef} />}
      {why.model.key === "concept_bottleneck" && <ConceptsBlock datasetId={datasetId} runId={why.run_id} row={row} />}

      <section className="min-w-0 space-y-3" aria-labelledby="why-evidence">
        <div className="space-y-1">
          <h3 id="why-evidence" className="flex items-center gap-1.5 text-sm font-semibold text-ink-1">
            {t("vision.why.evidence")}
            <InfoTip label={t("vision.why.evidence")} text={t("vision.why.evidenceTip")} />
          </h3>
          <p className="max-w-[75ch] text-ink-1">{evidence.sentence}</p>
          <p className="max-w-[75ch] text-xs text-ink-3">{t("vision.why.space", { space: evidence.space })}</p>
        </div>
        <CompareStrip panels={panels} label={t("vision.why.stripLabel")} />
        {patch && <p className="max-w-[75ch] text-ink-1">{patch.sentence}</p>}
        {anomaly && why.own_map && (own.isPending ? (
          <Loading compact label={t("vision.why.loadingOwnMap")} className="h-10 rounded-md" />
        ) : own.isError ? (
          <ErrorNote>{(own.error as Error).message}</ErrorNote>
        ) : (
          own.data && (
            <div className="space-y-2">
              <p className="max-w-[75ch] text-ink-1">{own.data.sentence}</p>
              <HeatmapControls map={own.data} settings={heat} onChange={setHeat} />
              <OwnNote />
            </div>
          )
        ))}
        {pool.length > 0 && (
          <div className="space-y-1.5">
            <p className="text-xs text-ink-2">{t(anomaly ? "vision.why.pickNormals" : "vision.why.pickEvidence")}</p>
            <ul className="flex min-w-0 gap-2 overflow-x-auto pb-1">
              {pool.map((item, index) => (
                <li key={`${item.kind}-${item.ref.row}`} className="shrink-0">
                  <button
                    type="button"
                    aria-pressed={index === picked}
                    onClick={() => setPicked(index)}
                    title={nameOf(item.ref)}
                    className="group/ref block w-[84px] rounded-md text-left focus-visible:outline-2 focus-visible:outline-accent"
                  >
                    <span className="relative block size-[84px] overflow-hidden rounded-md bg-surface-2">
                      <img src={imageUrl(datasetId, item.ref.image_id, "thumb")} alt={nameOf(item.ref)} loading="lazy" decoding="async" className="size-full object-cover" />
                      <span
                        aria-hidden
                        className={`pointer-events-none absolute inset-0 rounded-md ring-inset ${index === picked ? "ring-2 ring-accent" : "ring-1 ring-ink-1/8 group-hover/ref:ring-ink-1/30"}`}
                      />
                    </span>
                    <span className="mt-1 block truncate text-[11px] text-ink-1">
                      {item.kind === "prototype" ? t("vision.why.prototype") : (item.ref.label ?? nameOf(item.ref))}
                    </span>
                    <span className="block text-[11px] text-ink-3 tabular-nums">
                      {anomaly ? t("vision.why.distanceShort", { value: formatValue(item.ref.distance ?? 0) }) : t("vision.why.similarityShort", { value: (item.ref.similarity ?? 0).toFixed(2) })}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
            {evidence.prototype_sentence && <p className="max-w-[75ch] text-xs text-ink-2">{evidence.prototype_sentence}</p>}
          </div>
        )}
        {why.notes.map((note) => (
          <p key={note} className="flex max-w-[75ch] gap-1.5 text-xs text-ink-2">
            <CircleAlert size={13} strokeWidth={1.5} className="mt-px shrink-0 text-warn" aria-hidden />
            {note}
          </p>
        ))}
      </section>

      {(!anomaly || why.methods.some((method) => method.level !== "first")) && (
        <Folded title={t("vision.why.advanced")} hint={t(anomaly ? "vision.why.advancedHintAnomaly" : "vision.why.advancedHint")}>
          <Advanced datasetId={datasetId} row={row} why={why} nameOf={nameOf} openRef={openRef} />
        </Folded>
      )}
    </div>
  );
}

function RefCaption({ ref_, anomaly }: { ref_: WhyImage; anomaly: boolean }) {
  return (
    <span className="flex flex-wrap gap-x-2 tabular-nums">
      {ref_.label && <span className="font-mono text-ink-1">{ref_.label}</span>}
      {anomaly ? (
        <span>{t("vision.why.distance", { value: formatValue(ref_.distance ?? 0) })}</span>
      ) : (
        ref_.similarity !== null && ref_.similarity !== undefined && <span>{t("vision.why.similarity", { value: ref_.similarity.toFixed(2) })}</span>
      )}
      {ref_.share !== null && ref_.share !== undefined && <span>{t("vision.why.stands", { share: `${Math.round(ref_.share * 100)} %` })}</span>}
    </span>
  );
}

function OwnNote() {
  return (
    <p className="flex max-w-[75ch] gap-1.5 text-xs text-ink-2">
      <ShieldCheck size={13} strokeWidth={1.5} className="mt-px shrink-0 text-ok" aria-hidden />
      {t("vision.why.ownMap")}
    </p>
  );
}

function PredictionBlock({ why }: { why: WhyExplanation }) {
  const p = why.prediction;
  const verdict =
    p.correct === null || p.correct === undefined ? null : p.correct ? (
      <span className="inline-flex items-center gap-1 text-ok">
        <Check size={14} strokeWidth={1.75} aria-hidden />
        {t("vision.why.right")}
      </span>
    ) : (
      <span className="inline-flex items-center gap-1 text-bad">
        <X size={14} strokeWidth={1.75} aria-hidden />
        {t("vision.why.wrong", { truth: p.truth ?? "" })}
      </span>
    );
  return (
    <section className="min-w-0 space-y-2" aria-label={t("vision.why.prediction")}>
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        {why.task === "classification" ? (
          <>
            <span className="font-mono text-base font-semibold text-ink-1">{p.label}</span>
            <span className="text-base text-ink-1 tabular-nums">{pct(p.probability)}</span>
          </>
        ) : (
          <>
            <span className="text-base font-semibold text-ink-1">{p.flagged ? t("vision.why.unusual") : t("vision.why.normal")}</span>
            <span className="text-ink-2 tabular-nums">
              {t("vision.why.score", { score: formatValue(p.score ?? 0), threshold: p.threshold === null || p.threshold === undefined ? "-" : formatValue(p.threshold) })}
            </span>
          </>
        )}
        {verdict}
        {!p.tested && <span className="text-xs text-ink-3">{t("vision.why.trained")}</span>}
      </div>
      <p className="max-w-[75ch] text-ink-2">{p.sentence}</p>
      {p.probabilities.length > 1 && (
        <ul className="grid max-w-md grid-cols-[minmax(0,auto)_minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 text-xs" aria-label={t("vision.why.probabilities")}>
          {p.probabilities.map((item, index) => (
            <li key={item.label} className="contents">
              <span className="truncate font-mono text-ink-1">{item.label}</span>
              <span className="h-1.5 min-w-0" aria-hidden>
                {/* one series, no background track: the bar's length is the probability */}
                <span
                  className={`block h-full rounded-r-[2px] ${index === 0 ? "bg-(--chart-1)" : "bg-ink-3/60"}`}
                  style={{ width: `${Math.max(1, item.probability * 100)}%` }}
                />
              </span>
              <span className="text-right text-ink-1 tabular-nums">{pct(item.probability)}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

// --- Advanced and More -----------------------------------------------------------------------

function Advanced({
  datasetId,
  row,
  why,
  nameOf,
  openRef,
}: {
  datasetId: string;
  row: ImageRow;
  why: WhyExplanation;
  nameOf: (ref: WhyImage) => string;
  openRef: (ref: WhyImage) => (() => void) | undefined;
}) {
  const methods = new Map(why.methods.map((method) => [method.key, method]));
  const anomaly = why.task === "anomaly";
  const classes = why.prediction.probabilities.map((item) => item.label);
  const [method, setMethod] = useState<MapMethod | null>(null);
  const [target, setTarget] = useState<string | null>(null);
  const source = imageUrl(datasetId, row.id, "file");

  return (
    <div className="min-w-0 space-y-6">
      {why.counterfactual && (
        <section className="min-w-0 space-y-2">
          <h4 className="text-[13px] font-semibold text-ink-1">{t("vision.why.counterfactual")}</h4>
          <p className="max-w-[75ch] text-ink-2">{why.counterfactual.sentence}</p>
          <BeforeAfter
            label={t("vision.why.counterfactual")}
            panels={[
              { src: source, label: t("vision.why.thisImageAs", { label: why.prediction.label ?? "" }), alt: row.name },
              {
                src: imageUrl(datasetId, why.counterfactual.image.image_id, "file"),
                label: t("vision.why.nearestOf", { label: why.counterfactual.image.label ?? "" }),
                alt: nameOf(why.counterfactual.image),
                caption: (
                  <span className="tabular-nums">
                    {t("vision.why.similarity", { value: (why.counterfactual.image.similarity ?? 0).toFixed(2) })}
                    {openRef(why.counterfactual.image) && (
                      <button type="button" onClick={openRef(why.counterfactual.image)} className="ml-2 text-accent hover:underline">
                        {t("vision.why.openWhy")}
                      </button>
                    )}
                  </span>
                ),
              },
            ]}
            defaultMode="side"
          />
        </section>
      )}

      {anomaly && <PatchCoreSection datasetId={datasetId} row={row} why={why} method={methods.get("patchcore")} nameOf={nameOf} />}

      {!anomaly && (
        <section className="min-w-0 space-y-3">
          {/* no InfoTip inside a fold: the fold is a hover group of its own; the caveat is written */}
          <h4 className="text-[13px] font-semibold text-ink-1">{t("vision.why.maps")}</h4>
          <Warning />
          <div className="flex flex-wrap items-end gap-x-4 gap-y-2">
            <MethodButtons methods={POST_HOC.map((key) => methods.get(key)).filter((m): m is WhyMethod => Boolean(m))} value={method} onChange={setMethod} />
            {classes.length > 1 && (
              <label className="flex flex-col gap-1">
                <span className="text-xs text-ink-2">{t("vision.why.target")}</span>
                <Select value={target ?? ""} onChange={(event) => setTarget(event.target.value || null)}>
                  <option value="">{t("vision.why.targetPredicted", { label: why.prediction.label ?? "" })}</option>
                  {classes.map((label) => (
                    <option key={label} value={label}>
                      {label}
                    </option>
                  ))}
                </Select>
              </label>
            )}
          </div>
          <Unavailable methods={POST_HOC.map((key) => methods.get(key)).filter((m): m is WhyMethod => Boolean(m))} />
          {method && (POST_HOC.includes(method) || COSTLY.includes(method)) && (
            <MapView datasetId={datasetId} row={row} why={why} method={method} target={target} />
          )}
          <Folded nested title={t("vision.why.more")} hint={t("vision.why.moreHint")}>
            <div className="min-w-0 space-y-5">
              <div className="space-y-2">
                <p className="max-w-[75ch] text-ink-2">{t("vision.why.costlyIntro")}</p>
                <MethodButtons methods={COSTLY.map((key) => methods.get(key)).filter((m): m is WhyMethod => Boolean(m))} value={method} onChange={setMethod} />
                <Unavailable methods={COSTLY.map((key) => methods.get(key)).filter((m): m is WhyMethod => Boolean(m))} />
                {method && COSTLY.includes(method) && <p className="text-xs text-ink-3">{t("vision.why.shownAbove")}</p>}
              </div>
              <SanitySection datasetId={datasetId} row={row} why={why} method={methods.get("sanity")} />
            </div>
          </Folded>
        </section>
      )}
    </div>
  );
}

function Warning({ text }: { text?: string }) {
  return (
    <p className="flex max-w-[75ch] gap-2 rounded-md bg-surface-2 px-3 py-2 text-xs leading-relaxed text-ink-2">
      <ShieldAlert size={14} strokeWidth={1.5} className="mt-0.5 shrink-0 text-warn" aria-hidden />
      <span>{text ?? t("vision.why.warning")}</span>
    </p>
  );
}

function MethodButtons({ methods, value, onChange }: { methods: readonly WhyMethod[]; value: MapMethod | null; onChange: (method: MapMethod) => void }) {
  const usable = methods.filter((method) => method.available);
  if (usable.length === 0) return null;
  return (
    <div className="space-y-1">
      <span className="block text-xs text-ink-2">{t("vision.why.method")}</span>
      <div role="radiogroup" aria-label={t("vision.why.method")} className="inline-flex flex-wrap rounded-md border border-line bg-surface-2 p-0.5">
        {usable.map((method) => {
          const selected = method.key === value;
          return (
            <button
              key={method.key}
              type="button"
              role="radio"
              aria-checked={selected}
              aria-label={`${method.title}, ${method.cost}`}
              onClick={() => onChange(method.key as MapMethod)}
              className={`inline-flex h-[calc(var(--control-h)-6px)] items-center gap-1.5 rounded-[4px] px-2 text-xs transition-colors pointer-coarse:min-h-11 ${
                selected ? "bg-surface-1 text-ink-1 shadow-[0_0_0_1px_var(--line)]" : "text-ink-2 hover:text-ink-1"
              }`}
            >
              {method.title}
              <span className="text-[10px] text-ink-3">{method.cost}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function Unavailable({ methods }: { methods: readonly WhyMethod[] }) {
  const off = methods.filter((method) => !method.available);
  if (off.length === 0) return null;
  const reasons = [...new Set(off.map((method) => method.reason ?? ""))];
  return (
    <div className="space-y-1">
      {reasons.map((reason) => (
        <p key={reason} className="max-w-[75ch] text-xs text-ink-3">
          {t("vision.why.unavailable", { methods: off.filter((m) => (m.reason ?? "") === reason).map((m) => m.title).join(", "), reason })}
        </p>
      ))}
    </div>
  );
}

function MapView({ datasetId, row, why, method, target }: { datasetId: string; row: ImageRow; why: WhyExplanation; method: MapMethod; target: string | null }) {
  const map = useWhyMap(datasetId, why.run_id, row.id, why.model.key, method, target);
  const [heat, setHeat] = useState<HeatSettings>(DEFAULT_HEAT);
  useOpacityKeys(heat, setHeat, Boolean(map.data));
  if (map.isPending) return <Loading label={t("vision.why.loadingMap", { method: methodTitle(why, method) })} className="h-60 rounded-lg" />;
  if (map.isError) return <ErrorNote>{(map.error as Error).message}</ErrorNote>;
  return <MapResult datasetId={datasetId} row={row} why={why} map={map.data} heat={heat} onHeat={setHeat} />;
}

const methodTitle = (why: WhyExplanation, method: string) => why.methods.find((m) => m.key === method)?.title ?? method;

function MapResult({
  datasetId,
  row,
  why,
  map,
  heat,
  onHeat,
}: {
  datasetId: string;
  row: ImageRow;
  why: WhyExplanation;
  map: WhyMap;
  heat: HeatSettings;
  onHeat: (heat: HeatSettings) => void;
}) {
  const source = imageUrl(datasetId, row.id, "file");
  const nearest = why.evidence.nearest[0];
  const panels: StripPanel[] = [
    { key: "image", label: t("vision.why.thisImage"), src: source, alt: row.name },
    { key: "map", label: map.target ? t("vision.why.mapFor", { title: map.title, label: map.target }) : map.title, src: source, alt: row.name, heat: { map, settings: heat } },
  ];
  if (nearest) {
    panels.push({
      key: "nearest",
      label: t("vision.why.nearestExample"),
      src: imageUrl(datasetId, nearest.image_id, "file"),
      alt: nearest.label ?? nearest.image_id,
      caption: <RefCaption ref_={nearest} anomaly={false} />,
    });
  }
  return (
    <div className="min-w-0 space-y-3">
      <CompareStrip panels={panels} label={t("vision.why.stripLabel")} />
      <p className="max-w-[75ch] text-ink-1">{map.sentence}</p>
      <HeatmapControls map={map} settings={heat} onChange={onHeat} />
      {map.notes.map((note) => (
        <p key={note} className="max-w-[75ch] text-xs text-ink-2">
          {note}
        </p>
      ))}
      {map.warning && <Warning text={map.warning} />}
      <p className="text-xs text-ink-3 tabular-nums">{t(map.cached ? "vision.why.cached" : "vision.why.computed", { seconds: map.seconds.toFixed(1) })}</p>
    </div>
  );
}

function PatchCoreSection({ datasetId, row, why, method, nameOf }: { datasetId: string; row: ImageRow; why: WhyExplanation; method: WhyMethod | undefined; nameOf: (ref: WhyImage) => string }) {
  const [asked, setAsked] = useState(false);
  const map = useWhyMap(datasetId, why.run_id, row.id, why.model.key, asked ? "patchcore" : null, null);
  const [heat, setHeat] = useState<HeatSettings>({ ...DEFAULT_HEAT, threshold: 0.5, mode: "contours" });
  useOpacityKeys(heat, setHeat, Boolean(map.data));
  if (!method) return null;
  const patch = map.data?.patch ?? null;
  return (
    <section className="min-w-0 space-y-3">
      <h4 className="text-[13px] font-semibold text-ink-1">{t("vision.why.patchcore")}</h4>
      <p className="max-w-[75ch] text-ink-2">{t("vision.why.patchcoreIntro")}</p>
      {!method.available ? (
        <p className="max-w-[75ch] text-xs text-ink-3">{method.reason}</p>
      ) : !asked ? (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <Button variant="secondary" onClick={() => setAsked(true)}>
            {t("vision.why.patchcoreRun")}
          </Button>
          <span className="text-xs text-ink-3">{method.cost}</span>
        </div>
      ) : map.isPending ? (
        <Loading label={t("vision.why.loadingPatchcore")} className="h-60 rounded-lg" />
      ) : map.isError ? (
        <ErrorNote>{(map.error as Error).message}</ErrorNote>
      ) : (
        <div className="min-w-0 space-y-3">
          <CompareStrip
            label={t("vision.why.patchcore")}
            panels={[
              {
                key: "image",
                label: t("vision.why.thisImage"),
                src: imageUrl(datasetId, row.id, "file"),
                alt: row.name,
                heat: { map: map.data, settings: heat },
                boxes: patch ? [{ box: patch.box, label: t("vision.why.unusualArea") }] : [],
              },
              ...(patch
                ? [
                    {
                      key: "normal",
                      label: t("vision.why.nearestNormal", { name: nameOf(patch.normal) }),
                      src: imageUrl(datasetId, patch.normal.image_id, "file"),
                      alt: nameOf(patch.normal),
                      boxes: [{ box: patch.normal_box, label: t("vision.why.normalPatch") }],
                      caption: <span className="tabular-nums">{t("vision.why.distance", { value: formatValue(patch.distance) })}</span>,
                    },
                  ]
                : []),
            ]}
          />
          {patch && <p className="max-w-[75ch] text-ink-1">{patch.sentence}</p>}
          <p className="max-w-[75ch] text-ink-2">{map.data.sentence}</p>
          <HeatmapControls map={map.data} settings={heat} onChange={setHeat} />
          <OwnNote />
          <p className="text-xs text-ink-3 tabular-nums">{t(map.data.cached ? "vision.why.cached" : "vision.why.computed", { seconds: map.data.seconds.toFixed(1) })}</p>
        </div>
      )}
    </section>
  );
}

function SanitySection({ datasetId, row, why, method }: { datasetId: string; row: ImageRow; why: WhyExplanation; method: WhyMethod | undefined }) {
  const [choice, setChoice] = useState<SanityMethod>("grad_cam");
  const [asked, setAsked] = useState<SanityMethod | null>(null);
  const result = useWhySanity(datasetId, why.run_id, row.id, why.model.key, asked);
  if (!method) return null;
  const source = imageUrl(datasetId, row.id, "file");
  return (
    <section className="min-w-0 space-y-3">
      <h4 className="text-[13px] font-semibold text-ink-1">{t("vision.why.sanity")}</h4>
      <p className="max-w-[75ch] text-ink-2">
        {t("vision.why.sanityIntro")} {t("vision.why.sanityTip")}
      </p>
      {!method.available ? (
        <p className="max-w-[75ch] text-xs text-ink-3">{method.reason}</p>
      ) : (
        <div className="flex flex-wrap items-end gap-3">
          <SegmentedControl<SanityMethod>
            label={t("vision.why.sanityMethod")}
            value={choice}
            onChange={setChoice}
            options={[
              { value: "grad_cam", label: "Grad-CAM" },
              { value: "saliency", label: t("vision.why.saliency") },
              { value: "integrated_gradients", label: "Integrated Gradients" },
            ]}
          />
          <Button variant="secondary" onClick={() => setAsked(choice)} disabled={asked === choice && result.isFetching}>
            {t("vision.why.sanityRun")}
          </Button>
          <span className="self-center text-xs text-ink-3">{method.cost}</span>
        </div>
      )}
      {asked &&
        (result.isPending ? (
          <Loading label={t("vision.why.loadingSanity")} className="h-48 rounded-lg" />
        ) : result.isError ? (
          <ErrorNote>{(result.error as Error).message}</ErrorNote>
        ) : (
          <div className="min-w-0 space-y-3">
            <p className={`flex max-w-[75ch] gap-1.5 ${result.data.passes ? "text-ink-1" : "text-bad"}`}>
              {result.data.passes ? (
                <ShieldCheck size={15} strokeWidth={1.5} className="mt-px shrink-0 text-ok" aria-hidden />
              ) : (
                <ShieldAlert size={15} strokeWidth={1.5} className="mt-px shrink-0 text-bad" aria-hidden />
              )}
              <span>
                <span className="font-semibold">{result.data.passes ? t("vision.why.sanityPass") : t("vision.why.sanityFail")}</span> {result.data.sentence}
              </span>
            </p>
            <ol className="grid min-w-0 grid-cols-2 gap-3 sm:grid-cols-5">
              {result.data.stages.map((stage) => (
                <li key={stage.label} className="min-w-0 space-y-1">
                  <HeatmapOverlay
                    src={source}
                    alt={row.name}
                    className="aspect-square"
                    map={{ ...stageMap(result.data.rows, result.data.columns, result.data.box), png: stage.png, low: stage.low, high: stage.high }}
                  />
                  <span className="block truncate text-[11px] text-ink-1">{stage.label}</span>
                  <span className="block text-[11px] text-ink-3 tabular-nums">
                    {t("vision.why.rho", { value: stage.correlation === null || stage.correlation === undefined ? "-" : stage.correlation.toFixed(2) })}
                  </span>
                </li>
              ))}
            </ol>
          </div>
        ))}
    </section>
  );
}

/** A sanity stage drawn as a map (its picture and range; the rest only places it). */
function stageMap(rows: number, columns: number, box: readonly number[]): WhyMap {
  return {
    method: "sanity",
    title: "",
    unit: "",
    sentence: "",
    warning: "",
    own: false,
    blocky: false,
    box: [box[0] ?? 0, box[1] ?? 0, box[2] ?? 1, box[3] ?? 1],
    target: null,
    rows,
    columns,
    low: 0,
    high: 1,
    png: "",
    seconds: 0,
  } as WhyMap;
}
