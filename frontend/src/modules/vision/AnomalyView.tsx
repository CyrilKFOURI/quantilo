import { ChevronLeft, ChevronRight, ScanSearch } from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";

import { imageUrl } from "../../features/import/imageSetApi";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { BeforeAfter } from "../../ui/BeforeAfter";
import { sequentialStops } from "../../ui/colorRamp";
import { InfoTip } from "../../ui/InfoTip";
import { MapOverlay, useRampedMap } from "./MapOverlay";
import type { ImageRow } from "./model";
import { runImageUrl, type Prediction, type VisionModel } from "./modelApi";
import { num, text } from "./scores";
import { useChartTokens } from "./Threshold";

type Props = {
  datasetId: string;
  runId: string;
  model: VisionModel;
  rows: readonly ImageRow[];
  prediction: Prediction;
  threshold: number;
  list: number[];
  onNavigate: (row: number) => void;
  onClose: () => void;
  /** 117: this image's "Why?" (the model's evidence, then post-hoc maps). */
  onWhy?: (row: number) => void;
};

const POST_HOC = "vision.results.mapCaveat";

/** A box of an image's own pixels as percentages of that image (for an absolutely placed frame). */
function boxStyle(box: unknown, image: ImageRow | undefined) {
  if (!Array.isArray(box) || box.length !== 4 || !image?.width || !image.height) return null;
  const [x0, y0, x1, y1] = box.map(Number);
  if (![x0, y0, x1, y1].every(Number.isFinite)) return null;
  const w = image.width;
  const h = image.height;
  return {
    left: `${((x0 ?? 0) / w) * 100}%`,
    top: `${((y0 ?? 0) / h) * 100}%`,
    width: `${(((x1 ?? 0) - (x0 ?? 0)) / w) * 100}%`,
    height: `${(((y1 ?? 0) - (y0 ?? 0)) / h) * 100}%`,
  };
}

/** The image whole, with what is laid on it. The map covers the whole image (R-VAD-03). */
function Framed({ src, alt, children, className = "" }: { src: string; alt: string; children?: ReactNode; className?: string }) {
  return (
    <div className={`relative mx-auto w-fit max-w-full overflow-hidden rounded-md bg-surface-2 outline outline-1 -outline-offset-1 outline-ink-1/10 ${className}`}>
      <img src={src} alt={alt} className="block max-h-[52dvh] max-w-full object-contain" decoding="async" />
      {children}
    </div>
  );
}

/**
 * 113: one scored image of an anomaly run. Its map laid over it (opacity at hand, `[` and `]` on the
 * keyboard), then the model's own evidence: for PatchCore the most unusual area and the closest
 * normal patch of the training images (§ 8); then, in `BeforeAfter` (110), `image → map` or, for an
 * autoencoder, `image → what it expected → where they differ` (side by side, slider, toggle).
 */
export function AnomalyView({ datasetId, runId, model, rows, prediction, threshold, list, onNavigate, onClose, onWhy }: Props) {
  const tokens = useChartTokens();
  const [opacity, setOpacity] = useState(0.6);
  const [mapState, setMapState] = useState<"loading" | "ready" | "missing">("loading");
  const image = rows[prediction.row];
  const key = model.key;
  const score = num(prediction[`${key}_score`]);
  const flagged = score >= threshold;
  const defect = prediction.is_defect === true;
  const at = list.indexOf(prediction.row);
  const previous = at > 0 ? list[at - 1] : undefined;
  const next = at >= 0 && at < list.length - 1 ? list[at + 1] : undefined;
  const mapSrc = model.map_source ? runImageUrl(datasetId, runId, "maps", model.map_source, prediction.row) : null;
  const box = boxStyle(prediction[`${key}_box`], image);
  const normalRow = prediction[`${key}_normal_row`];
  const normal = typeof normalRow === "number" ? rows[normalRow] : undefined;
  const normalBox = boxStyle(prediction[`${key}_normal_box`], normal);
  // the same hue as the sequential scale of the map panel below (047)
  const colour = tokens.palette[0] ?? tokens.accent;
  const stops = useMemo(() => sequentialStops(tokens), [tokens]);
  const ramped = useRampedMap(mapSrc, stops);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement) return;
      if (event.key === "[") setOpacity((value) => Math.max(0, value - 0.1));
      if (event.key === "]") setOpacity((value) => Math.min(1, value + 0.1));
      if (event.key === "ArrowLeft" && previous !== undefined) onNavigate(previous);
      if (event.key === "ArrowRight" && next !== undefined) onNavigate(next);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [previous, next, onNavigate]);

  const verdict = defect
    ? flagged
      ? t("vision.results.view.found")
      : t("vision.results.view.missed")
    : flagged
      ? t("vision.results.view.falseAlarm")
      : t("vision.results.view.normal");

  return (
    <Dialog
      open
      onOpenChange={(open) => !open && onClose()}
      size="lg"
      title={image?.name ?? String(prediction.row)}
      description={`${text(prediction.truth)} · ${t("vision.results.view.score", { value: Number.isFinite(score) ? score.toPrecision(4) : "—" })} · ${verdict}`}
      footer={
        <div className="flex w-full items-center justify-between gap-2">
          <span className="text-xs text-ink-3 tabular-nums">{at >= 0 ? t("vision.results.view.position", { at: String(at + 1), count: String(list.length) }) : ""}</span>
          <span className="flex gap-2">
            {onWhy && (
              <Button variant="secondary" icon={ScanSearch} onClick={() => onWhy(prediction.row)} title={t("vision.why.open")}>
                {t("vision.why.button")}
              </Button>
            )}
            <Button variant="secondary" icon={ChevronLeft} disabled={previous === undefined} onClick={() => previous !== undefined && onNavigate(previous)}>
              {t("vision.results.view.previous")}
            </Button>
            <Button variant="secondary" disabled={next === undefined} onClick={() => next !== undefined && onNavigate(next)}>
              {t("vision.results.view.next")} <ChevronRight size={15} strokeWidth={1.5} aria-hidden />
            </Button>
          </span>
        </div>
      }
    >
      <div className="flex flex-col gap-5">
        <div className="grid gap-5 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
          <figure className="min-w-0">
            {image && (
              <Framed src={imageUrl(datasetId, image.id, "file")} alt={t("vision.results.view.alt", { name: image.name })}>
                {mapSrc && <MapOverlay src={mapSrc} opacity={opacity} colour={colour} onState={setMapState} />}
                {box && <span aria-hidden className="pointer-events-none absolute rounded-[2px] border-2 border-ink-1 shadow-[0_0_0_1px_var(--surface-1)]" style={box} />}
              </Framed>
            )}
            <figcaption className="mt-2.5 flex flex-col gap-2">
              {mapSrc ? (
                <label className="flex items-center gap-3 text-xs text-ink-2">
                  <span className="shrink-0">{t("vision.results.view.opacity")}</span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={5}
                    value={Math.round(opacity * 100)}
                    onChange={(event) => setOpacity(Number(event.target.value) / 100)}
                    className="h-11 min-w-0 flex-1 accent-(--accent) sm:h-auto"
                    aria-valuetext={`${Math.round(opacity * 100)} %`}
                  />
                  <span className="w-10 text-right tabular-nums">{Math.round(opacity * 100)} %</span>
                  <InfoTip label={t("vision.results.view.mapAbout")} text={t(POST_HOC)} />
                </label>
              ) : (
                <p className="text-xs text-ink-3">{t("vision.results.view.noMap")}</p>
              )}
              {mapState === "missing" && mapSrc && <p className="text-xs text-ink-3">{t("vision.results.view.mapMissing")}</p>}
              {model.map_source && model.map_source !== key && <p className="text-xs text-ink-3">{t("vision.results.view.mapFrom")}</p>}
              <p className="flex items-center gap-2 text-xs text-ink-3">
                <span aria-hidden className="inline-block h-2 w-16 rounded-full" style={{ background: `linear-gradient(90deg, transparent, ${colour})` }} />
                {t("vision.results.view.legend")}
              </p>
            </figcaption>
          </figure>

          <div className="flex min-w-0 flex-col gap-4">
            {normal && (
              <section aria-labelledby="vision-nearest-normal">
                <h4 id="vision-nearest-normal" className="text-[13px] font-medium text-ink-1">
                  {t("vision.results.view.nearest")}
                </h4>
                <p className="mt-1 text-[13px] text-pretty text-ink-2">{t("vision.results.view.nearestSentence", { name: normal.name, distance: num(prediction[`${key}_distance`]).toPrecision(3) })}</p>
                <Framed src={imageUrl(datasetId, normal.id, "file")} alt={t("vision.results.view.alt", { name: normal.name })} className="mt-2">
                  {normalBox && <span aria-hidden className="pointer-events-none absolute rounded-[2px] border-2 border-ink-1 shadow-[0_0_0_1px_var(--surface-1)]" style={normalBox} />}
                </Framed>
              </section>
            )}
            <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-[13px]">
              <div>
                <dt className="text-xs text-ink-3">{t("vision.results.view.truth")}</dt>
                <dd className="text-ink-1">{text(prediction.truth)}</dd>
              </div>
              <div>
                <dt className="text-xs text-ink-3">{t("vision.results.view.atThreshold")}</dt>
                <dd className={flagged ? "font-medium text-ink-1" : "text-ink-2"}>{flagged ? t("vision.results.view.flagged") : t("vision.results.view.notFlagged")}</dd>
              </div>
              <div>
                <dt className="text-xs text-ink-3">{t("vision.results.view.scoreLabel")}</dt>
                <dd className="text-ink-1 tabular-nums">{Number.isFinite(score) ? score.toPrecision(4) : "—"}</dd>
              </div>
              <div>
                <dt className="text-xs text-ink-3">{t("vision.results.view.thresholdLabel")}</dt>
                <dd className="text-ink-1 tabular-nums">{threshold.toPrecision(4)}</dd>
              </div>
            </dl>
          </div>
        </div>

        {image && mapSrc && (
          <section aria-labelledby="vision-before-after">
            <h4 id="vision-before-after" className="text-[13px] font-medium text-ink-1">
              {model.reconstructions ? t("vision.results.view.reconstruction") : t("vision.results.view.mapCompare")}
            </h4>
            <p className="mt-1 text-[13px] text-pretty text-ink-2">
              {model.reconstructions
                ? t("vision.results.view.reconstructionSentence", { model: model.title })
                : t("vision.results.view.mapCompareSentence")}
            </p>
            <div className="mt-2.5">
              <BeforeAfter
                label={t("vision.results.view.compareLabel", { name: image.name })}
                differenceBlocked={t("vision.results.view.noDifference")}
                aspect="1 / 1"
                busy={ramped === null}
                panels={
                  model.reconstructions
                    ? [
                        { src: imageUrl(datasetId, image.id, "file"), label: t("vision.results.view.input"), alt: t("vision.results.view.alt", { name: image.name }) },
                        { src: runImageUrl(datasetId, runId, "reconstructions", key, prediction.row), label: t("vision.results.view.expected"), alt: t("vision.results.view.expectedAlt", { name: image.name }) },
                        { src: ramped, label: t("vision.results.view.difference"), alt: t("vision.results.view.mapAlt", { name: image.name }) },
                      ]
                    : [
                        { src: imageUrl(datasetId, image.id, "file"), label: t("vision.results.view.input"), alt: t("vision.results.view.alt", { name: image.name }) },
                        { src: ramped, label: t("vision.results.view.mapPanel"), alt: t("vision.results.view.mapAlt", { name: image.name }) },
                      ]
                }
              />
            </div>
          </section>
        )}
      </div>
    </Dialog>
  );
}
