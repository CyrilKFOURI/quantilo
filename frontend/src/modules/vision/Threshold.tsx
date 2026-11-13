import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";

import { readTokens, type ChartTokens } from "../../charts/theme";
import { t } from "../../lib/i18n";
import type { Histogram } from "./scores";

/** Theme tokens, read again when the theme or the palette changes (as `charts/Plot` does). */
export function useChartTokens(): ChartTokens {
  const [version, setVersion] = useState(0);
  useEffect(() => {
    const bump = () => setVersion((v) => v + 1);
    const observer = new MutationObserver(bump);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "data-palette"] });
    const media = window.matchMedia?.("(prefers-color-scheme: dark)");
    media?.addEventListener?.("change", bump);
    return () => {
      observer.disconnect();
      media?.removeEventListener?.("change", bump);
    };
  }, []);
  // eslint-disable-next-line react-hooks/exhaustive-deps -- `version` is the theme's change counter
  return useMemo(() => readTokens(), [version]);
}

type ThresholdProps = {
  histogram: Histogram;
  threshold: number;
  /** The rule's threshold (R-VTH-01 / 02), marked so the user can find it again. */
  rule: number | null;
  onChange: (value: number) => void;
  /** No defect to show (R-VTH-02 without labels): one series. */
  single?: boolean;
  label: string;
};

const HEIGHT = 196;
const PAD = { left: 8, right: 8, top: 22, bottom: 22 };

/**
 * 113 (R-VTH-03): the scores of normal images (below the axis) and of defects (above), on shared
 * bins, with the threshold as a line the user drags with the mouse or a finger, or moves with the
 * arrows (Shift: ten times further, Home / End). Mirrored rather than overlaid, so no bar hides
 * another. Images at or right of the line are flagged.
 */
export function ThresholdHistogram({ histogram, threshold, rule, onChange, single = false, label }: ThresholdProps) {
  const tokens = useChartTokens();
  const id = useId();
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(640);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    const element = box.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver((entries) => {
      const next = entries[0]?.contentRect.width;
      if (next) setWidth(Math.max(240, Math.round(next)));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const low = histogram.edges[0] ?? 0;
  const high = histogram.edges[histogram.edges.length - 1] ?? 1;
  const span = high > low ? high - low : 1;
  const inner = width - PAD.left - PAD.right;
  const x = (value: number) => PAD.left + ((value - low) / span) * inner;
  const valueAt = (px: number) => low + ((px - PAD.left) / inner) * span;
  const clamp = (value: number) => Math.min(high, Math.max(low, value));

  const top = Math.max(1, ...histogram.defect);
  const bottom = Math.max(1, ...histogram.normal);
  const plot = HEIGHT - PAD.top - PAD.bottom;
  // defects take the upper part, normals the lower, in proportion to their tallest bar
  const upShare = single ? 0 : top / (top + bottom);
  const axis = PAD.top + plot * upShare;
  const upScale = (plot * upShare) / top;
  const downScale = (plot * (1 - upShare)) / bottom;

  const at = x(clamp(threshold));
  const tickCount = width < 420 ? 3 : 5;
  const ticks = Array.from({ length: tickCount }, (_, k) => low + (span * k) / (tickCount - 1));
  const bins = histogram.normal.length;

  const move = (event: PointerEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    onChange(clamp(valueAt(((event.clientX - rect.left) / rect.width) * width)));
  };

  const onKey = (event: KeyboardEvent<SVGGElement>) => {
    const step = (span / 200) * (event.shiftKey ? 10 : 1);
    const next =
      event.key === "ArrowRight" || event.key === "ArrowUp"
        ? threshold + step
        : event.key === "ArrowLeft" || event.key === "ArrowDown"
          ? threshold - step
          : event.key === "PageUp"
            ? threshold + span / 10
            : event.key === "PageDown"
              ? threshold - span / 10
              : event.key === "Home"
                ? low
                : event.key === "End"
                  ? high
                  : null;
    if (next === null) return;
    event.preventDefault();
    onChange(clamp(next));
  };

  const defectColour = tokens.palette[1] ?? tokens.accent;
  const normalColour = tokens.palette[0] ?? tokens.numeric;
  const format = (value: number) => (Math.abs(value) >= 100 ? value.toFixed(0) : Math.abs(value) >= 1 ? value.toFixed(2) : value.toPrecision(3));

  return (
    <div ref={box} className="w-full min-w-0 touch-pan-y select-none">
      <p className="mb-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2" aria-hidden>
        {!single && (
          <span className="inline-flex items-center gap-1.5">
            <span className="inline-block size-2.5 rounded-[2px]" style={{ backgroundColor: defectColour }} />
            {t("vision.results.histDefects")}
          </span>
        )}
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block size-2.5 rounded-[2px]" style={{ backgroundColor: normalColour }} />
          {single ? t("vision.results.histScored") : t("vision.results.histNormals")}
        </span>
        <span className="text-ink-3">{t("vision.results.histHint")}</span>
      </p>
      <svg
        width="100%"
        height={HEIGHT}
        viewBox={`0 0 ${width} ${HEIGHT}`}
        role="img"
        aria-labelledby={`${id}-title`}
        className={dragging ? "cursor-grabbing" : "cursor-col-resize"}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          setDragging(true);
          move(event);
        }}
        onPointerMove={(event) => dragging && move(event)}
        onPointerUp={() => setDragging(false)}
        onPointerCancel={() => setDragging(false)}
      >
        <title id={`${id}-title`}>{label}</title>
        {/* flagged side, lightly shaded */}
        <rect x={at} y={PAD.top - 6} width={Math.max(0, PAD.left + inner - at)} height={plot + 12} fill={tokens.ink1} opacity={0.045} />
        {histogram.edges.slice(0, -1).map((edge, k) => {
          const left = x(edge) + 1;
          const w = Math.max(1, x(histogram.edges[k + 1] ?? edge) - x(edge) - 2);
          const up = (histogram.defect[k] ?? 0) * upScale;
          const down = (histogram.normal[k] ?? 0) * downScale;
          return (
            <g key={k}>
              {!single && up > 0 && <rect x={left} y={axis - up} width={w} height={up} rx={Math.min(2, w / 2)} fill={defectColour} />}
              {down > 0 && <rect x={left} y={axis} width={w} height={down} rx={Math.min(2, w / 2)} fill={normalColour} />}
            </g>
          );
        })}
        <line x1={PAD.left} x2={PAD.left + inner} y1={axis} y2={axis} stroke={tokens.line} strokeWidth={1} />
        {/* the rule's threshold, dotted, so it can be found again */}
        {rule !== null && Number.isFinite(rule) && Math.abs(x(clamp(rule)) - at) > 2 && (
          <line x1={x(clamp(rule))} x2={x(clamp(rule))} y1={PAD.top - 6} y2={PAD.top + plot + 6} stroke={tokens.ink3} strokeWidth={1} strokeDasharray="3 3" />
        )}
        {ticks.map((tick, k) => (
          <text key={k} x={x(tick)} y={HEIGHT - 5} fontSize={11} fill={tokens.ink3} textAnchor={k === 0 ? "start" : k === ticks.length - 1 ? "end" : "middle"} className="tabular-nums">
            {format(tick)}
          </text>
        ))}
        <g
          role="slider"
          tabIndex={0}
          aria-label={t("vision.results.thresholdSlider")}
          aria-valuemin={low}
          aria-valuemax={high}
          aria-valuenow={threshold}
          aria-valuetext={format(threshold)}
          onKeyDown={onKey}
          className="outline-none [&:focus-visible_.knob]:stroke-accent"
        >
          {/* a 44 px wide invisible hit area around the line (044) */}
          <rect x={at - 22} y={0} width={44} height={HEIGHT} fill="transparent" />
          <line x1={at} x2={at} y1={PAD.top - 6} y2={PAD.top + plot + 6} stroke={tokens.ink1} strokeWidth={2} />
          <rect className="knob" x={at - 7} y={PAD.top - 18} width={14} height={14} rx={3} fill={tokens.surface} stroke={tokens.ink1} strokeWidth={1.5} />
          <text x={Math.min(PAD.left + inner, Math.max(PAD.left, at + 10))} y={PAD.top - 7} fontSize={11} fill={tokens.ink1} textAnchor={at > width - 90 ? "end" : "start"} className="tabular-nums" dx={at > width - 90 ? -20 : 0}>
            {format(threshold)}
          </text>
        </g>
      </svg>
      <p className="sr-only" aria-live="polite">
        {bins > 0 && t("vision.results.thresholdAt", { value: format(threshold) })}
      </p>
    </div>
  );
}
