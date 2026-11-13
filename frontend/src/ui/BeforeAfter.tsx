import { ArrowDown, ArrowRight, ChevronsLeftRight, Maximize, Minus, Plus } from "lucide-react";
import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  type WheelEvent,
} from "react";

import { readTokens } from "../charts/theme";
import { t, type MessageKey } from "../lib/i18n";
import { PHONE, useMedia } from "../lib/useMedia";
import { rampGradient, rampTable, sequentialStops, type Rgb } from "./colorRamp";
import { SegmentedControl } from "./SegmentedControl";

/**
 * 110 (study § 5.2, § 9.2): an image and what a step makes of it, `[before] → [after]`; or three
 * images for a reconstruction, `input → reconstruction → error`.
 *
 * Four ways to compare, in a SegmentedControl (keys 1 to 4 when focused):
 * - **Side by side**: the panels with an arrow between them (pointing down on a phone);
 * - **Slider**: the after image revealed over the before one by a handle (44 px to grab; ← → move it);
 * - **Toggle**: one or the other in the same place (Space);
 * - **Difference**: |after − before| on the theme's sequential scale, scaled to its 99th percentile
 *   (said), only when the step keeps the geometry (R-VPR-08).
 *
 * Zoom (wheel, pinch, + − 0) and pan (drag) are shared by every panel: one view in the images' own
 * coordinates, so the same detail stays under the eye in all of them, whatever their pixel sizes.
 * Moves go straight to the elements' style, never through a React render.
 */

export type CompareMode = "side" | "slider" | "toggle" | "difference";
export const COMPARE_MODES: readonly CompareMode[] = ["side", "slider", "toggle", "difference"];

export type BeforeAfterPanel = {
  src: string | null;
  label: string;
  alt: string;
  /** Under the image: its size, its measures and their changes. */
  caption?: ReactNode;
};

type BeforeAfterProps = {
  panels: readonly [BeforeAfterPanel, BeforeAfterPanel] | readonly [BeforeAfterPanel, BeforeAfterPanel, BeforeAfterPanel];
  /** Accessible name of the whole comparison. */
  label: string;
  mode?: CompareMode;
  defaultMode?: CompareMode;
  onModeChange?: (mode: CompareMode) => void;
  /** Why the difference cannot be shown (the step changes the geometry); null when it can. */
  differenceBlocked?: string | null;
  /** Hide the mode switch (a page that switches every comparison at once). */
  showModes?: boolean;
  /** Width / height of each frame. */
  aspect?: string;
  /** Shown between the toolbar and the images (3 panels: the score). */
  header?: ReactNode;
  busy?: boolean;
  /** A single frame (slider, toggle, difference) as wide as the screen allows (a dialog); otherwise
   *  no wider than a pair of side-by-side panels. */
  wide?: boolean;
  className?: string;
};

type View = { zoom: number; cx: number; cy: number };
type Layer = { frame: HTMLElement; element: HTMLElement; width: number; height: number };

const MAX_ZOOM = 16;
const FIT: View = { zoom: 1, cx: 0.5, cy: 0.5 };

/** Where an image of `width` x `height` sits in a frame for a view (scale, then offset). */
export function placement(frameWidth: number, frameHeight: number, width: number, height: number, view: View) {
  const fit = Math.min(frameWidth / width, frameHeight / height);
  const scale = fit * view.zoom;
  return { scale, x: frameWidth / 2 - view.cx * width * scale, y: frameHeight / 2 - view.cy * height * scale };
}

/** Keep the centre where some of the image fills the frame: centred when it fits. */
export function clampView(view: View, frameWidth: number, frameHeight: number, width: number, height: number): View {
  const { scale } = placement(frameWidth, frameHeight, width, height, view);
  const hx = frameWidth / (2 * width * scale);
  const hy = frameHeight / (2 * height * scale);
  const zoom = Math.min(MAX_ZOOM, Math.max(1, view.zoom));
  return {
    zoom,
    cx: hx >= 0.5 ? 0.5 : Math.min(1 - hx, Math.max(hx, view.cx)),
    cy: hy >= 0.5 ? 0.5 : Math.min(1 - hy, Math.max(hy, view.cy)),
  };
}

/** |after − before| per pixel (largest of the three channels), coloured on a ramp scaled to the
 *  99th percentile of the changes; the before image is drawn at the after image's size. */
export function differencePixels(before: Uint8ClampedArray, after: Uint8ClampedArray, ramp: Uint8ClampedArray) {
  const n = Math.min(before.length, after.length) / 4;
  const deltas = new Uint8Array(n);
  const histogram = new Uint32Array(256);
  let total = 0;
  for (let i = 0; i < n; i += 1) {
    const o = i * 4;
    const d = Math.max(
      Math.abs((after[o] ?? 0) - (before[o] ?? 0)),
      Math.abs((after[o + 1] ?? 0) - (before[o + 1] ?? 0)),
      Math.abs((after[o + 2] ?? 0) - (before[o + 2] ?? 0)),
    );
    deltas[i] = d;
    histogram[d] = (histogram[d] ?? 0) + 1;
    total += d;
  }
  let seen = 0;
  let top = 255;
  for (let v = 0; v < 256; v += 1) {
    seen += histogram[v] ?? 0;
    if (seen >= n * 0.99) {
      top = v;
      break;
    }
  }
  const scale = Math.max(1, top);
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i += 1) {
    const level = Math.min(255, Math.round(((deltas[i] ?? 0) / scale) * 255));
    out[i * 4] = ramp[level * 3] ?? 0;
    out[i * 4 + 1] = ramp[level * 3 + 1] ?? 0;
    out[i * 4 + 2] = ramp[level * 3 + 2] ?? 0;
    out[i * 4 + 3] = 255;
  }
  return { pixels: out, mean: n ? total / n : 0, top: scale };
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = src;
  });
}

type Difference = { src: string; mean: number; top: number; stops: Rgb[] };

/** The difference image of two sources, as a PNG data URL (null where canvas is missing: tests). */
async function makeDifference(beforeSrc: string, afterSrc: string): Promise<Difference | null> {
  const [before, after] = await Promise.all([loadImage(beforeSrc), loadImage(afterSrc)]);
  const width = after.naturalWidth;
  const height = after.naturalHeight;
  if (!width || !height) return null;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return null;
  context.drawImage(before, 0, 0, width, height);
  const a = context.getImageData(0, 0, width, height).data;
  context.drawImage(after, 0, 0, width, height);
  const b = context.getImageData(0, 0, width, height).data;
  const stops = sequentialStops(readTokens());
  const found = differencePixels(a, b, rampTable(stops));
  const image = context.createImageData(width, height);
  image.data.set(found.pixels);
  context.putImageData(image, 0, 0);
  return { src: canvas.toDataURL("image/png"), mean: found.mean, top: found.top, stops };
}

export function BeforeAfter({
  panels,
  label,
  mode: controlled,
  defaultMode = "side",
  onModeChange,
  differenceBlocked = null,
  showModes = true,
  aspect = "1 / 1",
  header,
  busy = false,
  wide = false,
  className = "",
}: BeforeAfterProps) {
  const phone = useMedia(PHONE);
  const [own, setOwn] = useState<CompareMode>(defaultMode);
  const wanted = controlled ?? own;
  const three = panels.length === 3;
  // three panels: the third already is the error map, so no computed difference
  const mode: CompareMode = wanted === "difference" && (differenceBlocked || three) ? "side" : wanted;
  const setMode = (next: CompareMode) => {
    if (controlled === undefined) setOwn(next);
    onModeChange?.(next);
  };
  const [showAfter, setShowAfter] = useState(true);
  const [split, setSplit] = useState(50);
  const [zoomShown, setZoomShown] = useState(1);
  const [difference, setDifference] = useState<{ key: string; value: Difference | null } | null>(null);
  const hintId = useId();

  const view = useRef<View>(FIT);
  const layers = useRef(new Map<string, Layer>());
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const gesture = useRef<{ x: number; y: number; distance: number; view: View; frame: HTMLElement } | null>(null);

  const render = useCallback(() => {
    for (const layer of layers.current.values()) {
      const { scale, x, y } = placement(layer.frame.clientWidth, layer.frame.clientHeight, layer.width, layer.height, view.current);
      layer.element.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
      layer.element.style.visibility = "visible";
    }
  }, []);

  const setView = useCallback(
    (next: View) => {
      const reference = layers.current.values().next().value as Layer | undefined;
      view.current = reference
        ? clampView(next, reference.frame.clientWidth, reference.frame.clientHeight, reference.width, reference.height)
        : next;
      render();
      setZoomShown(Math.round(view.current.zoom * 100) / 100);
    },
    [render],
  );

  /** Each image (or difference) registers its frame and natural size once loaded. */
  const register = useCallback(
    (key: string) => (element: HTMLImageElement | null) => {
      if (!element) {
        layers.current.delete(key);
        return;
      }
      const attach = () => {
        const frame = element.parentElement;
        if (!frame || !element.naturalWidth) return;
        element.style.width = `${element.naturalWidth}px`;
        element.style.height = `${element.naturalHeight}px`;
        layers.current.set(key, { frame, element, width: element.naturalWidth, height: element.naturalHeight });
        render();
      };
      if (element.complete) attach();
      element.onload = attach;
    },
    [render],
  );

  useEffect(() => {
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => render());
    for (const layer of layers.current.values()) observer.observe(layer.frame);
    return () => observer.disconnect();
  }, [render, mode, panels]);

  // the difference is computed once per pair of images, in the browser (no server call)
  const pairKey = `${panels[0].src ?? ""}|${panels[1].src ?? ""}`;
  useEffect(() => {
    if (mode !== "difference" || !panels[0].src || !panels[1].src || difference?.key === pairKey) return;
    let cancelled = false;
    makeDifference(panels[0].src, panels[1].src)
      .then((value) => !cancelled && setDifference({ key: pairKey, value }))
      .catch(() => !cancelled && setDifference({ key: pairKey, value: null }));
    return () => {
      cancelled = true;
    };
  }, [mode, pairKey, panels, difference?.key]);
  const shownDifference = difference?.key === pairKey ? difference.value : null;

  // --- zoom and pan, shared -----------------------------------------------------------------------

  const zoomAround = (factor: number, frame: HTMLElement | null, clientX?: number, clientY?: number) => {
    const layer = [...layers.current.values()].find((l) => l.frame === frame) ?? layers.current.values().next().value;
    if (!layer) return;
    const box = layer.frame.getBoundingClientRect();
    const current = view.current;
    const before = placement(layer.frame.clientWidth, layer.frame.clientHeight, layer.width, layer.height, current);
    const px = clientX === undefined ? layer.frame.clientWidth / 2 : clientX - box.left;
    const py = clientY === undefined ? layer.frame.clientHeight / 2 : clientY - box.top;
    const u = (px - before.x) / (layer.width * before.scale);
    const v = (py - before.y) / (layer.height * before.scale);
    const zoom = Math.min(MAX_ZOOM, Math.max(1, current.zoom * factor));
    const after = placement(layer.frame.clientWidth, layer.frame.clientHeight, layer.width, layer.height, { ...current, zoom });
    setView({
      zoom,
      cx: u - (px - layer.frame.clientWidth / 2) / (layer.width * after.scale),
      cy: v - (py - layer.frame.clientHeight / 2) / (layer.height * after.scale),
    });
  };

  const onWheel = (event: WheelEvent<HTMLElement>) => {
    if (!event.ctrlKey && !event.metaKey && view.current.zoom === 1 && event.deltaY > 0) return; // let the page scroll
    event.preventDefault();
    zoomAround(Math.exp(-event.deltaY * 0.0015), event.currentTarget, event.clientX, event.clientY);
  };

  const onPointerDown = (event: ReactPointerEvent<HTMLElement>) => {
    if ((event.target as HTMLElement).dataset.handle) return;
    capture(event.currentTarget, event.pointerId);
    pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    const points = [...pointers.current.values()];
    const [a, b] = points;
    gesture.current = {
      x: a && b ? (a.x + b.x) / 2 : event.clientX,
      y: a && b ? (a.y + b.y) / 2 : event.clientY,
      distance: a && b ? Math.hypot(a.x - b.x, a.y - b.y) : 0,
      view: view.current,
      frame: event.currentTarget,
    };
    if (mode === "slider" && points.length === 1 && view.current.zoom === 1) moveSplit(event.clientX, event.currentTarget);
  };

  const onPointerMove = (event: ReactPointerEvent<HTMLElement>) => {
    if (!pointers.current.has(event.pointerId) || !gesture.current) return;
    pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    const start = gesture.current;
    const layer = [...layers.current.values()].find((l) => l.frame === start.frame) ?? layers.current.values().next().value;
    if (!layer) return;
    const points = [...pointers.current.values()];
    if (points.length >= 2) {
      const [a, b] = points;
      if (!a || !b || !start.distance) return;
      const factor = Math.hypot(a.x - b.x, a.y - b.y) / start.distance;
      view.current = start.view;
      zoomAround(factor, start.frame, (a.x + b.x) / 2, (a.y + b.y) / 2);
      return;
    }
    if (mode === "slider" && start.view.zoom === 1) {
      moveSplit(event.clientX, start.frame);
      return;
    }
    const { scale } = placement(layer.frame.clientWidth, layer.frame.clientHeight, layer.width, layer.height, start.view);
    setView({
      zoom: start.view.zoom,
      cx: start.view.cx - (event.clientX - start.x) / (layer.width * scale),
      cy: start.view.cy - (event.clientY - start.y) / (layer.height * scale),
    });
  };

  const onPointerUp = (event: ReactPointerEvent<HTMLElement>) => {
    pointers.current.delete(event.pointerId);
    gesture.current = null;
  };

  function moveSplit(clientX: number, frame: HTMLElement) {
    const box = frame.getBoundingClientRect();
    if (box.width > 0) setSplit(Math.min(100, Math.max(0, ((clientX - box.left) / box.width) * 100)));
  }

  const onHandleDown = (event: ReactPointerEvent<HTMLElement>) => {
    event.stopPropagation();
    const frame = event.currentTarget.parentElement;
    if (!frame) return;
    capture(event.currentTarget, event.pointerId);
    const move = (e: PointerEvent) => moveSplit(e.clientX, frame);
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
    const numbered = COMPARE_MODES[Number(event.key) - 1];
    if (numbered && showModes) {
      if (numbered !== "difference" || (!differenceBlocked && !three)) setMode(numbered);
      return;
    }
    if (event.key === " " && mode === "toggle") {
      event.preventDefault();
      setShowAfter((shown) => !shown);
    } else if ((event.key === "ArrowLeft" || event.key === "ArrowRight") && mode === "slider") {
      event.preventDefault();
      const step = event.shiftKey ? 10 : 2;
      setSplit((value) => Math.min(100, Math.max(0, value + (event.key === "ArrowLeft" ? -step : step))));
    } else if (event.key === "+" || event.key === "=") zoomAround(1.25, null);
    else if (event.key === "-") zoomAround(0.8, null);
    else if (event.key === "0") setView(FIT);
  };

  // a new mode lays the frames out again: the next frame's layers register, then this re-applies
  useEffect(() => {
    render();
  }, [mode, render]);

  const options = useMemo(
    () =>
      COMPARE_MODES.filter((value) => value !== "difference" || !three).map((value) => ({
        value,
        label: t(`beforeAfter.mode.${value}` as MessageKey),
      })),
    [three],
  );

  const frameClass =
    "relative min-w-0 touch-none overflow-hidden rounded-[5px] bg-surface-2 outline outline-1 -outline-offset-1 outline-line select-none cursor-grab active:cursor-grabbing";
  const imageClass = "pointer-events-none absolute top-0 left-0 max-w-none origin-top-left [image-rendering:auto]";
  const Arrow = phone ? ArrowDown : ArrowRight;

  const frameHandlers = {
    onWheel,
    onPointerDown,
    onPointerMove,
    onPointerUp,
    onPointerCancel: onPointerUp,
    onDoubleClick: () => setView(FIT),
  };

  const img = (key: string, panel: BeforeAfterPanel, extra = "") =>
    panel.src ? (
      // the visibility waits for the first placement, so nothing jumps
      <img key={`${key}:${panel.src.length}:${panel.src.slice(-24)}`} ref={register(key)} src={panel.src} alt={panel.alt} draggable={false} decoding="async" className={`${imageClass} ${extra}`} style={{ visibility: "hidden" }} />
    ) : null;

  const [first, second] = panels;
  // a single frame in a dialog: as large as the height allows
  const single = wide ? { maxWidth: "min(100%, 66dvh)" } : undefined;
  const third = panels[2];

  return (
    <div
      role="group"
      aria-label={label}
      aria-describedby={hintId}
      tabIndex={0}
      onKeyDown={onKeyDown}
      className={`min-w-0 space-y-2 rounded-md focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-accent ${className}`}
    >
      {showModes && (
        <div className="flex flex-wrap items-center gap-2">
          <SegmentedControl<CompareMode> label={t("beforeAfter.modes")} value={mode} options={options} onChange={(next) => (next === "difference" && differenceBlocked ? undefined : setMode(next))} />
          {differenceBlocked && !three && <span className="text-xs text-ink-3">{differenceBlocked}</span>}
          <ZoomButtons zoom={zoomShown} onIn={() => zoomAround(1.25, null)} onOut={() => zoomAround(0.8, null)} onFit={() => setView(FIT)} />
        </div>
      )}
      <p id={hintId} className="sr-only">
        {t("beforeAfter.keys")}
      </p>
      {header}

      {mode === "side" && (
        <div
          className={
            phone
              ? "grid grid-cols-1 gap-1.5"
              : three
                ? "grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)_auto_minmax(0,1fr)] items-start gap-2"
                : "grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-start gap-2"
          }
        >
          {[first, second, third].map((panel, index) =>
            panel ? (
              <PanelBlock key={index} panel={panel} arrow={index > 0 ? <Arrow size={16} strokeWidth={1.5} aria-hidden /> : null} phone={phone}>
                <div className={frameClass} style={{ aspectRatio: aspect }} {...frameHandlers}>
                  {img(`side-${index}`, panel)}
                  {busy && index > 0 && <Busy />}
                </div>
              </PanelBlock>
            ) : null,
          )}
        </div>
      )}

      {mode === "slider" && (
        <figure className={`m-0 space-y-1.5 ${wide ? "" : "max-w-md"}`} style={single}>
          <div className={frameClass} style={{ aspectRatio: aspect }} {...frameHandlers}>
            {img("slider-before", first)}
            <div className="pointer-events-none absolute inset-0" style={{ clipPath: `inset(0 0 0 ${split}%)` }}>
              {img("slider-after", second)}
            </div>
            <div
              data-handle="1"
              role="slider"
              aria-label={t("beforeAfter.handle")}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(split)}
              aria-valuetext={t("beforeAfter.handleValue", { value: String(Math.round(split)) })}
              onPointerDown={onHandleDown}
              className="absolute inset-y-0 z-[1] flex w-11 -translate-x-1/2 cursor-ew-resize touch-none justify-center"
              style={{ left: `${split}%` }}
            >
              <span data-handle="1" className="h-full w-0.5 bg-surface-1 shadow-[0_0_0_1px_rgb(0_0_0/0.35)]" />
              <span data-handle="1" className="absolute top-1/2 flex size-7 -translate-y-1/2 items-center justify-center rounded-full border border-line bg-surface-1 text-ink-2 shadow-[0_2px_8px_-2px_rgb(0_0_0/0.35)]">
                <ChevronsLeftRight data-handle="1" size={14} strokeWidth={1.5} aria-hidden className="pointer-events-none" />
              </span>
            </div>
            {busy && <Busy />}
          </div>
          <figcaption className="grid grid-cols-2 gap-3 text-xs">
            <span className="min-w-0 space-y-0.5">
              <span className="block font-medium text-ink-1">{first.label}</span>
              {first.caption}
            </span>
            <span className="min-w-0 space-y-0.5 text-right">
              <span className="block font-medium text-ink-1">{second.label}</span>
              {second.caption}
            </span>
          </figcaption>
        </figure>
      )}

      {mode === "toggle" && (
        <figure className={`m-0 space-y-1.5 ${wide ? "" : "max-w-md"}`} style={single}>
          <div className={frameClass} style={{ aspectRatio: aspect }} {...frameHandlers} onClick={() => setShowAfter((shown) => !shown)}>
            <div className={`absolute inset-0 motion-safe:transition-opacity motion-safe:duration-150 ${showAfter ? "opacity-0" : "opacity-100"}`}>{img("toggle-before", first)}</div>
            <div className={`absolute inset-0 motion-safe:transition-opacity motion-safe:duration-150 ${showAfter ? "opacity-100" : "opacity-0"}`}>{img("toggle-after", second)}</div>
            {busy && <Busy />}
          </div>
          <figcaption className="flex flex-wrap items-center justify-between gap-2 text-xs">
            <span className="min-w-0 space-y-0.5" aria-live="polite">
              <span className="block font-medium text-ink-1">{t("beforeAfter.showing", { name: showAfter ? second.label : first.label })}</span>
              {showAfter ? second.caption : first.caption}
            </span>
            <button type="button" onClick={() => setShowAfter((shown) => !shown)} className="inline-flex min-h-6 items-center rounded px-1.5 text-accent hover:underline pointer-coarse:min-h-11">
              {t("beforeAfter.switch", { name: showAfter ? first.label : second.label })}
            </button>
          </figcaption>
        </figure>
      )}

      {mode === "difference" && (
        <figure className={`m-0 space-y-1.5 ${wide ? "" : "max-w-md"}`} style={single}>
          <div className={frameClass} style={{ aspectRatio: aspect }} {...frameHandlers}>
            {shownDifference
              ? img("difference", { src: shownDifference.src, label: t("beforeAfter.mode.difference"), alt: t("beforeAfter.differenceAlt", { name: second.alt }) }, "[image-rendering:pixelated]")
              : null}
            {(busy || !shownDifference) && <Busy />}
          </div>
          {shownDifference && <DifferenceLegend difference={shownDifference} />}
        </figure>
      )}
    </div>
  );
}

/** Follow the pointer outside the element; a pointer already gone (or synthetic) is fine. */
function capture(element: Element, pointerId: number) {
  try {
    element.setPointerCapture?.(pointerId);
  } catch {
    // nothing to capture
  }
}

function PanelBlock({ panel, arrow, phone, children }: { panel: BeforeAfterPanel; arrow: ReactNode; phone: boolean; children: ReactNode }) {
  return (
    <>
      {arrow && (
        <span className={phone ? "flex justify-center text-ink-3" : "flex h-full items-center self-stretch pb-8 text-ink-3"} aria-hidden>
          {arrow}
        </span>
      )}
      <figure className={phone ? "m-0 mx-auto w-full max-w-[280px] min-w-0 space-y-1" : "m-0 min-w-0 space-y-1"}>
        {children}
        <figcaption className="min-w-0 space-y-0.5 text-xs">
          <span className="block font-medium text-ink-1">{panel.label}</span>
          {panel.caption}
        </figcaption>
      </figure>
    </>
  );
}

function Busy() {
  return (
    <span className="loading-sheen pointer-events-none absolute inset-0 bg-surface-1/30" aria-hidden />
  );
}

function ZoomButtons({ zoom, onIn, onOut, onFit }: { zoom: number; onIn: () => void; onOut: () => void; onFit: () => void }) {
  const button =
    "relative inline-flex size-7 items-center justify-center rounded-md text-ink-2 hover:bg-surface-2 hover:text-ink-1 disabled:opacity-40 pointer-coarse:after:absolute pointer-coarse:after:-inset-2 pointer-coarse:after:content-['']";
  return (
    <span className="ml-auto inline-flex items-center gap-0.5">
      <button type="button" className={button} onClick={onOut} disabled={zoom <= 1} aria-label={t("beforeAfter.zoomOut")} title={t("beforeAfter.zoomOut")}>
        <Minus size={14} strokeWidth={1.5} aria-hidden />
      </button>
      <span className="w-10 text-center text-xs text-ink-3 tabular-nums" aria-live="polite">
        {Math.round(zoom * 100)}%
      </span>
      <button type="button" className={button} onClick={onIn} disabled={zoom >= MAX_ZOOM} aria-label={t("beforeAfter.zoomIn")} title={t("beforeAfter.zoomIn")}>
        <Plus size={14} strokeWidth={1.5} aria-hidden />
      </button>
      <button type="button" className={button} onClick={onFit} disabled={zoom === 1} aria-label={t("beforeAfter.fit")} title={t("beforeAfter.fit")}>
        <Maximize size={13} strokeWidth={1.5} aria-hidden />
      </button>
    </span>
  );
}

function DifferenceLegend({ difference }: { difference: Difference }) {
  return (
    <figcaption className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-2">
      <span className="inline-flex items-center gap-1.5">
        <span className="tabular-nums">0</span>
        <span className="h-2 w-28 rounded-[2px] outline outline-1 -outline-offset-1 outline-line" style={{ background: rampGradient(difference.stops) }} aria-hidden />
        <span className="tabular-nums">{difference.top}</span>
      </span>
      <span>{t("beforeAfter.differenceNote", { mean: difference.mean.toFixed(1), top: String(difference.top) })}</span>
    </figcaption>
  );
}
