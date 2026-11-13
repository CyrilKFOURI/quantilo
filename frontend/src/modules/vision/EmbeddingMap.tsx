import { Expand, Hand, Images, LassoSelect, Minus, Plus, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent as ReactPointerEvent, type WheelEvent } from "react";

import { readTokens, type ChartTokens } from "../../charts/theme";
import { formatNumber } from "../../charts/figures";
import { t, type MessageKey } from "../../lib/i18n";
import { TOUCH, useMedia } from "../../lib/useMedia";
import { parseColor, rampTable, sequentialStops } from "../../ui/colorRamp";
import type { ImageMap } from "./api";
import { baseTransform, boundsOf, insidePolygon, nearest, START, THUMB_FROM, THUMBS_MAX, thumbSide, zoomView, type Colouring, type MapView } from "./mapModel";
import { OTHER, type ImageRow } from "./model";

export type MapTool = "pan" | "lasso";

type EmbeddingMapProps = {
  datasetId: string;
  map: ImageMap;
  rows: readonly ImageRow[];
  colouring: Colouring;
  /** The legend entry put forward (the others fade), or null. */
  focus: string | null;
  /** Table rows selected (lasso, a group): the others fade. */
  selected: ReadonlySet<number> | null;
  onSelect: (tableRows: number[]) => void;
  onOpen: (row: ImageRow) => void;
  /** Select the group of an image (the phone's "tap a cluster", R-VMAP-08 and § 9.3). */
  onSelectGroup: (cluster: number) => void;
  label: string;
  height: number;
  describe: (row: ImageRow) => string | null;
};

const POINT = 3.5;
const HIT = 10;
const TOUCH_HIT = 22;
const FADED = 0.16;
/** The lowest values stay visible on the surface (sequential scale starts 30 % of the way). */
export const MARK_FLOOR = 0.3;

type Point = { x: number; y: number };

/** 109 (study § 4.4, § 9.2): the images as points of a plane, drawn on a canvas. Wheel or pinch to
 *  zoom, drag to pan; from 28 px per image the points become their thumbnails (R-VMAP-07); hover
 *  shows the image; the lasso (mouse or pen) selects; a click opens the image. On a touch screen a
 *  tap shows the image and offers its group instead of the lasso. Colours come from the theme. */
export function EmbeddingMap({ datasetId, map, rows, colouring, focus, selected, onSelect, onOpen, onSelectGroup, label, height, describe }: EmbeddingMapProps) {
  const touch = useMedia(TOUCH);
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ width: 0, height });
  const [tool, setTool] = useState<MapTool>("pan");
  const [forceThumbs, setForceThumbs] = useState(false);
  const [hover, setHover] = useState<{ index: number; x: number; y: number; pinned: boolean } | null>(null);
  const [zoomShown, setZoomShown] = useState(1);
  const view = useRef<MapView>(START);
  const lasso = useRef<Point[] | null>(null);
  const pointers = useRef(new Map<number, Point>());
  const gesture = useRef<{ start: Point; distance: number; view: MapView; moved: boolean } | null>(null);
  const frame = useRef(0);
  const tokens = useRef<ChartTokens | null>(null);
  const thumbs = useRef<ThumbStore>({ images: new Map(), loading: 0 });
  const redraw = useRef<() => void>(() => {});

  // --- layout: the base positions of the points, once per size and map ---------------------------
  useEffect(() => {
    const element = box.current;
    if (!element) return;
    const measure = () => setSize({ width: element.clientWidth, height });
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [height]);

  const base = useMemo(() => {
    const bounds = boundsOf(map.x, map.y);
    const to = baseTransform(bounds, size.width, size.height);
    const xs = new Float32Array(map.x.length);
    const ys = new Float32Array(map.y.length);
    map.x.forEach((value, i) => {
      xs[i] = to.x(value);
      ys[i] = to.y(map.y[i] ?? 0);
    });
    return { xs, ys };
  }, [map.x, map.y, size.width, size.height]);

  const screen = useCallback(() => {
    const { scale, x, y } = view.current;
    const xs = new Float32Array(base.xs.length);
    const ys = new Float32Array(base.ys.length);
    for (let i = 0; i < xs.length; i += 1) {
      xs[i] = (base.xs[i] ?? 0) * scale + x;
      ys[i] = (base.ys[i] ?? 0) * scale + y;
    }
    return { xs, ys };
  }, [base]);

  const colourOf = useCallback(
    (i: number, palette: string[], neutral: string, table: Uint8ClampedArray | null): string => {
      if (colouring.kind === "sequential") {
        const level = colouring.level[i];
        if (level === null || level === undefined || !table) return neutral;
        const at = Math.round(level * 255) * 3;
        return `rgb(${table[at]} ${table[at + 1]} ${table[at + 2]})`;
      }
      const key = colouring.keyOf[i];
      const entry = colouring.entries.find((e) => e.key === key);
      return entry && entry.slot >= 0 ? (palette[entry.slot] ?? neutral) : neutral;
    },
    [colouring],
  );

  const faded = useCallback(
    (i: number): boolean => {
      const row = map.rows[i] ?? -1;
      if (selected && selected.size > 0 && !selected.has(row)) return true;
      if (focus !== null && colouring.kind === "categorical" && colouring.keyOf[i] !== focus) return true;
      return false;
    },
    [map.rows, selected, focus, colouring],
  );

  // --- drawing -----------------------------------------------------------------------------------
  const draw = useCallback(() => {
    frame.current = 0;
    const element = canvas.current;
    if (!element || size.width === 0) return;
    const context = element.getContext("2d");
    if (!context) return;
    tokens.current ??= readTokens();
    const theme = tokens.current;
    const table = rampTable(sequentialStops(theme, MARK_FLOOR));
    const ratio = window.devicePixelRatio || 1;
    if (element.width !== Math.round(size.width * ratio) || element.height !== Math.round(size.height * ratio)) {
      element.width = Math.round(size.width * ratio);
      element.height = Math.round(size.height * ratio);
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, size.width, size.height);
    const { xs, ys } = screen();
    const neutral = theme.ink3;
    const surface = theme.surface;
    const side = thumbSide(view.current.scale, forceThumbs);
    const showThumbs = side >= THUMB_FROM;
    let drawnThumbs = 0;
    // faded first, then the others on top
    for (const pass of [true, false]) {
      for (let i = 0; i < xs.length; i += 1) {
        if (faded(i) !== pass) continue;
        const x = xs[i] ?? 0;
        const y = ys[i] ?? 0;
        if (x < -side || y < -side || x > size.width + side || y > size.height + side) continue;
        const colour = colourOf(i, theme.palette, neutral, table);
        context.globalAlpha = pass ? FADED : 1;
        if (showThumbs && !pass && drawnThumbs < THUMBS_MAX) {
          const id = rows[map.rows[i] ?? -1]?.id;
          const image = id ? thumbnail(thumbs.current, thumbUrl(datasetId, id), () => redraw.current()) : null;
          if (image) {
            drawnThumbs += 1;
            drawCover(context, image, x - side / 2, y - side / 2, side);
            context.lineWidth = 2;
            context.strokeStyle = colour;
            context.strokeRect(x - side / 2 + 1, y - side / 2 + 1, side - 2, side - 2);
            continue;
          }
        }
        context.beginPath();
        context.arc(x, y, POINT, 0, Math.PI * 2);
        context.fillStyle = colour;
        context.fill();
        context.lineWidth = 1;
        context.strokeStyle = surface; // a ring of the surface between overlapping marks (dataviz)
        context.stroke();
      }
    }
    context.globalAlpha = 1;
    // hovered point: a ring in the ink colour
    if (hover && hover.index >= 0) {
      const x = xs[hover.index] ?? 0;
      const y = ys[hover.index] ?? 0;
      context.beginPath();
      context.arc(x, y, showThumbs ? side / 2 + 3 : POINT + 3, 0, Math.PI * 2);
      context.lineWidth = 2;
      context.strokeStyle = theme.ink1;
      context.stroke();
    }
    // the lasso being drawn
    const path = lasso.current;
    if (path && path.length > 1) {
      context.beginPath();
      path.forEach((p, k) => (k === 0 ? context.moveTo(p.x, p.y) : context.lineTo(p.x, p.y)));
      context.closePath();
      context.fillStyle = withAlpha(theme.accent, 0.08);
      context.fill();
      context.setLineDash([5, 4]);
      context.lineWidth = 1.5;
      context.strokeStyle = theme.accent;
      context.stroke();
      context.setLineDash([]);
    }
  }, [size, screen, forceThumbs, faded, colourOf, rows, map.rows, hover, datasetId]);

  // one frame at a time, always with the latest drawing (a thumbnail that arrives redraws)
  const latest = useRef(draw);
  useEffect(() => {
    latest.current = draw;
  });
  const schedule = useCallback(() => {
    if (frame.current) return;
    frame.current = requestAnimationFrame(() => latest.current());
  }, []);

  useEffect(() => {
    latest.current = draw;
    redraw.current = schedule;
    schedule();
  }, [draw, schedule]);

  useEffect(() => () => cancelAnimationFrame(frame.current), []);

  // the theme or palette changes: read the tokens again
  useEffect(() => {
    const observer = new MutationObserver(() => {
      tokens.current = null;
      schedule();
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "data-palette"] });
    return () => observer.disconnect();
  }, [schedule]);

  const setView = (next: MapView) => {
    view.current = next;
    setZoomShown(Math.round(next.scale * 10) / 10);
    schedule();
  };

  // --- pointer -----------------------------------------------------------------------------------
  const local = (event: { clientX: number; clientY: number }): Point => {
    const rect = canvas.current?.getBoundingClientRect();
    return { x: event.clientX - (rect?.left ?? 0), y: event.clientY - (rect?.top ?? 0) };
  };

  const hit = (p: Point, radius: number) => {
    const { xs, ys } = screen();
    const side = thumbSide(view.current.scale, forceThumbs);
    return nearest(xs, ys, p.x, p.y, side >= THUMB_FROM ? Math.max(radius, side / 2) : radius, (i) => !faded(i));
  };

  const onWheel = (event: WheelEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    const p = local(event);
    setView(zoomView(view.current, Math.exp(-event.deltaY * 0.0015), p.x, p.y));
  };

  const onPointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    try {
      event.currentTarget.setPointerCapture?.(event.pointerId);
    } catch {
      // a pointer already gone: nothing to capture
    }
    const p = local(event);
    pointers.current.set(event.pointerId, p);
    const points = [...pointers.current.values()];
    const [a, b] = points;
    gesture.current = { start: p, distance: a && b ? Math.hypot(a.x - b.x, a.y - b.y) : 0, view: view.current, moved: false };
    if (tool === "lasso" && event.pointerType !== "touch" && points.length === 1) lasso.current = [p];
  };

  const onPointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const p = local(event);
    const start = gesture.current;
    if (!pointers.current.has(event.pointerId) || !start) {
      if (event.pointerType === "mouse" && !hover?.pinned) {
        const index = hit(p, HIT);
        if (index !== (hover?.index ?? -1)) setHover(index >= 0 ? { index, x: p.x, y: p.y, pinned: false } : null);
        else if (index >= 0 && hover) setHover({ ...hover, x: p.x, y: p.y });
      }
      return;
    }
    pointers.current.set(event.pointerId, p);
    if (Math.hypot(p.x - start.start.x, p.y - start.start.y) > 4) start.moved = true;
    const points = [...pointers.current.values()];
    if (points.length >= 2) {
      const [a, b] = points;
      if (!a || !b || !start.distance) return;
      lasso.current = null;
      const factor = Math.hypot(a.x - b.x, a.y - b.y) / start.distance;
      setView(zoomView(start.view, factor, (a.x + b.x) / 2, (a.y + b.y) / 2));
      return;
    }
    if (lasso.current) {
      lasso.current.push(p);
      schedule();
      return;
    }
    if (start.view.scale > 1 || tool === "pan") {
      const next = { scale: start.view.scale, x: start.view.x + p.x - start.start.x, y: start.view.y + p.y - start.start.y };
      setView(next.scale === 1 ? START : next);
    }
  };

  const onPointerUp = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const start = gesture.current;
    pointers.current.delete(event.pointerId);
    if (pointers.current.size > 0) return;
    gesture.current = null;
    const path = lasso.current;
    lasso.current = null;
    if (path && path.length > 3 && start?.moved) {
      const { xs, ys } = screen();
      const picked: number[] = [];
      for (let i = 0; i < xs.length; i += 1) if (insidePolygon(xs[i] ?? 0, ys[i] ?? 0, path)) picked.push(map.rows[i] ?? -1);
      onSelect(picked.filter((row) => row >= 0));
      schedule();
      return;
    }
    schedule();
    if (start && !start.moved) {
      const p = local(event);
      const index = hit(p, event.pointerType === "touch" ? TOUCH_HIT : HIT);
      if (index < 0) {
        setHover(null);
        return;
      }
      const row = rows[map.rows[index] ?? -1];
      if (event.pointerType === "touch" || touch) setHover({ index, x: p.x, y: p.y, pinned: true });
      else if (row) onOpen(row);
    }
  };

  const onKeyDown = (event: KeyboardEvent<HTMLCanvasElement>) => {
    const { scale, x, y } = view.current;
    const step = 40;
    const moves: Record<string, MapView> = {
      ArrowLeft: { scale, x: x + step, y },
      ArrowRight: { scale, x: x - step, y },
      ArrowUp: { scale, x, y: y + step },
      ArrowDown: { scale, x, y: y - step },
    };
    const moved = moves[event.key];
    if (moved && scale > 1) {
      event.preventDefault();
      setView(moved);
    } else if (event.key === "+" || event.key === "=") setView(zoomView(view.current, 1.4, size.width / 2, size.height / 2));
    else if (event.key === "-") setView(zoomView(view.current, 1 / 1.4, size.width / 2, size.height / 2));
    else if (event.key === "0") setView(START);
    else if (event.key === "Escape") setHover(null);
  };

  const hovered = hover ? rows[map.rows[hover.index] ?? -1] : undefined;
  const cluster = hovered ? map.clusters[hovered.i] : undefined;
  const groupName = cluster !== undefined && cluster >= 0 ? t("vision.map.groupN", { n: String(cluster + 1) }) : null;
  const iconButton =
    "relative inline-flex size-8 items-center justify-center rounded-[5px] text-ink-2 hover:bg-surface-2 hover:text-ink-1 disabled:opacity-40 aria-pressed:bg-surface-2 aria-pressed:text-ink-1 pointer-coarse:after:absolute pointer-coarse:after:-inset-1.5 pointer-coarse:after:content-['']";

  return (
    <div ref={box} className="relative min-w-0 overflow-hidden rounded-md border border-line bg-surface-1" style={{ height }}>
      <canvas
        ref={canvas}
        role="img"
        aria-label={label}
        tabIndex={0}
        className={`block size-full touch-none focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-accent ${tool === "lasso" ? "cursor-crosshair" : hovered && !hover?.pinned ? "cursor-pointer" : "cursor-grab active:cursor-grabbing"}`}
        style={{ width: size.width || "100%", height }}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerLeave={(event) => event.pointerType === "mouse" && !hover?.pinned && setHover(null)}
        onDoubleClick={() => setView(START)}
        onKeyDown={onKeyDown}
      />

      <div className="absolute top-2 right-2 flex items-center gap-0.5 rounded-md border border-line bg-surface-1/95 p-0.5 shadow-[0_4px_12px_-8px_rgb(0_0_0/0.35)]">
        {!touch && (
          <>
            <button type="button" className={iconButton} aria-pressed={tool === "pan"} onClick={() => setTool("pan")} aria-label={t("vision.map.tool.pan")} title={t("vision.map.tool.pan")}>
              <Hand size={15} strokeWidth={1.5} aria-hidden />
            </button>
            <button type="button" className={iconButton} aria-pressed={tool === "lasso"} onClick={() => setTool("lasso")} aria-label={t("vision.map.tool.lasso")} title={t("vision.map.tool.lasso")}>
              <LassoSelect size={15} strokeWidth={1.5} aria-hidden />
            </button>
            <span className="mx-0.5 h-5 w-px bg-line" aria-hidden />
          </>
        )}
        <button type="button" className={iconButton} aria-pressed={forceThumbs} onClick={() => { setForceThumbs((on) => !on); schedule(); }} aria-label={t("vision.map.thumbs")} title={t("vision.map.thumbs")}>
          <Images size={15} strokeWidth={1.5} aria-hidden />
        </button>
        <button type="button" className={iconButton} onClick={() => setView(zoomView(view.current, 1 / 1.5, size.width / 2, size.height / 2))} disabled={zoomShown <= 1} aria-label={t("vision.map.zoomOut")} title={t("vision.map.zoomOut")}>
          <Minus size={15} strokeWidth={1.5} aria-hidden />
        </button>
        <button type="button" className={iconButton} onClick={() => setView(zoomView(view.current, 1.5, size.width / 2, size.height / 2))} aria-label={t("vision.map.zoomIn")} title={t("vision.map.zoomIn")}>
          <Plus size={15} strokeWidth={1.5} aria-hidden />
        </button>
        <button type="button" className={iconButton} onClick={() => setView(START)} disabled={zoomShown <= 1} aria-label={t("vision.map.reset")} title={t("vision.map.reset")}>
          <Expand size={14} strokeWidth={1.5} aria-hidden />
        </button>
      </div>

      {hovered && hover && !hover.pinned && (
        <div
          role="tooltip"
          aria-label={hovered.name}
          className="pointer-events-none absolute z-[2] w-[184px] rounded-md border border-line bg-surface-1 p-2 text-xs shadow-[0_10px_28px_-12px_rgb(0_0_0/0.45)]"
          style={tipPosition(hover, size)}
        >
          <img src={thumbUrl(datasetId, hovered.id)} alt="" className="mb-1.5 aspect-square w-full rounded-[3px] bg-surface-2 object-cover outline outline-1 -outline-offset-1 outline-line" />
          <p className="truncate font-medium text-ink-1">{hovered.name}</p>
          <p className="truncate text-ink-2">{[hovered.label, groupName].filter(Boolean).join(" · ")}</p>
          {describe(hovered) && <p className="truncate text-ink-3 tabular-nums">{describe(hovered)}</p>}
        </div>
      )}

      {/* a tap (§ 9.3): a small sheet at the foot of the map, the image and what to do with it */}
      {hovered && hover?.pinned && (
        <div role="dialog" aria-label={hovered.name} className="absolute inset-x-2 bottom-2 z-[2] flex gap-2.5 rounded-md border border-line bg-surface-1 p-2 text-xs shadow-[0_10px_28px_-12px_rgb(0_0_0/0.45)]">
          <img src={thumbUrl(datasetId, hovered.id)} alt="" className="size-[88px] shrink-0 rounded-[3px] bg-surface-2 object-cover outline outline-1 -outline-offset-1 outline-line" />
          <div className="flex min-w-0 flex-1 flex-col">
            <div className="flex items-start gap-1">
              <div className="min-w-0 flex-1">
                <p className="truncate font-medium text-ink-1">{hovered.name}</p>
                <p className="truncate text-ink-2">{[hovered.label, groupName].filter(Boolean).join(" · ")}</p>
                {describe(hovered) && <p className="truncate text-ink-3 tabular-nums">{describe(hovered)}</p>}
              </div>
              <button type="button" onClick={() => setHover(null)} aria-label={t("vision.map.closeCard")} className="-mt-1 -mr-1 inline-flex size-11 shrink-0 items-center justify-center rounded-md text-ink-3 hover:bg-surface-2 hover:text-ink-1">
                <X size={16} strokeWidth={1.5} aria-hidden />
              </button>
            </div>
            <div className="mt-auto flex gap-1.5">
              <button type="button" className="min-h-11 flex-1 rounded-md border border-line px-2 text-ink-1 hover:bg-surface-2" onClick={() => onOpen(hovered)}>
                {t("vision.map.openImage")}
              </button>
              {cluster !== undefined && cluster >= 0 && (
                <button
                  type="button"
                  className="min-h-11 flex-1 rounded-md border border-line px-2 text-ink-1 hover:bg-surface-2"
                  onClick={() => {
                    onSelectGroup(cluster);
                    setHover(null);
                  }}
                >
                  {t("vision.map.selectGroup", { n: String(cluster + 1) })}
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      <p className="pointer-events-none absolute bottom-2 left-3 text-[11px] text-ink-3 tabular-nums" aria-live="polite">
        {zoomShown > 1 ? `${formatNumber(zoomShown)}×` : touch ? t("vision.map.hintTouch") : tool === "lasso" ? t("vision.map.hintLasso") : t("vision.map.hint")}
      </p>
    </div>
  );
}

type ThumbStore = { images: Map<string, HTMLImageElement | "failed">; loading: number };

const thumbUrl = (datasetId: string, imageId: string) => `/api/v1/datasets/${encodeURIComponent(datasetId)}/images/${imageId}/thumb`;

/** A thumbnail when loaded; otherwise starts loading it (24 at a time) and redraws when it comes. */
function thumbnail(store: ThumbStore, url: string, onLoad: () => void): HTMLImageElement | null {
  const found = store.images.get(url);
  if (found === "failed") return null;
  if (found) return found.complete && found.naturalWidth ? found : null;
  if (store.loading >= 24) return null;
  const image = new Image();
  image.decoding = "async";
  store.loading += 1;
  image.onload = () => {
    store.loading -= 1;
    onLoad();
  };
  image.onerror = () => {
    store.loading -= 1;
    store.images.set(url, "failed");
  };
  image.src = url;
  store.images.set(url, image);
  return null;
}

function tipPosition(hover: { x: number; y: number }, size: { width: number; height: number }) {
  const width = 184;
  const height = 250;
  const left = hover.x + 14 + width > size.width ? Math.max(4, hover.x - 14 - width) : hover.x + 14;
  const top = Math.min(Math.max(4, hover.y - height / 2), Math.max(4, size.height - height - 4));
  return { left, top };
}

function drawCover(context: CanvasRenderingContext2D, image: HTMLImageElement, x: number, y: number, side: number) {
  const w = image.naturalWidth;
  const h = image.naturalHeight;
  const crop = Math.min(w, h);
  context.drawImage(image, (w - crop) / 2, (h - crop) / 2, crop, crop, x, y, side, side);
}

function withAlpha(colour: string, alpha: number): string {
  const [r, g, b] = parseColor(colour);
  return `rgb(${r} ${g} ${b} / ${alpha})`;
}

/** The legend's name of a categorical key. */
export function legendName(key: string, by: string, ruleTitle: (code: string) => string | undefined): string {
  if (key === OTHER) return t("vision.map.other");
  if (key === "__none__") return t(`vision.map.none.${by}` as MessageKey);
  if (by === "cluster") return t("vision.map.groupN", { n: String(Number(key) + 1) });
  if (by === "quality") return ruleTitle(key) ? `${key} ${ruleTitle(key)}` : key;
  return key;
}
