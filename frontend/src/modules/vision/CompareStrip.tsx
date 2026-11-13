import { Maximize, Minus, Plus } from "lucide-react";
import { useCallback, useEffect, useRef, type CSSProperties, type KeyboardEvent, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";

import { t } from "../../lib/i18n";
import { clampView, placement } from "../../ui/BeforeAfter";
import type { WhyMap } from "./explainApi";
import { boxStyle, HeatmapLayer, type HeatSettings } from "./HeatmapOverlay";

/**
 * 117 (study § 9.2): two to five images side by side with one zoom and one pan, so the same
 * detail stays under the eye in all of them, whatever their pixel sizes: the image, its map, the
 * nearest training image, a prototype, the nearest normal patch. The view lives in the images' own
 * relative coordinates (the geometry of `ui/BeforeAfter`); Ctrl + wheel, pinch, drag, + − 0 and a
 * double click move it, straight on the elements' style, never through a React render.
 */

export type StripBox = { box: readonly number[]; label: string };

export type StripPanel = {
  key: string;
  label: string;
  src: string | null;
  alt: string;
  heat?: { map: WhyMap; settings: HeatSettings } | null;
  boxes?: readonly StripBox[];
  caption?: ReactNode;
  /** A click on the label (or Enter on it): open that image's own "Why?". */
  onOpen?: () => void;
};

type View = { zoom: number; cx: number; cy: number };
type Frame = { frame: HTMLElement; stage: HTMLElement; src: string | null };
type Size = { width: number; height: number };

const FIT: View = { zoom: 1, cx: 0.5, cy: 0.5 };
const STEP = 1.5;

type StripProps = {
  panels: readonly StripPanel[];
  label: string;
  /** Width / height of every frame. */
  aspect?: string;
  className?: string;
};

export function CompareStrip({ panels, label, aspect = "1 / 1", className = "" }: StripProps) {
  const view = useRef<View>(FIT);
  const frames = useRef(new Map<string, Frame>());
  const sizes = useRef(new Map<string, Size>()); // by image source: kept across renders
  const readout = useRef<HTMLSpanElement>(null);

  /** The view, clamped on the first measured image, placed on every stage. */
  const apply = useCallback(() => {
    const sized = [...frames.current.values()].flatMap((f) => {
      const size = f.src ? sizes.current.get(f.src) : undefined;
      return size ? [{ ...f, ...size }] : [];
    });
    const first = sized[0];
    if (first) {
      const rect = first.frame.getBoundingClientRect();
      view.current = clampView(view.current, rect.width, rect.height, first.width, first.height);
    }
    for (const f of sized) {
      const rect = f.frame.getBoundingClientRect();
      const place = placement(rect.width, rect.height, f.width, f.height, view.current);
      f.stage.style.width = `${f.width}px`;
      f.stage.style.height = `${f.height}px`;
      f.stage.style.transform = `translate(${place.x}px, ${place.y}px) scale(${place.scale})`;
      f.stage.style.setProperty("--inverse", String(1 / place.scale));
      f.stage.style.visibility = "visible";
    }
    if (readout.current) readout.current.textContent = `${Math.round(view.current.zoom * 100)} %`;
  }, []);

  const register = (key: string, src: string | null) => (frame: HTMLElement | null) => {
    if (!frame) {
      frames.current.delete(key);
      return;
    }
    const stage = frame.querySelector<HTMLElement>("[data-stage]");
    if (stage) frames.current.set(key, { frame, stage, src });
  };

  const measured = (src: string, image: HTMLImageElement) => {
    sizes.current.set(src, { width: image.naturalWidth, height: image.naturalHeight });
    apply();
  };

  /** A frame with the size of its image, once the image has loaded. */
  const sized = useCallback((f: Frame | undefined): (Frame & Size) | undefined => {
    const size = f?.src ? sizes.current.get(f.src) : undefined;
    return f && size ? { ...f, ...size } : undefined;
  }, []);

  /** Zoom by `factor` keeping the image point under (px, py) of `key`'s frame where it is. */
  const zoomAt = useCallback(
    (factor: number, key?: string, px?: number, py?: number) => {
      const f = sized(key ? frames.current.get(key) : undefined) ?? [...frames.current.values()].map(sized).find(Boolean);
      const before = view.current;
      const zoom = Math.min(16, Math.max(1, before.zoom * factor));
      if (!f) {
        view.current = { ...before, zoom };
        apply();
        return;
      }
      const rect = f.frame.getBoundingClientRect();
      const x = px ?? rect.width / 2;
      const y = py ?? rect.height / 2;
      const old = placement(rect.width, rect.height, f.width, f.height, before);
      const u = (x - old.x) / (f.width * old.scale);
      const v = (y - old.y) / (f.height * old.scale);
      const scale = old.scale * (zoom / before.zoom);
      view.current = { zoom, cx: u - (x - rect.width / 2) / (f.width * scale), cy: v - (y - rect.height / 2) / (f.height * scale) };
      apply();
    },
    [apply, sized],
  );

  const panBy = (key: string, dx: number, dy: number) => {
    const f = sized(frames.current.get(key));
    if (!f) return;
    const rect = f.frame.getBoundingClientRect();
    const { scale } = placement(rect.width, rect.height, f.width, f.height, view.current);
    view.current = { ...view.current, cx: view.current.cx - dx / (f.width * scale), cy: view.current.cy - dy / (f.height * scale) };
    apply();
  };

  const fit = useCallback(() => {
    view.current = FIT;
    apply();
  }, [apply]);

  // wheel zoom needs a non-passive listener (React's is passive); resizes re-place every stage
  const keys = panels.map((panel) => `${panel.key}:${panel.src ?? ""}`).join("|");
  useEffect(() => {
    const cleanups: (() => void)[] = [];
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(() => apply());
    for (const [key, f] of frames.current) {
      const onWheel = (event: WheelEvent) => {
        // the plain wheel scrolls the page: zoom with Ctrl (a trackpad pinch sends Ctrl + wheel)
        if (!event.ctrlKey && !event.metaKey) return;
        event.preventDefault();
        const rect = f.frame.getBoundingClientRect();
        zoomAt(event.deltaY < 0 ? 1.2 : 1 / 1.2, key, event.clientX - rect.left, event.clientY - rect.top);
      };
      f.frame.addEventListener("wheel", onWheel, { passive: false });
      observer?.observe(f.frame);
      cleanups.push(() => f.frame.removeEventListener("wheel", onWheel));
    }
    apply();
    return () => {
      for (const cleanup of cleanups) cleanup();
      observer?.disconnect();
    };
  }, [keys, apply, zoomAt]);

  // drag to pan, two fingers to pinch
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const pinch = useRef<number | null>(null);
  const onPointerDown = (event: ReactPointerEvent<HTMLElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    pinch.current = null;
  };
  const onPointerMove = (key: string) => (event: ReactPointerEvent<HTMLElement>) => {
    const previous = pointers.current.get(event.pointerId);
    if (!previous) return;
    const now = { x: event.clientX, y: event.clientY };
    pointers.current.set(event.pointerId, now);
    const all = [...pointers.current.values()];
    if (all.length >= 2) {
      const [a, b] = all as [{ x: number; y: number }, { x: number; y: number }];
      const distance = Math.hypot(a.x - b.x, a.y - b.y);
      const rect = event.currentTarget.getBoundingClientRect();
      if (pinch.current) zoomAt(distance / pinch.current, key, (a.x + b.x) / 2 - rect.left, (a.y + b.y) / 2 - rect.top);
      pinch.current = distance;
      return;
    }
    if (view.current.zoom > 1) panBy(key, now.x - previous.x, now.y - previous.y);
  };
  const onPointerUp = (event: ReactPointerEvent<HTMLElement>) => {
    pointers.current.delete(event.pointerId);
    if (pointers.current.size < 2) pinch.current = null;
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "+" || event.key === "=") zoomAt(STEP);
    else if (event.key === "-") zoomAt(1 / STEP);
    else if (event.key === "0") fit();
    else return;
    event.preventDefault();
  };

  // two panels stay a pair of comfortable squares rather than filling a wide dialog
  const columns = { "--strip-columns": `repeat(${panels.length}, minmax(0, 1fr))`, maxWidth: `${panels.length * 300}px` } as CSSProperties;

  return (
    <div className={`min-w-0 space-y-2 ${className}`} role="group" aria-label={label} tabIndex={0} onKeyDown={onKeyDown}>
      <div
        className="grid min-w-0 grid-cols-2 gap-3 sm:grid-cols-(--strip-columns)"
        style={columns}
      >
        {panels.map((panel) => (
          <figure key={panel.key} className="m-0 min-w-0 space-y-1.5">
            <figcaption className="flex min-w-0 items-baseline justify-between gap-2 text-xs">
              {panel.onOpen ? (
                <button type="button" onClick={panel.onOpen} className="min-w-0 truncate text-left font-medium text-ink-1 underline-offset-2 hover:underline pointer-coarse:min-h-11">
                  {panel.label}
                </button>
              ) : (
                <span className="min-w-0 truncate font-medium text-ink-1">{panel.label}</span>
              )}
            </figcaption>
            <div
              ref={register(panel.key, panel.src)}
              className="relative cursor-grab touch-none overflow-hidden rounded-md bg-surface-2 select-none active:cursor-grabbing"
              style={{ aspectRatio: aspect }}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove(panel.key)}
              onPointerUp={onPointerUp}
              onPointerCancel={onPointerUp}
              onDoubleClick={(event) => {
                const rect = event.currentTarget.getBoundingClientRect();
                if (view.current.zoom > 1.01) fit();
                else zoomAt(3, panel.key, event.clientX - rect.left, event.clientY - rect.top);
              }}
            >
              <div data-stage className="invisible absolute top-0 left-0 origin-top-left will-change-transform">
                {panel.src ? (
                  <img
                    src={panel.src}
                    alt={panel.alt}
                    draggable={false}
                    decoding="async"
                    onLoad={(event) => measured(panel.src ?? "", event.currentTarget)}
                    className="block size-full max-w-none"
                  />
                ) : (
                  <span className="block size-full bg-surface-2" />
                )}
                {panel.heat && (
                  <div className="absolute" style={boxStyle(panel.heat.map.box)}>
                    <HeatmapLayer map={panel.heat.map} settings={panel.heat.settings} />
                  </div>
                )}
                {panel.boxes?.map((box) => (
                  <div
                    key={box.label}
                    title={box.label}
                    className="pointer-events-none absolute rounded-[2px] border-accent"
                    style={{
                      ...boxStyle(box.box),
                      borderStyle: "solid",
                      borderWidth: "calc(2px * var(--inverse, 1))",
                      boxShadow: "0 0 0 calc(1px * var(--inverse, 1)) rgb(255 255 255 / 0.9)",
                    }}
                  />
                ))}
              </div>
              <span aria-hidden className="pointer-events-none absolute inset-0 rounded-md ring-1 ring-ink-1/8 ring-inset" />
            </div>
            {panel.caption && <div className="min-w-0 text-xs text-ink-2">{panel.caption}</div>}
          </figure>
        ))}
      </div>
      <div className="flex items-center gap-1 text-xs text-ink-3">
        <ZoomButton label={t("vision.why.strip.zoomOut")} onClick={() => zoomAt(1 / STEP)}>
          <Minus size={14} strokeWidth={1.5} aria-hidden />
        </ZoomButton>
        <span ref={readout} className="w-11 text-center text-ink-2 tabular-nums" aria-live="polite">
          100 %
        </span>
        <ZoomButton label={t("vision.why.strip.zoomIn")} onClick={() => zoomAt(STEP)}>
          <Plus size={14} strokeWidth={1.5} aria-hidden />
        </ZoomButton>
        <ZoomButton label={t("vision.why.strip.fit")} onClick={fit}>
          <Maximize size={13} strokeWidth={1.5} aria-hidden />
        </ZoomButton>
        <span className="ml-2 max-sm:hidden">{t("vision.why.strip.hint")}</span>
      </div>
    </div>
  );
}

function ZoomButton({ label, onClick, children }: { label: string; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className="inline-flex size-7 items-center justify-center rounded-md text-ink-2 hover:bg-surface-2 hover:text-ink-1 pointer-coarse:size-11"
    >
      {children}
    </button>
  );
}
