import { Layers, Maximize, Minus, Plus, ScanSearch } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type PointerEvent, type SyntheticEvent, type WheelEvent } from "react";

import { t } from "../../lib/i18n";

const MAX_SCALE = 16;
const SWIPE = 60; // px of horizontal travel that turns to the next image (not zoomed in)
const INSPECT_SIDE = 2048; // the copy read by the pixel inspector, long side

type View = { scale: number; x: number; y: number };

type ViewerProps = {
  src: string;
  maskSrc?: string | null;
  alt: string;
  /** Swipe left / right on a touch screen when not zoomed in. */
  onSwipe?: (direction: 1 | -1) => void;
  className?: string;
};

/**
 * 107: the original image, fitted to its frame. Wheel or pinch to zoom (around the pointer), drag to
 * pan, Fit and 1:1; + − 0 on the keyboard. The pixel inspector reads RGB and luma under the
 * pointer; the defect mask (R-VIS-LAY-04) can be laid over in the theme's alert colour. Moves go
 * straight to the element's style (no re-render per frame).
 */
export function ImageViewer({ src, maskSrc, alt, onSwipe, className = "" }: ViewerProps) {
  const frame = useRef<HTMLDivElement>(null);
  const layer = useRef<HTMLDivElement>(null);
  const readout = useRef<HTMLSpanElement>(null);
  const view = useRef<View>({ scale: 1, x: 0, y: 0 });
  const natural = useRef({ width: 0, height: 0 });
  const fit = useRef(1);
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const gesture = useRef<{ x: number; y: number; distance: number; view: View; startX: number } | null>(null);
  const sampler = useRef<{ context: CanvasRenderingContext2D; ratio: number } | null>(null);
  // shown (percentage, cursor); the transform itself lives in `view`
  const [zoom, setZoom] = useState({ scale: 1, fit: 1 });
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  const [mask, setMask] = useState(true);
  const [inspect, setInspect] = useState(false);

  const apply = useCallback((next: View) => {
    view.current = next;
    if (layer.current) layer.current.style.transform = `translate(${next.x}px, ${next.y}px) scale(${next.scale})`;
    setZoom((shown) => ({ ...shown, scale: next.scale }));
  }, []);

  /** Keep the image on screen: centred when smaller than the frame, edges never past the frame. */
  const clamp = useCallback((next: View): View => {
    const box = frame.current;
    if (!box) return next;
    const w = natural.current.width * next.scale;
    const h = natural.current.height * next.scale;
    const fw = box.clientWidth;
    const fh = box.clientHeight;
    const x = w <= fw ? (fw - w) / 2 : Math.min(0, Math.max(fw - w, next.x));
    const y = h <= fh ? (fh - h) / 2 : Math.min(0, Math.max(fh - h, next.y));
    return { scale: next.scale, x, y };
  }, []);

  const toFit = useCallback(() => {
    const box = frame.current;
    if (!box || !natural.current.width) return;
    fit.current = Math.min(box.clientWidth / natural.current.width, box.clientHeight / natural.current.height, 8);
    apply(clamp({ scale: fit.current, x: 0, y: 0 }));
    setZoom({ scale: fit.current, fit: fit.current });
  }, [apply, clamp]);

  const zoomAt = useCallback(
    (factor: number, cx?: number, cy?: number) => {
      const box = frame.current;
      if (!box) return;
      const current = view.current;
      const scale = Math.min(MAX_SCALE, Math.max(fit.current, current.scale * factor));
      const px = cx ?? box.clientWidth / 2;
      const py = cy ?? box.clientHeight / 2;
      const ratio = scale / current.scale;
      apply(clamp({ scale, x: px - (px - current.x) * ratio, y: py - (py - current.y) * ratio }));
    },
    [apply, clamp],
  );

  // A new image: back to "fit"; the frame resizing (rotation, dialog) refits too.
  useEffect(() => {
    const box = frame.current;
    if (!box || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => {
      if (view.current.scale <= fit.current + 1e-6) toFit();
      else apply(clamp(view.current));
    });
    observer.observe(box);
    return () => observer.disconnect();
  }, [apply, clamp, toFit]);

  const onLoad = (event: SyntheticEvent<HTMLImageElement>) => {
    const image = event.currentTarget;
    natural.current = { width: image.naturalWidth, height: image.naturalHeight };
    sampler.current = null;
    setLoaded(true);
    toFit();
  };

  const sample = (clientX: number, clientY: number) => {
    const box = frame.current?.getBoundingClientRect();
    const target = readout.current;
    const image = layer.current?.querySelector("img");
    if (!box || !target || !image || !natural.current.width) return;
    const { scale, x, y } = view.current;
    const ix = Math.floor((clientX - box.left - x) / scale);
    const iy = Math.floor((clientY - box.top - y) / scale);
    if (ix < 0 || iy < 0 || ix >= natural.current.width || iy >= natural.current.height) {
      target.textContent = t("vision.viewer.outside");
      return;
    }
    if (!sampler.current) {
      const ratio = Math.min(1, INSPECT_SIDE / Math.max(natural.current.width, natural.current.height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(natural.current.width * ratio));
      canvas.height = Math.max(1, Math.round(natural.current.height * ratio));
      const context = canvas.getContext("2d", { willReadFrequently: true });
      if (!context) return;
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      sampler.current = { context, ratio };
    }
    const { context, ratio } = sampler.current;
    const [r = 0, g = 0, b = 0] = context.getImageData(Math.floor(ix * ratio), Math.floor(iy * ratio), 1, 1).data;
    const luma = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    target.textContent = t("vision.viewer.pixel", {
      x: String(ix),
      y: String(iy),
      r: String(r),
      g: String(g),
      b: String(b),
      luma: luma.toFixed(2),
    });
  };

  const onWheel = (event: WheelEvent<HTMLDivElement>) => {
    const box = frame.current?.getBoundingClientRect();
    if (!box) return;
    zoomAt(Math.exp(-event.deltaY * 0.0015), event.clientX - box.left, event.clientY - box.top);
  };

  // React attaches wheel listeners as passive: block the page scroll from the frame itself.
  useEffect(() => {
    const box = frame.current;
    if (!box) return;
    const stop = (event: globalThis.WheelEvent) => event.preventDefault();
    box.addEventListener("wheel", stop, { passive: false });
    return () => box.removeEventListener("wheel", stop);
  }, []);

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    frame.current?.setPointerCapture(event.pointerId);
    pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    const points = [...pointers.current.values()];
    const [a, b] = points;
    gesture.current = {
      x: a && b ? (a.x + b.x) / 2 : event.clientX,
      y: a && b ? (a.y + b.y) / 2 : event.clientY,
      distance: a && b ? Math.hypot(a.x - b.x, a.y - b.y) : 0,
      view: view.current,
      startX: event.clientX,
    };
  };

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (inspect && event.pointerType === "mouse") sample(event.clientX, event.clientY);
    if (!pointers.current.has(event.pointerId)) return;
    pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    const start = gesture.current;
    if (!start) return;
    const box = frame.current?.getBoundingClientRect();
    const points = [...pointers.current.values()];
    const [a, b] = points;
    if (a && b && start.distance > 0 && box) {
      const middle = { x: (a.x + b.x) / 2 - box.left, y: (a.y + b.y) / 2 - box.top };
      const scale = Math.min(MAX_SCALE, Math.max(fit.current, start.view.scale * (Math.hypot(a.x - b.x, a.y - b.y) / start.distance)));
      const ratio = scale / start.view.scale;
      apply(clamp({ scale, x: middle.x - (start.x - box.left - start.view.x) * ratio, y: middle.y - (start.y - box.top - start.view.y) * ratio }));
      return;
    }
    if (start.view.scale > fit.current + 1e-6) {
      apply(clamp({ ...start.view, x: start.view.x + event.clientX - start.x, y: start.view.y + event.clientY - start.y }));
    }
  };

  const onPointerUp = (event: PointerEvent<HTMLDivElement>) => {
    const start = gesture.current;
    const single = pointers.current.size === 1;
    pointers.current.delete(event.pointerId);
    if (single && start && onSwipe && view.current.scale <= fit.current + 1e-6) {
      const travel = event.clientX - start.startX;
      if (Math.abs(travel) > SWIPE) onSwipe(travel < 0 ? 1 : -1);
    }
    gesture.current = null;
  };

  const percent = Math.round(zoom.scale * 100);
  const zoomed = zoom.scale > zoom.fit + 1e-6;
  const control =
    "inline-flex h-7 min-w-7 items-center justify-center gap-1 rounded-md px-1.5 text-xs text-ink-2 hover:bg-surface-2 hover:text-ink-1 aria-pressed:bg-surface-2 aria-pressed:text-ink-1 pointer-coarse:min-h-11 pointer-coarse:min-w-11";

  return (
    <div className={`flex min-h-0 flex-col gap-2 ${className}`}>
      <div
        ref={frame}
        role="img"
        aria-label={alt}
        tabIndex={0}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onDoubleClick={(event) => {
          const box = frame.current?.getBoundingClientRect();
          if (view.current.scale > fit.current + 1e-6) toFit();
          else if (box) zoomAt(2.5, event.clientX - box.left, event.clientY - box.top);
        }}
        onKeyDown={(event) => {
          if (event.key === "+" || event.key === "=") zoomAt(1.25);
          else if (event.key === "-") zoomAt(0.8);
          else if (event.key === "0") toFit();
          else return;
          event.preventDefault();
        }}
        className={`relative min-h-0 flex-1 touch-none overflow-hidden rounded-md bg-surface-2 outline-none focus-visible:ring-2 focus-visible:ring-accent/40 ${
          inspect ? "cursor-crosshair" : zoomed ? "cursor-grab active:cursor-grabbing" : "cursor-zoom-in"
        }`}
      >
        {failed ? (
          <p className="absolute inset-0 flex items-center justify-center p-6 text-center text-[13px] text-ink-2">{t("vision.viewer.failed")}</p>
        ) : (
          <div ref={layer} className="absolute top-0 left-0 origin-top-left will-change-transform" style={{ opacity: loaded ? 1 : 0 }}>
            <img
              src={src}
              alt=""
              draggable={false}
              onLoad={onLoad}
              onError={() => setFailed(true)}
              className="block max-w-none select-none [image-rendering:auto]"
              style={zoom.scale >= 4 ? { imageRendering: "pixelated" } : undefined}
            />
            {maskSrc && mask && loaded && (
              <span
                aria-hidden
                className="absolute inset-0 bg-bad opacity-55"
                // the mask is a grey PNG (white = defect): read by its luminance, not its alpha
                style={{ maskImage: `url("${maskSrc}")`, maskMode: "luminance", maskSize: "100% 100%", WebkitMaskImage: `url("${maskSrc}")`, WebkitMaskSize: "100% 100%" }}
              />
            )}
          </div>
        )}
        {inspect && (
          <span
            ref={readout}
            aria-live="off"
            className="pointer-events-none absolute bottom-2 left-2 rounded-[4px] border border-line bg-surface-1/95 px-2 py-0.5 font-mono text-[11px] text-ink-1 tabular-nums"
          >
            {t("vision.viewer.hover")}
          </span>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-1" role="toolbar" aria-label={t("vision.viewer.tools")}>
        <button type="button" className={control} onClick={() => zoomAt(0.8)} aria-label={t("vision.viewer.zoomOut")} title={t("vision.viewer.zoomOut")}>
          <Minus size={14} strokeWidth={1.5} aria-hidden />
        </button>
        <span className="w-12 text-center text-xs text-ink-2 tabular-nums" aria-live="polite">
          {percent}%
        </span>
        <button type="button" className={control} onClick={() => zoomAt(1.25)} aria-label={t("vision.viewer.zoomIn")} title={t("vision.viewer.zoomIn")}>
          <Plus size={14} strokeWidth={1.5} aria-hidden />
        </button>
        <button type="button" className={control} onClick={toFit}>
          <Maximize size={13} strokeWidth={1.5} aria-hidden />
          {t("vision.viewer.fit")}
        </button>
        <button type="button" className={control} onClick={() => apply(clamp({ ...view.current, scale: 1 }))}>
          1:1
        </button>
        <span className="mx-1 h-4 w-px bg-line" aria-hidden />
        <button type="button" className={`${control} max-sm:hidden`} aria-pressed={inspect} onClick={() => setInspect(!inspect)}>
          <ScanSearch size={14} strokeWidth={1.5} aria-hidden />
          {t("vision.viewer.inspect")}
        </button>
        {maskSrc && (
          <button type="button" className={control} aria-pressed={mask} onClick={() => setMask(!mask)}>
            <Layers size={14} strokeWidth={1.5} aria-hidden />
            {t("vision.viewer.mask")}
          </button>
        )}
      </div>
    </div>
  );
}
