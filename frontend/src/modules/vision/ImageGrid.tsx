import { useVirtualizer } from "@tanstack/react-virtual";
import { Ban, Check, FileWarning, TriangleAlert } from "lucide-react";
import {
  memo,
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent,
  type PointerEvent,
} from "react";

import { imageUrl } from "../../features/import/imageSetApi";
import { t } from "../../lib/i18n";
import { PHONE, TOUCH, useMedia } from "../../lib/useMedia";
import {
  clickSelection,
  exclusionCause,
  isExcluded,
  positionsInRect,
  type Decision,
  type Draft,
  type ImageRow,
} from "./model";
import type { Density } from "./search";

/** Side of a tile's square per density (R-107-05); the columns stretch to fill the width. */
export const CELL: Record<Density, number> = { s: 112, m: 160, l: 224 };
const GAP = 8;
const CAPTION = 36;
const PHONE_COLUMNS = 3;
const DRAG_THRESHOLD = 5;
const EDGE = 48; // px from the scroller's edge where a selection rectangle scrolls it

type GridProps = {
  datasetId: string;
  rows: readonly ImageRow[];
  density: Density;
  draft: Draft;
  selected: ReadonlySet<string>;
  onSelectedChange: (next: Set<string>) => void;
  onOpen: (row: ImageRow) => void;
  onDecide: (keys: string[], decision: Decision | null) => void;
};

export function gridLayout(width: number, density: Density, phone: boolean) {
  const usable = Math.max(width, 1);
  const columns = phone ? PHONE_COLUMNS : Math.max(2, Math.floor((usable + GAP) / (CELL[density] + GAP)));
  const cellWidth = Math.max(48, (usable - GAP * (columns - 1)) / columns);
  return { columns, cellWidth, rowHeight: cellWidth + CAPTION };
}

/**
 * 107: every image of the shown list as a grid of thumbnails, virtualized by rows of fixed height
 * (thousands stay fluid: only the rows on screen, plus a few, are in the page). It scrolls with
 * the page's <main>, the toolbar above stays stuck. Selection: click, Shift+click (range),
 * Ctrl/⌘+click, a rectangle dragged with the mouse, the arrows with Shift, Ctrl+A; Enter or a
 * double click opens the image; e / k / r decide on the selection (108).
 */
export function ImageGrid({ datasetId, rows, density, draft, selected, onSelectedChange, onOpen, onDecide }: GridProps) {
  const id = useId();
  const phone = useMedia(PHONE);
  const touch = useMedia(TOUCH);
  const container = useRef<HTMLDivElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(960);
  const [margin, setMargin] = useState(0);
  const [active, setActive] = useState<number | null>(null);
  const anchor = useRef<number | null>(null);
  const { columns, cellWidth, rowHeight } = gridLayout(width, density, phone);
  const rowCount = Math.ceil(rows.length / columns);
  const focus = active !== null && active < rows.length ? active : null;

  // Width (columns) and where the grid starts in <main>'s scroll (the stats above may fold).
  useEffect(() => {
    const element = container.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    const main = element.closest("main");
    const measure = () => {
      setWidth(element.clientWidth || 960);
      if (main) {
        const top = element.getBoundingClientRect().top - main.getBoundingClientRect().top + main.scrollTop;
        setMargin(Math.max(0, Math.round(top)));
      }
    };
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    if (main?.firstElementChild) observer.observe(main.firstElementChild);
    measure();
    return () => observer.disconnect();
  }, []);

  // The compiler skips memoizing this component (the virtualizer's functions change on scroll).
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => container.current?.closest("main") ?? null,
    estimateSize: () => rowHeight + GAP,
    overscan: 4,
    scrollMargin: margin,
    initialRect: { width: 1200, height: 900 },
  });
  useEffect(() => {
    virtualizer.measure();
  }, [rowHeight, virtualizer]);

  // Stable handlers for the memoized tiles, reading the latest state from refs.
  const latest = useRef({ rows, selected, onSelectedChange, onOpen, touch });
  useEffect(() => {
    latest.current = { rows, selected, onSelectedChange, onOpen, touch };
  });
  const suppressClick = useRef(false);

  const onTileClick = useCallback((position: number, event: MouseEvent) => {
    if (suppressClick.current) {
      suppressClick.current = false;
      return;
    }
    const state = latest.current;
    const row = state.rows[position];
    if (!row) return;
    const toggle = event.ctrlKey || event.metaKey;
    // On a touch screen a tap opens the image, unless a selection is under way.
    if (state.touch && !event.shiftKey && !toggle && state.selected.size === 0) {
      state.onOpen(row);
      return;
    }
    const next = clickSelection(state.selected, state.rows, position, anchor.current, {
      shift: event.shiftKey,
      toggle: toggle || (state.touch && state.selected.size > 0),
    });
    if (!event.shiftKey) anchor.current = position;
    setActive(position);
    state.onSelectedChange(next);
  }, []);

  const onTileToggle = useCallback((position: number) => {
    const state = latest.current;
    const row = state.rows[position];
    if (!row) return;
    const next = new Set(state.selected);
    if (next.has(row.key)) next.delete(row.key);
    else next.add(row.key);
    anchor.current = position;
    setActive(position);
    state.onSelectedChange(next);
  }, []);

  const onTileOpen = useCallback((position: number) => {
    const row = latest.current.rows[position];
    if (row) latest.current.onOpen(row);
  }, []);

  // --- keyboard ---------------------------------------------------------------------------------

  const moveTo = (position: number, extend: boolean) => {
    const target = Math.max(0, Math.min(rows.length - 1, position));
    setActive(target);
    virtualizer.scrollToIndex(Math.floor(target / columns), { align: "auto" });
    if (extend) {
      onSelectedChange(clickSelection(new Set(), rows, target, anchor.current ?? focus ?? target, { shift: true, toggle: false }));
    } else {
      anchor.current = target;
    }
  };

  const keysOfSelection = () => (selected.size > 0 ? [...selected] : focus !== null && rows[focus] ? [rows[focus].key] : []);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (rows.length === 0) return;
    const at = focus ?? 0;
    const page = columns * Math.max(1, Math.floor(window.innerHeight / (rowHeight + GAP)) - 1);
    const moves: Record<string, number> = {
      ArrowRight: 1,
      ArrowLeft: -1,
      ArrowDown: columns,
      ArrowUp: -columns,
      PageDown: page,
      PageUp: -page,
    };
    const toggle = event.ctrlKey || event.metaKey;
    if (event.key in moves) {
      event.preventDefault();
      moveTo(focus === null ? 0 : at + (moves[event.key] ?? 0), event.shiftKey);
    } else if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      moveTo(event.key === "Home" ? 0 : rows.length - 1, event.shiftKey);
    } else if (event.key === "Enter" && focus !== null) {
      event.preventDefault();
      const row = rows[focus];
      if (row) onOpen(row);
    } else if (event.key === " " && focus !== null) {
      event.preventDefault();
      onTileToggle(focus);
    } else if (toggle && event.key.toLowerCase() === "a") {
      event.preventDefault();
      onSelectedChange(new Set(rows.map((row) => row.key)));
    } else if (event.key === "Escape" && selected.size > 0) {
      event.preventDefault();
      event.stopPropagation();
      onSelectedChange(new Set());
    } else if (!toggle && !event.altKey && ["e", "k", "r"].includes(event.key.toLowerCase())) {
      const keys = keysOfSelection();
      if (keys.length === 0) return;
      event.preventDefault();
      const letter = event.key.toLowerCase();
      onDecide(keys, letter === "e" ? "exclude" : letter === "k" ? "keep" : null);
    }
  };

  // --- the selection rectangle (mouse only) -------------------------------------------------------

  const drag = useRef<{ x: number; y: number; base: Set<string>; moved: boolean; pointer: number } | null>(null);
  const [rect, setRect] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);

  const local = (event: PointerEvent) => {
    const box = content.current?.getBoundingClientRect();
    return { x: event.clientX - (box?.left ?? 0), y: event.clientY - (box?.top ?? 0) };
  };

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.pointerType !== "mouse" || event.button !== 0) return;
    if ((event.target as HTMLElement).closest("button")) return;
    const start = local(event);
    const additive = event.ctrlKey || event.metaKey || event.shiftKey;
    drag.current = { ...start, base: additive ? new Set(selected) : new Set(), moved: false, pointer: event.pointerId };
  };

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const state = drag.current;
    if (!state || state.pointer !== event.pointerId) return;
    const here = local(event);
    if (!state.moved && Math.hypot(here.x - state.x, here.y - state.y) < DRAG_THRESHOLD) return;
    if (!state.moved) {
      state.moved = true;
      content.current?.setPointerCapture(event.pointerId);
    }
    const next = { x0: state.x, y0: state.y, x1: here.x, y1: here.y };
    setRect(next);
    const hit = positionsInRect(next, { columns, cellWidth, rowHeight, gap: GAP, count: rows.length });
    const keys = new Set(state.base);
    for (const position of hit) {
      const key = rows[position]?.key;
      if (key !== undefined) keys.add(key);
    }
    onSelectedChange(keys);
    // near the top or bottom of the scroller, scroll it
    const main = container.current?.closest("main");
    if (main) {
      const box = main.getBoundingClientRect();
      if (event.clientY > box.bottom - EDGE) main.scrollBy({ top: 24 });
      else if (event.clientY < box.top + EDGE) main.scrollBy({ top: -24 });
    }
  };

  const endDrag = (event: PointerEvent<HTMLDivElement>) => {
    const state = drag.current;
    if (!state) return;
    if (state.moved) {
      suppressClick.current = true;
      content.current?.releasePointerCapture(event.pointerId);
    }
    drag.current = null;
    setRect(null);
  };

  const items = virtualizer.getVirtualItems();
  const activeId = focus !== null ? `${id}-tile-${focus}` : undefined;
  const checks = touch || selected.size > 0;

  return (
    <div ref={container} className="min-w-0">
      <div
        role="listbox"
        aria-multiselectable="true"
        aria-label={t("vision.gallery.gridLabel", { count: String(rows.length) })}
        aria-activedescendant={activeId}
        tabIndex={0}
        onKeyDown={onKeyDown}
        onFocus={() => {
          if (focus === null && rows.length > 0) setActive(0);
        }}
        className="rounded-md outline-none focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:ring-offset-4 focus-visible:ring-offset-surface-0"
      >
        <div
          ref={content}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          className="relative w-full touch-pan-y"
          style={{ height: Math.max(0, virtualizer.getTotalSize() - GAP) }}
        >
          {items.map((item) => {
            const first = item.index * columns;
            return (
              <div
                key={item.key}
                className="absolute top-0 left-0 grid w-full"
                style={{
                  transform: `translateY(${item.start - virtualizer.options.scrollMargin}px)`,
                  gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
                  columnGap: GAP,
                }}
              >
                {rows.slice(first, first + columns).map((row, offset) => {
                  const position = first + offset;
                  return (
                    <Tile
                      key={row.key}
                      id={`${id}-tile-${position}`}
                      datasetId={datasetId}
                      row={row}
                      position={position}
                      size={cellWidth}
                      compact={density === "s" || phone}
                      selected={selected.has(row.key)}
                      active={position === focus}
                      excluded={isExcluded(row, draft)}
                      cause={exclusionCause(row, draft)}
                      checks={checks}
                      onClick={onTileClick}
                      onOpen={onTileOpen}
                      onToggle={onTileToggle}
                    />
                  );
                })}
              </div>
            );
          })}
          {rect && (
            <div
              aria-hidden
              className="pointer-events-none absolute rounded-[2px] border border-accent bg-accent/10"
              style={{
                left: Math.min(rect.x0, rect.x1),
                top: Math.min(rect.y0, rect.y1),
                width: Math.abs(rect.x1 - rect.x0),
                height: Math.abs(rect.y1 - rect.y0),
              }}
            />
          )}
        </div>
      </div>
    </div>
  );
}

type TileProps = {
  id: string;
  datasetId: string;
  row: ImageRow;
  position: number;
  size: number;
  compact: boolean;
  selected: boolean;
  active: boolean;
  excluded: boolean;
  cause: ReturnType<typeof exclusionCause>;
  checks: boolean;
  onClick: (position: number, event: MouseEvent) => void;
  onOpen: (position: number) => void;
  onToggle: (position: number) => void;
};

/** The words of an image for a screen reader and a tooltip: name, label, problems (R-107-07). */
export function describeImage(row: ImageRow): string {
  return [row.name, row.label, row.issues.length > 0 ? row.issues.join(", ") : null].filter(Boolean).join(", ");
}

const Tile = memo(function Tile({
  id,
  datasetId,
  row,
  position,
  size,
  compact,
  selected,
  active,
  excluded,
  cause,
  checks,
  onClick,
  onOpen,
  onToggle,
}: TileProps) {
  const shownIssues = compact ? 1 : 2;
  const frame = selected
    ? "ring-2 ring-accent ring-offset-2 ring-offset-surface-0"
    : active
      ? "ring-1 ring-ink-3 ring-offset-2 ring-offset-surface-0"
      : "group-hover/tile:ring-1 group-hover/tile:ring-line group-hover/tile:ring-offset-2 group-hover/tile:ring-offset-surface-0";
  return (
    <div
      role="option"
      id={id}
      aria-selected={selected}
      aria-label={describeImage(row)}
      onClick={(event) => onClick(position, event)}
      onDoubleClick={() => onOpen(position)}
      className="group/tile relative flex min-w-0 cursor-default flex-col select-none"
    >
      <div className={`relative overflow-hidden rounded-md bg-surface-2 transition-shadow duration-150 ${frame}`} style={{ height: size }}>
        {row.decodeOk ? (
          <img
            src={imageUrl(datasetId, row.id, "thumb")}
            alt=""
            loading="lazy"
            decoding="async"
            draggable={false}
            className={`size-full object-contain transition-opacity duration-150 ${excluded ? "opacity-35" : ""}`}
          />
        ) : (
          <span className="flex size-full flex-col items-center justify-center gap-1 px-2 text-center text-[11px] text-ink-3">
            <FileWarning size={18} strokeWidth={1.5} aria-hidden />
            {t("vision.tile.notDecoded")}
          </span>
        )}
        {/* a hairline inside the frame keeps light thumbnails off the background */}
        <span aria-hidden className="pointer-events-none absolute inset-0 rounded-md ring-1 ring-ink-1/8 ring-inset" />
        {row.issues.length > 0 && (
          <span className="absolute top-1.5 left-1.5 flex max-w-[calc(100%-36px)] flex-wrap gap-1">
            {row.issues.slice(0, shownIssues).map((code) => (
              <span
                key={code}
                className="inline-flex items-center gap-0.5 rounded-[4px] border border-line bg-surface-1/95 px-1 py-px font-mono text-[10px] leading-4 text-ink-1"
              >
                <TriangleAlert size={10} strokeWidth={1.75} className="text-warn" aria-hidden />
                {code.replace("R-", "")}
              </span>
            ))}
            {row.issues.length > shownIssues && (
              <span className="rounded-[4px] border border-line bg-surface-1/95 px-1 py-px text-[10px] leading-4 text-ink-2 tabular-nums">
                +{row.issues.length - shownIssues}
              </span>
            )}
          </span>
        )}
        <button
          type="button"
          tabIndex={-1}
          aria-label={selected ? t("vision.tile.unselect") : t("vision.tile.select")}
          aria-pressed={selected}
          onClick={(event) => {
            event.stopPropagation();
            onToggle(position);
          }}
          className={`absolute top-1 right-1 inline-flex size-7 items-center justify-center rounded-md transition-opacity duration-150 pointer-coarse:size-11 ${
            checks || selected ? "opacity-100" : "opacity-0 group-hover/tile:opacity-100 focus-visible:opacity-100"
          }`}
        >
          <span
            className={`inline-flex size-[18px] items-center justify-center rounded-[4px] border ${
              selected ? "border-accent bg-accent text-accent-ink" : "border-line bg-surface-1/95 text-transparent"
            }`}
          >
            <Check size={12} strokeWidth={2.25} aria-hidden />
          </span>
        </button>
        {(excluded || cause.kind === "kept") && (
          <span
            className={`absolute right-1.5 bottom-1.5 left-1.5 inline-flex min-w-0 items-center gap-1 truncate rounded-[4px] border border-line bg-surface-1/95 px-1.5 py-px text-[10px] leading-4 ${
              excluded ? "text-ink-1" : "text-ok"
            }`}
          >
            {excluded ? <Ban size={10} strokeWidth={1.75} aria-hidden /> : <Check size={10} strokeWidth={2} aria-hidden />}
            <span className="truncate">
              {excluded
                ? cause.kind === "user"
                  ? t("vision.tile.excludedByYou")
                  : t("vision.tile.excludedBy", { code: cause.code ?? "" })
                : t("vision.tile.kept")}
            </span>
          </span>
        )}
      </div>
      {/* the label first; the file name only when the tile is wide enough to read it (container query) */}
      <div className="@container/caption flex h-[36px] min-w-0 items-center gap-1.5 pt-1" title={row.key}>
        {row.label !== null && (
          <span className="max-w-full min-w-0 shrink-0 truncate rounded-[4px] bg-surface-2 px-1.5 text-[11px] leading-5 text-ink-1">{row.label}</span>
        )}
        <span
          className={`min-w-0 flex-1 truncate font-mono text-[11px] text-ink-2 ${row.label !== null ? "hidden @[200px]/caption:block" : ""} ${compact && row.label !== null ? "!hidden" : ""}`}
        >
          {row.name}
        </span>
      </div>
    </div>
  );
});
