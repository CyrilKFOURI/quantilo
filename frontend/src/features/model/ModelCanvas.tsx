import { KeyRound } from "lucide-react";
import { useRef, useState, type KeyboardEvent, type PointerEvent } from "react";

import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import type { Model, ModelTable, Relation } from "./api";
import { cardCounts, cardinalityLabel, joinLabel, marks, primaryKey, relationTitle, roleLabel, tableNames } from "./labels";
import { CARD_H, CARD_W, edgeBetween, planeSize, type Point } from "./layout";

export type Selection = { kind: "relation" | "table"; id: string } | null;

type CanvasProps = {
  model: Model;
  /** Top-left corner of each card, by root id. */
  positions: Record<string, Point>;
  selected: Selection;
  onSelect: (selection: Selection) => void;
  /** A card was dropped somewhere else (saved for the project). */
  onMove: (rootId: string, point: Point) => void;
  /** 1 = full size; the page fits the diagram's width (`fitScale`) or applies the user's zoom. */
  scale?: number;
  /** The room is measured: centre the primary fact once. */
  ready?: boolean;
};

/** The scale that fits a plane in the room it has, not below 70 % (text stays readable). */
export function fitScale(positions: Record<string, Point>, room: number | undefined): number {
  const resting = planeSize(Object.values(positions));
  return room && resting.width > 0 ? Math.min(1, Math.max(0.7, room / resting.width)) : 1;
}

type Drag = { root: string; dx: number; dy: number; x: number; y: number; moved: boolean };

const MOVE_THRESHOLD = 3;

/** The diagram (088): cards absolutely placed on a plane, the lines in an SVG underneath.
 *  A card moves by its header (pointer events, captured); a line is selected by click or by
 *  keyboard (each is a focusable button in the SVG, and the side panel lists them too). */
export function ModelCanvas({ model, positions, selected, onSelect, onMove, scale = 1, ready = false }: CanvasProps) {
  const [drag, setDrag] = useState<Drag | null>(null);
  const plane = useRef<HTMLDivElement>(null);
  const names = tableNames(model);
  const placed = (table: ModelTable): Point | undefined =>
    drag?.root === table.root_id ? { x: drag.x, y: drag.y } : positions[table.root_id];
  const pointOf = new Map(model.tables.flatMap((table) => {
    const point = placed(table);
    return point ? [[table.id, point] as const] : [];
  }));
  const size = planeSize([...pointOf.values()]);
  // the first time the diagram shows, the primary fact is in view, centred
  const centred = useRef(false);
  const centre = (node: HTMLDivElement | null) => {
    if (!node || centred.current || typeof node.scrollIntoView !== "function") return;
    centred.current = true;
    node.scrollIntoView({ block: "nearest", inline: "center" });
  };

  function start(event: PointerEvent<HTMLElement>, table: ModelTable) {
    if (event.button !== 0) return;
    const point = pointOf.get(table.id);
    if (!point) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const box = plane.current?.getBoundingClientRect();
    const x = (event.clientX - (box?.left ?? 0)) / scale;
    const y = (event.clientY - (box?.top ?? 0)) / scale;
    setDrag({ root: table.root_id, dx: x - point.x, dy: y - point.y, x: point.x, y: point.y, moved: false });
  }

  function move(event: PointerEvent<HTMLElement>) {
    if (!drag) return;
    const box = plane.current?.getBoundingClientRect();
    const x = Math.max(0, Math.round((event.clientX - (box?.left ?? 0)) / scale - drag.dx));
    const y = Math.max(0, Math.round((event.clientY - (box?.top ?? 0)) / scale - drag.dy));
    const far = Math.abs(x - drag.x) + Math.abs(y - drag.y) >= MOVE_THRESHOLD;
    if (drag.moved || far) setDrag({ ...drag, x, y, moved: true });
  }

  function end(table: ModelTable) {
    if (!drag) return;
    if (drag.moved) onMove(drag.root, { x: drag.x, y: drag.y });
    else onSelect({ kind: "table", id: table.id });
    setDrag(null);
  }

  // several relations between the same two tables are drawn side by side
  const pairs = new Map<string, Relation[]>();
  for (const relation of model.relations) {
    const key = [relation.child_id, relation.parent_id].sort().join("|");
    pairs.set(key, [...(pairs.get(key) ?? []), relation]);
  }

  return (
    <div
      role="group"
      aria-label={t("model.diagram")}
      className="relative bg-[radial-gradient(var(--line)_1px,transparent_1px)] bg-size-[20px_20px]"
      style={{ width: size.width * scale, height: size.height * scale, minWidth: "100%", minHeight: "100%" }}
      onPointerDown={(event) => {
        if (event.target === event.currentTarget || event.target === plane.current) onSelect(null);
      }}
    >
      <div ref={plane} className="absolute top-0 left-0 origin-top-left" style={{ width: size.width, height: size.height, transform: scale < 1 ? `scale(${scale})` : undefined }}>
      <svg className="absolute inset-0 overflow-visible" width={size.width} height={size.height}>
        {[...pairs.values()].flatMap((group) =>
          group.map((relation, index) => {
            const child = pointOf.get(relation.child_id);
            const parent = pointOf.get(relation.parent_id);
            if (!child || !parent) return null;
            const offset = (index - (group.length - 1) / 2) * 16;
            const edge = edgeBetween(child, parent, offset);
            const active = selected?.kind === "relation" && selected.id === relation.id;
            const near =
              selected?.kind === "table" &&
              (selected.id === relation.child_id || selected.id === relation.parent_id);
            return (
              <RelationLine
                key={relation.id}
                relation={relation}
                edge={edge}
                active={active}
                near={near}
                label={relationTitle(relation, names)}
                onSelect={() => onSelect({ kind: "relation", id: relation.id })}
              />
            );
          }),
        )}
      </svg>
      {model.tables.map((table) => {
        const point = pointOf.get(table.id);
        if (!point) return null;
        const dragging = drag?.root === table.root_id && drag.moved;
        return (
          <TableCard
            key={table.root_id}
            table={table}
            point={point}
            selected={selected?.kind === "table" && selected.id === table.id}
            dragging={dragging}
            onPointerDown={(event) => start(event, table)}
            onPointerMove={move}
            onPointerUp={() => end(table)}
            onSelect={() => onSelect({ kind: "table", id: table.id })}
            anchor={table.primary && ready ? centre : undefined}
          />
        );
      })}
      </div>
    </div>
  );
}

function RelationLine({
  relation,
  edge,
  active,
  near,
  label,
  onSelect,
}: {
  relation: Relation;
  edge: ReturnType<typeof edgeBetween>;
  active: boolean;
  near: boolean;
  label: string;
  onSelect: () => void;
}) {
  const suggested = relation.status === "suggested";
  const many = relation.cardinality === "n:m";
  const [childMark, parentMark] = marks(relation);
  const stroke = active ? "var(--accent)" : many ? "var(--bad)" : near ? "var(--ink-1)" : "var(--ink-3)";
  const width = active ? 2 : 1.25;
  const pill = joinLabel(many ? "none" : relation.join_type);
  const pillWidth = pill.length * 6.6 + 14;
  const onKey = (event: KeyboardEvent<SVGGElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect();
    }
  };
  return (
    <g
      role="button"
      tabIndex={0}
      aria-label={`${label}, ${cardinalityLabel(relation)}, ${pill}${suggested ? `, ${t("model.relation.suggested")}` : ""}`}
      aria-pressed={active}
      onClick={onSelect}
      onKeyDown={onKey}
      className="cursor-pointer outline-none [&:focus-visible>.halo]:stroke-accent/40"
    >
      <line className="halo" x1={edge.start.x} y1={edge.start.y} x2={edge.end.x} y2={edge.end.y} stroke="transparent" strokeWidth={14} />
      <line
        x1={edge.start.x}
        y1={edge.start.y}
        x2={edge.end.x}
        y2={edge.end.y}
        stroke={stroke}
        strokeWidth={width}
        strokeDasharray={suggested || many ? "5 4" : undefined}
        strokeLinecap="round"
      />
      <circle cx={edge.end.x} cy={edge.end.y} r={2.5} fill={stroke} />
      <Mark at={edge.startMark} text={childMark} color={stroke} />
      <Mark at={edge.endMark} text={parentMark} color={stroke} />
      <g transform={`translate(${edge.middle.x - pillWidth / 2} ${edge.middle.y - 9})`}>
        <rect
          width={pillWidth}
          height={18}
          rx={4}
          fill="var(--surface-1)"
          stroke={active ? "var(--accent)" : "var(--line)"}
          strokeDasharray={suggested ? "3 2" : undefined}
        />
        <text x={pillWidth / 2} y={12.5} textAnchor="middle" fontSize={10.5} fontWeight={500} fill={active ? "var(--accent)" : many ? "var(--bad)" : "var(--ink-2)"}>
          {pill}
        </text>
      </g>
    </g>
  );
}

function Mark({ at, text, color }: { at: Point; text: string; color: string }) {
  return (
    <text
      x={at.x}
      y={at.y + 4}
      textAnchor="middle"
      fontSize={text === "*" ? 14 : 11.5}
      fontFamily="var(--font-code)"
      fontWeight={600}
      fill={color}
      stroke="var(--surface-0)"
      strokeWidth={3}
      paintOrder="stroke"
    >
      {text}
    </text>
  );
}

type CardProps = {
  table: ModelTable;
  point: Point;
  selected: boolean;
  dragging: boolean;
  onPointerDown: (event: PointerEvent<HTMLElement>) => void;
  onPointerMove: (event: PointerEvent<HTMLElement>) => void;
  onPointerUp: () => void;
  onSelect: () => void;
  anchor?: (node: HTMLDivElement | null) => void;
};

/** A table: name and role, rows, key, what it holds. The whole card is the drag handle; a click
 *  without moving selects it (and Enter on the focused card). */
function TableCard({ table, point, selected, dragging, onPointerDown, onPointerMove, onPointerUp, onSelect, anchor }: CardProps) {
  const key = primaryKey(table);
  const version = table.versions.find((item) => item.id === table.id);
  const fact = table.role === "fact";
  return (
    <div
      ref={anchor}
      role="button"
      tabIndex={0}
      aria-label={t("model.card.select", { name: table.name })}
      aria-pressed={selected}
      title={t("model.card.drag", { name: table.name })}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect();
        }
      }}
      data-role={table.role}
      data-selected={selected || undefined}
      data-dragging={dragging || undefined}
      className="absolute flex touch-none flex-col gap-1 rounded-md border border-line bg-surface-1 px-3 py-2.5 text-left shadow-[0_1px_2px_rgb(0_0_0/0.04)] select-none data-dragging:z-10 data-dragging:cursor-grabbing data-dragging:shadow-[0_10px_28px_-10px_rgb(0_0_0/0.35)] data-[role=fact]:border-ink-3 data-[role=unrelated]:border-dashed data-selected:outline-2 data-selected:outline-offset-1 data-selected:outline-accent cursor-grab transition-shadow duration-150 motion-reduce:transition-none"
      style={{ left: point.x, top: point.y, width: CARD_W, height: CARD_H }}
    >
      <div className="flex min-w-0 items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-[13px] font-semibold text-ink-1">{table.name}</span>
        <span
          className={`shrink-0 rounded-[4px] px-1.5 py-px text-[11px] font-medium ${
            fact ? "bg-accent-soft text-accent" : table.role === "dimension" ? "bg-surface-2 text-ink-2" : "text-ink-3"
          }`}
        >
          {roleLabel(table)}
        </span>
      </div>
      <div className="flex items-center gap-2 text-xs text-ink-2 tabular-nums">
        {t("model.card.rows", { rows: formatCount(table.rows) })}
        {version && version.seq > 0 && <span className="text-ink-3">{t("model.card.version", { seq: String(version.seq) })}</span>}
      </div>
      <div className="flex min-w-0 items-center gap-1.5 text-xs">
        <KeyRound size={12} strokeWidth={1.5} className="shrink-0 text-ink-3" aria-hidden />
        {key ? (
          <span className="truncate font-mono text-ink-1">{key.join(" + ")}</span>
        ) : (
          <span className="text-ink-3">{t("model.card.noKey")}</span>
        )}
      </div>
      <div className="truncate text-xs text-ink-3 tabular-nums">
        {cardCounts(table)}
      </div>
    </div>
  );
}
