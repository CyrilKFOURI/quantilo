/** Where the cards of the Model view go (088), and the geometry of the lines between them.
 *  Pure functions. The star layout is a tidy tree on two sides: each fact in the centre, its
 *  dimensions split left and right, a snowflake's outer dimensions one column further out on the
 *  same side; every subtree owns a band of rows, so no two cards ever overlap. */

export const CARD_W = 216;
export const CARD_H = 112;
export const PAD = 32;

export type Point = { x: number; y: number };
export type LayoutTable = { id: string; root_id: string; role: "fact" | "dimension" | "unrelated"; depth: number | null; primary: boolean; name: string };
export type LayoutRelation = { child_id: string; parent_id: string };

const COLUMN = CARD_W + 128; // from one column of cards to the next (room for the join label)
const SLOT = CARD_H + 44; // one row of the band a subtree owns
const GAP = 56; // between unrelated cards in their row
const FACT_GAP = 96; // between the bands of two facts

/** Positions by root id (top-left corner of each card), moved so the plane starts at PAD. */
export function starLayout(tables: LayoutTable[], relations: LayoutRelation[], facts: string[]): Record<string, Point> {
  const byId = new Map(tables.map((table) => [table.id, table]));
  const parentsOf = new Map<string, string[]>();
  for (const relation of relations) {
    if (!byId.has(relation.child_id) || !byId.has(relation.parent_id)) continue;
    parentsOf.set(relation.child_id, [...(parentsOf.get(relation.child_id) ?? []), relation.parent_id]);
  }
  const roots = facts.filter((id) => byId.has(id));
  // breadth first: each dimension belongs to the first table that reaches it
  const children = new Map<string, string[]>();
  const owned = new Set(roots);
  let frontier = [...roots];
  while (frontier.length > 0) {
    const next: string[] = [];
    for (const id of frontier) {
      const fresh = [...new Set(parentsOf.get(id) ?? [])]
        .filter((parent) => !owned.has(parent))
        .sort((a, b) => nameOf(byId, a).localeCompare(nameOf(byId, b)));
      fresh.forEach((parent) => owned.add(parent));
      children.set(id, fresh);
      next.push(...fresh);
    }
    frontier = next;
  }
  const size = (id: string): number => Math.max(1, (children.get(id) ?? []).reduce((sum, child) => sum + size(child), 0));
  const centres = new Map<string, Point>();
  const place = (id: string, column: number, top: number) => {
    centres.set(id, { x: column * COLUMN, y: top + (size(id) * SLOT) / 2 });
    let cursor = top;
    for (const child of children.get(id) ?? []) {
      place(child, column + Math.sign(column), cursor);
      cursor += size(child) * SLOT;
    }
  };
  let bandTop = 0;
  for (const fact of roots) {
    // the biggest subtrees first, each to the lighter side; then by name within a side
    const sides: [string[], string[]] = [[], []];
    const weight = [0, 0];
    const first = [...(children.get(fact) ?? [])].sort((a, b) => size(b) - size(a));
    for (const child of first) {
      const side = (weight[0] ?? 0) <= (weight[1] ?? 0) ? 0 : 1;
      sides[side].push(child);
      weight[side] = (weight[side] ?? 0) + size(child);
    }
    const band = Math.max(1, weight[0] ?? 0, weight[1] ?? 0) * SLOT;
    centres.set(fact, { x: 0, y: bandTop + band / 2 });
    sides.forEach((group, side) => {
      const column = side === 0 ? -1 : 1;
      let cursor = bandTop + (band - (weight[side] ?? 0) * SLOT) / 2;
      for (const child of group.sort((a, b) => nameOf(byId, a).localeCompare(nameOf(byId, b)))) {
        place(child, column, cursor);
        cursor += size(child) * SLOT;
      }
    });
    bandTop += band + FACT_GAP;
  }
  // tables no fact reaches (a related pair without a fact, or unrelated): a row underneath
  const rest = tables.filter((table) => !centres.has(table.id)).sort((a, b) => a.name.localeCompare(b.name));
  const lowest = centres.size > 0 ? Math.max(...[...centres.values()].map((point) => point.y)) + CARD_H + 64 : 0;
  rest.forEach((table, index) => {
    centres.set(table.id, { x: index * (CARD_W + GAP), y: lowest });
  });
  const points = [...centres.values()];
  const minX = Math.min(...points.map((point) => point.x - CARD_W / 2));
  const minY = Math.min(...points.map((point) => point.y - CARD_H / 2));
  const out: Record<string, Point> = {};
  for (const [id, centre] of centres) {
    const table = byId.get(id);
    if (!table) continue;
    out[table.root_id] = { x: Math.round(centre.x - CARD_W / 2 - minX + PAD), y: Math.round(centre.y - CARD_H / 2 - minY + PAD) };
  }
  return out;
}

function nameOf(byId: Map<string, LayoutTable>, id: string): string {
  return byId.get(id)?.name ?? id;
}

/** The plane's size: every card inside, plus the padding. */
export function planeSize(positions: Point[]): { width: number; height: number } {
  if (positions.length === 0) return { width: 0, height: 0 };
  return {
    width: Math.max(...positions.map((point) => point.x)) + CARD_W + PAD,
    height: Math.max(...positions.map((point) => point.y)) + CARD_H + PAD,
  };
}

/** Where the segment from the centre of `from` towards `to` leaves the card `from`. */
function exitPoint(from: Point, to: Point, height: number): Point {
  const cx = from.x + CARD_W / 2;
  const cy = from.y + height / 2;
  const dx = to.x - cx;
  const dy = to.y - cy;
  if (dx === 0 && dy === 0) return { x: cx, y: cy };
  const scale = Math.min(Math.abs(CARD_W / 2 / (dx || 1e-9)), Math.abs(height / 2 / (dy || 1e-9)));
  return { x: cx + dx * scale, y: cy + dy * scale };
}

export type Edge = {
  /** On the child's edge (the "many" side) and on the parent's edge (the "one" side). */
  start: Point;
  end: Point;
  middle: Point;
  /** Where the cardinality marks go, a little inside the line's ends. */
  startMark: Point;
  endMark: Point;
};

/** The line between two cards, offset sideways when several relations join the same pair. */
export function edgeBetween(child: Point, parent: Point, offset = 0, childH = CARD_H, parentH = CARD_H): Edge {
  const childCentre = { x: child.x + CARD_W / 2, y: child.y + childH / 2 };
  const parentCentre = { x: parent.x + CARD_W / 2, y: parent.y + parentH / 2 };
  const length = Math.hypot(parentCentre.x - childCentre.x, parentCentre.y - childCentre.y) || 1;
  const nx = -(parentCentre.y - childCentre.y) / length;
  const ny = (parentCentre.x - childCentre.x) / length;
  const shift = (point: Point): Point => ({ x: point.x + nx * offset, y: point.y + ny * offset });
  const start = shift(exitPoint(child, parentCentre, childH));
  const end = shift(exitPoint(parent, childCentre, parentH));
  const span = Math.hypot(end.x - start.x, end.y - start.y) || 1;
  const ux = (end.x - start.x) / span;
  const uy = (end.y - start.y) / span;
  const mark = 13;
  const side = 9;
  return {
    start,
    end,
    middle: { x: (start.x + end.x) / 2, y: (start.y + end.y) / 2 },
    startMark: { x: start.x + ux * mark + nx * side, y: start.y + uy * mark + ny * side },
    endMark: { x: end.x - ux * mark + nx * side, y: end.y - uy * mark + ny * side },
  };
}
