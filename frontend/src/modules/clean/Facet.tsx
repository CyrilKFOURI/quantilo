import { Plus, Search, X } from "lucide-react";
import { useDeferredValue, useEffect, useId, useRef, useState } from "react";

import { Folded } from "../../features/text/Folded";
import { formatCount } from "../../lib/format";
import { locale, t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote, Select, TextInput } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { SegmentedControl } from "../../ui/SegmentedControl";
import type { Step } from "./api";
import { ColumnBar } from "./ColumnQuality";
import { actionsOf, inRange, isEmpty, NOTHING, rangeOf, rowsOf, stepOf, toggled, useFacet, type Action, type Facet, type FacetSort, type Selection } from "./facets";
import { moveFocus, reviewKey, rowKeys } from "./keys";
import { visible } from "./parts";

/** N40, under the Steps tab and folded (onion rule): a facet of one column of the draft, as
 *  OpenRefine has them. A text lists its values with their rows (by count or by name, a search);
 *  a number or a date is a histogram whose bars pick a range; the blank cells and those its type
 *  cannot read are values of their own. Picking filters the rows shown beside it (the draft's,
 *  every step applied); a gesture then writes one step (merge, read as missing, remove or keep the
 *  rows, empty the values outside a range), after its reason and a click on Add step. */
export function FacetPanel({ datasetId, steps, columns, onChange }: { datasetId: string; steps: Step[]; columns: string[]; onChange: (steps: Step[]) => void }) {
  return (
    <Folded title={t("clean.facet.title")} hint={t("clean.facet.hint")}>
      <FacetBody datasetId={datasetId} steps={steps} columns={columns} onChange={onChange} />
    </Folded>
  );
}

function FacetBody({ datasetId, steps, columns, onChange }: { datasetId: string; steps: Step[]; columns: string[]; onChange: (steps: Step[]) => void }) {
  const [column, setColumn] = useState<string | null>(columns[0] ?? null);
  const [selection, setSelection] = useState<Selection>(NOTHING);
  const [sort, setSort] = useState<FacetSort>("count");
  const [search, setSearch] = useState("");
  const [added, setAdded] = useState<string | null>(null);
  const deferred = useDeferredValue(search);
  const facet = useFacet(datasetId, steps, column, selection, sort, deferred);
  const data = facet.data && facet.data.column === column ? facet.data : undefined;
  const pick = (next: Selection) => {
    setSelection(next);
    setAdded(null);
  };
  const choose = (next: string) => {
    setColumn(next);
    setSelection(NOTHING);
    setSearch("");
    setAdded(null);
  };
  const add = (step: Step) => {
    onChange([...steps, step]);
    setSelection(NOTHING);
    setAdded(step.id);
  };
  const columnId = useId();
  const keys = useListKeys(); // one listener for the values, the blank and the unreadable cells
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end gap-x-3 gap-y-2">
        <label htmlFor={columnId} className="flex flex-col gap-1 text-xs text-ink-3">
          {t("clean.facet.column")}
          <Select id={columnId} value={column ?? ""} onChange={(event) => choose(event.target.value)} className="min-w-44 font-mono">
            {columns.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </Select>
        </label>
        {data?.kind === "text" && (
          <>
            <SegmentedControl
              label={t("clean.facet.sort")}
              value={sort}
              options={[
                { value: "count", label: t("clean.facet.sort.count") },
                { value: "name", label: t("clean.facet.sort.name") },
              ]}
              onChange={setSort}
            />
            <label className="relative block min-w-0 flex-1 basis-48">
              <Search size={14} strokeWidth={1.5} className="pointer-events-none absolute top-1/2 left-2 -translate-y-1/2 text-ink-3" aria-hidden />
              <TextInput type="search" aria-label={t("clean.facet.search")} placeholder={t("clean.facet.search")} value={search} onChange={(event) => setSearch(event.target.value)} className="w-full pl-7" />
            </label>
          </>
        )}
        {!isEmpty(selection) && (
          <Button variant="ghost" icon={X} onClick={() => pick(NOTHING)}>
            {t("clean.facet.clear")}
          </Button>
        )}
      </div>
      <p className="text-xs text-ink-3">{t("clean.facet.onDraft")}</p>
      {facet.isError ? (
        <ErrorNote>{facet.error.message}</ErrorNote>
      ) : !data ? (
        column && <Loading label={t("clean.facet.loading")} className="h-40 rounded-md" compact />
      ) : (
        <div aria-busy={facet.isFetching} className={`grid gap-4 transition-opacity lg:grid-cols-[minmax(0,360px)_minmax(0,1fr)] ${facet.isFetching ? "opacity-70" : ""}`}>
          <div ref={keys} className="min-w-0 space-y-2">
            {data.kind === "text" ? <ValueList facet={data} selection={selection} onPick={pick} /> : <Histogram facet={data} selection={selection} onPick={pick} />}
            <Specials facet={data} selection={selection} onPick={pick} />
          </div>
          <div className="min-w-0 space-y-3">
            <Picked facet={data} selection={selection} steps={steps} onAdd={add} added={added} />
          </div>
        </div>
      )}
    </div>
  );
}

/** N42's keys in the list: j / k from a value to the next, Space (or x) picks it. */
function useListKeys() {
  const list = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      if ((key !== "j" && key !== "k") || event.shiftKey || !reviewKey(event) || !list.current) return;
      if (moveFocus(list.current, key === "j" ? 1 : -1)) event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  return list;
}

/** A row's share of the largest, as a thin neutral bar (magnitude in one quiet hue; the accent is
 *  for what is picked). */
function Share({ rows, most, picked }: { rows: number; most: number; picked: boolean }) {
  const width = most > 0 ? Math.max(2, (rows / most) * 100) : 0;
  return (
    <span className="block h-1 w-full rounded-full bg-surface-2" aria-hidden>
      <span className={`block h-full rounded-full ${picked ? "bg-accent" : "bg-ink-3/45"}`} style={{ width: `${width}%` }} />
    </span>
  );
}

function FacetLine({ label, mono = true, rows, most, picked, onToggle }: { label: string; mono?: boolean; rows: number; most: number; picked: boolean; onToggle: () => void }) {
  const id = useId();
  const keys = rowKeys({ onTick: onToggle });
  return (
    <li {...keys} aria-label={label} className={`grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-x-2 gap-y-0.5 rounded-sm px-2 py-1 hover:bg-surface-2 focus-visible:-outline-offset-2 ${picked ? "bg-accent-soft" : ""}`}>
      <input id={id} type="checkbox" className="size-3.5 shrink-0 accent-(--accent)" checked={picked} onChange={onToggle} />
      <label htmlFor={id} className={`min-w-0 cursor-pointer truncate text-[12px] text-ink-1 ${mono ? "font-mono" : "italic"}`} title={label}>
        {label}
      </label>
      <span className="text-[11px] text-ink-3 tabular-nums">{formatCount(rows)}</span>
      <span className="col-start-2 col-end-4">
        <Share rows={rows} most={most} picked={picked} />
      </span>
    </li>
  );
}

function ValueList({ facet, selection, onPick }: { facet: Facet; selection: Selection; onPick: (selection: Selection) => void }) {
  const most = Math.max(0, ...facet.values.map((value) => value.rows));
  return (
    <div className="space-y-1">
      <p className="text-xs text-ink-3 tabular-nums">
        {facet.values.length < facet.listed_from ? t("clean.facet.listed", { shown: formatCount(facet.values.length), count: formatCount(facet.listed_from) }) : t("clean.facet.listedAll", { count: formatCount(facet.listed_from) })}
      </p>
      {facet.values.length === 0 ? (
        <p className="rounded-md border border-dashed border-line px-3 py-4 text-center text-[13px] text-ink-2">{t("clean.facet.noValues")}</p>
      ) : (
        <ul aria-label={t("clean.facet.values", { column: facet.column })} className="max-h-80 overflow-auto rounded-md border border-line bg-surface-1 p-1">
          {facet.values.map((value) => (
            <FacetLine key={value.value} label={visible(value.value)} rows={value.rows} most={most} picked={selection.values.includes(value.value)} onToggle={() => onPick(toggled(selection, value.value))} />
          ))}
        </ul>
      )}
    </div>
  );
}

/** The blank cells and, for a text read as numbers or dates, those it cannot read. */
function Specials({ facet, selection, onPick }: { facet: Facet; selection: Selection; onPick: (selection: Selection) => void }) {
  const unreadable = facet.unreadable ?? 0;
  const most = Math.max(facet.blank, unreadable);
  if (facet.blank === 0 && unreadable === 0) return null;
  const set = (patch: Partial<Selection>) => onPick({ ...selection, low: null, high: null, ...patch });
  return (
    <ul aria-label={t("clean.facet.specials")} className="rounded-md border border-line bg-surface-1 p-1">
      {facet.blank > 0 && <FacetLine label={t("clean.facet.blank")} mono={false} rows={facet.blank} most={most} picked={selection.blank} onToggle={() => set({ blank: !selection.blank })} />}
      {unreadable > 0 && (
        <FacetLine
          label={t("clean.facet.unreadable", { reading: t(`clean.quality.reading.${facet.reading ?? "number"}` as MessageKey) })}
          mono={false}
          rows={unreadable}
          most={most}
          picked={selection.unreadable}
          onToggle={() => set({ unreadable: !selection.unreadable })}
        />
      )}
    </ul>
  );
}

/** A bound as the user reads it: a number with up to 4 significant digits, a date as a day. */
function shownBound(value: number | null, facet: Facet): string {
  if (value === null) return "";
  if (facet.kind === "date") return new Date(value * 1000).toISOString().slice(0, facet.dtype === "DATE" ? 10 : 16).replace("T", " ");
  return new Intl.NumberFormat(locale(), { maximumSignificantDigits: 6 }).format(value);
}

/** Read a bound typed in its field: a number, or a date (seconds since 1970); null when empty. */
function typedBound(text: string, facet: Facet): number | null | undefined {
  if (text.trim() === "") return null;
  if (facet.kind === "date") {
    const time = Date.parse(`${text}T00:00:00Z`);
    return Number.isNaN(time) ? undefined : time / 1000;
  }
  const number = Number(text.replace(",", "."));
  return Number.isFinite(number) ? number : undefined;
}

function fieldValue(value: number | null, facet: Facet): string {
  if (value === null) return "";
  return facet.kind === "date" ? new Date(value * 1000).toISOString().slice(0, 10) : String(Number(value.toPrecision(12)));
}

/** A number's or a date's rows by value, in bins of equal width; a bar picks its range, Shift
 *  extends it; From / To type it. */
function Histogram({ facet, selection, onPick }: { facet: Facet; selection: Selection; onPick: (selection: Selection) => void }) {
  const [anchor, setAnchor] = useState<number | null>(null);
  const most = Math.max(0, ...facet.bins.map((bin) => bin.rows));
  if (!facet.bins.length) return <p className="rounded-md border border-dashed border-line px-3 py-4 text-center text-[13px] text-ink-2">{t("clean.facet.noValues")}</p>;
  const choose = (index: number, extend: boolean) => {
    const from = extend && anchor !== null ? anchor : index;
    if (!extend || anchor === null) setAnchor(index);
    onPick(rangeOf(facet.bins, from, index));
  };
  const typed = (which: "low" | "high", text: string) => {
    const value = typedBound(text, facet);
    if (value === undefined) return;
    const next = { ...NOTHING, low: selection.low, high: selection.high, [which]: value };
    onPick(next.low === null && next.high === null ? NOTHING : next);
  };
  return (
    <figure className="space-y-2">
      <figcaption className="text-xs text-ink-3">{t("clean.facet.histogram", { column: facet.column })}</figcaption>
      <div role="group" aria-label={t("clean.facet.histogram", { column: facet.column })} className="flex h-28 items-end gap-0.5 border-b border-line">
        {facet.bins.map((bin, index) => {
          const picked = inRange(bin, selection);
          const label = t("clean.facet.bin", { low: shownBound(bin.low, facet), high: shownBound(bin.high, facet), count: formatCount(bin.rows) });
          return (
            <button
              key={index}
              type="button"
              aria-label={label}
              aria-pressed={picked}
              title={label}
              onClick={(event) => choose(index, event.shiftKey)}
              className="flex h-full min-w-0 flex-1 items-end rounded-t-sm hover:bg-surface-2 focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
            >
              <span className={`block w-full rounded-t-sm ${picked ? "bg-accent" : "bg-ink-3/45"}`} style={{ height: `${most > 0 && bin.rows > 0 ? Math.max(4, (bin.rows / most) * 100) : 0}%` }} />
            </button>
          );
        })}
      </div>
      <div className="flex justify-between font-mono text-[11px] text-ink-3 tabular-nums">
        <span>{shownBound(facet.low, facet)}</span>
        <span>{shownBound(facet.high, facet)}</span>
      </div>
      <p className="text-xs text-ink-3">{t("clean.facet.rangeHint")}</p>
      <div className="grid grid-cols-2 gap-2">
        <BoundField label={t("clean.facet.from")} facet={facet} value={selection.low} onCommit={(text) => typed("low", text)} />
        <BoundField label={t("clean.facet.to")} facet={facet} value={selection.high} onCommit={(text) => typed("high", text)} />
      </div>
    </figure>
  );
}

/** A bound typed and read when the field is left or Enter is pressed (not at each key). */
function BoundField({ label, facet, value, onCommit }: { label: string; facet: Facet; value: number | null; onCommit: (text: string) => void }) {
  const id = useId();
  const shown = fieldValue(value, facet);
  const [draft, setDraft] = useState<{ base: string; text: string } | null>(null);
  const text = draft && draft.base === shown ? draft.text : shown;
  const commit = () => {
    if (draft) onCommit(draft.text);
    setDraft(null);
  };
  return (
    <label htmlFor={id} className="flex flex-col gap-1 text-xs text-ink-3">
      {label}
      <TextInput
        id={id}
        type={facet.kind === "date" ? "date" : "text"}
        inputMode={facet.kind === "date" ? undefined : "decimal"}
        value={text}
        onChange={(event) => setDraft({ base: shown, text: event.target.value })}
        onBlur={commit}
        onKeyDown={(event) => event.key === "Enter" && commit()}
        className="w-full font-mono"
      />
    </label>
  );
}

const ACTION_LABEL: Record<Action, MessageKey> = {
  merge: "clean.facet.action.merge",
  missing: "clean.facet.action.missing",
  remove: "clean.facet.action.remove",
  keep: "clean.facet.action.keep",
  outside: "clean.facet.action.outside",
};

/** What is picked: its rows, the gestures it allows and, once one is chosen, its reason and Add
 *  step; then the first rows picked. */
function Picked({ facet, selection, steps, onAdd, added }: { facet: Facet; selection: Selection; steps: Step[]; onAdd: (step: Step) => void; added: string | null }) {
  const [action, setAction] = useState<{ action: Action; key: string } | null>(null);
  const [target, setTarget] = useState<{ key: string; value: string } | null>(null);
  const listId = useId();
  const key = JSON.stringify([facet.column, selection]);
  const chosen = action?.key === key ? action.action : null;
  const rows = rowsOf(selection, facet);
  if (isEmpty(selection))
    return (
      <div className="space-y-2">
        {added && (
          <p role="status" className="text-[13px] text-ok">
            {t("clean.facet.added", { id: added })}
          </p>
        )}
        <p className="rounded-md border border-dashed border-line px-3 py-6 text-center text-[13px] text-ink-2">{t("clean.facet.none")}</p>
      </div>
    );
  const counts = new Map(facet.values.map((value) => [value.value, value.rows]));
  const most = [...selection.values].sort((a, b) => (counts.get(b) ?? 0) - (counts.get(a) ?? 0))[0] ?? "";
  const into = target?.key === key ? target.value : most;
  const valuesNamed = selection.values.length + (selection.unreadable ? facet.unreadable_values.length : 0);
  const reason = chosen ? reasonOf(chosen, facet, selection, rows, into, counts, valuesNamed) : null;
  const ready = chosen !== "merge" || (into !== "" && selection.values.some((value) => value !== into));
  return (
    <div className="space-y-3">
      <p className="text-[13px] text-ink-1 tabular-nums" aria-live="polite">
        {t("clean.facet.picked", { rows: formatCount(rows), total: formatCount(facet.rows) })}
      </p>
      <div className="flex flex-wrap gap-1.5" role="group" aria-label={t("clean.facet.actions")}>
        {actionsOf(selection, facet.kind).map((item) => (
          <Button key={item} variant={chosen === item ? "primary" : "secondary"} aria-pressed={chosen === item} onClick={() => setAction(chosen === item ? null : { action: item, key })}>
            {t(ACTION_LABEL[item])}
          </Button>
        ))}
      </div>
      {chosen && (
        <div className="space-y-2 rounded-md border border-line bg-surface-0 px-3 py-2.5">
          {chosen === "merge" && (
            <label className="flex flex-col gap-1 text-xs text-ink-3">
              {t("clean.facet.target")}
              <TextInput list={listId} value={into} onChange={(event) => setTarget({ key, value: event.target.value })} className="max-w-72 font-mono" />
              <datalist id={listId}>
                {selection.values.map((value) => (
                  <option key={value} value={value} />
                ))}
              </datalist>
            </label>
          )}
          <p className="max-w-prose text-[13px] text-ink-1">{reason}</p>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="primary"
              icon={Plus}
              disabled={!ready}
              onClick={() => {
                onAdd(stepOf(chosen, selection, facet, steps, into));
                setAction(null);
                setTarget(null);
              }}
            >
              {t("clean.facet.add")}
            </Button>
            <Button variant="ghost" onClick={() => setAction(null)}>
              {t("clean.facet.cancel")}
            </Button>
          </div>
        </div>
      )}
      {facet.sample.length > 0 && <Sample facet={facet} />}
    </div>
  );
}

function reasonOf(action: Action, facet: Facet, selection: Selection, rows: number, into: string, counts: Map<string, number>, named: number): string {
  const column = facet.column;
  switch (action) {
    case "merge": {
      const moving = selection.values.filter((value) => value !== into);
      const changed = moving.reduce((sum, value) => sum + (counts.get(value) ?? 0), 0);
      const changes = formatCount(changed);
      return moving.length === 1 ? t("clean.facet.reason.mergeOne", { value: moving[0]!, target: into, column, rows: changes }) : t("clean.facet.reason.merge", { count: formatCount(moving.length), target: into, column, rows: changes });
    }
    case "missing": {
      const only = selection.values.length === 1 && !selection.unreadable ? selection.values[0] : undefined;
      return only !== undefined ? t("clean.facet.reason.missingOne", { value: only, column, rows: formatCount(rows) }) : t("clean.facet.reason.missing", { count: formatCount(named), column, rows: formatCount(rows) });
    }
    case "remove":
      return t("clean.facet.reason.remove", { rows: formatCount(rows), total: formatCount(facet.rows) });
    case "keep":
      return t("clean.facet.reason.keep", { rows: formatCount(rows), gone: formatCount(facet.rows - rows) });
    case "outside":
      return t("clean.facet.reason.outside", {
        column,
        low: shownBound(selection.low ?? facet.low, facet),
        high: shownBound(selection.high ?? facet.high, facet),
        cells: formatCount(Math.max(0, facet.rows - facet.blank - rows)),
      });
  }
}

/** The first rows of the draft the selection picks, the faceted column first. */
function Sample({ facet }: { facet: Facet }) {
  return (
    <div className="space-y-1">
      <p className="text-xs text-ink-3">{t("clean.facet.first", { count: formatCount(facet.sample.length) })}</p>
      <div className="max-h-[28rem] overflow-auto rounded-lg border border-line bg-surface-1">
        <table data-keep-line className="w-full border-collapse text-[12px] tabular-nums">
          <thead className="sticky top-0 bg-surface-1">
            <tr className="border-b border-line text-left text-[11px] text-ink-3">
              <th scope="col" className="px-2 py-1.5 text-right font-normal">
                {t("clean.preview.row")}
              </th>
              {facet.shown.map((name) => (
                <th key={name} scope="col" className="px-2 py-1.5 font-mono font-normal whitespace-nowrap">
                  {name}
                  <ColumnBar column={name} dense className="mt-1 min-w-12" />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {facet.sample.map((row) => (
              <tr key={row.row} className="border-b border-line align-top last:border-b-0">
                <th scope="row" className="px-2 py-1 text-right font-normal whitespace-nowrap text-ink-3">
                  {formatCount(row.row + 1)}
                </th>
                {row.values.map((value, index) => (
                  <td key={facet.shown[index] ?? index} className={`px-2 py-1 whitespace-nowrap ${index === 0 ? "text-ink-1" : "text-ink-2"}`}>
                    {value === null ? <em className="text-ink-3">{t("clean.empty")}</em> : visible(value)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
