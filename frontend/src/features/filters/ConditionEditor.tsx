import { useMemo, useState } from "react";

import type { Column } from "../../api/client";
import type { Condition } from "../../api/filters";
import { useChart } from "../../api/hooks";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Select, TextInput } from "../../ui/fields";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { choiceOf, choicesWith, daysBefore, PARTS, pastedValues, upgraded, type Choice } from "./conditions";

type ConditionEditorProps = {
  datasetId: string;
  columns: Column[];
  /** Columns offered for "+ column" and "ordered by" (default: `columns`). */
  others?: Column[];
  initial?: Condition;
  onSave: (condition: Condition) => void;
  onCancel: () => void;
};

type Draft = {
  column: string;
  choice: string;
  value: string;
  low: string;
  high: string;
  values: string[];
  pasted: string; // "is any of" on an identifier: the pasted list, as typed
  days: string;
  anchor: string; // "in the last N days": the last date it counts back from (R-FLT-08)
  part: string;
  extra: string[];
  multi: boolean; // the value is a combination of columns (a composite key)
  keep: "first" | "last";
  orderBy: string;
};

const MAX_VALUES = 1_000;
const MAX_DAYS = 36_500;

/** Where a column's values lie, from its summary computed at import (R-FLT-09): no query. */
type Bounds =
  | { kind: "number"; min: number; max: number; median: number }
  | { kind: "date"; min: string; max: string }
  | null;

function boundsOf(column: Column | undefined): Bounds {
  const summary = column?.summary;
  if (!summary) return null;
  if (column.type === "datetime") {
    const { min, max } = summary.date ?? {};
    return min && max ? { kind: "date", min: min.slice(0, 10), max: max.slice(0, 10) } : null;
  }
  return summary.numeric ? { kind: "number", ...summary.numeric } : null;
}

const plain = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
const small = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 3 });

function shown(value: number): string {
  return Math.abs(value) >= 1 || value === 0 ? plain.format(value) : small.format(value);
}

/** A number as a placeholder (no thousands separator: it reads as something to type). */
function typed(value: number): string {
  return String(Number(value.toPrecision(6)));
}

const today = () => new Date().toISOString().slice(0, 10);

/** Dates open inside the data (R-FLT-10): after = the first date, before = the last, a range
 *  = first to last. A field still on a bound (not touched) follows the new choice. */
function prefilled(draft: Draft, column: Column | undefined, choice: Choice | undefined): Draft {
  if (!choice) return draft;
  const next = choice.input === "days" && !draft.days ? { ...draft, days: "30" } : draft;
  const bounds = boundsOf(column);
  if (bounds?.kind !== "date") return next;
  const untouched = (raw: string) => raw === "" || raw === bounds.min || raw === bounds.max;
  if (choice.input === "range" && untouched(next.low) && untouched(next.high)) {
    return { ...next, low: bounds.min, high: bounds.max };
  }
  if (choice.input === "value" && untouched(next.value)) {
    return { ...next, value: choice.op === "lt" || choice.op === "lte" ? bounds.max : bounds.min };
  }
  return next;
}

function text(value: Condition["value"] | undefined, date: boolean): string {
  if (value === undefined || value === null) return "";
  return date ? String(value).slice(0, 10) : String(value);
}

function draftOf(condition: Condition | undefined, columns: Column[]): Draft {
  const column = columns.find((c) => c.name === condition?.column) ?? columns[0];
  const current = condition && column ? upgraded(condition, column.type) : undefined;
  const choice = current && column ? choiceOf(current, column.type) : undefined;
  const first = column ? choicesWith(column.type, undefined)[0] : undefined;
  const date = column?.type === "datetime";
  const values = current?.values ?? [];
  const lastDays = current?.op === "last_days";
  const draft: Draft = {
    column: column?.name ?? "",
    choice: choice?.id ?? first?.id ?? "",
    value: lastDays ? "" : text(current?.value, date),
    low: text(current?.low, date),
    high: lastDays ? "" : text(current?.high, date),
    values,
    pasted: choice?.input === "paste" ? values.join("\n") : "",
    days: lastDays ? text(current?.value, false) : "",
    anchor: lastDays ? text(current?.high, true) : "",
    part: current?.part ?? "day_of_week",
    extra: current?.columns ?? [],
    multi: Boolean(current?.columns?.length),
    keep: current?.keep ?? "first",
    orderBy: current?.order_by ?? "",
  };
  return condition ? draft : prefilled(draft, column, first);
}

/** Builds the API condition, or explains what is missing or wrong (R-FLT-11). */
function build(draft: Draft, column: Column, choice: Choice): Condition | string {
  const base = { column: column.name, op: choice.op, negate: choice.negate };
  const numeric = column.type === "numeric_continuous" || column.type === "numeric_discrete";
  const read = (raw: string) => (numeric ? Number(raw) : raw);
  const filled = (raw: string) => raw.trim() !== "" && (!numeric || Number.isFinite(Number(raw)));
  if (choice.input === "none") return base;
  if (choice.input === "rows") {
    const columns = draft.multi ? draft.extra.filter((name) => name !== column.name) : [];
    if (draft.multi && !columns.length) return t("filters.needKeyColumns");
    return choice.op === "unique"
      ? { ...base, columns, keep: draft.keep, order_by: draft.orderBy || null }
      : { ...base, columns };
  }
  if (choice.input === "value") {
    return filled(draft.value) ? { ...base, value: read(draft.value) } : t("filters.needValue");
  }
  if (choice.input === "range") {
    if (!filled(draft.low) || !filled(draft.high)) return t("filters.needRange");
    const reversed = numeric ? Number(draft.low) > Number(draft.high) : draft.low > draft.high;
    if (reversed) return t("filters.rangeOrder");
    return { ...base, low: read(draft.low), high: read(draft.high) };
  }
  if (choice.input === "days") {
    const days = Number(draft.days);
    if (!Number.isInteger(days) || days < 1 || days > MAX_DAYS) return t("filters.needDays");
    return { ...base, value: days, high: anchorOf(draft, column) };
  }
  const values = choice.input === "paste" ? pastedValues(draft.pasted) : draft.values;
  if (!values.length) return t("filters.needValues");
  if (values.length > MAX_VALUES) return t("filters.tooManyPasted", { max: formatCount(MAX_VALUES) });
  if (choice.input === "part") return { ...base, part: draft.part as Condition["part"], values };
  return { ...base, values: choice.input === "one" ? values.slice(0, 1) : values };
}

/** The last date "in the last N days" counts back from: kept from the condition, else the
 *  column's last date, else today (R-FLT-08). */
function anchorOf(draft: Draft, column: Column): string {
  const bounds = boundsOf(column);
  return draft.anchor || (bounds?.kind === "date" ? bounds.max : today());
}

/** Whether a threshold misses the data: it keeps no row, or every row (R-FLT-11, a warning). */
function reach(choice: Choice, draft: Draft, bounds: Bounds): "none" | "all" | null {
  if (!bounds || choice.legacy) return null;
  const date = bounds.kind === "date";
  // dates compare as days (a number of milliseconds), like numbers
  const scale = (raw: string | number): number | null => {
    if (typeof raw === "number") return raw;
    if (raw.trim() === "") return null;
    const number = date ? Date.parse(raw.slice(0, 10)) : Number(raw);
    return Number.isFinite(number) ? number : null;
  };
  const min = scale(bounds.min);
  const max = scale(bounds.max);
  if (min === null || max === null) return null;
  const read = scale;
  if (choice.input === "value") {
    const v = read(draft.value);
    if (v === null) return null;
    const outside = v < min || v > max;
    const cases: Record<string, [boolean, boolean]> = date
      ? { lt: [v <= min, v > max], gt: [v > max, v < min] }
      : {
          eq: [outside, false],
          neq: [false, outside],
          gt: [v >= max, v < min],
          gte: [v > max, v <= min],
          lt: [v <= min, v > max],
          lte: [v < min, v >= max],
        };
    const [none, all] = cases[choice.id] ?? [false, false];
    return none ? "none" : all ? "all" : null;
  }
  if (choice.input === "range") {
    const [low, high] = [read(draft.low), read(draft.high)];
    if (low === null || high === null || low > high) return null;
    const disjoint = high < min || low > max;
    const covers = !date && low <= min && high >= max;
    if (choice.op !== "between") return null;
    if (choice.negate) return covers ? "none" : disjoint && !date ? "all" : null;
    return disjoint ? "none" : covers ? "all" : null;
  }
  return null;
}

export function ConditionEditor({ datasetId, columns, others = columns, initial, onSave, onCancel }: ConditionEditorProps) {
  const [draft, setDraft] = useState<Draft>(() => draftOf(initial, columns));
  const [error, setError] = useState<string | null>(null);
  const column = columns.find((c) => c.name === draft.column);
  // the saved condition's older operator stays offered while it is the one being edited
  const choices = column ? choicesWith(column.type, initial?.column === column.name ? initial : undefined) : [];
  const choice = choices.find((item) => item.id === draft.choice) ?? choices[0];
  const bounds = boundsOf(column);
  const update = (patch: Partial<Draft>) => {
    setError(null);
    setDraft((current) => ({ ...current, ...patch }));
  };

  const submit = () => {
    if (!column || !choice) return;
    const result = build(draft, column, choice);
    if (typeof result === "string") setError(result);
    else onSave(result);
  };

  const isDate = column?.type === "datetime";
  const inputType = isDate ? "date" : column?.type.startsWith("numeric") ? "number" : "text";
  // dates: the picker opens inside the data (R-FLT-10); typing outside is warned, not blocked
  const limits = bounds?.kind === "date" ? { min: bounds.min, max: bounds.max } : {};
  const number = bounds?.kind === "number" ? bounds : null;
  const reached = choice ? reach(choice, draft, bounds) : null;

  return (
    <form
      className="space-y-2.5"
      noValidate
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
    >
      <div className="grid grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)] gap-2">
        <Select
          aria-label={t("filters.column")}
          className="w-full min-w-0"
          value={draft.column}
          onChange={(event) => {
            const next = columns.find((c) => c.name === event.target.value);
            const first = next ? choicesWith(next.type, undefined)[0] : undefined;
            setError(null);
            setDraft((current) =>
              prefilled(
                {
                  ...current,
                  column: event.target.value,
                  choice: first?.id ?? "",
                  values: [],
                  value: "",
                  low: "",
                  high: "",
                  pasted: "",
                  anchor: "",
                  extra: [],
                  multi: false,
                },
                next,
                first,
              ),
            );
          }}
        >
          {columns.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name}
            </option>
          ))}
        </Select>
        <Select
          aria-label={t("filters.condition")}
          className="w-full min-w-0"
          value={choice?.id ?? ""}
          onChange={(event) => {
            const next = choices.find((item) => item.id === event.target.value);
            setError(null);
            setDraft((current) => prefilled({ ...current, choice: event.target.value }, column, next));
          }}
        >
          {choices.map((item) => (
            <option key={item.id} value={item.id}>
              {item.label}
            </option>
          ))}
        </Select>
      </div>

      {choice?.input === "value" && (
        <TextInput
          aria-label={t("filters.value")}
          type={inputType}
          step="any"
          {...limits}
          placeholder={number ? t("filters.example", { value: typed(number.median) }) : isDate ? undefined : examplesOf(column)}
          value={draft.value}
          onChange={(event) => update({ value: event.target.value })}
          className="w-full"
        />
      )}
      {choice?.input === "range" && (
        <div className="grid grid-cols-2 gap-2">
          <TextInput
            aria-label={t("filters.from")}
            placeholder={number ? typed(number.min) : t("filters.from")}
            className="w-full min-w-0"
            type={inputType}
            step="any"
            {...limits}
            value={draft.low}
            onChange={(event) => update({ low: event.target.value })}
          />
          <TextInput
            aria-label={t("filters.to")}
            placeholder={number ? typed(number.max) : t("filters.to")}
            className="w-full min-w-0"
            type={inputType}
            step="any"
            {...limits}
            value={draft.high}
            onChange={(event) => update({ high: event.target.value })}
          />
        </div>
      )}
      {choice?.input === "days" && column && (
        <DaysInput days={draft.days} anchor={anchorOf(draft, column)} dated={bounds?.kind === "date" || Boolean(draft.anchor)} onChange={(days) => update({ days })} />
      )}
      {(choice?.input === "value" || choice?.input === "range") && bounds && (
        <p className="text-xs text-ink-3 tabular-nums">
          {bounds.kind === "date"
            ? t("filters.dateRange", { min: bounds.min, max: bounds.max })
            : t("filters.numberRange", { min: shown(bounds.min), max: shown(bounds.max), median: shown(bounds.median) })}
        </p>
      )}
      {reached && bounds && (
        <p className="text-xs text-warn" role="status">
          {t(reached === "none" ? "filters.keepsNone" : "filters.keepsAll", {
            min: bounds.kind === "date" ? bounds.min : shown(bounds.min),
            max: bounds.kind === "date" ? bounds.max : shown(bounds.max),
          })}
        </p>
      )}
      {choice?.input === "one" && column && (
        <OnePicker datasetId={datasetId} column={column} value={draft.values[0] ?? ""} onChange={(value) => update({ values: value ? [value] : [] })} />
      )}
      {choice?.input === "values" && column && (
        <ValuesPicker datasetId={datasetId} column={column} selected={draft.values} onChange={(values) => update({ values })} />
      )}
      {choice?.input === "paste" && (
        <PasteList text={draft.pasted} placeholder={examplesOf(column, "\n")} onChange={(pasted) => update({ pasted })} />
      )}
      {choice?.input === "part" && (
        <PartPicker part={draft.part} selected={draft.values} onChange={(part, values) => update({ part, values })} />
      )}
      {choice?.input === "rows" && column && (
        <RowsOptions
          column={column.name}
          unique={choice.op === "unique"}
          columns={others.filter((c) => c.type !== "ignored" && !c.excluded)}
          draft={draft}
          onChange={update}
        />
      )}

      {error && <p className="text-xs text-bad">{error}</p>}
      <div className="flex justify-end gap-2">
        <Button variant="ghost" onClick={onCancel}>
          {t("filters.cancel")}
        </Button>
        <Button variant="primary" type="submit">
          {initial ? t("filters.update") : t("filters.apply")}
        </Button>
      </div>
    </form>
  );
}

/** "e.g. FR-0012" from the column's examples (summary), as a placeholder. */
function examplesOf(column: Column | undefined, separator = ", "): string | undefined {
  const examples = column?.summary?.examples.slice(0, 2) ?? [];
  return examples.length ? t("filters.example", { value: examples.join(separator) }) : undefined;
}

/** "in the last N days": N, and the dates it covers (R-FLT-08). */
function DaysInput({
  days,
  anchor,
  dated,
  onChange,
}: {
  days: string;
  anchor: string;
  dated: boolean;
  onChange: (days: string) => void;
}) {
  const count = Number(days);
  const valid = Number.isInteger(count) && count >= 1 && count <= MAX_DAYS;
  return (
    <div className="space-y-1.5">
      <TextInput
        aria-label={t("filters.days")}
        type="number"
        min={1}
        max={MAX_DAYS}
        step={1}
        value={days}
        onChange={(event) => onChange(event.target.value)}
        className="w-full"
      />
      {valid && (
        <p className="text-xs text-ink-3 tabular-nums">
          {t(dated ? "filters.lastDays" : "filters.lastDaysToday", { start: daysBefore(anchor, count - 1), end: anchor })}
        </p>
      )}
    </div>
  );
}

/** "is any of" on an identifier: a list pasted from elsewhere (R-FLT-05). */
function PasteList({ text, placeholder, onChange }: { text: string; placeholder?: string; onChange: (text: string) => void }) {
  const count = pastedValues(text).length;
  return (
    <div className="space-y-1.5">
      <textarea
        aria-label={t("filters.pasteLabel")}
        placeholder={placeholder}
        value={text}
        onChange={(event) => onChange(event.target.value)}
        rows={5}
        className="block w-full rounded-md border border-line bg-surface-1 px-2 py-1.5 font-mono text-xs text-ink-1 outline-none focus-visible:border-accent"
      />
      <p className="text-xs text-ink-3">
        {t("filters.pasteHint")}
        {count > 0 && ` ${t("filters.pasted", { count: formatCount(count) })}`}
      </p>
    </div>
  );
}

/** "is" / "is not" on a category: one value from the data, with its count (R-FLT-03). */
function OnePicker({
  datasetId,
  column,
  value,
  onChange,
}: {
  datasetId: string;
  column: Column;
  value: string;
  onChange: (value: string) => void;
}) {
  const chart = useChart(datasetId, column.name, { full: true }, true, { unfiltered: true });
  if (chart.data && chart.data.kind !== "bars") {
    return (
      <TextInput aria-label={t("filters.value")} value={value} onChange={(event) => onChange(event.target.value)} className="w-full" />
    );
  }
  const bars = chart.data?.bars ?? [];
  const known = !value || bars.some((bar) => bar.label === value);
  return (
    <Select aria-label={t("filters.value")} className="w-full" value={value} aria-busy={!chart.data} onChange={(event) => onChange(event.target.value)}>
      <option value="">{t("filters.pickValue")}</option>
      {!known && <option value={value}>{value}</option>}
      {bars.map((bar) => (
        <option key={bar.label} value={bar.label}>
          {`${bar.label} (${formatCount(bar.count)})`}
        </option>
      ))}
    </Select>
  );
}

/** Checkboxes over the column's values (every row, not only the filtered ones). */
function ValuesPicker({
  datasetId,
  column,
  selected,
  onChange,
}: {
  datasetId: string;
  column: Column;
  selected: string[];
  onChange: (values: string[]) => void;
}) {
  const [query, setQuery] = useState("");
  const chart = useChart(datasetId, column.name, { full: true }, true, { unfiltered: true });
  const values = useMemo(() => chart.data?.bars ?? [], [chart.data]);
  const needle = query.trim().toLowerCase();
  const shown = needle ? values.filter((bar) => bar.label.toLowerCase().includes(needle)) : values;
  const toggle = (label: string) =>
    onChange(selected.includes(label) ? selected.filter((item) => item !== label) : [...selected, label]);

  if (chart.data && chart.data.kind !== "bars") {
    return <p className="text-xs text-ink-3">{t("filters.tooManyValues")}</p>;
  }
  return (
    <div className="rounded-md border border-line">
      <div className="flex items-center gap-2 border-b border-line p-1.5">
        <TextInput
          aria-label={t("filters.searchValues")}
          placeholder={t("filters.searchValues")}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          className="h-7 flex-1 text-xs"
        />
        <button type="button" className="text-xs text-ink-2 hover:text-ink-1" onClick={() => onChange([...new Set([...selected, ...shown.map((bar) => bar.label)])])}>
          {t("filters.all")}
        </button>
        <button type="button" className="text-xs text-ink-2 hover:text-ink-1" onClick={() => onChange([])}>
          {t("filters.none")}
        </button>
      </div>
      <ul className="max-h-52 overflow-y-auto py-1" aria-busy={!chart.data}>
        {shown.map((bar) => (
          <li key={bar.label}>
            <label className="flex cursor-pointer items-center gap-2 px-2.5 py-1 text-[13px] hover:bg-surface-2">
              <input type="checkbox" className="size-3.5 accent-(--accent)" checked={selected.includes(bar.label)} onChange={() => toggle(bar.label)} />
              <span className="min-w-0 flex-1 truncate" title={bar.label}>
                {bar.label}
              </span>
              <span className="text-xs text-ink-3 tabular-nums">{formatCount(bar.count)}</span>
            </label>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Unique values / Duplicates: more columns making up the value; which row to keep. */
/** Unique values / Duplicates: on this column, or on a combination of columns (a key). */
function RowsOptions({
  column,
  unique,
  columns,
  draft,
  onChange,
}: {
  column: string;
  unique: boolean;
  columns: Column[];
  draft: Draft;
  onChange: (patch: Partial<Draft>) => void;
}) {
  const candidates = columns.filter((c) => c.name !== column);
  const toggle = (name: string) =>
    onChange({ extra: draft.extra.includes(name) ? draft.extra.filter((item) => item !== name) : [...draft.extra, name] });
  return (
    <div className="space-y-2.5 rounded-md border border-line p-2.5">
      <div className="grid grid-cols-[auto_minmax(0,1fr)] items-center gap-x-3 gap-y-2">
        <span className="text-xs text-ink-3">{t("filters.basedOn")}</span>
        <SegmentedControl
          label={t("filters.basedOn")}
          value={draft.multi ? "key" : "single"}
          options={[
            { value: "single", label: t("filters.basedOnColumn") },
            { value: "key", label: t("filters.basedOnKey") },
          ]}
          onChange={(mode) => onChange({ multi: mode === "key" })}
        />
      </div>
      <p className="text-xs text-ink-3">
        {draft.multi
          ? t(unique ? "filters.keyUniqueHint" : "filters.keyDuplicatesHint", { key: [column, ...draft.extra].join(" + ") })
          : t(unique ? "filters.uniqueHint" : "filters.duplicatesHint", { column })}
      </p>
      {draft.multi && (
        <div className="rounded-md border border-line">
          <label className="flex items-center gap-2 border-b border-line px-2.5 py-1 text-[13px] text-ink-3">
            <input type="checkbox" className="size-3.5" checked disabled />
            <span className="truncate font-mono text-xs">{column}</span>
          </label>
          <ul className="max-h-40 overflow-y-auto py-1" aria-label={t("filters.keyColumns")}>
            {candidates.map((c) => (
              <li key={c.name}>
                <label className="flex cursor-pointer items-center gap-2 px-2.5 py-1 text-[13px] hover:bg-surface-2">
                  <input
                    type="checkbox"
                    className="size-3.5 accent-(--accent)"
                    checked={draft.extra.includes(c.name)}
                    onChange={() => toggle(c.name)}
                  />
                  <span className="truncate font-mono text-xs">{c.name}</span>
                </label>
              </li>
            ))}
          </ul>
        </div>
      )}
      {unique && (
        <div className="grid grid-cols-[auto_minmax(0,1fr)] items-center gap-x-3 gap-y-2">
          <span className="text-xs text-ink-3">{t("filters.keep")}</span>
          <SegmentedControl
            label={t("filters.keep")}
            value={draft.keep}
            options={[
              { value: "first", label: t("filters.keepFirst") },
              { value: "last", label: t("filters.keepLast") },
            ]}
            onChange={(keep) => onChange({ keep })}
          />
          <span className="text-xs text-ink-3">{t("filters.orderedBy")}</span>
          <Select aria-label={t("filters.orderedBy")} className="w-full min-w-0" value={draft.orderBy} onChange={(event) => onChange({ orderBy: event.target.value })}>
            <option value="">{t("filters.fileOrder")}</option>
            {columns.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name}
              </option>
            ))}
          </Select>
        </div>
      )}
    </div>
  );
}

function PartPicker({
  part,
  selected,
  onChange,
}: {
  part: string;
  selected: string[];
  onChange: (part: string, values: string[]) => void;
}) {
  const spec = PARTS[part] ?? PARTS.day_of_week;
  return (
    <div className="space-y-2">
      <Select aria-label={t("filters.part")} value={part} onChange={(event) => onChange(event.target.value, [])} className="w-full">
        {Object.entries(PARTS).map(([id, item]) => (
          <option key={id} value={id}>
            {item.label}
          </option>
        ))}
      </Select>
      <div className="flex flex-wrap gap-1">
        {spec?.values.map((item) => {
          const on = selected.includes(item.value);
          return (
            <button
              key={item.value}
              type="button"
              aria-pressed={on}
              onClick={() => onChange(part, on ? selected.filter((v) => v !== item.value) : [...selected, item.value])}
              className={`h-7 min-w-9 rounded-md border px-2 text-xs ${on ? "border-accent bg-accent-soft text-ink-1" : "border-line text-ink-2 hover:text-ink-1"}`}
            >
              {item.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
