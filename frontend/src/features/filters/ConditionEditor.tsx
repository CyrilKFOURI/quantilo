import { useMemo, useState } from "react";

import type { Column } from "../../api/client";
import type { Condition } from "../../api/filters";
import { useChart } from "../../api/hooks";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Select, TextInput } from "../../ui/fields";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { choiceOf, choicesFor, PARTS, type Choice } from "./conditions";

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
  part: string;
  extra: string[];
  multi: boolean; // the value is a combination of columns (a composite key)
  keep: "first" | "last";
  orderBy: string;
};

function draftOf(condition: Condition | undefined, columns: Column[]): Draft {
  const column = columns.find((c) => c.name === condition?.column) ?? columns[0];
  const choice = condition && column ? choiceOf(condition, column.type) : undefined;
  return {
    column: column?.name ?? "",
    choice: choice?.id ?? (column ? (choicesFor(column.type)[0]?.id ?? "") : ""),
    value: condition?.value !== undefined && condition.value !== null ? String(condition.value) : "",
    low: condition?.low !== undefined && condition.low !== null ? String(condition.low) : "",
    high: condition?.high !== undefined && condition.high !== null ? String(condition.high) : "",
    values: condition?.values ?? [],
    part: condition?.part ?? "day_of_week",
    extra: condition?.columns ?? [],
    multi: Boolean(condition?.columns?.length),
    keep: condition?.keep ?? "first",
    orderBy: condition?.order_by ?? "",
  };
}

/** Builds the API condition, or explains what is missing. */
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
    return { ...base, low: read(draft.low), high: read(draft.high) };
  }
  if (!draft.values.length) return t("filters.needValues");
  return choice.input === "part"
    ? { ...base, part: draft.part as Condition["part"], values: draft.values }
    : { ...base, values: draft.values };
}

export function ConditionEditor({ datasetId, columns, others = columns, initial, onSave, onCancel }: ConditionEditorProps) {
  const [draft, setDraft] = useState<Draft>(() => draftOf(initial, columns));
  const [error, setError] = useState<string | null>(null);
  const column = columns.find((c) => c.name === draft.column);
  const choices = column ? choicesFor(column.type) : [];
  const choice = choices.find((item) => item.id === draft.choice) ?? choices[0];
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

  return (
    <form
      className="space-y-2.5"
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
            update({
              column: event.target.value,
              choice: next ? (choicesFor(next.type)[0]?.id ?? "") : "",
              values: [],
              value: "",
              extra: [],
              multi: false,
            });
          }}
        >
          {columns.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name}
            </option>
          ))}
        </Select>
        <Select aria-label={t("filters.condition")} className="w-full min-w-0" value={choice?.id ?? ""} onChange={(event) => update({ choice: event.target.value })}>
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
          value={draft.value}
          onChange={(event) => update({ value: event.target.value })}
          className="w-full"
        />
      )}
      {choice?.input === "range" && (
        <div className="grid grid-cols-2 gap-2">
          <TextInput aria-label={t("filters.from")} placeholder={t("filters.from")} className="w-full min-w-0" type={inputType} step="any" value={draft.low} onChange={(event) => update({ low: event.target.value })} />
          <TextInput aria-label={t("filters.to")} placeholder={t("filters.to")} className="w-full min-w-0" type={inputType} step="any" value={draft.high} onChange={(event) => update({ high: event.target.value })} />
        </div>
      )}
      {choice?.input === "values" && column && (
        <ValuesPicker datasetId={datasetId} column={column} selected={draft.values} onChange={(values) => update({ values })} />
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
