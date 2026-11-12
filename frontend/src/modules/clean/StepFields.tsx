import { Plus, X } from "lucide-react";
import type { ReactNode } from "react";

import { t, type MessageKey } from "../../lib/i18n";
import { Checkbox, Select } from "../../ui/fields";
import type { RowCondition, Step, StepOf } from "./api";
import { Cell, CommitInput } from "./Cell";

const field = "h-8 w-full min-w-0 text-[13px]";
const split = (text: string) =>
  text
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
const joined = (values: string[] | null | undefined) => (values ?? []).join(", ");

type Props = {
  step: Step;
  /** The step as its rule proposed it: a setting that differs gets the reset icon. */
  original?: Step | null;
  columns: string[];
  onChange: (step: Step) => void;
};

/** The settings of one step, in cells three per row (045, 046). */
export function StepFields({ step, original, columns, onChange }: Props) {
  const set = <S extends Step>(patch: Partial<S>) => onChange({ ...step, ...patch } as Step);
  const differs = (key: string) =>
    Boolean(original && JSON.stringify((original as Record<string, unknown>)[key]) !== JSON.stringify((step as Record<string, unknown>)[key]));
  const reset = (key: string) => () => original && onChange({ ...step, [key]: (original as Record<string, unknown>)[key] } as Step);
  const cell = (key: string, label: MessageKey, control: ReactNode, wide: boolean | "full" = false, why?: string) => (
    <Cell key={key} label={t(label)} why={why} changed={differs(key)} onReset={reset(key)} wide={wide}>
      {control}
    </Cell>
  );
  const columnSelect = (value: string, onPick: (column: string) => void, label: MessageKey = "clean.field.column") => (
    <Select aria-label={t(label)} value={value} onChange={(event) => onPick(event.target.value)} className={field}>
      {[...new Set([value, ...columns])].map((name) => (
        <option key={name} value={name}>
          {name}
        </option>
      ))}
    </Select>
  );
  const text = (key: string, label: MessageKey, value: string, commit: (value: string) => void, placeholder?: string) => (
    <CommitInput key={`${key}:${value}`} label={t(label)} value={value} onCommit={commit} placeholder={placeholder} />
  );
  const choice = <V extends string>(label: MessageKey, value: V, options: V[], prefix: string, onPick: (value: V) => void) => (
    <Select aria-label={t(label)} value={value} onChange={(event) => onPick(event.target.value as V)} className={field}>
      {options.map((option) => (
        <option key={option} value={option}>
          {t(`${prefix}.${option}` as MessageKey)}
        </option>
      ))}
    </Select>
  );

  const cells: ReactNode[] = [];
  if ("column" in step) cells.push(cell("column", "clean.field.column", columnSelect(step.column, (column) => set({ column }))));
  if ("columns" in step)
    cells.push(
      cell(
        "columns",
        "clean.field.columns",
        text("columns", "clean.field.columns", joined(step.columns), (value) => split(value).length && set({ columns: split(value) }), t("clean.field.columnsHint")),
        "full",
        t("clean.field.columnsHint"),
      ),
    );

  switch (step.op) {
    case "rename_column":
    case "duplicate_column":
      cells.push(cell("to", "clean.field.to", text("to", "clean.field.to", step.to, (to) => to.trim() && set({ to: to.trim() }))));
      break;
    case "assemble_date": {
      const s = step as StepOf<"assemble_date">;
      const optional = (key: "day" | "hour" | "minute", label: MessageKey, why?: string) =>
        cell(
          key,
          label,
          <Select aria-label={t(label)} value={s[key] ?? ""} onChange={(event) => set({ [key]: event.target.value || null })} className={field}>
            <option value="">{t("clean.field.noPart")}</option>
            {[...new Set([...(s[key] ? [s[key]] : []), ...columns])].map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </Select>,
          false,
          why,
        );
      cells.push(cell("year", "clean.field.year", columnSelect(s.year, (year) => set({ year }), "clean.field.year")));
      cells.push(cell("month", "clean.field.month", columnSelect(s.month, (month) => set({ month }), "clean.field.month")));
      cells.push(optional("day", "clean.field.day", t("clean.field.why.day")));
      cells.push(optional("hour", "clean.field.hour"));
      if (s.hour) cells.push(optional("minute", "clean.field.minute"));
      cells.push(cell("to", "clean.field.to", text("to", "clean.field.to", s.to, (to) => to.trim() && set({ to: to.trim() }))));
      break;
    }
    case "change_type": {
      const s = step as StepOf<"change_type">;
      cells.push(cell("to", "clean.field.type", choice("clean.field.type", s.to, ["number", "integer", "date", "datetime", "boolean", "text"], "clean.type", (to) => set({ to }))));
      if (s.to === "number" || s.to === "integer")
        cells.push(cell("decimal", "clean.field.decimal", choice("clean.field.decimal", s.decimal, [".", ","], "clean.decimal", (decimal) => set({ decimal })), false, t("clean.field.why.decimal")));
      if (s.to === "date" || s.to === "datetime") {
        cells.push(cell("formats", "clean.field.formats", text("formats", "clean.field.formats", joined(s.formats), (value) => set({ formats: split(value) }), "%Y-%m-%d"), true, t("clean.field.formatsHint")));
        cells.push(
          cell(
            "excel_serial",
            "clean.field.excel",
            <Checkbox checked={s.excel_serial} onChange={(event) => set({ excel_serial: event.target.checked })} label={t("clean.field.excel")} className="h-8 text-[13px]" />,
            false,
            t("clean.field.why.excel"),
          ),
        );
      }
      if (s.to === "boolean") {
        cells.push(cell("true_values", "clean.field.trueValues", text("true_values", "clean.field.trueValues", joined(s.true_values), (value) => set({ true_values: split(value) }), "yes, y, 1")));
        cells.push(cell("false_values", "clean.field.falseValues", text("false_values", "clean.field.falseValues", joined(s.false_values), (value) => set({ false_values: split(value) }), "no, n, 0")));
      }
      break;
    }
    case "change_case":
      cells.push(cell("case", "clean.field.case", choice("clean.field.case", step.case, ["lower", "upper", "title"], "clean.case", (value) => set({ case: value }))));
      break;
    case "replace_values":
      cells.push(cell("find", "clean.field.find", text("find", "clean.field.find", step.find, (find) => find && set({ find }))));
      cells.push(cell("replace", "clean.field.replace", text("replace", "clean.field.replace", step.replace, (replace) => set({ replace }))));
      cells.push(cell("match", "clean.field.match", choice("clean.field.match", step.match, ["exact", "contains", "regex"], "clean.match", (match) => set({ match }))));
      cells.push(
        cell(
          "case_sensitive",
          "clean.field.caseSensitive",
          <Checkbox checked={step.case_sensitive} onChange={(event) => set({ case_sensitive: event.target.checked })} label={t("clean.field.caseSensitive")} className="h-8 text-[13px]" />,
        ),
      );
      break;
    case "map_values":
      cells.push(
        <Cell key="map" label={t("clean.field.map")} why={step.rule ? t("clean.field.why.mapFromRule") : null} changed={differs("map")} onReset={reset("map")} wide="full">
          <MapEditor value={step.map} onChange={(map) => set({ map })} />
        </Cell>,
      );
      break;
    case "fill":
      cells.push(cell("direction", "clean.field.direction", choice("clean.field.direction", step.direction, ["down", "up"], "clean.direction", (direction) => set({ direction }))));
      break;
    case "fill_empty":
      cells.push(cell("value", "clean.field.value", text("value", "clean.field.value", step.value, (value) => set({ value }))));
      break;
    case "null_values":
      cells.push(cell("values", "clean.field.values", text("values", "clean.field.values", joined(step.values), (value) => split(value).length && set({ values: split(value) })), true, t("clean.field.valuesHint")));
      cells.push(cell("match", "clean.field.match", choice("clean.field.match", step.match, ["exact", "token"], "clean.match", (match) => set({ match }))));
      break;
    case "null_outside": {
      const number = (value: string) => (value.trim() === "" ? null : Number(value));
      cells.push(cell("min", "clean.field.min", text("min", "clean.field.min", step.min === null ? "" : String(step.min), (value) => set({ min: number(value) }))));
      cells.push(cell("max", "clean.field.max", text("max", "clean.field.max", step.max === null ? "" : String(step.max), (value) => set({ max: number(value) }))));
      break;
    }
    case "pad_zeros":
      cells.push(cell("length", "clean.field.length", text("length", "clean.field.length", String(step.length), (value) => Number(value) >= 2 && set({ length: Math.round(Number(value)) }))));
      break;
    case "filter_rows":
      cells.push(cell("action", "clean.field.action", choice("clean.field.action", step.action, ["remove", "keep"], "clean.action", (action) => set({ action }))));
      cells.push(cell("combine", "clean.field.combine", choice("clean.field.combine", step.combine, ["all", "any"], "clean.combine", (combine) => set({ combine }))));
      cells.push(
        <Cell key="conditions" label={t("clean.field.conditions")} changed={differs("conditions")} onReset={reset("conditions")} wide="full">
          <Conditions value={step.conditions} columns={columns} onChange={(conditions) => set({ conditions })} />
        </Cell>,
      );
      break;
    case "drop_duplicates":
      cells.push(cell("keys", "clean.field.keys", text("keys", "clean.field.keys", joined(step.keys), (value) => set({ keys: split(value).length ? split(value) : null }), t("clean.field.keysHint")), true, t("clean.field.keysHint")));
      cells.push(cell("keep", "clean.field.keep", choice("clean.field.keep", step.keep, ["first", "last", "most_complete"], "clean.keep", (keep) => set({ keep }))));
      break;
    default:
      break;
  }
  return <div className="params-grid">{cells}</div>;
}

/** Old value → new value, one line each (also the cluster editor of a merge, study § 5). */
function MapEditor({ value, onChange }: { value: Record<string, string | null>; onChange: (map: Record<string, string | null>) => void }) {
  const entries = Object.entries(value);
  const replace = (index: number, old: string, next: string | null) =>
    onChange(Object.fromEntries(entries.map((entry, i) => (i === index ? [old, next] : entry))));
  return (
    <div className="space-y-1.5">
      <div className="max-h-64 overflow-y-auto rounded-md border border-line">
        <table className="w-full text-[13px]">
          <thead>
            <tr className="border-b border-line text-left text-[11px] text-ink-3">
              <th scope="col" className="px-2 py-1 font-normal">{t("clean.field.old")}</th>
              <th scope="col" className="px-2 py-1 font-normal">{t("clean.field.new")}</th>
              <th scope="col" className="w-0"><span className="sr-only">{t("clean.field.removeRow")}</span></th>
            </tr>
          </thead>
          <tbody>
            {entries.slice(0, 200).map(([old, next], index) => (
              <tr key={`${index}:${old}`} className="border-b border-line last:border-b-0">
                <td className="max-w-0 px-2 py-1">
                  <CommitInput key={old} label={t("clean.field.old")} value={old} onCommit={(changed) => onChange(Object.fromEntries(entries.map((entry, i) => (i === index ? [changed, next] : entry))))} />
                </td>
                <td className="max-w-0 px-2 py-1">
                  <CommitInput key={next ?? ""} label={t("clean.field.new")} value={next ?? ""} placeholder={t("clean.empty")} onCommit={(changed) => replace(index, old, changed === "" ? null : changed)} />
                </td>
                <td className="px-1">
                  <button
                    type="button"
                    aria-label={t("clean.field.removeRow")}
                    title={t("clean.field.removeRow")}
                    disabled={entries.length < 2}
                    onClick={() => onChange(Object.fromEntries(entries.filter((_, i) => i !== index)))}
                    className="inline-flex size-7 items-center justify-center rounded-md text-ink-3 hover:bg-surface-2 hover:text-ink-1 disabled:opacity-40"
                  >
                    <X size={13} aria-hidden />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <button
        type="button"
        onClick={() => onChange({ ...value, "": "" })}
        disabled={"" in value}
        className="inline-flex items-center gap-1 text-xs text-accent hover:underline disabled:opacity-40"
      >
        <Plus size={12} aria-hidden />
        {t("clean.field.addRow")}
      </button>
    </div>
  );
}

const OPERATORS: RowCondition["op"][] = ["eq", "ne", "lt", "le", "gt", "ge", "contains", "empty", "not_empty", "in"];

function Conditions({ value, columns, onChange }: { value: RowCondition[]; columns: string[]; onChange: (conditions: RowCondition[]) => void }) {
  const update = (index: number, patch: Partial<RowCondition>) =>
    onChange(value.map((condition, i) => (i === index ? { ...condition, ...patch } : condition)));
  return (
    <div className="space-y-1.5">
      {value.map((condition, index) => (
        <div key={index} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)_auto] items-center gap-1.5">
          <Select aria-label={t("clean.field.column")} value={condition.column} onChange={(event) => update(index, { column: event.target.value })} className={field}>
            {[...new Set([condition.column, ...columns])].map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </Select>
          <Select
            aria-label={t("clean.field.operator")}
            value={condition.op}
            onChange={(event) => {
              const op = event.target.value as RowCondition["op"];
              update(index, op === "in" ? { op, values: condition.values ?? split(condition.value ?? ""), value: null } : { op, value: condition.value ?? (op === "empty" || op === "not_empty" ? null : ""), values: null });
            }}
            className={field}
          >
            {OPERATORS.map((op) => (
              <option key={op} value={op}>
                {t(`clean.operator.${op}` as MessageKey)}
              </option>
            ))}
          </Select>
          {condition.op === "empty" || condition.op === "not_empty" ? (
            <span />
          ) : condition.op === "in" ? (
            <CommitInput key={joined(condition.values)} label={t("clean.field.values")} value={joined(condition.values)} onCommit={(text) => split(text).length && update(index, { values: split(text) })} />
          ) : (
            <CommitInput key={condition.value ?? ""} label={t("clean.field.value")} value={condition.value ?? ""} onCommit={(text) => update(index, { value: text })} />
          )}
          <button
            type="button"
            aria-label={t("clean.field.removeCondition")}
            title={t("clean.field.removeCondition")}
            disabled={value.length < 2}
            onClick={() => onChange(value.filter((_, i) => i !== index))}
            className="inline-flex size-7 items-center justify-center rounded-md text-ink-3 hover:bg-surface-2 hover:text-ink-1 disabled:opacity-40"
          >
            <X size={13} aria-hidden />
          </button>
        </div>
      ))}
      <button
        type="button"
        disabled={value.length >= 20}
        onClick={() => onChange([...value, { column: columns[0] ?? value[0]?.column ?? "", op: "empty", value: null, values: null }])}
        className="inline-flex items-center gap-1 text-xs text-accent hover:underline"
      >
        <Plus size={12} aria-hidden />
        {t("clean.field.addCondition")}
      </button>
    </div>
  );
}
