import { Plus, X } from "lucide-react";
import type { ReactNode } from "react";

import { t, type MessageKey } from "../../lib/i18n";
import { Checkbox, Select } from "../../ui/fields";
import type { RowCondition, Step, StepOf } from "./api";
import { Cell, CommitInput } from "./Cell";
import { OtherDataset } from "./OtherDataset";
import { newCase, type Case } from "./steps";

type Aggregation = StepOf<"group_by">["aggregations"][number];

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
      const optional = (key: "day" | "hour" | "minute" | "time", label: MessageKey, why?: string) =>
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
      if (!s.time) cells.push(optional("hour", "clean.field.hour"));
      if (s.hour) cells.push(optional("minute", "clean.field.minute"));
      // C1 (fix1): the hour and minute from one time-of-day column ("07:40", "0740")
      if (!s.hour) cells.push(optional("time", "clean.field.time", t("clean.field.why.time")));
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
      // N12: only the rows of the countries whose codes have this length (none: every row)
      cells.push(
        <Cell key="conditions" label={t("clean.field.onlyRows")} why={t("clean.field.why.onlyRows")} changed={differs("conditions")} onReset={reset("conditions")} wide="full">
          <Conditions value={step.conditions} columns={columns} allowNone onChange={(conditions) => set({ conditions })} />
        </Cell>,
      );
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
    case "label_rows": {
      const s = step as StepOf<"label_rows">;
      const names = [...new Set(Object.values(s.labels))];
      cells.push(cell("to", "clean.field.to", text("to", "clean.field.to", s.to, (to) => to.trim() && set({ to: to.trim() }))));
      cells.push(cell("other", "clean.field.other", text("other", "clean.field.other", s.other ?? "", (other) => set({ other: other === "" ? null : other }), t("clean.empty")), false, t("clean.field.why.other")));
      cells.push(
        <Cell key="labels" label={t("clean.field.pinned")} why={t("clean.opHint.label_rows")} changed={differs("labels")} onReset={reset("labels")}>
          <p className="py-1.5 text-[13px] text-ink-2 tabular-nums">
            {t("clean.field.pinnedSummary", { rows: String(Object.keys(s.labels).length), labels: String(names.length), names: names.slice(0, 4).join(", ") })}
          </p>
        </Cell>,
      );
      break;
    }
    // 101: made by More checks; the cells and rows are listed, not edited (row numbers from 1)
    case "set_cells": {
      const s = step as StepOf<"set_cells">;
      const entries = Object.entries(s.cells);
      const examples = entries.slice(0, 4).map(([row, value]) => `${Number(row) + 1}: ${value ?? t("clean.empty")}`).join(", ");
      cells.push(
        <Cell key="cells" label={t("clean.field.cells")} why={t("clean.opHint.set_cells")} changed={differs("cells")} onReset={reset("cells")} wide="full">
          <p className="py-1.5 text-[13px] text-ink-2 tabular-nums">{t("clean.field.cellsSummary", { cells: String(entries.length), examples: entries.length > 4 ? `${examples}…` : examples })}</p>
        </Cell>,
      );
      break;
    }
    case "drop_rows": {
      const s = step as StepOf<"drop_rows">;
      const examples = s.rows.slice(0, 8).map((row) => String(row + 1)).join(", ");
      cells.push(
        <Cell key="rows" label={t("clean.field.rows")} why={t("clean.opHint.drop_rows")} changed={differs("rows")} onReset={reset("rows")} wide="full">
          <p className="py-1.5 text-[13px] text-ink-2 tabular-nums">{t("clean.field.rowsSummary", { rows: String(s.rows.length), examples: s.rows.length > 8 ? `${examples}…` : examples })}</p>
        </Cell>,
      );
      break;
    }
    case "drop_empty_rows": {
      const s = step as StepOf<"drop_empty_rows">;
      const share = (value: string) => {
        const read = Number(value.replace("%", "")) / (value.includes("%") || Number(value) > 1 ? 100 : 1);
        return value.trim() === "" ? null : read > 0 && read < 1 ? read : undefined;
      };
      cells.push(
        cell(
          "share",
          "clean.field.share",
          text("share", "clean.field.share", s.share === null ? "" : `${Math.round(s.share * 100)}%`, (value) => {
            const read = share(value);
            if (read !== undefined) set({ share: read });
          }, t("clean.field.shareAll")),
          false,
          t("clean.field.why.share"),
        ),
      );
      break;
    }
    // M5 (B1): numbers, splitting, extracting, calculating; the conditional step (C-45)
    case "transform_number": {
      const s = step as StepOf<"transform_number">;
      const number = (value: string) => (value.trim() === "" ? null : Number(value));
      cells.push(cell("how", "clean.field.how", choice("clean.field.how", s.how, ["round", "abs", "multiply", "divide", "clip", "bin"], "clean.how", (how) => set({ how }))));
      if (s.how === "round") cells.push(cell("decimals", "clean.field.decimals", text("decimals", "clean.field.decimals", String(s.decimals), (value) => Number.isInteger(Number(value)) && set({ decimals: Number(value) }))));
      if (s.how === "multiply" || s.how === "divide") cells.push(cell("factor", "clean.field.factor", text("factor", "clean.field.factor", String(s.factor), (value) => Number(value) !== 0 && Number.isFinite(Number(value)) && set({ factor: Number(value) }))));
      if (s.how === "clip") {
        cells.push(cell("min", "clean.field.min", text("min", "clean.field.min", s.min === null ? "" : String(s.min), (value) => set({ min: number(value) }))));
        cells.push(cell("max", "clean.field.max", text("max", "clean.field.max", s.max === null ? "" : String(s.max), (value) => set({ max: number(value) }))));
      }
      if (s.how === "bin") cells.push(cell("width", "clean.field.width", text("width", "clean.field.width", s.width === null ? "" : String(s.width), (value) => Number(value) > 0 && set({ width: Number(value) }))));
      break;
    }
    case "split_column": {
      const s = step as StepOf<"split_column">;
      const numbers = (value: string) => split(value).map(Number).filter((n) => Number.isInteger(n) && n >= 1);
      cells.push(cell("by", "clean.field.by", choice("clean.field.by", s.by, ["delimiter", "positions", "digits"], "clean.by", (by) => set({ by }))));
      if (s.by === "delimiter") cells.push(cell("delimiter", "clean.field.delimiter", text("delimiter", "clean.field.delimiter", s.delimiter, (delimiter) => delimiter && set({ delimiter }))));
      if (s.by === "positions")
        cells.push(cell("positions", "clean.field.positions", text("positions", "clean.field.positions", joined(s.positions.map(String)), (value) => set({ positions: numbers(value) }), "3, 7"), false, t("clean.field.why.positions")));
      cells.push(cell("to", "clean.field.parts", text("to", "clean.field.parts", joined(s.to), (value) => split(value).length >= 2 && set({ to: split(value) })), "full", t("clean.field.why.parts")));
      break;
    }
    case "merge_columns": {
      const s = step as StepOf<"merge_columns">;
      cells.push(cell("separator", "clean.field.separator", text("separator", "clean.field.separator", s.separator, (separator) => set({ separator }))));
      cells.push(cell("to", "clean.field.to", text("to", "clean.field.to", s.to, (to) => to.trim() && set({ to: to.trim() }))));
      break;
    }
    case "extract_text": {
      const s = step as StepOf<"extract_text">;
      cells.push(cell("how", "clean.field.how", choice("clean.field.how", s.how, ["first", "last", "between", "regex"], "clean.extract", (how) => set({ how }))));
      if (s.how === "first" || s.how === "last") cells.push(cell("count", "clean.field.count", text("count", "clean.field.count", String(s.count), (value) => Number(value) >= 1 && set({ count: Math.round(Number(value)) }))));
      if (s.how === "between") {
        cells.push(cell("start", "clean.field.start", text("start", "clean.field.start", s.start, (start) => set({ start }))));
        cells.push(cell("end", "clean.field.end", text("end", "clean.field.end", s.end, (end) => set({ end }))));
      }
      if (s.how === "regex") cells.push(cell("pattern", "clean.field.pattern", text("pattern", "clean.field.pattern", s.pattern ?? "", (pattern) => pattern && set({ pattern }), "([0-9]+)"), true, t("clean.field.why.pattern")));
      cells.push(cell("to", "clean.field.toOptional", text("to", "clean.field.toOptional", s.to ?? "", (to) => set({ to: to.trim() || null }), t("clean.field.inPlace")), false, t("clean.field.why.toOptional")));
      break;
    }
    case "format_text": {
      const s = step as StepOf<"format_text">;
      cells.push(cell("pad_to", "clean.field.padTo", text("pad_to", "clean.field.padTo", s.pad_to === null ? "" : String(s.pad_to), (value) => set({ pad_to: value.trim() === "" ? null : Math.max(1, Math.round(Number(value)) || 1) }))));
      cells.push(cell("pad_with", "clean.field.padWith", text("pad_with", "clean.field.padWith", s.pad_with, (value) => value.length === 1 && set({ pad_with: value }))));
      cells.push(cell("prefix", "clean.field.prefix", text("prefix", "clean.field.prefix", s.prefix, (prefix) => set({ prefix }))));
      cells.push(cell("suffix", "clean.field.suffix", text("suffix", "clean.field.suffix", s.suffix, (suffix) => set({ suffix }))));
      break;
    }
    case "parse_quantity": {
      // N23: one new column per kind of unit; an empty name leaves that kind out
      const s = step as StepOf<"parse_quantity">;
      const optional = (key: "to_grams" | "to_litres" | "to_metres" | "unit_to", label: MessageKey) =>
        cell(key, label, text(key, label, s[key] ?? "", (value) => set({ [key]: value.trim() || null }), t("clean.field.notMade")), false, t("clean.field.why.quantity"));
      cells.push(optional("to_grams", "clean.field.toGrams"));
      cells.push(optional("to_litres", "clean.field.toLitres"));
      cells.push(optional("to_metres", "clean.field.toMetres"));
      cells.push(optional("unit_to", "clean.field.unitTo"));
      break;
    }
    case "entity_column": {
      // N36: the anchor tells the entities apart; the clusters come from A-20 and its review
      const s = step as StepOf<"entity_column">;
      cells.push(
        cell(
          "anchor",
          "clean.field.anchor",
          <Select aria-label={t("clean.field.anchor")} value={s.anchor ?? ""} onChange={(event) => set({ anchor: event.target.value || null })} className={field}>
            <option value="">{t("clean.field.noAnchor")}</option>
            {[...new Set([...(s.anchor ? [s.anchor] : []), ...columns])].map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </Select>,
          false,
          t("clean.field.why.anchor"),
        ),
      );
      if (s.anchor) cells.push(cell("anchor_kind", "clean.field.anchorKind", choice("clean.field.anchorKind", s.anchor_kind, ["key", "postal_code", "phone", "email"], "clean.anchorKind", (anchor_kind) => set({ anchor_kind }))));
      cells.push(cell("to", "clean.field.to", text("to", "clean.field.to", s.to, (to) => to.trim() && set({ to: to.trim() }))));
      if (s.groups.length) cells.push(<p key="groups" className="text-xs text-ink-3 sm:col-span-2">{t("clean.field.entityClusters", { count: String(s.groups.length) })}</p>);
      break;
    }
    case "set_where":
    case "conditional_column": {
      const s = step as StepOf<"set_where"> | StepOf<"conditional_column">;
      if (s.op === "conditional_column") {
        cells.push(cell("to", "clean.field.to", text("to", "clean.field.to", s.to, (to) => to.trim() && set({ to: to.trim() }))));
        cells.push(cell("otherwise", "clean.field.otherwise", text("otherwise", "clean.field.otherwise", s.otherwise ?? "", (otherwise) => set({ otherwise: otherwise === "" ? null : otherwise }), t("clean.empty"))));
      }
      cells.push(
        <Cell key="cases" label={t("clean.field.cases")} why={t("clean.field.why.cases")} changed={differs("cases")} onReset={reset("cases")} wide="full">
          <Cases value={s.cases} columns={columns} onChange={(cases) => set({ cases })} />
        </Cell>,
      );
      break;
    }
    case "computed_column": {
      const s = step as StepOf<"computed_column">;
      cells.push(cell("to", "clean.field.to", text("to", "clean.field.to", s.to, (to) => to.trim() && set({ to: to.trim() }))));
      cells.push(cell("formula", "clean.field.formula", text("formula", "clean.field.formula", s.formula, (formula) => formula.trim() && set({ formula })), "full", t("clean.field.why.formula")));
      break;
    }
    case "swap_values": {
      const s = step as StepOf<"swap_values">;
      cells.push(cell("when", "clean.field.when", text("when", "clean.field.when", s.when, (when) => when.trim() && set({ when })), "full", t("clean.field.why.formula")));
      break;
    }
    case "drop_duplicates":
      cells.push(cell("keys", "clean.field.keys", text("keys", "clean.field.keys", joined(step.keys), (value) => set({ keys: split(value).length ? split(value) : null }), t("clean.field.keysHint")), true, t("clean.field.keysHint")));
      cells.push(
        cell(
          "keep",
          "clean.field.keep",
          choice("clean.field.keep", step.keep, ["first", "last", "most_complete", "latest"], "clean.keep", (keep) =>
            set({ keep, order_by: keep === "latest" ? (step.order_by ?? columns[0] ?? null) : null }),
          ),
        ),
      );
      // N22: the latest row by a date (an update time), rows without one last
      if (step.keep === "latest")
        cells.push(cell("order_by", "clean.field.orderBy", columnSelect(step.order_by ?? "", (order_by) => set({ order_by }), "clean.field.orderBy"), false, t("clean.field.why.orderBy")));
      break;
    // C1: dates, rows and header, structure (C-52 to C-59)
    case "shift_date": {
      const s = step as StepOf<"shift_date">;
      cells.push(cell("amount", "clean.field.amount", text("amount", "clean.field.amount", String(s.amount), (value) => Number.isInteger(Number(value)) && set({ amount: Number(value) })), false, t("clean.field.why.amount")));
      cells.push(cell("unit", "clean.field.unit", choice("clean.field.unit", s.unit, ["years", "months", "days", "hours", "minutes", "seconds"], "clean.unit", (unit) => set({ unit }))));
      break;
    }
    case "convert_timezone": {
      const s = step as StepOf<"convert_timezone">;
      cells.push(cell("from_zone", "clean.field.fromZone", text("from_zone", "clean.field.fromZone", s.from_zone, (zone) => zone.trim() && set({ from_zone: zone.trim() }), "UTC"), false, t("clean.field.why.zone")));
      cells.push(cell("to_zone", "clean.field.toZone", text("to_zone", "clean.field.toZone", s.to_zone, (zone) => zone.trim() && set({ to_zone: zone.trim() }), "Europe/Paris"), false, t("clean.field.why.zone")));
      break;
    }
    case "keep_rows": {
      const s = step as StepOf<"keep_rows">;
      cells.push(cell("how", "clean.field.how", choice("clean.field.how", s.how, ["first", "last", "remove_first", "remove_last"], "clean.keepRows", (how) => set({ how }))));
      cells.push(cell("count", "clean.field.count", text("count", "clean.field.count", String(s.count), (value) => Number(value) >= 1 && set({ count: Math.round(Number(value)) }))));
      break;
    }
    case "promote_header": {
      // N47: the names may be on a later row, under a note (the rows above leave the table)
      const s = step as StepOf<"promote_header">;
      cells.push(cell("row", "clean.field.headerRow", text("row", "clean.field.headerRow", String(s.row), (value) => Number(value) >= 1 && Number(value) <= 100 && set({ row: Math.round(Number(value)) })), false, t("clean.field.why.headerRow")));
      break;
    }
    case "split_rows": {
      const s = step as StepOf<"split_rows">;
      cells.push(cell("delimiter", "clean.field.delimiter", text("delimiter", "clean.field.delimiter", s.delimiter, (delimiter) => delimiter && set({ delimiter }))));
      cells.push(cell("trim", "clean.field.trim", <Checkbox checked={s.trim} onChange={(event) => set({ trim: event.target.checked })} label={t("clean.field.trim")} className="h-8 text-[13px]" />));
      break;
    }
    case "unpivot": {
      const s = step as StepOf<"unpivot">;
      cells.push(cell("names_to", "clean.field.namesTo", text("names_to", "clean.field.namesTo", s.names_to, (to) => to.trim() && set({ names_to: to.trim() }))));
      cells.push(cell("values_to", "clean.field.valuesTo", text("values_to", "clean.field.valuesTo", s.values_to, (to) => to.trim() && set({ values_to: to.trim() }))));
      cells.push(cell("drop_empty", "clean.field.dropEmpty", <Checkbox checked={s.drop_empty} onChange={(event) => set({ drop_empty: event.target.checked })} label={t("clean.field.dropEmpty")} className="h-8 text-[13px]" />));
      break;
    }
    case "pivot": {
      const s = step as StepOf<"pivot">;
      cells.push(
        cell(
          "value_column",
          "clean.field.valueColumn",
          <Select aria-label={t("clean.field.valueColumn")} value={s.value_column ?? ""} onChange={(event) => set(event.target.value ? { value_column: event.target.value } : { value_column: null, aggregate: "count" })} className={field}>
            <option value="">{t("clean.field.rowsCounted")}</option>
            {columns.filter((name) => name !== s.column).map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </Select>,
        ),
      );
      if (s.value_column) cells.push(cell("aggregate", "clean.field.aggregate", choice("clean.field.aggregate", s.aggregate, AGGREGATES, "clean.aggregate", (aggregate) => set({ aggregate }))));
      cells.push(cell("headers", "clean.field.headers", text("headers", "clean.field.headers", joined(s.headers), (value) => set({ headers: split(value) }), t("clean.field.headersAll")), "full", t("clean.field.why.headers")));
      break;
    }
    case "group_by": {
      const s = step as StepOf<"group_by">;
      cells.push(cell("keys", "clean.field.groupKeys", text("keys", "clean.field.groupKeys", joined(s.keys), (value) => split(value).length && set({ keys: split(value) })), "full", t("clean.field.columnsHint")));
      cells.push(
        <Cell key="aggregations" label={t("clean.field.aggregations")} changed={differs("aggregations")} onReset={reset("aggregations")} wide="full">
          <Aggregations value={s.aggregations} columns={columns} onChange={(aggregations) => set({ aggregations })} />
        </Cell>,
      );
      break;
    }
    case "append_rows":
    case "merge_dataset":
      cells.push(<OtherDataset key="dataset" step={step as StepOf<"append_rows"> | StepOf<"merge_dataset">} onChange={onChange} />);
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

/** C-45: the cases of a conditional step; the first a row meets gives the value. A case tests
 *  conditions on values, a formula, or both; it sets a value, or a formula's result. */
function Cases({ value, columns, onChange }: { value: Case[]; columns: string[]; onChange: (cases: Case[]) => void }) {
  const update = (index: number, patch: Partial<Case>) => onChange(value.map((item, i) => (i === index ? { ...item, ...patch } : item)));
  return (
    <div className="space-y-2">
      {value.slice(0, 50).map((item, index) => {
        const formula = item.formula !== null;
        return (
          <fieldset key={index} className="min-w-0 space-y-1.5 rounded-md border border-line p-2">
            <legend className="flex w-full items-center justify-between gap-2 px-1 text-[11px] text-ink-3">
              <span>{t("clean.field.caseN", { n: String(index + 1) })}</span>
              <button
                type="button"
                aria-label={t("clean.field.removeCase")}
                title={t("clean.field.removeCase")}
                disabled={value.length < 2}
                onClick={() => onChange(value.filter((_, i) => i !== index))}
                className="inline-flex size-7 items-center justify-center rounded-md text-ink-3 hover:bg-surface-2 hover:text-ink-1 disabled:opacity-40"
              >
                <X size={13} aria-hidden />
              </button>
            </legend>
            {(
              <Conditions value={item.conditions} columns={columns} allowNone={item.when !== null} onChange={(conditions) => update(index, { conditions })} />
            )}
            <div className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,12rem),1fr))] gap-1.5">
              <CommitInput key={`when:${item.when ?? ""}`} label={t("clean.field.when")} value={item.when ?? ""} placeholder={t("clean.field.whenHint")} onCommit={(when) => update(index, { when: when.trim() || null })} />
              <CommitInput
                key={`set:${formula ? item.formula : item.value}`}
                label={t("clean.field.setTo")}
                value={(formula ? item.formula : item.value) ?? ""}
                placeholder={t("clean.empty")}
                onCommit={(text) => update(index, formula ? { formula: text.trim() || null } : { value: text === "" ? null : text })}
              />
              <Checkbox
                checked={formula}
                onChange={(event) => update(index, event.target.checked ? { formula: item.value || "0", value: null } : { value: item.formula ?? "", formula: null })}
                label={t("clean.field.isFormula")}
                className="h-8 text-[13px]"
              />
            </div>
          </fieldset>
        );
      })}
      {value.length > 50 && <p className="text-xs text-ink-3">{t("clean.field.moreCases", { count: String(value.length - 50) })}</p>}
      <button
        type="button"
        disabled={value.length >= 500}
        onClick={() => onChange([...value, newCase(columns[0] ?? "")])}
        className="inline-flex items-center gap-1 text-xs text-accent hover:underline"
      >
        <Plus size={12} aria-hidden />
        {t("clean.field.addCase")}
      </button>
    </div>
  );
}

const AGGREGATES: Aggregation["fn"][] = ["sum", "mean", "median", "min", "max", "count", "count_distinct", "first", "last"];

/** C-56: the columns of a group, one line each: what, of which column, named how. */
function Aggregations({ value, columns, onChange }: { value: Aggregation[]; columns: string[]; onChange: (aggregations: Aggregation[]) => void }) {
  const update = (index: number, patch: Partial<Aggregation>) => onChange(value.map((item, i) => (i === index ? { ...item, ...patch } : item)));
  return (
    <div className="space-y-1.5">
      {value.map((item, index) => (
        <div key={index} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)_auto] items-center gap-1.5">
          <Select
            aria-label={t("clean.field.aggregate")}
            value={item.fn}
            onChange={(event) => {
              const fn = event.target.value as Aggregation["fn"];
              update(index, fn === "count" ? { fn } : { fn, column: item.column ?? columns[0] ?? null });
            }}
            className={field}
          >
            {AGGREGATES.map((fn) => (
              <option key={fn} value={fn}>
                {t(`clean.aggregate.${fn}` as MessageKey)}
              </option>
            ))}
          </Select>
          <Select aria-label={t("clean.field.column")} value={item.column ?? ""} onChange={(event) => update(index, { column: event.target.value || null })} className={field}>
            {item.fn === "count" && <option value="">{t("clean.field.rowsCounted")}</option>}
            {[...new Set([...(item.column ? [item.column] : []), ...columns])].map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </Select>
          <CommitInput key={item.to} label={t("clean.field.to")} value={item.to} onCommit={(to) => to.trim() && update(index, { to: to.trim() })} />
          <button
            type="button"
            aria-label={t("clean.field.removeRow")}
            title={t("clean.field.removeRow")}
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
        disabled={value.length >= 50}
        onClick={() => onChange([...value, { column: columns[0] ?? null, fn: "sum", to: `total ${value.length + 1}` }])}
        className="inline-flex items-center gap-1 text-xs text-accent hover:underline"
      >
        <Plus size={12} aria-hidden />
        {t("clean.field.addAggregation")}
      </button>
    </div>
  );
}

const OPERATORS: RowCondition["op"][] =["eq", "ne", "lt", "le", "gt", "ge", "contains", "empty", "not_empty", "in"];

export function Conditions({ value, columns, allowNone = false, onChange }: { value: RowCondition[]; columns: string[]; allowNone?: boolean; onChange: (conditions: RowCondition[]) => void }) {
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
            disabled={value.length < 2 && !allowNone}
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
