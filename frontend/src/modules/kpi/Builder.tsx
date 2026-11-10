import { ChevronDown, Plus, Save, X } from "lucide-react";
import { useState, type ReactNode } from "react";

import type { Column, KpiDefinition, KpiResult } from "../../api/client";
import { t, type MessageKey } from "../../lib/i18n";
import { PHONE, useMedia } from "../../lib/useMedia";
import { Button } from "../../ui/Button";
import { Checkbox, Select, TextInput } from "../../ui/fields";
import { ParamPanel } from "../../ui/ParamPanel";
import { InfoTip } from "../../ui/InfoTip";
import { Popover } from "../../ui/Popover";
import {
  AGGS,
  columnsFor,
  dateColumns,
  defaultMeasure,
  DELAY_AGGS,
  GRAINS,
  isDelay,
  keyColumns,
  levelsOf,
  MAX_MEASURES,
  MAX_SPLITS,
  nextMeasureId,
  RATIO_KINDS,
  splitColumns,
  weightColumns,
  type Agg,
  type Measure,
  type RatioKind,
} from "./definition";

type Period = NonNullable<NonNullable<KpiDefinition["time"]>["period"]>;
const PERIOD_KINDS: Period["kind"][] = ["all", "last", "ytd", "range"];

const small = "h-7 text-xs";

/** One group of the bar: a short label, then its controls. */
function Group({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5 max-sm:w-full" role="group" aria-label={label}>
      <span className="text-ink-3 max-sm:basis-full">{label}</span>
      {children}
    </div>
  );
}

const Divider = () => <span className="hidden h-5 w-px bg-line sm:block" aria-hidden />;

function Add({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & { children: ReactNode }) {
  return (
    <button
      type="button"
      {...props}
      className="inline-flex h-7 items-center gap-0.5 rounded-md px-1.5 text-accent hover:bg-accent-soft disabled:text-ink-3 disabled:hover:bg-transparent"
    >
      <Plus size={12} aria-hidden />
      {children}
    </button>
  );
}

function Remove({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button type="button" aria-label={label} title={label} onClick={onClick} className="-ml-1 inline-flex size-6 items-center justify-center rounded text-ink-3 hover:text-ink-1">
      <X size={12} aria-hidden />
    </button>
  );
}

/** The values a "% of values" measure counts (053, R-MES-05): the column's most frequent values,
 *  those already chosen, and any other one typed in. At least one stays checked. */
function ValuesPicker({ measure, column, onChange }: { measure: Measure; column: Column | undefined; onChange: (values: string[]) => void }) {
  const [open, setOpen] = useState(false);
  const [extra, setExtra] = useState("");
  const chosen = measure.values ?? [];
  const levels = [...new Set([...levelsOf(column), ...chosen])];
  const toggle = (value: string, on: boolean) => {
    const next = on ? [...chosen, value] : chosen.filter((v) => v !== value);
    if (next.length > 0) onChange(next);
  };
  const add = () => {
    const value = extra.trim();
    if (value && !chosen.includes(value)) onChange([...chosen, value]);
    setExtra("");
  };
  return (
    <Popover
      open={open}
      onOpenChange={setOpen}
      label={t("kpi.values")}
      align="start"
      pinBelowLaptop
      trigger={
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          title={chosen.join(", ")}
          className={`${small} inline-flex max-w-40 items-center gap-1 rounded-md border border-line bg-surface-1 px-2 text-ink-1 hover:border-ink-3 pointer-coarse:min-h-8 pointer-coarse:text-base`}
        >
          <span className="truncate">{chosen.length === 1 ? chosen[0] : t("kpi.valuesCount", { count: String(chosen.length) })}</span>
          <ChevronDown size={12} aria-hidden className="shrink-0 text-ink-3" />
        </button>
      }
    >
      <div className="space-y-2 p-3 text-[13px]">
        <p className="text-xs text-ink-3">
          {measure.key ? t("kpi.valuesHelpKey", { column: measure.column ?? "", key: measure.key }) : t("kpi.valuesHelp", { column: measure.column ?? "" })}
        </p>
        <div className="grid max-h-56 grid-cols-2 gap-x-3 gap-y-1 overflow-auto">
          {levels.map((value) => (
            <Checkbox key={value} label={value} checked={chosen.includes(value)} onChange={(e) => toggle(value, e.target.checked)} className="min-h-7" />
          ))}
        </div>
        <div className="flex gap-1.5">
          <TextInput
            aria-label={t("kpi.valuesAdd")}
            placeholder={t("kpi.valuesAdd")}
            value={extra}
            maxLength={200}
            onChange={(e) => setExtra(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && add()}
            className="h-7 min-w-0 flex-1 text-xs"
          />
          <Button onClick={add} disabled={!extra.trim()} className="h-7 text-xs">
            {t("kpi.valuesAddButton")}
          </Button>
        </div>
      </div>
    </Popover>
  );
}

/** "+ Ratio" (053, P4): a ratio, a margin or a difference of the first two measures. */
function AddRatio({ onPick }: { onPick: (kind: RatioKind) => void }) {
  const [open, setOpen] = useState(false);
  return (
    <Popover
      open={open}
      onOpenChange={setOpen}
      label={t("kpi.addRatio")}
      align="end"
      pinBelowLaptop
      trigger={
        <Add onClick={() => setOpen(!open)} aria-expanded={open}>
          {t("kpi.addRatio")}
        </Add>
      }
    >
      <ul className="p-1.5 text-[13px]">
        {RATIO_KINDS.map((kind) => (
          <li key={kind}>
            <button
              type="button"
              onClick={() => {
                setOpen(false);
                onPick(kind);
              }}
              className="flex w-full flex-col items-start rounded-md px-2 py-1.5 text-left hover:bg-surface-2"
            >
              <span className="text-ink-1">{t(`kpi.ratioKind.${kind}`)}</span>
              <span className="text-xs text-ink-3">{t(`kpi.ratioKind.${kind}.help`)}</span>
            </button>
          </li>
        ))}
      </ul>
    </Popover>
  );
}

/** The KPI's settings in one compact bar, like the Time series page: what to measure, over
 *  which dates, split by what. Changes take effect with Calculate (spec 022, story 6). */
export function Builder({
  definition,
  columns,
  result,
  invalid,
  dirty,
  saving,
  saved,
  changed,
  onChange,
  onCalculate,
  onSave,
}: {
  definition: KpiDefinition;
  columns: Column[];
  result: KpiResult | undefined;
  invalid: string | null; // the field the server asked to fix
  dirty: boolean;
  saving: boolean;
  saved: boolean; // a saved KPI is open
  changed: boolean; // what is shown differs from what was saved
  onChange: (next: KpiDefinition) => void;
  onCalculate: () => void;
  onSave: () => void;
}) {
  const phone = useMedia(PHONE);
  const measures = definition.measures;
  const dates = dateColumns(columns);
  const splits = splitColumns(columns);
  const time = definition.time ?? null;
  const groups = definition.groups ?? [];
  const grain = result?.resolved.grain;
  const mark = (field: string) => (invalid === field ? "border-bad" : "");

  const setMeasure = (id: string, patch: Partial<Measure>) =>
    onChange({ ...definition, measures: measures.map((m) => (m.id === id ? { ...m, ...patch } : m)) });
  /** The fields an aggregate needs, kept when they still fit (053): values, weight. */
  const needs = (agg: Agg, column: string | null, measure: Measure): Partial<Measure> => {
    if (agg === "share") {
      const values = measure.agg === "share" && measure.column === column ? (measure.values ?? []) : [];
      const first = levelsOf(columns.find((c) => c.name === column))[0];
      return { values: values.length ? values : first ? [first] : [] };
    }
    if (agg === "weighted_mean") {
      const weights = weightColumns(columns, column);
      return { weight: weights.some((c) => c.name === measure.weight) ? measure.weight : (weights[0]?.name ?? null) };
    }
    return {};
  };
  const setAgg = (measure: Measure, agg: Agg) => {
    if (isDelay(measure)) return setMeasure(measure.id, { agg });
    const usable = columnsFor(agg, columns);
    const column = agg === "count" ? null : usable.some((c) => c.name === measure.column) ? measure.column : usable[0]?.name ?? null;
    setMeasure(measure.id, { agg, column: column ?? null, values: [], key: null, weight: null, ...needs(agg, column ?? null, measure) });
  };
  const setColumn = (measure: Measure, name: string) => {
    if (isDelay(measure)) return setMeasure(measure.id, { column: name });
    const column = columns.find((c) => c.name === name);
    // R-MES-01: a column that doesn't fit the aggregate takes its own default (a weighted average for a price, 053)
    const patch = column && !columnsFor(measure.agg, columns).some((c) => c.name === name) ? defaultMeasure(column, columns) : { agg: measure.agg };
    const agg = patch.agg ?? measure.agg;
    setMeasure(measure.id, { column: name, values: [], key: null, weight: null, ...needs(agg, name, { ...measure, ...patch }), ...patch });
  };
  const addDelay = () => {
    const [start, end] = dates;
    if (start && end) onChange({ ...definition, measures: [...measures, { id: nextMeasureId(measures), agg: "mean", column: start.name, until: end.name }] });
  };
  const addRatio = (kind: RatioKind) =>
    measures[0] && measures[1] && onChange({ ...definition, ratio: { numerator: measures[0].id, denominator: measures[1].id, kind } });
  const removeMeasure = (id: string) => {
    const ratio = definition.ratio && (definition.ratio.numerator === id || definition.ratio.denominator === id) ? null : definition.ratio;
    onChange({ ...definition, measures: measures.filter((m) => m.id !== id), ratio });
  };
  const setPeriod = (patch: Partial<Period>) =>
    time && onChange({ ...definition, time: { ...time, period: { ...(time.period ?? { kind: "all" }), ...patch } } });
  const period = time?.period ?? { kind: "all" as const };
  const measureName = (m: Measure) => {
    if (m.label) return m.label;
    if (m.agg === "count") return t("kpi.agg.count");
    if (m.until) return `${t(`kpi.aggShort.${m.agg}` as MessageKey)} ${m.column} → ${m.until}`;
    return `${t(`kpi.aggShort.${m.agg}` as MessageKey)} ${m.column}`;
  };
  const keys = keyColumns(columns);
  const kind: RatioKind = definition.ratio?.kind ?? "ratio";

  const bar = (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
      <Group label={t("kpi.measure")}>
        {measures.map((measure) => {
          const delay = isDelay(measure);
          const usable = delay ? dates : columnsFor(measure.agg, columns);
          const columnSelect = (field: "column" | "until" | "weight", value: string | null | undefined, onPick: (name: string) => void, options: Column[], label: string) => (
            <Select aria-label={label} value={value ?? ""} onChange={(e) => onPick(e.target.value)} className={`${small} max-w-36 ${mark(`measures.${measure.id}.${field}`)}`}>
              {!options.some((c) => c.name === value) && value && <option value={value}>{value}</option>}
              {options.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name}
                </option>
              ))}
            </Select>
          );
          return (
            <span key={measure.id} className="inline-flex flex-wrap items-center gap-1">
              {delay && <span className="text-ink-3">{t("kpi.delay")}</span>}
              <Select aria-label={t("kpi.aggregate")} value={measure.agg} onChange={(e) => setAgg(measure, e.target.value as Agg)} className={`${small} ${mark(`measures.${measure.id}.agg`)}`}>
                {(delay ? DELAY_AGGS : AGGS.filter((agg) => agg === "count" || columnsFor(agg, columns).length > 0)).map((agg) => (
                  <option key={agg} value={agg}>
                    {t(`kpi.aggShort.${agg}` as MessageKey)}
                  </option>
                ))}
              </Select>
              {measure.agg !== "count" && columnSelect("column", measure.column, (name) => setColumn(measure, name), usable, delay ? t("kpi.delayFrom") : t("kpi.column"))}
              {delay && (
                <>
                  <span className="text-ink-3" aria-hidden>
                    →
                  </span>
                  {columnSelect("until", measure.until, (name) => setMeasure(measure.id, { until: name }), dates, t("kpi.delayTo"))}
                  <span className="text-ink-3">{t("kpi.delayUnit")}</span>
                </>
              )}
              {measure.agg === "share" && (
                <>
                  <ValuesPicker measure={measure} column={columns.find((c) => c.name === measure.column)} onChange={(values) => setMeasure(measure.id, { values })} />
                  <span className="text-ink-3">{t("kpi.shareOf")}</span>
                  <Select
                    aria-label={t("kpi.shareKey")}
                    title={t("kpi.shareKey.help")}
                    value={measure.key ?? ""}
                    onChange={(e) => setMeasure(measure.id, { key: e.target.value || null })}
                    className={`${small} max-w-36 ${mark(`measures.${measure.id}.key`)}`}
                  >
                    <option value="">{t("kpi.shareKey.rows")}</option>
                    {keys
                      .filter((c) => c.name !== measure.column)
                      .map((c) => (
                        <option key={c.name} value={c.name}>
                          {c.name}
                        </option>
                      ))}
                  </Select>
                </>
              )}
              {measure.agg === "weighted_mean" && (
                <>
                  <span className="text-ink-3">{t("kpi.weightBy")}</span>
                  {columnSelect(
                    "weight",
                    measure.weight,
                    (name) => setMeasure(measure.id, { weight: name }),
                    weightColumns(columns, measure.column),
                    t("kpi.weight"),
                  )}
                  <InfoTip label={t("kpi.weight")} text={t("kpi.weight.help", { column: measure.column ?? "", weight: measure.weight ?? "" })} />
                </>
              )}
              {measures.length > 1 && <Remove label={t("kpi.removeMeasure")} onClick={() => removeMeasure(measure.id)} />}
            </span>
          );
        })}
        {definition.ratio && (
          <span className="inline-flex flex-wrap items-center gap-1">
            <Select
              aria-label={t("kpi.ratioKind")}
              title={t(`kpi.ratioKind.${kind}.help`)}
              value={kind}
              onChange={(e) => definition.ratio && onChange({ ...definition, ratio: { ...definition.ratio, kind: e.target.value as RatioKind } })}
              className={small}
            >
              {RATIO_KINDS.map((k) => (
                <option key={k} value={k}>
                  {t(`kpi.ratioKind.${k}`)}
                </option>
              ))}
            </Select>
            {(["numerator", "denominator"] as const).map((side, index) => (
              <span key={side} className="contents">
                {index === 1 && <span className="text-ink-3">{kind === "ratio" ? "÷" : "−"}</span>}
                <Select
                  aria-label={t(`kpi.ratio.${side}`)}
                  value={definition.ratio?.[side]}
                  onChange={(e) => definition.ratio && onChange({ ...definition, ratio: { ...definition.ratio, [side]: e.target.value } })}
                  className={`${small} max-w-36`}
                >
                  {measures.map((m) => (
                    <option key={m.id} value={m.id}>
                      {measureName(m)}
                    </option>
                  ))}
                </Select>
              </span>
            ))}
            {kind === "ratio" && (
              <Checkbox
                label="%"
                title={t("kpi.ratio.percent")}
                checked={Boolean(definition.ratio.percent)}
                onChange={(e) => definition.ratio && onChange({ ...definition, ratio: { ...definition.ratio, percent: e.target.checked } })}
                className="ml-1 text-ink-2"
              />
            )}
            <Remove label={t("kpi.removeRatio")} onClick={() => onChange({ ...definition, ratio: null })} />
          </span>
        )}
        <Add disabled={measures.length >= MAX_MEASURES} onClick={() => onChange({ ...definition, measures: [...measures, { id: nextMeasureId(measures), agg: "count" }] })}>
          {t("kpi.addMeasure")}
        </Add>
        {!definition.ratio && measures.length >= 2 && <AddRatio onPick={addRatio} />}
        {dates.length >= 2 && (
          <Add disabled={measures.length >= MAX_MEASURES} onClick={addDelay} title={t("kpi.addDelay.help")}>
            {t("kpi.addDelay")}
          </Add>
        )}
      </Group>

      <Divider />
      <Group label={t("kpi.time")}>
        {dates.length === 0 ? (
          <span className="text-ink-3">{t("kpi.noDate")}</span>
        ) : (
          <>
            <Select
              aria-label={t("kpi.dateColumn")}
              value={time?.column ?? ""}
              onChange={(e) => onChange({ ...definition, time: e.target.value ? { ...(time ?? {}), column: e.target.value } : null })}
              className={`${small} max-w-36 ${mark("time.column")}`}
            >
              <option value="">{t("kpi.noTime")}</option>
              {dates.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name}
                </option>
              ))}
            </Select>
            {time && (
              <>
                <Select
                  aria-label={t("kpi.grain")}
                  value={time.grain ?? "auto"}
                  title={grain?.reason}
                  onChange={(e) => onChange({ ...definition, time: { ...time, grain: e.target.value as typeof time.grain } })}
                  className={`${small} ${mark("time.grain")}`}
                >
                  <option value="auto">
                    {grain && (time.grain ?? "auto") === "auto" ? t("kpi.grainAuto", { grain: t(`kpi.grain.${grain.value}` as MessageKey) }) : t("kpi.autoPlain")}
                  </option>
                  {GRAINS.map((g) => {
                    const option = grain?.options.find((o) => o.value === g);
                    return (
                      <option key={g} value={g} disabled={option ? !option.allowed : false} title={option?.reason ?? undefined}>
                        {t(`kpi.grain.${g}` as MessageKey)}
                      </option>
                    );
                  })}
                </Select>
                <Select aria-label={t("kpi.period")} value={period.kind ?? "all"} onChange={(e) => setPeriod({ kind: e.target.value as Period["kind"] })} className={`${small} ${mark("time.period")}`}>
                  {PERIOD_KINDS.map((kind) => (
                    <option key={kind} value={kind}>
                      {t(`kpi.period.${kind}` as MessageKey)}
                    </option>
                  ))}
                </Select>
                {period.kind === "last" && (
                  <TextInput type="number" min={1} max={2000} aria-label={t("kpi.period.n")} value={period.n ?? 12} onChange={(e) => setPeriod({ n: Math.max(1, Number(e.target.value) || 1) })} className={`${small} w-14`} />
                )}
                {period.kind === "range" && (
                  <>
                    <TextInput type="date" aria-label={t("kpi.period.from")} value={period.from ?? ""} onChange={(e) => setPeriod({ from: e.target.value || null })} className={small} />
                    <span className="text-ink-3">→</span>
                    <TextInput type="date" aria-label={t("kpi.period.to")} value={period.to ?? ""} onChange={(e) => setPeriod({ to: e.target.value || null })} className={small} />
                  </>
                )}
                <Checkbox
                  label={t("kpi.pageFiltersShort")}
                  title={t("kpi.pageFilters.help")}
                  checked={time.page_filters ?? true}
                  onChange={(e) => onChange({ ...definition, time: { ...time, page_filters: e.target.checked } })}
                  className="ml-1 text-ink-2"
                />
              </>
            )}
          </>
        )}
      </Group>

      <Divider />
      <Group label={t("kpi.splitBy")}>
        {groups.map((group, index) => (
          <span key={index} className="inline-flex items-center gap-1">
            {index > 0 && <span className="text-ink-3">{t("kpi.then")}</span>}
            <Select
              aria-label={t("kpi.splitColumn")}
              value={group.column}
              onChange={(e) => onChange({ ...definition, groups: groups.map((g, i) => (i === index ? { column: e.target.value } : g)) })}
              className={`${small} max-w-36 ${mark(`groups.${index}.column`)}`}
            >
              {splits
                .filter((c) => c.name === group.column || !groups.some((g) => g.column === c.name))
                .map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name}
                  </option>
                ))}
            </Select>
            <Remove label={t("kpi.removeSplit")} onClick={() => onChange({ ...definition, groups: groups.filter((_, i) => i !== index) })} />
          </span>
        ))}
        {groups.length === 0 && <span className="text-ink-3">{t("kpi.none")}</span>}
        {groups.length < MAX_SPLITS && splits.some((c) => !groups.some((g) => g.column === c.name)) && (
          <Add
            onClick={() => {
              const free = splits.find((c) => !groups.some((g) => g.column === c.name));
              if (free) onChange({ ...definition, groups: [...groups, { column: free.name }] });
            }}
          >
            {t("kpi.split")}
          </Add>
        )}
      </Group>

      <div className="ml-auto flex items-center gap-2">
        {dirty && <span className="text-ink-1" aria-live="polite">{t("kpi.notApplied")}</span>}
        <Button variant="primary" onClick={onCalculate} disabled={!dirty} className="h-7 text-xs">
          {t("kpi.calculate")}
        </Button>
        <Button variant="ghost" icon={Save} onClick={onSave} disabled={saving || (saved && !changed && !dirty)} className="h-7 text-xs">
          {saved ? t("kpi.saveChanges") : t("kpi.save")}
        </Button>
      </div>
    </div>
  );
  if (phone) {
    // 041: on a phone the builder folds into one line, so the KPI keeps the screen.
    return (
      <div className="py-2">
        <ParamPanel
          label={t("kpi.builder")}
          layout="stack"
          summary={[
            measures.map(measureName).join(", "),
            time?.column,
            groups.length > 0 && groups.map((g) => g.column).join(" › "),
            dirty && t("kpi.notApplied"),
          ]}
        >
          {bar}
        </ParamPanel>
      </div>
    );
  }
  return (
    <div className="sticky top-0 z-10 -mx-(--gutter) border-y border-line bg-surface-0/95 px-(--gutter) py-2 backdrop-blur" role="group" aria-label={t("kpi.builder")}>
      {bar}
    </div>
  );
}
