import { Plus, Save, X } from "lucide-react";
import type { ReactNode } from "react";

import type { Column, KpiDefinition, KpiResult } from "../../api/client";
import { t, type MessageKey } from "../../lib/i18n";
import { PHONE, useMedia } from "../../lib/useMedia";
import { Button } from "../../ui/Button";
import { Checkbox, Select, TextInput } from "../../ui/fields";
import { ParamPanel } from "../../ui/ParamPanel";
import {
  AGGS,
  columnsFor,
  dateColumns,
  defaultAgg,
  GRAINS,
  MAX_MEASURES,
  MAX_SPLITS,
  nextMeasureId,
  splitColumns,
  type Agg,
  type Measure,
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
  const setAgg = (measure: Measure, agg: Agg) => {
    const usable = columnsFor(agg, columns);
    const column = agg === "count" ? null : usable.some((c) => c.name === measure.column) ? measure.column : usable[0]?.name ?? null;
    setMeasure(measure.id, { agg, column });
  };
  const setColumn = (measure: Measure, name: string) => {
    const column = columns.find((c) => c.name === name);
    const agg = column && !columnsFor(measure.agg, columns).some((c) => c.name === name) ? defaultAgg(column) : measure.agg;
    setMeasure(measure.id, { column: name, agg });
  };
  const removeMeasure = (id: string) => {
    const ratio = definition.ratio && (definition.ratio.numerator === id || definition.ratio.denominator === id) ? null : definition.ratio;
    onChange({ ...definition, measures: measures.filter((m) => m.id !== id), ratio });
  };
  const setPeriod = (patch: Partial<Period>) =>
    time && onChange({ ...definition, time: { ...time, period: { ...(time.period ?? { kind: "all" }), ...patch } } });
  const period = time?.period ?? { kind: "all" as const };
  const measureName = (m: Measure) =>
    m.label || (m.agg === "count" ? t("kpi.agg.count") : `${t(`kpi.aggShort.${m.agg}` as MessageKey)} ${m.column}`);

  const bar = (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
      <Group label={t("kpi.measure")}>
        {measures.map((measure) => (
          <span key={measure.id} className="inline-flex items-center gap-1">
            <Select aria-label={t("kpi.aggregate")} value={measure.agg} onChange={(e) => setAgg(measure, e.target.value as Agg)} className={`${small} ${mark(`measures.${measure.id}.agg`)}`}>
              {AGGS.filter((agg) => agg === "count" || columnsFor(agg, columns).length > 0).map((agg) => (
                <option key={agg} value={agg}>
                  {t(`kpi.aggShort.${agg}` as MessageKey)}
                </option>
              ))}
            </Select>
            {measure.agg !== "count" && (
              <Select aria-label={t("kpi.column")} value={measure.column ?? ""} onChange={(e) => setColumn(measure, e.target.value)} className={`${small} max-w-36 ${mark(`measures.${measure.id}.column`)}`}>
                {!columnsFor(measure.agg, columns).some((c) => c.name === measure.column) && measure.column && (
                  <option value={measure.column}>{measure.column}</option>
                )}
                {columnsFor(measure.agg, columns).map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name}
                  </option>
                ))}
              </Select>
            )}
            {measures.length > 1 && <Remove label={t("kpi.removeMeasure")} onClick={() => removeMeasure(measure.id)} />}
          </span>
        ))}
        {definition.ratio && (
          <span className="inline-flex items-center gap-1">
            <span className="text-ink-3">{t("kpi.ratio")}</span>
            {(["numerator", "denominator"] as const).map((side, index) => (
              <span key={side} className="contents">
                {index === 1 && <span className="text-ink-3">÷</span>}
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
            <Checkbox
              label="%"
              title={t("kpi.ratio.percent")}
              checked={Boolean(definition.ratio.percent)}
              onChange={(e) => definition.ratio && onChange({ ...definition, ratio: { ...definition.ratio, percent: e.target.checked } })}
              className="ml-1 text-ink-2"
            />
            <Remove label={t("kpi.removeRatio")} onClick={() => onChange({ ...definition, ratio: null })} />
          </span>
        )}
        <Add disabled={measures.length >= MAX_MEASURES} onClick={() => onChange({ ...definition, measures: [...measures, { id: nextMeasureId(measures), agg: "count" }] })}>
          {t("kpi.addMeasure")}
        </Add>
        {!definition.ratio && measures.length >= 2 && (
          <Add onClick={() => measures[0] && measures[1] && onChange({ ...definition, ratio: { numerator: measures[0].id, denominator: measures[1].id } })}>
            {t("kpi.addRatio")}
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
