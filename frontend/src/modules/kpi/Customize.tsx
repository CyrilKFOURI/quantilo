import { ChevronDown, RotateCcw, SlidersHorizontal } from "lucide-react";
import { useId, type ReactNode } from "react";

import type { KpiDefinition, KpiResult } from "../../api/client";
import { t, type MessageKey } from "../../lib/i18n";
import { Checkbox, Select, TextInput } from "../../ui/fields";
import { chartFor, chartName } from "./KpiCharts";
import {
  customised,
  formatOf,
  RATIO,
  resetView,
  viewOf,
  withFormat,
  withLabel,
  withTable,
  withView,
  type ShowAs,
} from "./definition";
import { lineAllowed } from "./figures";
import { ADDITIVE_MODES, effectiveMode, TIME_MODES } from "./pivot";

const SHOW_AS: ShowAs[] = ["value", "share_total", "share_row", "share_column", "change", "change_pct", "change_yoy", "running"];
const BARS = new Set(["bars", "stacked_bars", "grouped_bars"]);
const field = "h-8 w-full min-w-0 text-[13px]";

/** One setting in the grid (045): a short label (Auto's choice and why in its tooltip), its
 *  control, and a reset icon once it differs from Auto. */
function Cell({
  label,
  why,
  changed,
  onReset,
  wide = false,
  children,
}: {
  label: string;
  why?: string | null;
  changed: boolean;
  onReset: () => void;
  /** Two columns: a control with two choices side by side. */
  wide?: boolean;
  children: ReactNode;
}) {
  return (
    <div className={`min-w-0 space-y-1 ${wide ? "col-span-2" : ""}`}>
      <div className="flex min-w-0 items-center gap-1 text-[11px] text-ink-2">
        {changed && <span className="size-1.5 shrink-0 rounded-full bg-accent" aria-label={t("kpi.customised")} />}
        <span className="truncate" title={why ?? undefined}>
          {label}
        </span>
        {changed && (
          <button
            type="button"
            onClick={onReset}
            aria-label={`${t("kpi.resetAuto")}: ${label}`}
            title={t("kpi.resetAuto")}
            className="ml-auto shrink-0 text-ink-3 hover:text-accent"
          >
            <RotateCcw size={11} aria-hidden />
          </button>
        )}
      </div>
      {children}
    </div>
  );
}

function Switch3({
  label,
  value,
  onChange,
}: {
  label: string;
  value: "auto" | "on" | "off";
  onChange: (value: "auto" | "on" | "off") => void;
}) {
  return (
    <Select aria-label={label} value={value} onChange={(e) => onChange(e.target.value as "auto" | "on" | "off")} className={field}>
      {(["auto", "on", "off"] as const).map((v) => (
        <option key={v} value={v}>
          {t(`kpi.switch.${v}` as MessageKey)}
        </option>
      ))}
    </Select>
  );
}

/** The KPI's display settings (045): one block above the tiles, chart and table, opened from its
 *  own header line; the settings sit three per row (six on a laptop) so they take little room. */
export function Customize({
  definition,
  result,
  draft,
  open,
  onToggle,
  onView,
  onData,
}: {
  definition: KpiDefinition; // applied: display changes are drawn at once
  result: KpiResult;
  draft: KpiDefinition; // what the builder holds: data changes wait for Calculate
  open: boolean;
  onToggle: () => void;
  onView: (next: KpiDefinition) => void;
  onData: (next: KpiDefinition) => void;
}) {
  const id = useId();
  const view = viewOf(definition);
  const main = result.measures[0];
  const time = result.resolved.time;
  const grouped = result.splits.length > 0;
  const autoChart = main ? chartFor(main, "auto") : "none";
  const drawn = main ? chartFor(main, view.chart) : "none";
  const bars = Boolean(time) && BARS.has(drawn);
  const series = grouped ? (result.splits[result.splits.length === 2 ? 1 : 0]?.levels.length ?? 1) : 1;
  const mode = main ? effectiveMode(view.show_as, main.additive, time) : view.show_as;
  const canLine = lineAllowed(mode, series);
  const reasonFor = (m: ShowAs): string | null => {
    if (TIME_MODES.has(m) && !time) return t("kpi.reason.needsTime");
    if (ADDITIVE_MODES.has(m) && main && !main.additive) return t("kpi.reason.notAdditive", { measure: main.label });
    if ((m === "share_row" || m === "share_column") && !grouped) return t("kpi.reason.needsSplit");
    return null;
  };
  const layoutOptions = result.resolved.layout.options;
  const count = customised(definition);
  const summary = [main && chartName(drawn), t(`kpi.showAs.${view.show_as}` as MessageKey), view.line && t("kpi.line")]
    .filter(Boolean)
    .join(" · ");

  return (
    <section aria-label={t("kpi.customize")} className="rounded-lg border border-line bg-surface-1">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={onToggle}
        className="flex w-full items-center gap-2.5 px-3 py-2 text-left"
      >
        <SlidersHorizontal size={15} strokeWidth={1.5} className="shrink-0 text-ink-3" aria-hidden />
        <span className="shrink-0 text-[13px] font-medium text-ink-1">{t("kpi.customize")}</span>
        {count > 0 && <span className="shrink-0 rounded-full bg-accent px-1.5 text-[11px] text-accent-ink">{count}</span>}
        <span className="min-w-0 flex-1 truncate text-xs text-ink-3">{summary}</span>
        <span className="shrink-0 text-xs text-accent">{open ? t("params.close") : t("params.edit")}</span>
        <ChevronDown size={15} strokeWidth={1.5} className={`shrink-0 text-ink-3 transition-transform ${open ? "rotate-180" : ""}`} aria-hidden />
      </button>

      {open && (
        <div id={id} className="space-y-3 border-t border-line px-3 py-3">
          {main && main.chart.options.length > 1 && (
            <div className="space-y-1">
              <div className="flex items-center gap-1 text-[11px] text-ink-2">
                {view.chart !== "auto" && <span className="size-1.5 rounded-full bg-accent" aria-label={t("kpi.customised")} />}
                <span title={`${main.chart.rule} · ${main.chart.reason}`}>
                  {t("kpi.chartType")} · {t("kpi.auto", { choice: chartName(autoChart) })}
                </span>
              </div>
              <div className="flex flex-wrap gap-1" role="radiogroup" aria-label={t("kpi.chartType")}>
                {main.chart.options.map((option) => {
                  const selected = drawn === option.value;
                  return (
                    <button
                      key={option.value}
                      type="button"
                      role="radio"
                      aria-checked={selected}
                      disabled={!option.allowed}
                      title={option.reason ?? undefined}
                      onClick={() => onView(withView(definition, { chart: option.value === autoChart ? "auto" : (option.value as never) }))}
                      className={`rounded-md border px-2 py-1 text-xs ${selected ? "border-accent bg-accent-soft text-ink-1" : "border-line text-ink-2 hover:text-ink-1"} disabled:cursor-not-allowed disabled:border-dashed disabled:text-ink-3`}
                    >
                      {chartName(option.value)}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          <div className="grid grid-cols-3 gap-x-2 gap-y-2.5 lg:grid-cols-6">
            <Cell
              label={t("kpi.valueLabels")}
              why={t("kpi.valueLabels.auto")}
              changed={view.labels !== "auto"}
              onReset={() => onView(withView(definition, { labels: "auto" }))}
            >
              <Switch3 label={t("kpi.valueLabels")} value={view.labels} onChange={(labels) => onView(withView(definition, { labels }))} />
            </Cell>
            <Cell label={t("kpi.showAs")} changed={view.show_as !== "value"} onReset={() => onView(withView(definition, { show_as: "value" }))}>
              <Select aria-label={t("kpi.showAs")} value={view.show_as} onChange={(e) => onView(withView(definition, { show_as: e.target.value as ShowAs }))} className={field}>
                {SHOW_AS.map((m) => {
                  const reason = reasonFor(m);
                  return (
                    <option key={m} value={m} disabled={Boolean(reason)} title={reason ?? undefined}>
                      {t(`kpi.showAs.${m}` as MessageKey)}
                      {reason ? ` — ${reason}` : ""}
                    </option>
                  );
                })}
              </Select>
            </Cell>
            {bars && (
              <Cell
                label={t("kpi.line")}
                why={canLine ? t("kpi.lineWhy") : t("kpi.lineNotAllowed")}
                changed={view.line}
                onReset={() => onView(withView(definition, { line: false }))}
              >
                <Checkbox
                  label={series > 1 ? t("kpi.lineTotal") : t("kpi.lineOn")}
                  checked={view.line && canLine}
                  disabled={!canLine}
                  title={canLine ? undefined : t("kpi.lineNotAllowed")}
                  onChange={(e) => onView(withView(definition, { line: e.target.checked }))}
                  className="h-8 text-[13px]"
                />
              </Cell>
            )}
            {bars && (
              <Cell label={t("kpi.spacing")} changed={view.spacing !== "normal"} onReset={() => onView(withView(definition, { spacing: "normal" }))}>
                <Select
                  aria-label={t("kpi.spacing")}
                  value={view.spacing}
                  onChange={(e) => onView(withView(definition, { spacing: e.target.value as "narrow" | "normal" | "wide" }))}
                  className={field}
                >
                  {(["narrow", "normal", "wide"] as const).map((v) => (
                    <option key={v} value={v}>
                      {t(`kpi.spacing.${v}`)}
                    </option>
                  ))}
                </Select>
              </Cell>
            )}
            {main && time && grouped && (
              <Cell label={t("kpi.share")} why={main.share.reason} changed={view.share} onReset={() => onView(withView(definition, { share: false }))}>
                <Checkbox
                  label={t("kpi.shareOn")}
                  checked={view.share}
                  disabled={!main.share.allowed}
                  title={main.share.reason ?? t("kpi.shareLabel")}
                  onChange={(e) => onView(withView(definition, { share: e.target.checked }))}
                  className="h-8 text-[13px]"
                />
              </Cell>
            )}
            {layoutOptions.length > 0 && (
              <Cell
                label={t("kpi.tableLayout")}
                why={`${t("kpi.auto", { choice: t(`kpi.layout.${result.resolved.layout.value}` as MessageKey) })} — ${result.resolved.layout.reason}`}
                changed={view.table.layout !== "auto"}
                onReset={() => onView(withTable(definition, { layout: "auto" }))}
              >
                <Select aria-label={t("kpi.tableLayout")} value={view.table.layout} onChange={(e) => onView(withTable(definition, { layout: e.target.value as never }))} className={field}>
                  <option value="auto">{t("kpi.autoPlain")}</option>
                  {layoutOptions.map((o) => (
                    <option key={o.value} value={o.value}>
                      {t(`kpi.layout.${o.value}` as MessageKey)}
                    </option>
                  ))}
                </Select>
              </Cell>
            )}
            <Cell label={t("kpi.tableTotals")} changed={view.table.totals !== "auto"} onReset={() => onView(withTable(definition, { totals: "auto" }))}>
              <Switch3 label={t("kpi.tableTotals")} value={view.table.totals} onChange={(totals) => onView(withTable(definition, { totals }))} />
            </Cell>
            <Cell
              label={t("kpi.tableShading")}
              why={result.resolved.shading.reason}
              changed={view.table.shading !== "auto"}
              onReset={() => onView(withTable(definition, { shading: "auto" }))}
            >
              <Switch3 label={t("kpi.tableShading")} value={view.table.shading} onChange={(shading) => onView(withTable(definition, { shading }))} />
            </Cell>
            {time && (
              <Cell wide label={t("kpi.compare")} changed={view.table.compare.length > 0} onReset={() => onView(withTable(definition, { compare: [] }))}>
                <div className="flex min-h-8 flex-wrap items-center gap-x-3 gap-y-1 text-[13px]">
                  {(["previous", "last_year"] as const).map((kind) => (
                    <Checkbox
                      key={kind}
                      label={t(`kpi.compare.${kind}.short`)}
                      title={t(`kpi.compare.${kind}`)}
                      checked={view.table.compare.includes(kind)}
                      onChange={(e) =>
                        onView(withTable(definition, { compare: e.target.checked ? [...view.table.compare, kind] : view.table.compare.filter((k) => k !== kind) }))
                      }
                    />
                  ))}
                </div>
              </Cell>
            )}
            {grouped && (
              <>
                <Cell label={t("kpi.top")} why={t("kpi.top.auto")} changed={(draft.top ?? 8) !== 8} onReset={() => onData({ ...draft, top: 8 })}>
                  <TextInput
                    type="number"
                    min={1}
                    max={20}
                    aria-label={t("kpi.top")}
                    value={draft.top ?? 8}
                    onChange={(e) => onData({ ...draft, top: Math.min(20, Math.max(1, Number(e.target.value) || 1)) })}
                    className={field}
                  />
                </Cell>
                <Cell label={t("kpi.other")} changed={(definition.other ?? "show") !== "show"} onReset={() => onView({ ...definition, other: "show" })}>
                  <Checkbox
                    label={t("kpi.hideOtherShort")}
                    title={t("kpi.hideOther")}
                    checked={definition.other === "hide"}
                    onChange={(e) => onView({ ...definition, other: e.target.checked ? "hide" : "show" })}
                    className="h-8 text-[13px]"
                  />
                </Cell>
                <Cell label={t("kpi.order")} changed={(draft.group_order ?? "value") !== "value"} onReset={() => onData({ ...draft, group_order: "value" })}>
                  <Select aria-label={t("kpi.order")} value={draft.group_order ?? "value"} onChange={(e) => onData({ ...draft, group_order: e.target.value as "value" | "alpha" })} className={field}>
                    <option value="value">{t("kpi.order.value")}</option>
                    <option value="alpha">{t("kpi.order.alpha")}</option>
                  </Select>
                </Cell>
              </>
            )}
          </div>

          {result.measures.map((measure) => {
            const custom = formatOf(definition, measure.id);
            const label = measure.id === RATIO ? definition.ratio?.label : definition.measures.find((m) => m.id === measure.id)?.label;
            const changed = Boolean(label) || (custom.decimals ?? "auto") !== "auto" || (custom.compact ?? "auto") !== "auto" || Boolean(custom.prefix || custom.suffix);
            return (
              <div key={measure.id} className="space-y-1">
                <div className="flex items-center gap-1 text-[11px] text-ink-2">
                  {changed && <span className="size-1.5 rounded-full bg-accent" aria-label={t("kpi.customised")} />}
                  <span className="truncate" title={t("kpi.format.auto", { reason: measure.format.reason })}>
                    {t("kpi.formatOf", { measure: measure.label })}
                  </span>
                  {changed && (
                    <button
                      type="button"
                      onClick={() => onView(withLabel(withFormat(definition, measure.id, { decimals: "auto", compact: "auto", prefix: "", suffix: "" }), measure.id, null))}
                      aria-label={`${t("kpi.resetAuto")}: ${measure.label}`}
                      title={t("kpi.resetAuto")}
                      className="ml-auto shrink-0 text-ink-3 hover:text-accent"
                    >
                      <RotateCcw size={11} aria-hidden />
                    </button>
                  )}
                </div>
                <div className="grid grid-cols-3 gap-2 lg:grid-cols-6">
                  <TextInput
                    aria-label={t("kpi.format.name")}
                    placeholder={t("kpi.format.name")}
                    value={label ?? ""}
                    maxLength={80}
                    onChange={(e) => onView(withLabel(definition, measure.id, e.target.value || null))}
                    className={`${field} col-span-3 lg:col-span-2`}
                  />
                  <Select
                    aria-label={t("kpi.format.decimals")}
                    value={String(custom.decimals ?? "auto")}
                    onChange={(e) => onView(withFormat(definition, measure.id, { decimals: e.target.value === "auto" ? "auto" : Number(e.target.value) }))}
                    className={field}
                  >
                    <option value="auto">{t("kpi.format.decimalsAuto", { count: String(measure.format.decimals) })}</option>
                    {[0, 1, 2, 3, 4].map((d) => (
                      <option key={d} value={d}>
                        {t("kpi.format.decimalsN", { count: String(d) })}
                      </option>
                    ))}
                  </Select>
                  <Select
                    aria-label={t("kpi.format.compact")}
                    value={custom.compact ?? "auto"}
                    onChange={(e) => onView(withFormat(definition, measure.id, { compact: e.target.value as "auto" | "on" | "off" }))}
                    className={field}
                  >
                    <option value="auto">{t("kpi.format.compactAuto")}</option>
                    <option value="on">{t("kpi.format.compactOn")}</option>
                    <option value="off">{t("kpi.format.compactOff")}</option>
                  </Select>
                  <TextInput aria-label={t("kpi.format.prefix")} placeholder={t("kpi.format.prefix")} maxLength={8} value={custom.prefix ?? ""} onChange={(e) => onView(withFormat(definition, measure.id, { prefix: e.target.value }))} className={field} />
                  <TextInput aria-label={t("kpi.format.suffix")} placeholder={t("kpi.format.suffix")} maxLength={8} value={custom.suffix ?? ""} onChange={(e) => onView(withFormat(definition, measure.id, { suffix: e.target.value }))} className={field} />
                </div>
              </div>
            );
          })}

          <div className="flex flex-wrap items-center justify-between gap-2 border-t border-line pt-2">
            <p className="text-xs text-ink-3">{grouped ? t("kpi.groupsNeedCalculate") : ""}</p>
            <button
              type="button"
              onClick={() => onView(resetView(definition))}
              className="inline-flex h-7 items-center gap-1 rounded-md px-2 text-xs text-ink-2 hover:bg-surface-2 hover:text-ink-1"
            >
              <RotateCcw size={12} aria-hidden /> {t("kpi.resetAll")}
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
