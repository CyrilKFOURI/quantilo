import { ChevronDown, Sparkles } from "lucide-react";
import { useId, useState } from "react";

import type { KpiDefinition, KpiSuggestions } from "../../api/client";
import { t } from "../../lib/i18n";
import { reasonTexts, type ReasonDetail } from "../../lib/i18n/reasons";
import { TipBox } from "../../ui/InfoTip";
import { Popover } from "../../ui/Popover";
import { kpiName } from "./names";

/** Chips shown in the row; the others wait under "More (n)" (048). */
export const VISIBLE_SUGGESTIONS = 6;

type Suggested = KpiSuggestions["kpis"][number];
type Hierarchy = KpiSuggestions["hierarchies"][number];

/** "R-SUG-12 · Sales ÷ distinct Order ID: …" → its rule and its sentence. */
export function splitReason(reason: string): { rule: string | null; text: string } {
  const at = reason.indexOf(" · ");
  if (at < 0 || !reason.startsWith("R-")) return { rule: null, text: reason };
  return { rule: reason.slice(0, at), text: reason.slice(at + 3) };
}

/** A suggested KPI's reasons in the interface language (M02). */
export function kpiReasons(kpi: { reasons: string[]; reason_details?: ReasonDetail[] | null }): string[] {
  return reasonTexts(kpi.reasons, kpi.reason_details);
}

function Reasons({ reasons }: { reasons: string[] }) {
  return (
    <ul className="space-y-1">
      {reasons.map((reason) => {
        const { rule, text } = splitReason(reason);
        return (
          <li key={reason} className="flex items-baseline gap-2">
            <span className="min-w-0 flex-1">{text}</span>
            {rule && <span className="shrink-0 font-mono text-[10px] text-ink-3">{rule}</span>}
          </li>
        );
      })}
    </ul>
  );
}

/** A suggestion as today's chip; its reasons in a "Why this KPI" tip on hover, focus or tap. */
function Chip({ kpi, active, onPick }: { kpi: Suggested; active: boolean; onPick: () => void }) {
  const id = useId();
  return (
    <span className="group relative inline-flex">
      <button
        type="button"
        aria-pressed={active}
        aria-describedby={id}
        onClick={onPick}
        className={`border-b-2 px-0.5 pb-0.5 text-[13px] ${active ? "border-accent font-medium text-ink-1" : "border-transparent text-ink-2 hover:border-line hover:text-ink-1"}`}
      >
        {kpiName(kpi)}
      </button>
      <TipBox id={id} placement="below">
        <span className="mb-1 block font-medium text-ink-1">{t("kpi.whyKpi", { score: String(kpi.score) })}</span>
        <Reasons reasons={kpiReasons(kpi)} />
      </TipBox>
    </span>
  );
}

/** The suggestions beyond the first six, each with its first reason. */
function More({ kpis, isActive, onPick }: { kpis: Suggested[]; isActive: (kpi: Suggested) => boolean; onPick: (kpi: Suggested) => void }) {
  const [open, setOpen] = useState(false);
  const holdsActive = kpis.some(isActive);
  return (
    <Popover
      open={open}
      onOpenChange={setOpen}
      label={t("kpi.moreSuggestions")}
      align="start"
      pinBelowLaptop
      trigger={
        <button
          type="button"
          aria-expanded={open}
          onClick={() => setOpen(!open)}
          className={`inline-flex items-center gap-0.5 border-b-2 px-0.5 pb-0.5 text-[13px] ${holdsActive ? "border-accent font-medium text-ink-1" : "border-transparent text-ink-2 hover:border-line hover:text-ink-1"}`}
        >
          {t("kpi.more", { count: String(kpis.length) })}
          <ChevronDown size={12} aria-hidden />
        </button>
      }
    >
      <ul className="max-h-80 overflow-auto p-1.5">
        {kpis.map((kpi) => (
          <li key={kpi.name}>
            <button
              type="button"
              aria-pressed={isActive(kpi)}
              onClick={() => {
                setOpen(false);
                onPick(kpi);
              }}
              className="w-full rounded-md px-2 py-1.5 text-left hover:bg-surface-2"
            >
              <span className={`block text-[13px] text-ink-1 ${isActive(kpi) ? "font-medium" : ""}`}>{kpiName(kpi)}</span>
              {kpi.reasons[0] && <span className="block text-xs text-ink-3">{splitReason(kpiReasons(kpi)[0] ?? "").text}</span>}
            </button>
          </li>
        ))}
      </ul>
    </Popover>
  );
}

/** "Try:" and the ranked suggestions: six chips, then "More (n)". */
export function SuggestionChips({ kpis, isActive, onPick }: { kpis: Suggested[]; isActive: (definition: KpiDefinition) => boolean; onPick: (definition: KpiDefinition) => void }) {
  const shown = kpis.slice(0, VISIBLE_SUGGESTIONS);
  const rest = kpis.slice(VISIBLE_SUGGESTIONS);
  return (
    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1" role="group" aria-label={t("kpi.suggested")}>
      <span className="inline-flex items-center gap-1 pr-1 text-xs text-ink-3">
        <Sparkles size={12} aria-hidden /> {t("kpi.suggested")}
      </span>
      {shown.map((kpi) => (
        <Chip key={kpi.name} kpi={kpi} active={isActive(kpi.definition)} onPick={() => onPick(kpi.definition)} />
      ))}
      {rest.length > 0 && <More kpis={rest} isActive={(kpi) => isActive(kpi.definition)} onPick={(kpi) => onPick(kpi.definition)} />}
    </div>
  );
}

/** "+ Region › State" under the chart: both splits at once, with the rule's reason (R-SUG-18). */
export function HierarchyChip({ hierarchy, onPick }: { hierarchy: Hierarchy; onPick: () => void }) {
  const id = useId();
  return (
    <span className="group relative inline-flex">
      <button type="button" aria-describedby={id} onClick={onPick} className="px-0.5 text-[13px] text-accent hover:underline">
        + {t("kpi.hierarchy", { parent: hierarchy.parent, child: hierarchy.child })}
      </button>
      <TipBox id={id}>{splitReason(hierarchy.reason).text}</TipBox>
    </span>
  );
}
