import { Check, LayoutGrid } from "lucide-react";
import { useState } from "react";

import type { KpiArchetype, KpiDefinition, KpiSuggestions } from "../../api/client";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Checkbox } from "../../ui/fields";
import { InfoTip } from "../../ui/InfoTip";
import { splitReason } from "./SuggestionChips";

type Suggested = KpiSuggestions["kpis"][number];

/** The suggested KPIs of the pack, in its order (names the server gave, R-ARC-02). */
export function packOf(archetype: KpiArchetype, kpis: Suggested[]): Suggested[] {
  return archetype.kpis.flatMap((name) => kpis.filter((k) => k.name === name));
}

/** 061: "Looks like sales data · 5 KPIs". A guess from written rules on the columns: the user
 *  reviews the pack and confirms before anything is saved (R-ARC-01, R-ARC-02). */
export function ArchetypeBanner({
  archetype,
  kpis,
  saving,
  onSave,
}: {
  archetype: KpiArchetype;
  kpis: Suggested[];
  saving: boolean;
  onSave: (picked: { name: string; definition: KpiDefinition }[]) => Promise<void>;
}) {
  const pack = packOf(archetype, kpis);
  const [open, setOpen] = useState(false);
  const [hidden, setHidden] = useState(false);
  const [left, setLeft] = useState<Set<string>>(new Set());
  const [saved, setSaved] = useState<number | null>(null);
  if (hidden || pack.length === 0) return null;
  const picked = pack.filter((k) => !left.has(k.name));
  const why = archetype.reasons.map((r) => splitReason(r).text).join(" ");
  return (
    <div className="space-y-2 border-l-2 border-accent pl-3" role="region" aria-label={archetype.message}>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="inline-flex items-center gap-1.5 text-[13px] text-ink-1">
          <LayoutGrid size={13} strokeWidth={1.75} aria-hidden className="text-ink-2" />
          {archetype.message}
          <InfoTip text={why} label={t("kpi.archetype.why")} />
        </span>
        {saved !== null ? (
          <span className="inline-flex items-center gap-1 text-xs text-ink-2">
            <Check size={12} aria-hidden /> {t("kpi.archetype.saved", { count: String(saved) })}
          </span>
        ) : (
          <>
            <button type="button" aria-expanded={open} onClick={() => setOpen(!open)} className="text-[13px] text-accent hover:underline">
              {t("kpi.archetype.review")}
            </button>
            <button type="button" onClick={() => setHidden(true)} className="text-[13px] text-ink-2 hover:text-ink-1">
              {t("kpi.archetype.hide")}
            </button>
          </>
        )}
      </div>
      {open && saved === null && (
        <div className="space-y-2">
          <p className="text-xs text-ink-2">{t("kpi.archetype.confirm")}</p>
          <ul className="space-y-1">
            {pack.map((kpi) => (
              <li key={kpi.name} className="flex min-w-0 flex-wrap items-baseline gap-x-2">
                <Checkbox
                  label={kpi.name}
                  checked={!left.has(kpi.name)}
                  onChange={(e) => {
                    const next = new Set(left);
                    if (e.target.checked) next.delete(kpi.name);
                    else next.add(kpi.name);
                    setLeft(next);
                  }}
                  className="text-[13px]"
                />
                {kpi.reasons[0] && <span className="min-w-0 text-xs text-ink-3">{splitReason(kpi.reasons[0]).text}</span>}
              </li>
            ))}
          </ul>
          <Button
            variant="primary"
            disabled={picked.length === 0 || saving}
            onClick={() =>
              void onSave(picked.map((k) => ({ name: k.name, definition: k.definition }))).then(() => {
                setSaved(picked.length);
                setOpen(false);
              })
            }
          >
            {t("kpi.archetype.save", { count: String(picked.length) })}
          </Button>
        </div>
      )}
    </div>
  );
}
