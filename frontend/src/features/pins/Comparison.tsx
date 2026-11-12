import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import type { PinnedComparison } from "./api";

const pValue = (p: number | null) => (p === null ? "—" : p < 0.001 ? "< 0.001" : p.toFixed(3));

/** "What distinguishes pinned rows" (099): the conclusion first, then one line per column that
 *  sets them apart, with the rule, the gap and the corrected test; rules when there are some. */
export function Comparison({ found }: { found: PinnedComparison }) {
  return (
    <section aria-label={t("pins.compare.title")} className="space-y-3">
      <p className="text-[13px] font-medium text-ink-1">{found.reading}</p>
      {found.pinned >= 2 && (
        <p className="text-xs text-ink-3 tabular-nums">
          {t("pins.compare.scope", {
            pinned: formatCount(found.pinned),
            others: formatCount(found.others),
            columns: String(found.columns.length),
          })}
        </p>
      )}
      {found.warnings.map((warning) => (
        <p key={warning} className="text-xs text-warn">
          {warning}
        </p>
      ))}
      {found.traits.length > 0 && (
        <ul className="divide-y divide-line rounded-md border border-line">
          {found.traits.map((trait) => (
            <li key={`${trait.rule}:${trait.column}`} className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-0.5 px-3 py-2">
              <span className={`min-w-0 text-[13px] ${trait.chance ? "text-ink-2" : "text-ink-1"}`}>{trait.text}</span>
              <span className="shrink-0 text-xs text-ink-3 tabular-nums" title={trait.rule}>
                {t("pins.compare.gap", { gap: trait.effect.toFixed(1), p: pValue(trait.p_value) })}
                {trait.chance && ` · ${t("pins.compare.chance")}`}
              </span>
            </li>
          ))}
        </ul>
      )}
      {found.rules && (
        <div className="space-y-1">
          <h4 className="text-xs text-ink-3">{t("pins.compare.rules")}</h4>
          <ul className="space-y-0.5 text-[13px] text-ink-1">
            {found.rules.rules.map((rule) => (
              <li key={rule.text} className="font-mono text-xs">
                {rule.text}
                <span className="ml-2 font-sans text-ink-3 tabular-nums">
                  {t("pins.compare.ruleRows", { rows: formatCount(rule.rows), share: `${Math.round(rule.precision * 100)}` })}
                </span>
              </li>
            ))}
          </ul>
          <p className="text-xs text-ink-3">{found.rules.text}</p>
        </div>
      )}
    </section>
  );
}
