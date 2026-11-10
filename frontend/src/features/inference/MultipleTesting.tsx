import type { MultipleTesting } from "../../api/client";
import { t } from "../../lib/i18n";
import { InfoTip } from "../../ui/InfoTip";

/** R-MT-06: significant before the correction, not after. */
export function noLongerHolds(p: number | null | undefined, adjusted: number | null | undefined, alpha = 0.05): boolean {
  return p !== null && p !== undefined && adjusted !== null && adjusted !== undefined && p < alpha && adjusted >= alpha;
}

/** The conclusion of a correction, in one sentence, with the reason for the method (051). */
export function MultipleTestingNote({ multiple, className = "" }: { multiple: MultipleTesting; className?: string }) {
  if (multiple.method === "none") return null;
  return (
    <p className={`flex items-start gap-1.5 text-[13px] text-ink-2 ${className}`}>
      <span>{multiple.sentence}</span>
      <InfoTip label={t("inference.why")} text={multiple.reason} />
    </p>
  );
}

/** A small mark beside an adjusted p-value that the correction pushed above α. */
export function NoLongerHolds() {
  return (
    <span className="ml-1.5 text-[11px] whitespace-nowrap text-warn" title={t("inference.noLongerHoldsHint")}>
      {t("inference.noLongerHolds")}
    </span>
  );
}
