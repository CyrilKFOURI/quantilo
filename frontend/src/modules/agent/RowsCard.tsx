import { TableProperties } from "lucide-react";
import { useState } from "react";

import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Checkbox } from "../../ui/fields";
import type { Mask, RowsRequest, RowsSharing } from "./api";

export const MASK_LABELS: Record<Mask, MessageKey> = {
  identifier: "agent.rows.mask.identifier",
  email: "agent.rows.mask.email",
  phone: "agent.rows.mask.phone",
  address: "agent.rows.mask.address",
  postal_code: "agent.rows.mask.postal_code",
};

/** What leaves this computer (R-AG-P1), said with the row-sharing level (R-AG-P2). */
export function privacyText(level: RowsSharing | undefined): string {
  if (!level || level === "never") return t("agent.privacy");
  return `${t("agent.rows.privacyShared")} ${t(level === "ask" ? "agent.rows.privacyAsk" : "agent.rows.privacyAllow")}`;
}

/** The card shown when the assistant asks to see rows (level "Ask each time"): what would be
 *  shared, the masked columns the user may let through for this request, Allow once / Dismiss. */
export function RowsCard({
  rows,
  onDecide,
}: {
  rows: RowsRequest;
  onDecide: (approved: boolean, unmask: string[]) => void;
}) {
  const [unmask, setUnmask] = useState<string[]>([]);
  const toggle = (name: string, checked: boolean) =>
    setUnmask((current) => (checked ? [...current, name] : current.filter((n) => n !== name)));
  return (
    <li
      className="space-y-2 rounded-lg border border-accent bg-accent-soft px-3 py-2.5 text-xs"
      role="group"
      aria-label={t("agent.rows.cardTitle")}
    >
      <p className="flex items-start gap-2 text-[13px] text-ink-1">
        <TableProperties size={15} className="mt-px shrink-0 text-accent" aria-hidden />
        <span>
          {t("agent.rows.cardText", { count: String(rows.count), columns: rows.columns.join(", ") })}
          {rows.filtered && ` ${t("agent.rows.cardFiltered")}`}
        </span>
      </p>
      {rows.masked.length > 0 && (
        <fieldset className="space-y-1">
          <legend className="text-ink-2">{t("agent.rows.cardMasked")}</legend>
          {rows.masked.map((column) => (
            <Checkbox
              key={column.name}
              className="flex text-ink-1"
              checked={unmask.includes(column.name)}
              onChange={(event) => toggle(column.name, event.target.checked)}
              label={t("agent.rows.shareUnmasked", { column: column.name, kind: t(MASK_LABELS[column.mask]) })}
            />
          ))}
        </fieldset>
      )}
      <div className="flex gap-2">
        <Button variant="primary" onClick={() => onDecide(true, unmask)}>
          {t("agent.rows.allowOnce")}
        </Button>
        <Button variant="ghost" onClick={() => onDecide(false, [])}>
          {t("agent.dismiss")}
        </Button>
      </div>
    </li>
  );
}
