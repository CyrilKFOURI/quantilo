import { TableProperties } from "lucide-react";

import { t, type MessageKey } from "../../lib/i18n";
import { Checkbox, ErrorNote, Field, Select } from "../../ui/fields";
import type { RowsSharing as Level } from "./api";
import { MASK_LABELS } from "./RowsCard";
import { usePersonalColumns, useSetUnmasked, useUpdateAgentSettings } from "./useAgent";

const LEVELS: { value: Level; label: MessageKey; hint: MessageKey }[] = [
  { value: "never", label: "agent.rows.never", hint: "agent.rows.hintNever" },
  { value: "ask", label: "agent.rows.ask", hint: "agent.rows.hintAsk" },
  { value: "allow", label: "agent.rows.allow", hint: "agent.rows.hintAllow" },
];

/** Settings › Assistant: whether the assistant may read raw rows (R-AG-P2), and which personal
 *  columns of this dataset leave unmasked. */
export function RowsSharing({ datasetId, level }: { datasetId: string; level: Level }) {
  const update = useUpdateAgentSettings();
  const personal = usePersonalColumns(datasetId, level !== "never");
  const unmask = useSetUnmasked(datasetId);
  const hint = LEVELS.find((l) => l.value === level)?.hint ?? "agent.rows.hintNever";
  const shared = (personal.data ?? []).filter((c) => c.unmasked).map((c) => c.name);

  const toggle = (name: string, checked: boolean) =>
    unmask.mutate(checked ? [...shared, name] : shared.filter((n) => n !== name));

  return (
    <section className="space-y-2">
      <h3 className="flex items-center gap-1.5 font-medium text-ink-1">
        <TableProperties size={14} aria-hidden />
        {t("agent.rows.title")}
      </h3>
      <Field label={t("agent.rows.level")}>
        <Select value={level} onChange={(event) => update.mutate({ rows_sharing: event.target.value as Level })}>
          {LEVELS.map((l) => (
            <option key={l.value} value={l.value}>
              {t(l.label)}
            </option>
          ))}
        </Select>
      </Field>
      <p className="text-xs text-ink-3">{t(hint)}</p>
      {level !== "never" && personal.data && (
        <fieldset className="space-y-1.5">
          <legend className="text-xs text-ink-2">{t("agent.rows.personal")}</legend>
          {personal.data.length === 0 ? (
            <p className="text-xs text-ink-3">{t("agent.rows.noPersonal")}</p>
          ) : (
            <>
              <p className="text-xs text-ink-3">{t("agent.rows.personalHint")}</p>
              {personal.data.map((column) => (
                <Checkbox
                  key={column.name}
                  className="flex text-ink-1"
                  checked={column.unmasked}
                  disabled={unmask.isPending}
                  onChange={(event) => toggle(column.name, event.target.checked)}
                  label={t("agent.rows.shareUnmasked", { column: column.name, kind: t(MASK_LABELS[column.mask]) })}
                />
              ))}
            </>
          )}
        </fieldset>
      )}
      {(update.isError || unmask.isError || personal.isError) && (
        <ErrorNote>{(update.error ?? unmask.error ?? personal.error)?.message}</ErrorNote>
      )}
    </section>
  );
}
