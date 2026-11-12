import { useState } from "react";

import type { Column, MlChoicesInput } from "../../api/client";
import { derivedNote } from "../../features/derived/derived";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Select } from "../../ui/fields";
import { Dialog } from "../../ui/Dialog";
import type { SegmentDecision, SegmentPlan } from "./api";
import type { Shared } from "./SegmentsPage";

type ColumnChoice = NonNullable<MlChoicesInput["columns"]>[string];
const SETTINGS = ["transform", "scaling", "encoding"] as const;

/** The columns used and left out, each with its rule and reason, and what can be changed on top
 * (use, transform, scaling, encoding): the ML preparation rules with the "distance" profile. */
export function ColumnsDialog({
  open,
  onOpenChange,
  plan,
  shared,
  onShared,
  derived = [],
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  plan: SegmentPlan;
  shared: Shared;
  onShared: (next: Shared) => void;
  /** 058: derived columns ("review › Length"), added only when ticked (R-DER-07). */
  derived?: Column[];
}) {
  const [draft, setDraft] = useState<Record<string, ColumnChoice>>(shared.preparation?.columns ?? {});
  const change = (column: string, patch: ColumnChoice) => setDraft({ ...draft, [column]: { ...draft[column], ...patch } });
  const apply = () => {
    onShared({ columns: null, preparation: Object.keys(draft).length ? { columns: draft } : null });
    onOpenChange(false);
  };
  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={t("segments.columnsTitle")}
      description={t("segments.columnsHint")}
      size="lg"
      footer={
        <>
          <Button variant="ghost" onClick={() => setDraft({})}>
            {t("segments.reset")}
          </Button>
          <Button variant="primary" onClick={apply}>
            {t("segments.apply")}
          </Button>
        </>
      }
    >
      <table className="w-full text-[13px]">
        <caption className="sr-only">{t("segments.columnsTitle")}</caption>
        <thead>
          <tr className="border-b border-line text-left text-xs text-ink-3">
            <th className="py-1.5 pr-3 font-normal">{t("segments.column")}</th>
            <th className="py-1.5 pr-3 font-normal">{t("segments.use")}</th>
            {SETTINGS.map((s) => (
              <th key={s} className="py-1.5 pr-3 font-normal">
                {t(`segments.setting.${s}` as MessageKey)}
              </th>
            ))}
            <th className="py-1.5 font-normal">{t("segments.why")}</th>
          </tr>
        </thead>
        <tbody>
          {plan.preparation.decisions.map((d) => (
            <Row key={d.column} decision={d} choice={draft[d.column]} onChange={(patch) => change(d.column, patch)} />
          ))}
        </tbody>
      </table>
      <DerivedChoices
        derived={derived.filter((c) => !plan.preparation.decisions.some((d) => d.column === c.name))}
        added={(name) => draft[name]?.use === true}
        onToggle={(name, on) => {
          if (on) change(name, { use: true });
          else setDraft(Object.fromEntries(Object.entries(draft).filter(([key]) => key !== name)));
        }}
      />
    </Dialog>
  );
}

function Row({ decision, choice, onChange }: { decision: SegmentDecision; choice: ColumnChoice | undefined; onChange: (patch: ColumnChoice) => void }) {
  const refused = decision.options.use?.find((o) => o.value === "yes")?.reason ?? null;
  const used = choice?.use ?? decision.use;
  return (
    <tr className="border-b border-line align-top last:border-0">
      <td className="py-1.5 pr-3 font-mono text-xs text-ink-1">{decision.column}</td>
      <td className="py-1.5 pr-3">
        <input
          type="checkbox"
          className="size-3.5 accent-(--accent)"
          aria-label={t("segments.useColumn", { column: decision.column })}
          title={refused ?? undefined}
          checked={used}
          disabled={Boolean(refused) && !decision.use}
          onChange={(e) => onChange({ use: e.target.checked })}
        />
      </td>
      {SETTINGS.map((setting) => {
        const options = decision.options[setting];
        if (!options || !used) return <td key={setting} className="py-1.5 pr-3 text-ink-3">—</td>;
        const current = (choice?.[setting] as string | undefined) ?? decision[setting] ?? "none";
        return (
          <td key={setting} className="py-1.5 pr-3">
            <Select
              aria-label={t("segments.settingOf", { setting: t(`segments.setting.${setting}` as MessageKey), column: decision.column })}
              value={current}
              onChange={(e) => onChange({ [setting]: e.target.value })}
              className="h-7 text-xs"
            >
              {options.map((o) => (
                <option key={o.value} value={o.value} disabled={Boolean(o.reason)} title={o.reason ?? undefined}>
                  {o.value}
                </option>
              ))}
            </Select>
          </td>
        );
      })}
      <td className="py-1.5 text-xs text-ink-2">
        {decision.why.map((sentence, i) => (
          <span key={i} className="block">
            {sentence}
          </span>
        ))}
      </td>
    </tr>
  );
}

/** 058: derived columns of the text columns, each added as a column of its own when ticked. */
function DerivedChoices({
  derived,
  added,
  onToggle,
}: {
  derived: Column[];
  added: (name: string) => boolean;
  onToggle: (name: string, on: boolean) => void;
}) {
  if (!derived.length) return null;
  return (
    <section className="mt-4 border-t border-line pt-3">
      <h3 className="text-xs font-medium text-ink-2">{t("segments.derived")}</h3>
      <p className="text-xs text-ink-3">{t("segments.derivedHint")}</p>
      <div className="mt-2 grid gap-x-4 sm:grid-cols-2 lg:grid-cols-3">
        {derived.map((column) => (
          <label
            key={column.name}
            title={derivedNote(column) ?? undefined}
            className="flex min-w-0 cursor-pointer items-center gap-2 rounded px-1 py-1 text-[13px] hover:bg-surface-2"
          >
            <input
              type="checkbox"
              className="size-3.5 accent-(--accent)"
              checked={added(column.name)}
              onChange={(event) => onToggle(column.name, event.target.checked)}
            />
            <span className="truncate font-mono text-xs">{column.name}</span>
          </label>
        ))}
      </div>
    </section>
  );
}
