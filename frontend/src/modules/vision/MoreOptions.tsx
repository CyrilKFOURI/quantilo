import { CircleSlash, Clock } from "lucide-react";
import { useId, type ReactNode } from "react";

import { Folded } from "../../features/text/Folded";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import type { TaskOut } from "./modelApi";
import type { ConceptOut, MoreKey, MoreOption } from "./moreApi";

/**
 * 118: Advanced › More of the Model tab (classification). Fine-tuning, ProtoPNet-lite and the
 * concept bottleneck, each ticked by the user and trained only when the run starts (the onion
 * rule); an option that cannot run here is greyed with its reason (R-VMO-01), every one shows an
 * estimated time (R-VMO-02). The bottleneck lists the set's concepts: columns of the table and
 * groups of images saved from the gallery's selection (R-VCB-01).
 */
export function MoreOptions({
  out,
  more,
  onMore,
  concepts,
  onConcepts,
}: {
  out: TaskOut;
  more: readonly MoreKey[];
  onMore: (next: MoreKey[]) => void;
  /** The bottleneck's concepts by key; null = every usable one. */
  concepts: readonly string[] | null;
  onConcepts: (next: string[] | null) => void;
}) {
  const options = out.more ?? [];
  if (options.length === 0) return null;
  const usable = (out.concepts ?? []).filter((c) => c.bottleneck);
  const toggle = (key: MoreKey, on: boolean) => onMore(on ? [...new Set([...more, key])] : more.filter((k) => k !== key));
  return (
    <Folded nested title={t("vision.more.title")} hint={t("vision.more.hint")}>
      <div className="flex flex-col gap-2">
        <p className="max-w-[72ch] text-xs text-pretty text-ink-3">{t("vision.more.intro")}</p>
        {options.map((option) => (
          <OptionRow key={option.key} option={option} checked={more.includes(option.key)} onChange={(on) => toggle(option.key, on)}>
            {option.key === "concept_bottleneck" && option.available && more.includes("concept_bottleneck") && (
              <ConceptPicker concepts={usable} chosen={concepts} onChange={onConcepts} />
            )}
          </OptionRow>
        ))}
        <p className="max-w-[72ch] text-xs text-pretty text-ink-3">{t("vision.more.conceptsHint")}</p>
      </div>
    </Folded>
  );
}

function OptionRow({ option, checked, onChange, children }: { option: MoreOption; checked: boolean; onChange: (on: boolean) => void; children?: ReactNode }) {
  const id = useId();
  const disabled = !option.available;
  return (
    <div className={`flex gap-3 rounded-md border border-line px-3 py-2.5 ${checked ? "bg-surface-2" : ""}`}>
      <input
        id={id}
        type="checkbox"
        className="mt-0.5 size-3.5 shrink-0 accent-(--accent) disabled:opacity-50 pointer-coarse:size-5"
        checked={checked && !disabled}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
      />
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <label htmlFor={id} className={`flex flex-wrap items-baseline gap-x-2 gap-y-0.5 text-[13px] ${disabled ? "text-ink-3" : "cursor-pointer text-ink-1"}`}>
          <span className="font-medium">{option.title}</span>
          <span className="text-xs font-normal text-ink-3">{t("vision.more.level")}</span>
        </label>
        <p className="text-xs text-pretty text-ink-3">{option.note}</p>
        {disabled && option.reason && (
          <p className="flex gap-1.5 text-xs text-pretty text-ink-2">
            <CircleSlash size={13} strokeWidth={1.5} className="mt-px shrink-0 text-ink-3" aria-hidden />
            {option.reason}
          </p>
        )}
        {option.estimate && (
          <p className="flex gap-1.5 text-xs text-ink-2 tabular-nums">
            <Clock size={13} strokeWidth={1.5} className="mt-px shrink-0 text-ink-3" aria-hidden />
            {t("vision.more.estimate", { estimate: option.estimate })}
          </p>
        )}
        {children}
      </div>
    </div>
  );
}

function ConceptPicker({ concepts, chosen, onChange }: { concepts: readonly ConceptOut[]; chosen: readonly string[] | null; onChange: (next: string[] | null) => void }) {
  const selected = new Set(chosen ?? concepts.map((c) => c.key));
  const set = (key: string, on: boolean) => {
    const next = concepts.map((c) => c.key).filter((k) => (k === key ? on : selected.has(k)));
    onChange(next.length === concepts.length ? null : next);
  };
  const count = concepts.filter((c) => selected.has(c.key)).length;
  return (
    <fieldset className="mt-1 flex flex-col gap-1">
      <legend className="mb-1 text-xs text-ink-2">{t("vision.more.concepts", { count: formatCount(count) })}</legend>
      <ul className="grid gap-x-4 gap-y-1 sm:grid-cols-2">
        {concepts.map((concept) => (
          <li key={concept.key} className="min-w-0">
            <label className="flex min-h-7 min-w-0 cursor-pointer items-center gap-2 text-xs text-ink-1 pointer-coarse:min-h-11">
              <input type="checkbox" className="size-3.5 shrink-0 accent-(--accent)" checked={selected.has(concept.key)} onChange={(event) => set(concept.key, event.target.checked)} />
              <span className="min-w-0 truncate">{concept.name}</span>
              <span className="shrink-0 text-ink-3 tabular-nums">
                {concept.source === "group" ? t("vision.more.group", { count: formatCount(concept.images) }) : concept.kind === "numeric" ? t("vision.more.numeric") : t("vision.more.column", { count: formatCount(concept.images) })}
              </span>
            </label>
          </li>
        ))}
      </ul>
      {count < 2 && <p className="text-xs text-warn">{t("vision.more.twoConcepts")}</p>}
    </fieldset>
  );
}
