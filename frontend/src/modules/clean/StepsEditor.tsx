import { ArrowDown, ArrowUp, Plus, Search, Trash2, TriangleAlert } from "lucide-react";
import { useState } from "react";

import { t, type MessageKey } from "../../lib/i18n";
import { PHONE, useMedia } from "../../lib/useMedia";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { TextInput } from "../../ui/fields";
import { ParamPanel } from "../../ui/ParamPanel";
import { usePreview, type Op, type Step, type Suggestion } from "./api";
import { PreviewPanel } from "./PreviewTable";
import { StepFields } from "./StepFields";
import { CATALOG, describeStep, move, newStep, nextId, opHint, opLabel, replaceStep, type CatalogGroup } from "./steps";

type Props = {
  datasetId: string;
  steps: Step[];
  problems: Record<string, string>;
  suggestions: Suggestion[];
  columns: string[];
  onChange: (steps: Step[]) => void;
};

/** The draft's steps in order (Power Query's "Applied steps"): pick one to see its settings and
 *  what it changes; move, turn off or remove it; add your own from the catalogue. */
export function StepsEditor({ datasetId, steps, problems, suggestions, columns, onChange }: Props) {
  const [picked, setPicked] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const phone = useMedia(PHONE);
  // the step shown: the one picked while it exists, else the first (derived, never set in an effect)
  const selected = steps.find((step) => step.id === picked) ?? steps[0] ?? null;
  const runs = Boolean(selected?.enabled && !problems[selected.id]);
  const preview = usePreview(datasetId, runs ? steps : undefined, selected?.id ?? null);
  const original = suggestions.find((suggestion) => suggestion.id === selected?.suggestion)?.step ?? null;
  // a step from a suggestion keeps its sentence ("Repair the encoding of brand")
  const titles = new Map(suggestions.map((suggestion) => [suggestion.id, suggestion.title]));

  const add = (op: Op) => {
    const step = newStep(op, nextId(steps), columns[0] ?? "");
    onChange([...steps, step]);
    setPicked(step.id);
    setAdding(false);
  };

  const fields = selected && <StepFields key={selected.id} step={selected} original={original as Step | null} columns={columns} onChange={(step) => onChange(replaceStep(steps, step))} />;

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)]">
      <section aria-label={t("clean.steps.label")} className="min-w-0 space-y-2">
        {steps.length === 0 ? (
          <p className="rounded-lg border border-dashed border-line px-3 py-6 text-center text-[13px] text-ink-2">{t("clean.steps.empty")}</p>
        ) : (
          <ol className="divide-y divide-line rounded-lg border border-line bg-surface-1">
            {steps.map((step, index) => (
              <StepRow
                key={step.id}
                step={step}
                label={titles.get(step.suggestion ?? "") ?? describeStep(step)}
                index={index}
                last={index === steps.length - 1}
                selected={step.id === selected?.id}
                problem={problems[step.id]}
                onSelect={() => setPicked(step.id)}
                onToggle={(enabled) => onChange(replaceStep(steps, { ...step, enabled } as Step))}
                onMove={(by) => onChange(move(steps, step.id, by))}
                onRemove={() => onChange(steps.filter((other) => other.id !== step.id))}
              />
            ))}
          </ol>
        )}
        <Button icon={Plus} onClick={() => setAdding(true)}>
          {t("clean.steps.add")}
        </Button>
      </section>

      <div className="min-w-0 space-y-4">
        {!selected ? (
          <p className="text-[13px] text-ink-2">{t("clean.steps.pick")}</p>
        ) : (
          <>
            {phone ? (
              <ParamPanel label={t("clean.steps.settings")} summary={[describeStep(selected)]}>
                {fields}
              </ParamPanel>
            ) : (
              <section aria-label={t("clean.steps.settings")} className="rounded-lg border border-line bg-surface-1 p-3">
                <h3 className="mb-0.5 text-[13px] font-medium text-ink-1">{opLabel(selected.op)}</h3>
                <p className="mb-3 text-xs text-ink-3">{opHint(selected.op)}</p>
                {fields}
              </section>
            )}
            {problems[selected.id] ? (
              <p role="alert" className="flex items-start gap-2 rounded-md border border-warn/40 bg-warn/8 px-3 py-2 text-[13px] text-ink-1">
                <TriangleAlert size={14} strokeWidth={1.75} className="mt-0.5 shrink-0 text-warn" aria-hidden />
                {problems[selected.id]}
              </p>
            ) : !selected.enabled ? (
              <p className="text-[13px] text-ink-2">{t("clean.steps.off")}</p>
            ) : (
              <PreviewPanel title={t("clean.preview.title")} preview={preview.data} loading={preview.isFetching} error={preview.error} />
            )}
          </>
        )}
      </div>

      {adding && <Catalog onPick={add} onClose={() => setAdding(false)} />}
    </div>
  );
}

function StepRow({
  step,
  label,
  index,
  last,
  selected,
  problem,
  onSelect,
  onToggle,
  onMove,
  onRemove,
}: {
  step: Step;
  label: string;
  index: number;
  last: boolean;
  selected: boolean;
  problem: string | undefined;
  onSelect: () => void;
  onToggle: (enabled: boolean) => void;
  onMove: (by: -1 | 1) => void;
  onRemove: () => void;
}) {
  return (
    <li className={`flex min-h-11 items-center gap-2 px-2 ${selected ? "bg-accent-soft" : ""}`}>
      <span className="w-5 shrink-0 text-right text-[11px] text-ink-3 tabular-nums">{index + 1}</span>
      <input
        type="checkbox"
        className="size-3.5 shrink-0 accent-(--accent)"
        checked={step.enabled}
        onChange={(event) => onToggle(event.target.checked)}
        aria-label={`${t("clean.steps.enabled")}: ${label}`}
      />
      <button
        type="button"
        onClick={onSelect}
        aria-current={selected ? "step" : undefined}
        className="flex min-w-0 flex-1 flex-col items-start py-1.5 text-left"
      >
        <span className={`flex w-full min-w-0 items-center gap-1.5 text-[13px] ${step.enabled ? "text-ink-1" : "text-ink-3 line-through"}`}>
          {problem && <TriangleAlert size={13} strokeWidth={1.75} className="shrink-0 text-warn" aria-label={t("clean.steps.problem")} />}
          <span className="truncate" title={label}>
            {label}
          </span>
        </span>
        <span className="text-[11px] text-ink-3">
          {[t(`clean.steps.origin.${step.origin}` as MessageKey), step.rule].filter(Boolean).join(" · ")}
        </span>
      </button>
      <span className="flex shrink-0">
        <Button variant="ghost" icon={ArrowUp} iconOnly aria-label={`${t("clean.steps.up")}: ${label}`} title={t("clean.steps.up")} disabled={index === 0} onClick={() => onMove(-1)} />
        <Button variant="ghost" icon={ArrowDown} iconOnly aria-label={`${t("clean.steps.down")}: ${label}`} title={t("clean.steps.down")} disabled={last} onClick={() => onMove(1)} />
        <Button variant="ghost" icon={Trash2} iconOnly aria-label={`${t("clean.steps.remove")}: ${label}`} title={t("clean.steps.remove")} onClick={onRemove} />
      </span>
    </li>
  );
}

const GROUPS: CatalogGroup[] = ["columns", "text", "values", "rows"];

/** "+ Add step": the M3 batch of transformations, with a search. */
function Catalog({ onPick, onClose }: { onPick: (op: Op) => void; onClose: () => void }) {
  const [query, setQuery] = useState("");
  const words = query.trim().toLowerCase();
  const found = CATALOG.filter(({ op }) => !words || `${opLabel(op)} ${opHint(op)}`.toLowerCase().includes(words));
  return (
    <Dialog open onOpenChange={(open) => !open && onClose()} title={t("clean.catalog.title")} size="md">
      <label className="relative mb-3 block">
        <Search size={14} strokeWidth={1.5} className="pointer-events-none absolute top-1/2 left-2 -translate-y-1/2 text-ink-3" aria-hidden />
        <TextInput autoFocus aria-label={t("clean.catalog.search")} placeholder={t("clean.catalog.search")} value={query} onChange={(event) => setQuery(event.target.value)} className="w-full pl-7" />
      </label>
      {found.length === 0 && <p className="text-[13px] text-ink-2">{t("clean.catalog.none")}</p>}
      <div className="space-y-4">
        {GROUPS.map((group) => {
          const items = found.filter((item) => item.group === group);
          if (!items.length) return null;
          return (
            <section key={group}>
              <h3 className="mb-1 text-xs text-ink-3">{t(`clean.catalog.group.${group}` as MessageKey)}</h3>
              <ul className="grid gap-1 sm:grid-cols-2">
                {items.map(({ op }) => (
                  <li key={op}>
                    <button
                      type="button"
                      onClick={() => onPick(op)}
                      className="flex w-full flex-col items-start rounded-md border border-line px-3 py-2 text-left hover:border-accent hover:bg-accent-soft/50"
                    >
                      <span className="text-[13px] text-ink-1">{opLabel(op)}</span>
                      <span className="text-xs text-ink-3">{opHint(op)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          );
        })}
      </div>
    </Dialog>
  );
}
