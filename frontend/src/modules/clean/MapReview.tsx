import { useId, useState } from "react";

import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { TextInput } from "../../ui/fields";
import type { Step, Suggestion } from "./api";
import type { Replace } from "./context";
import { Tick, visible } from "./parts";
import { readReview, reviewFacts, reviewGroups, writeReview, type ReviewGroup, type ReviewState } from "./review";

/** N38: a merge of spellings reviewed value by value, as OpenRefine's cluster dialog: each group
 *  with its target (editable) and its values, each with its rows; untick a value or a whole group,
 *  change a target. Every change writes the step again with exactly what remains ticked (one
 *  change, one Ctrl+Z); nothing ticked, the suggestion leaves the draft. */
export function MapReview({ suggestion, step, onReplace }: { suggestion: Suggestion; step: Step | undefined; onReplace: Replace }) {
  const groups = reviewGroups(suggestion);
  const state = readReview(groups, step);
  // a target typed while its group is not ticked waits here until a value of it is
  const [edits, setEdits] = useState<Record<number, string>>({});
  const targets = groups.map((group, index) => (state.ticked[index]?.some(Boolean) ? (state.targets[index] ?? group.target) : (edits[index] ?? group.target)));
  const facts = reviewFacts(groups, { ...state, targets });

  const write = (next: ReviewState) => {
    const map = writeReview(groups, next);
    const base = step ?? suggestion.step;
    onReplace(suggestion, base && Object.keys(map).length ? ({ ...base, map } as Step) : null);
  };
  const tick = (group: number, value: number | null, on: boolean) =>
    write({
      targets,
      ticked: state.ticked.map((values, index) => (index === group ? values.map((old, at) => (value === null || at === value ? on : old)) : values)),
    });
  const all = (on: boolean) => write({ targets, ticked: state.ticked.map((values) => values.map(() => on)) });
  const retarget = (group: number, target: string) => {
    if (state.ticked[group]?.some(Boolean)) write({ ticked: state.ticked, targets: targets.map((old, index) => (index === group ? target : old)) });
    else setEdits({ ...edits, [group]: target });
  };

  const summary = [
    t("clean.review.groups", { ticked: formatCount(facts.groups), count: formatCount(groups.length) }),
    t("clean.review.values", { ticked: formatCount(facts.values), count: formatCount(facts.total) }),
    facts.cells !== null && t("clean.count.cells", { count: formatCount(facts.cells) }),
  ].filter(Boolean);
  return (
    <section aria-label={t("clean.review.label")} className="mx-3 mt-1 mb-2 rounded-md border border-line bg-surface-0 sm:ml-9">
      <header className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 border-b border-line px-3 py-1.5">
        <p className="text-xs text-ink-2 tabular-nums" aria-live="polite">
          {summary.join(" · ")}
        </p>
        <span className="-mr-2 flex items-center gap-1">
          <Button variant="ghost" onClick={() => all(true)} disabled={facts.values === facts.total}>
            {t("clean.review.all")}
          </Button>
          <Button variant="ghost" onClick={() => all(false)} disabled={facts.values === 0}>
            {t("clean.review.none")}
          </Button>
        </span>
      </header>
      <p className="px-3 pt-1.5 text-xs text-ink-3">{t("clean.review.hint")}</p>
      <ul className="max-h-[28rem] divide-y divide-line overflow-y-auto">
        {groups.map((group, index) => (
          <GroupReview key={`${index}:${group.target}`} group={group} ticked={state.ticked[index] ?? []} target={targets[index] ?? group.target} onTick={(value, on) => tick(index, value, on)} onTarget={(target) => retarget(index, target)} />
        ))}
      </ul>
    </section>
  );
}

function GroupReview({ group, ticked, target, onTick, onTarget }: { group: ReviewGroup; ticked: boolean[]; target: string; onTick: (value: number | null, on: boolean) => void; onTarget: (target: string) => void }) {
  const on = ticked.filter(Boolean).length;
  const state = on === 0 ? false : on === ticked.length ? true : "mixed";
  const changed = target !== group.target;
  return (
    <li className="px-3 py-2">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <Tick state={state} label={t("clean.review.group", { target: group.target })} onChange={(next) => onTick(null, next)} className="mt-0" />
        <span className="text-xs text-ink-2">{t("clean.review.into")}</span>
        <TargetInput key={target} value={target} label={t("clean.review.target", { target: group.target })} onCommit={onTarget} />
        {group.rows !== null && <span className="text-[11px] text-ink-3 tabular-nums">{t(changed ? "clean.review.targetRowsChanged" : "clean.review.targetRows", { count: formatCount(group.rows), target: group.target })}</span>}
      </div>
      <ul className="mt-1 grid gap-x-4 pl-6 sm:grid-cols-2">
        {group.values.map((value, at) => (
          <li key={value.value}>
            <label className="flex min-h-7 min-w-0 cursor-pointer items-center gap-2" title={value.spellings.length > 1 ? value.spellings.map((spelling) => visible(spelling)).join(" · ") : undefined}>
              <input type="checkbox" aria-label={t("clean.review.value", { value: value.value, target })} className="size-3.5 shrink-0 accent-(--accent)" checked={ticked[at] ?? false} onChange={(event) => onTick(at, event.target.checked)} />
              <span className={`min-w-0 truncate font-mono text-[12px] ${ticked[at] ? "text-ink-1" : "text-ink-3 line-through"}`}>{visible(value.value)}</span>
              <span className="shrink-0 text-[11px] text-ink-3 tabular-nums">
                {value.rows !== null && t("clean.count.rows", { count: formatCount(value.rows) })}
                {value.spellings.length > 1 && ` · ${t("clean.review.spellings", { count: formatCount(value.spellings.length) })}`}
              </span>
            </label>
          </li>
        ))}
      </ul>
    </li>
  );
}

/** A target, written when the field is left or Enter pressed; Escape gives the last one back. An
 *  empty target is refused (a merge into nothing is not a merge): the field takes its value back. */
function TargetInput({ value, label, onCommit }: { value: string; label: string; onCommit: (target: string) => void }) {
  const [text, setText] = useState(value);
  const id = useId();
  const commit = () => {
    const next = text.trim();
    if (!next) setText(value);
    else if (next !== value) onCommit(next);
  };
  return (
    <TextInput
      id={id}
      aria-label={label}
      value={text}
      onChange={(event) => setText(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          commit();
        } else if (event.key === "Escape") {
          event.stopPropagation();
          setText(value);
        }
      }}
      className="w-48 max-w-full font-mono text-[12px]"
    />
  );
}
