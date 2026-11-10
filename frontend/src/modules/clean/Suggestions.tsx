import { ChevronDown } from "lucide-react";
import { useId, useState, type ReactNode } from "react";

import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { InfoTip } from "../../ui/InfoTip";
import type { Step, Suggestion } from "./api";

/** Suggestions in layers (study § 5): Basic open, Advanced and Deeper checks folded. A ticked
 *  suggestion is a step of the draft; unticking removes it. */
export function SuggestionsList({
  suggestions,
  steps,
  onToggle,
  onTickAll,
}: {
  suggestions: Suggestion[];
  steps: Step[];
  onToggle: (suggestion: Suggestion, on: boolean) => void;
  onTickAll: (on: boolean) => void;
}) {
  const ticked = new Set(steps.map((step) => step.suggestion));
  const automatic = suggestions.filter((suggestion) => suggestion.mode === "A").length;
  const allTicked = suggestions.every((suggestion) => !suggestion.step || ticked.has(suggestion.id));
  return (
    <div className="space-y-3">
      <Layer
        title={t("clean.layer.basic")}
        hint={t("clean.layer.basicHint", { count: String(suggestions.length), automatic: String(automatic) })}
        defaultOpen
        action={
          suggestions.length > 0 && (
            <Button variant="ghost" onClick={() => onTickAll(!allTicked)}>
              {t(allTicked ? "clean.untickAll" : "clean.tickAll")}
            </Button>
          )
        }
      >
        {suggestions.length === 0 ? (
          <p className="px-3 py-4 text-[13px] text-ink-2">{t("clean.layer.empty")}</p>
        ) : (
          <ul className="divide-y divide-line">
            {suggestions.map((suggestion) => (
              <SuggestionRow
                key={suggestion.id}
                suggestion={suggestion}
                checked={ticked.has(suggestion.id)}
                onToggle={(on) => onToggle(suggestion, on)}
              />
            ))}
          </ul>
        )}
      </Layer>
      {/* Owner's rule: the slower and more niche, the deeper. Advanced is folded; Advanced › More
          (Expert) is folded inside it; neither computes anything until opened. */}
      <Layer title={t("clean.layer.advanced")} hint={t("clean.layer.advancedHint")}>
        <p className="px-3 py-3 text-[13px] text-ink-2">{t("clean.layer.advancedBody")}</p>
        <div className="px-3 pb-3">
          <Layer title={t("clean.layer.expert")} hint={t("clean.layer.expertHint")} nested>
            <p className="px-3 py-3 text-[13px] text-ink-2">{t("clean.layer.expertBody")}</p>
          </Layer>
        </div>
      </Layer>
    </div>
  );
}

function Layer({
  title,
  hint,
  defaultOpen = false,
  action,
  nested = false,
  children,
}: {
  title: string;
  hint: string;
  defaultOpen?: boolean;
  action?: ReactNode;
  /** A layer inside a layer (Advanced › More): on the panel's own background. */
  nested?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();
  return (
    <section className={`rounded-lg border border-line ${nested ? "bg-surface-0" : "bg-surface-1"}`}>
      <div className="flex min-h-11 items-center gap-2 pr-2">
        <button
          type="button"
          aria-expanded={open}
          aria-controls={id}
          onClick={() => setOpen(!open)}
          className="flex min-w-0 flex-1 items-center gap-2 px-3 py-2 text-left"
        >
          <ChevronDown size={15} strokeWidth={1.5} className={`shrink-0 text-ink-3 transition-transform ${open ? "" : "-rotate-90"}`} aria-hidden />
          <span className="shrink-0 text-[13px] font-medium text-ink-1">{title}</span>
          <span className="min-w-0 truncate text-xs text-ink-3">{hint}</span>
        </button>
        {open && action}
      </div>
      {open && (
        <div id={id} className="border-t border-line">
          {children}
        </div>
      )}
    </section>
  );
}

/** An example value as it is: empty in words, blank as "", spaces at the ends made visible. */
function visible(value: string | null): string {
  if (value === null) return t("clean.empty");
  if (value.trim() === "") return `"${value}"`;
  return value.replace(/^ +| +$/g, (spaces) => "·".repeat(spaces.length));
}

function SuggestionRow({ suggestion, checked, onToggle }: { suggestion: Suggestion; checked: boolean; onToggle: (on: boolean) => void }) {
  const id = useId();
  const counts = [
    suggestion.cells > 0 && t("clean.count.cells", { count: formatCount(suggestion.cells) }),
    suggestion.rows > 0 && t("clean.count.rows", { count: formatCount(suggestion.rows) }),
  ].filter(Boolean);
  const examples = suggestion.examples.slice(0, 3);
  return (
    <li className="flex items-start gap-3 px-3 py-2.5">
      <input
        id={id}
        type="checkbox"
        className="mt-0.5 size-3.5 shrink-0 accent-(--accent)"
        checked={checked}
        disabled={!suggestion.step}
        onChange={(event) => onToggle(event.target.checked)}
      />
      <div className="min-w-0 flex-1 space-y-0.5">
        <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
          <label htmlFor={id} className="min-w-0 cursor-pointer text-[13px] text-ink-1">
            {suggestion.title}
          </label>
          <InfoTip
            label={`${t("clean.why")}: ${suggestion.title}`}
            text={`${t(`clean.mode.${suggestion.mode}` as MessageKey)} (${suggestion.rule}): ${t(`clean.modeHint.${suggestion.mode}` as MessageKey)}`}
          />
        </div>
        <p className="text-xs text-ink-2">{suggestion.reason}</p>
        {examples.length > 0 && (
          <p className="flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[11px] text-ink-3">
            {examples.map((example, index) => (
              <span key={index} className="min-w-0 truncate">
                {visible(example.before)} → {visible(example.after)}
              </span>
            ))}
          </p>
        )}
      </div>
      <div className="flex shrink-0 flex-col items-end gap-0.5 text-right text-[11px] text-ink-3 tabular-nums">
        <span>{counts.join(" · ")}</span>
        <span title={t(`clean.modeHint.${suggestion.mode}` as MessageKey)}>
          {suggestion.rule} · <span className={suggestion.mode === "A" ? "text-ok" : "text-warn"}>{t(`clean.mode.${suggestion.mode}` as MessageKey)}</span>
        </span>
      </div>
    </li>
  );
}
