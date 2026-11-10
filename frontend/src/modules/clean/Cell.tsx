import { RotateCcw } from "lucide-react";
import { useState, type ReactNode } from "react";

import { t } from "../../lib/i18n";
import { TextInput } from "../../ui/fields";

/** One setting in the grid, as Customize's `Cell` (045, not exported there): a short label (why
 *  in its tooltip), its control, and a reset icon once it differs from what the rule proposed. */
export function Cell({
  label,
  why,
  changed = false,
  onReset,
  wide = false,
  children,
}: {
  label: string;
  why?: string | null;
  changed?: boolean;
  onReset?: () => void;
  /** Two columns, or the whole row for `full`. */
  wide?: boolean | "full";
  children: ReactNode;
}) {
  const span = wide === "full" ? "col-span-full" : wide ? "col-span-2" : "";
  return (
    <div className={`min-w-0 space-y-1 ${span}`}>
      <div className="flex min-w-0 items-center gap-1 text-[11px] text-ink-2">
        {changed && <span className="size-1.5 shrink-0 rounded-full bg-accent" aria-label={t("kpi.customised")} />}
        <span className="truncate" title={why ?? undefined}>
          {label}
        </span>
        {changed && onReset && (
          <button
            type="button"
            onClick={onReset}
            aria-label={`${t("clean.reset.label")}: ${label}`}
            title={t("clean.reset.label")}
            className="ml-auto shrink-0 text-ink-3 hover:text-accent"
          >
            <RotateCcw size={11} aria-hidden />
          </button>
        )}
      </div>
      {children}
    </div>
  );
}

/** A text field that commits on Enter or when it loses focus (a step is not rebuilt at every key).
 *  Give it `key={value}` so a value changed elsewhere (undo) shows. */
export function CommitInput({
  value,
  onCommit,
  label,
  type = "text",
  placeholder,
}: {
  value: string;
  onCommit: (value: string) => void;
  label: string;
  type?: "text" | "number";
  placeholder?: string;
}) {
  const [draft, setDraft] = useState(value);
  const commit = () => draft !== value && onCommit(draft);
  return (
    <TextInput
      aria-label={label}
      type={type}
      value={draft}
      placeholder={placeholder}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => event.key === "Enter" && commit()}
      className="h-8 w-full min-w-0 text-[13px]"
    />
  );
}
