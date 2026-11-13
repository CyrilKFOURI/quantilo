import { t } from "../../lib/i18n";

/** A tick for several fixes or values: ticked, not ticked, or some of them (mixed, which the
 *  checkbox itself says to a screen reader: `indeterminate`). */
export function Tick({ id, state, label, className = "", disabled, onChange }: { id?: string; state: boolean | "mixed"; label?: string; className?: string; disabled?: boolean; onChange: (on: boolean) => void }) {
  return (
    <input
      id={id}
      type="checkbox"
      aria-label={label}
      className={`mt-0.5 size-3.5 shrink-0 accent-(--accent) ${className}`}
      ref={(box) => {
        if (box) box.indeterminate = state === "mixed";
      }}
      checked={state === true}
      disabled={disabled}
      onChange={() => onChange(state !== true)}
    />
  );
}

/** An example value as it is: empty in words, blank as "", spaces at the ends made visible. */
export function visible(value: string | null): string {
  if (value === null) return t("clean.empty");
  if (value.trim() === "") return `"${value}"`;
  return value.replace(/^ +| +$/g, (spaces) => "·".repeat(spaces.length));
}
