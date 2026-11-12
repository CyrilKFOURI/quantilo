import { useId, useRef, type KeyboardEvent, type ReactNode } from "react";

/** `disabled` (107): shown but not yet usable, `hint` says what it will do (its title). */
export type TabItem<T extends string> = { value: T; label: string; badge?: string; disabled?: boolean; hint?: string };

type TabsProps<T extends string> = {
  label: string;
  value: T;
  items: TabItem<T>[];
  onChange: (value: T) => void;
  children: ReactNode;
};

/** WAI-ARIA tabs: arrow keys move between tabs, only the active panel is rendered. */
export function Tabs<T extends string>({ label, value, items, onChange, children }: TabsProps<T>) {
  const id = useId();
  const list = useRef<HTMLDivElement>(null);

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const usable = items.filter((item) => !item.disabled);
    const index = usable.findIndex((item) => item.value === value);
    const moves: Record<string, number> = { ArrowRight: 1, ArrowLeft: -1, Home: -index, End: usable.length - 1 - index };
    const move = moves[event.key];
    if (move === undefined) return;
    event.preventDefault();
    const next = usable[(index + move + usable.length) % usable.length];
    if (!next) return;
    onChange(next.value);
    list.current?.querySelector<HTMLButtonElement>(`[data-value="${next.value}"]`)?.focus();
  }

  return (
    <div>
      <div
        ref={list}
        role="tablist"
        aria-label={label}
        onKeyDown={onKeyDown}
        className="flex gap-4 border-b border-line"
      >
        {items.map((item) => {
          const selected = item.value === value;
          return (
            <button
              key={item.value}
              type="button"
              role="tab"
              id={`${id}-tab-${item.value}`}
              data-value={item.value}
              aria-selected={selected}
              aria-controls={`${id}-panel`}
              tabIndex={selected ? 0 : -1}
              aria-disabled={item.disabled || undefined}
              title={item.hint}
              onClick={() => !item.disabled && onChange(item.value)}
              className={`-mb-px inline-flex h-9 shrink-0 items-center gap-1.5 border-b-2 text-[13px] whitespace-nowrap transition-colors ${
                selected
                  ? "border-accent font-medium text-ink-1"
                  : item.disabled
                    ? "cursor-default border-transparent text-ink-3"
                    : "border-transparent text-ink-2 hover:text-ink-1"
              }`}
            >
              {item.label}
              {item.badge && (
                <span className="rounded-full bg-surface-2 px-1.5 text-[11px] text-ink-2 tabular-nums">{item.badge}</span>
              )}
            </button>
          );
        })}
      </div>
      <div role="tabpanel" id={`${id}-panel`} aria-labelledby={`${id}-tab-${value}`} className="pt-4">
        {children}
      </div>
    </div>
  );
}
