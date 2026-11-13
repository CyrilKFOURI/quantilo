import type { KeyboardEvent as ReactKeyboardEvent } from "react";

/** N42: the keyboard of the review. j / k move from a fix to the next or the previous one (the
 *  list listens on the window), Space (or x) ticks the fix that has the focus, Enter (or e) opens
 *  it (a merge's review, a batch's columns, else its rows), u undoes (CleanPage). Never with Ctrl,
 *  Alt or ⌘ (Ctrl+K, Ctrl+J, Ctrl+Z are the app's), never while typing, never under a dialog. */

const NOT_TYPED = new Set(["checkbox", "radio", "button", "submit", "reset", "range", "color", "file"]);

/** The focus is in a field that takes text: its keys are the user's. */
export function typingIn(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false;
  if (target.closest('textarea, select, [contenteditable="true"], [contenteditable=""]')) return true;
  const input = target.closest("input");
  return input !== null && !NOT_TYPED.has(input.type);
}

/** A key the review may take: no modifier, not typed into a field, no dialog open over the page. */
export function reviewKey(event: KeyboardEvent | ReactKeyboardEvent): boolean {
  return !event.ctrlKey && !event.metaKey && !event.altKey && !event.defaultPrevented && !typingIn(event.target) && !document.querySelector('[role="dialog"]');
}

/** The attribute that makes a line a stop of j / k. */
export const ROW = "data-review-row";

/** The props of a line of the review: focusable by j / k (not by Tab: its controls are), Space to
 *  tick, Enter to open, only when the line itself has the focus (a control inside keeps its keys). */
export function rowKeys({ onTick, onOpen }: { onTick: () => void; onOpen?: () => void }) {
  return {
    [ROW]: "",
    tabIndex: -1,
    onKeyDown: (event: ReactKeyboardEvent<HTMLElement>) => {
      if (event.target !== event.currentTarget || !reviewKey(event)) return;
      const key = event.key.toLowerCase();
      if (key === " " || key === "x") {
        event.preventDefault();
        onTick();
      } else if ((key === "enter" || key === "e") && onOpen) {
        event.preventDefault();
        onOpen();
      }
    },
  };
}

/** The lines of a list that can be reached: not inside a folded (hidden) part. */
function rowsOf(container: HTMLElement): HTMLElement[] {
  return [...container.querySelectorAll<HTMLElement>(`[${ROW}]`)].filter((row) => !row.closest("[hidden]"));
}

/** j / k: the focus to the next or the previous line of `container` (the first when none has it). */
export function moveFocus(container: HTMLElement, by: 1 | -1): HTMLElement | null {
  const rows = rowsOf(container);
  if (!rows.length) return null;
  const at = rows.findIndex((row) => row.contains(document.activeElement));
  const next = rows[at < 0 ? 0 : Math.min(rows.length - 1, Math.max(0, at + by))] ?? null;
  next?.focus();
  next?.scrollIntoView?.({ block: "nearest" });
  return next;
}
