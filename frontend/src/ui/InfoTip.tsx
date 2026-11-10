import { Info } from "lucide-react";
import { useId, type ReactNode } from "react";

const PLACEMENT = {
  top: "bottom-full left-1/2 mb-1.5 w-64 -translate-x-1/2",
  // under a chip of a top row (048): opens downwards from its left edge, never off the left side
  below: "top-full left-0 mt-1.5 w-80",
} as const;

/** The explanation box of a hover tip: shown while its `group` parent is hovered or holds the
 *  focus. Below a laptop (041) a tap shows it pinned above the bottom of the screen; hidden, it
 *  takes no room (no sideways scroll). Its trigger points at it with `aria-describedby={id}`. */
export function TipBox({ id, children, placement = "top" }: { id: string; children: ReactNode; placement?: keyof typeof PLACEMENT }) {
  return (
    <span
      role="tooltip"
      id={id}
      className={`pointer-events-none invisible absolute z-30 rounded-md border border-line bg-surface-1 px-2.5 py-2 text-left text-xs leading-relaxed font-normal text-ink-2 opacity-0 shadow-[0_8px_24px_-12px_rgb(0_0_0/0.35)] transition-opacity group-focus-within:visible group-focus-within:opacity-100 group-hover:visible group-hover:opacity-100 max-lg:fixed max-lg:inset-x-4 max-lg:top-auto max-lg:bottom-[calc(16px+env(safe-area-inset-bottom))] max-lg:mt-0 max-lg:hidden max-lg:w-auto max-lg:translate-x-0 max-lg:text-[13px] max-lg:group-focus-within:block ${PLACEMENT[placement]}`}
    >
      {children}
    </span>
  );
}

/** A small "i" that reveals an explanation on hover or keyboard focus. */
export function InfoTip({ text, label }: { text: string; label: string }) {
  const id = useId();
  return (
    <span className="group relative inline-flex">
      <button
        type="button"
        aria-label={label}
        aria-describedby={id}
        className="inline-flex size-4 items-center justify-center rounded-full text-ink-3 hover:text-ink-1 focus-visible:text-ink-1"
      >
        <Info size={12} strokeWidth={1.75} aria-hidden />
      </button>
      <TipBox id={id}>{text}</TipBox>
    </span>
  );
}
