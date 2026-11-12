import { ChevronRight } from "lucide-react";
import { useState, type ReactNode } from "react";

/** A folded layer of the onion rule ("Advanced", then "More" inside it): its content is not
 *  rendered, so nothing is asked of the server, before the user opens it. */
export function Folded({
  title,
  hint,
  nested = false,
  onOpenChange,
  children,
}: {
  title: string;
  hint?: string;
  /** A layer inside a layer (Advanced › More): on the page's own background. */
  nested?: boolean;
  /** Told when the layer opens or closes (099: the Data table becomes selectable while open). */
  onOpenChange?: (open: boolean) => void;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <details
      className={`group rounded-lg border border-line ${nested ? "bg-surface-0" : "bg-surface-1"}`}
      onToggle={(event) => {
        setOpen(event.currentTarget.open);
        onOpenChange?.(event.currentTarget.open);
      }}
    >
      <summary className="flex min-h-11 cursor-pointer list-none items-center gap-2 px-3 py-2 select-none hover:bg-surface-2 [&::-webkit-details-marker]:hidden">
        <ChevronRight
          size={16}
          strokeWidth={1.5}
          aria-hidden
          className="shrink-0 text-ink-3 transition-transform group-open:rotate-90"
        />
        <span className="shrink-0 text-[13px] font-medium text-ink-1">{title}</span>
        {hint && <span className="min-w-0 truncate text-xs text-ink-3">{hint}</span>}
      </summary>
      {open && <div className="border-t border-line px-3 py-3">{children}</div>}
    </details>
  );
}
