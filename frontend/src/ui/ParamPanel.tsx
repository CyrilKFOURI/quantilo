import { ChevronDown, SlidersHorizontal } from "lucide-react";
import { useId, useState, type ReactNode } from "react";

import { t } from "../lib/i18n";
import { PHONE, useMedia } from "../lib/useMedia";

type ParamPanelProps = {
  /** What the settings are about, for screen readers ("Series settings"). */
  label: string;
  /** The current choices, joined into one line: "date · units · Week · Sum". */
  summary: (string | null | undefined | false)[];
  /** The fields: laid out two per row, label above (`.params-grid` in index.css). */
  children: ReactNode;
  /** Under the fields: a note, a full-width action. */
  footer?: ReactNode;
  /** Open at first (nothing chosen yet). */
  defaultOpen?: boolean;
  /** `grid`: fields two per row; `stack`: groups of controls one under the other. */
  layout?: "grid" | "stack";
};

/** A page's settings on a phone (041): one line that sums them up and opens them, so the page's
 *  content keeps the screen. Every page shows its settings the same way. */
export function ParamPanel({ label, summary, children, footer, defaultOpen = false, layout = "grid" }: ParamPanelProps) {
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();
  return (
    <section aria-label={label} className="rounded-lg border border-line bg-surface-1">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2.5 px-3 py-2 text-left"
      >
        <SlidersHorizontal size={15} strokeWidth={1.5} className="shrink-0 text-ink-3" aria-hidden />
        <span className="min-w-0 flex-1 truncate text-[13px] text-ink-1">
          {summary.filter(Boolean).join(" · ")}
        </span>
        <span className="shrink-0 text-xs text-accent">{open ? t("params.close") : t("params.edit")}</span>
        <ChevronDown
          size={15}
          strokeWidth={1.5}
          className={`shrink-0 text-ink-3 transition-transform ${open ? "rotate-180" : ""}`}
          aria-hidden
        />
      </button>
      {open && (
        <div id={id} className="space-y-3 border-t border-line px-3 py-3">
          <div className={layout === "grid" ? "params-grid" : "space-y-3 text-xs"}>{children}</div>
          {footer}
        </div>
      )}
    </section>
  );
}

/** A page's settings bar: stuck under the top bar on a laptop or a tablet; on a phone (041)
 *  folded into a ParamPanel, with the page's main action (Generate…) kept in sight below it. */
export function SettingsBar({
  label,
  summary,
  phoneAction,
  children,
}: {
  label: string;
  summary: ParamPanelProps["summary"];
  phoneAction?: ReactNode;
  children: ReactNode;
}) {
  const phone = useMedia(PHONE);
  if (phone) {
    return (
      <div className="space-y-2">
        <ParamPanel label={label} summary={summary} layout="stack">
          {children}
        </ParamPanel>
        {phoneAction}
      </div>
    );
  }
  return (
    <div
      className="sticky top-0 z-10 -mx-(--gutter) border-y border-line bg-surface-0/95 px-(--gutter) py-2 backdrop-blur"
      role="group"
      aria-label={label}
    >
      {children}
    </div>
  );
}
