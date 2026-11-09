import * as RadixDialog from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import type { ReactNode } from "react";

import { t } from "../lib/i18n";

type DialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description?: string;
  children: ReactNode;
  footer?: ReactNode;
  size?: "sm" | "md" | "lg";
};

const widths = { sm: "max-w-md", md: "max-w-2xl", lg: "max-w-5xl" };

export function Dialog({
  open,
  onOpenChange,
  title,
  description,
  children,
  footer,
  size = "md",
}: DialogProps) {
  return (
    <RadixDialog.Root open={open} onOpenChange={onOpenChange}>
      <RadixDialog.Portal>
        <RadixDialog.Overlay className="fixed inset-0 z-40 bg-ink-1/30" />
        <RadixDialog.Content
          // 041: a full-screen sheet on a phone (title bar on top, the action in the foot).
          className={`fixed top-1/2 left-1/2 z-50 flex max-h-[min(88dvh,900px)] w-[calc(100vw-32px)] -translate-x-1/2 -translate-y-1/2 flex-col rounded-lg border border-line bg-surface-1 shadow-[0_12px_40px_-12px_rgb(0_0_0/0.35)] max-sm:inset-0 max-sm:h-dvh max-sm:max-h-none max-sm:w-full max-sm:max-w-none max-sm:translate-x-0 max-sm:translate-y-0 max-sm:rounded-none max-sm:border-0 max-sm:pt-[env(safe-area-inset-top)] max-sm:pb-[env(safe-area-inset-bottom)] ${widths[size]}`}
        >
          <header className="flex items-start justify-between gap-4 border-b border-line px-5 py-3.5">
            <div className="min-w-0">
              <RadixDialog.Title className="truncate text-sm font-semibold">{title}</RadixDialog.Title>
              {description ? (
                <RadixDialog.Description className="mt-0.5 text-xs text-ink-3">
                  {description}
                </RadixDialog.Description>
              ) : (
                <RadixDialog.Description className="sr-only">{title}</RadixDialog.Description>
              )}
            </div>
            <RadixDialog.Close
              aria-label={t("dialog.close")}
              className="-mr-1.5 inline-flex size-7 items-center justify-center rounded-md text-ink-3 hover:bg-surface-2 hover:text-ink-1"
            >
              <X size={16} strokeWidth={1.5} aria-hidden />
            </RadixDialog.Close>
          </header>
          <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">{children}</div>
          {footer && (
            <footer className="flex items-center justify-end gap-2 border-t border-line px-5 py-3">
              {footer}
            </footer>
          )}
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </RadixDialog.Root>
  );
}
