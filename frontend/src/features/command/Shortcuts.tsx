import { useEffect, useState } from "react";

import { t, type MessageKey } from "../../lib/i18n";
import { Dialog } from "../../ui/Dialog";
import { CommandPalette, type PaletteModule } from "./CommandPalette";

const SHORTCUTS: { keys: string[]; label: MessageKey }[] = [
  { keys: ["F"], label: "shortcuts.filters" },
  { keys: ["Ctrl", "K"], label: "shortcuts.palette" },
  { keys: ["?"], label: "shortcuts.list" },
  { keys: ["Ctrl", "Z"], label: "shortcuts.undo" },
  { keys: ["Esc"], label: "shortcuts.close" },
];

/** Dispatched on window (by the top bar's search button) to open the palette. */
export const OPEN_PALETTE_EVENT = "de:open-palette";

const isTyping = (target: EventTarget | null) =>
  target instanceof Element && Boolean(target.closest("input, select, textarea, [contenteditable=true]"));

/** Global keyboard layer: Ctrl+K (or ⌘K) opens the palette, ? lists the shortcuts. */
export function KeyboardLayer({ modules = [] }: { modules?: PaletteModule[] }) {
  const [palette, setPalette] = useState(false);
  const [help, setHelp] = useState(false);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPalette((open) => !open);
      } else if (event.key === "?" && !isTyping(event.target) && !event.ctrlKey && !event.metaKey && !event.altKey) {
        event.preventDefault();
        setHelp(true);
      }
    };
    const onOpen = () => setPalette(true);
    window.addEventListener("keydown", onKey);
    window.addEventListener(OPEN_PALETTE_EVENT, onOpen);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener(OPEN_PALETTE_EVENT, onOpen);
    };
  }, []);

  return (
    <>
      <CommandPalette open={palette} onOpenChange={setPalette} onShortcuts={() => setHelp(true)} modules={modules} />
      <Dialog open={help} onOpenChange={setHelp} title={t("shortcuts.title")} size="sm">
        <dl className="divide-y divide-line">
          {SHORTCUTS.map((shortcut) => (
            <div key={shortcut.label} className="flex items-center justify-between gap-4 py-2">
              <dt className="text-ink-2">{t(shortcut.label)}</dt>
              <dd className="flex gap-1">
                {shortcut.keys.map((key) => (
                  <kbd key={key} className="min-w-6 rounded border border-line bg-surface-2 px-1.5 py-0.5 text-center font-mono text-xs text-ink-1">
                    {key}
                  </kbd>
                ))}
              </dd>
            </div>
          ))}
        </dl>
      </Dialog>
    </>
  );
}
