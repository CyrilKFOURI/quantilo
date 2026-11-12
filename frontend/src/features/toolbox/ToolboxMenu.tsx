import { Link } from "@tanstack/react-router";
import { Toolbox } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { useColumns, useDataset } from "../../api/hooks";
import { t } from "../../lib/i18n";
import { LockBadge } from "../../lite/Locked";
import { availableTools, THEMES, themeName, toolAnswer, toolName } from "./tools";

type ToolboxMenuProps = {
  datasetId: string;
  modules: { id: string; locked?: boolean }[];
  collapsed: boolean;
  /** In the phone and tablet drawer (041): the menu opens above the entry, at the drawer's width. */
  drawer: boolean;
  itemClass: string;
  active: boolean;
};

/** 098 (owner's revision): the Toolbox is a small menu opened by a click, not a page. Tools are
 *  listed by theme, one compact line each; what a tool answers shows on hover and on focus. */
export function ToolboxMenu({ datasetId, modules, collapsed, drawer, itemClass, active }: ToolboxMenuProps) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const dataset = useDataset(datasetId);
  const columns = useColumns(datasetId, open && dataset.data?.status === "ready");
  const tools = availableTools(modules);

  useEffect(() => {
    if (!open) return;
    const onPointer = (event: PointerEvent) => {
      if (root.current && !root.current.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    panel.current?.querySelector<HTMLElement>("a")?.focus();
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const place = drawer
    ? "bottom-full left-0 mb-1.5 w-full"
    : "bottom-0 left-full ml-2 w-[280px] max-w-[calc(100vw-80px)]";

  return (
    <div ref={root} className="relative">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-current={active ? "page" : undefined}
        title={collapsed ? t("nav.toolbox") : undefined}
        aria-label={collapsed ? t("nav.toolbox") : undefined}
        className={`${itemClass} w-full hover:bg-surface-2 hover:text-ink-1 ${active || open ? "bg-accent-soft !text-ink-1 font-medium" : ""}`}
      >
        <Toolbox size={16} strokeWidth={1.5} aria-hidden />
        {!collapsed && t("nav.toolbox")}
      </button>
      {open && (
        <div
          ref={panel}
          role="dialog"
          aria-label={t("nav.toolbox")}
          className={`absolute z-50 ${place} max-h-[min(70vh,520px)] overflow-y-auto rounded-lg border border-line bg-surface-1 p-1.5 shadow-[0_12px_40px_-12px_rgb(0_0_0/0.35)]`}
          onClick={(event) => {
            if ((event.target as Element).closest("a")) setOpen(false);
          }}
        >
          {THEMES.map((theme) => {
            const found = tools.filter((tool) => tool.theme === theme);
            if (found.length === 0) return null;
            return (
              <section key={theme} aria-label={t(themeName(theme))} className="py-1 first:pt-0.5">
                <h2 className="px-2 pb-0.5 text-[11px] font-medium tracking-wide text-ink-3 uppercase">
                  {t(themeName(theme))}
                </h2>
                <ul>
                  {found.map((tool) => {
                    const Icon = tool.icon;
                    const name = t(toolName(tool));
                    const missing =
                      tool.requires &&
                      columns.data &&
                      !tool.requires.every((type) =>
                        columns.data.some((column) => column.type === type && !column.excluded),
                      );
                    const line = "flex h-8 items-center gap-2 rounded-md px-2 text-[13px]";
                    if (missing) {
                      return (
                        <li key={tool.id}>
                          <span
                            aria-disabled="true"
                            title={tool.requiresKey ? t(tool.requiresKey) : undefined}
                            className={`${line} cursor-not-allowed text-ink-3 opacity-60`}
                          >
                            <Icon size={15} strokeWidth={1.5} aria-hidden className="shrink-0" />
                            <span className="min-w-0 flex-1 truncate">{name}</span>
                          </span>
                        </li>
                      );
                    }
                    return (
                      <li key={tool.id}>
                        <Link
                          to={tool.to as never}
                          params={{ datasetId } as never}
                          search={(tool.search ?? {}) as never}
                          title={t(toolAnswer(tool))}
                          className={`${line} text-ink-1 transition-colors hover:bg-surface-2 focus-visible:bg-surface-2 focus-visible:outline-none`}
                        >
                          <Icon size={15} strokeWidth={1.5} aria-hidden className="shrink-0 text-ink-3" />
                          <span className="min-w-0 flex-1 truncate">{name}</span>
                          {tool.locked && <LockBadge />}
                        </Link>
                      </li>
                    );
                  })}
                </ul>
              </section>
            );
          })}
          <div className="mt-1 border-t border-line px-2 pt-1.5 pb-0.5">
            <Link
              to={"/d/$datasetId/toolbox" as never}
              params={{ datasetId } as never}
              className="text-xs text-ink-3 hover:text-ink-1"
            >
              {t("toolbox.menu.describe")}
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
