import * as RadixDialog from "@radix-ui/react-dialog";
import { useNavigate } from "@tanstack/react-router";
import {
  BookOpen,
  Columns3,
  Database,
  Eraser,
  Filter,
  Grid3x3,
  House,
  Keyboard,
  LayoutGrid,
  Monitor,
  Moon,
  Rows3,
  ScatterChart,
  Search,
  Settings,
  Sun,
  Toolbox,
  TrendingUp,
  X,
  type LucideIcon,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { useColumns, useDataset, useDatasets } from "../../api/hooks";
import { meets, type Requirement } from "../../modules/types";
import { t, type MessageKey } from "../../lib/i18n";
import { savePreference, themePreference } from "../../lib/preferences";
import { TYPE_META } from "../columns/columnTypes";
import { closeCurrentDataset, useCurrentDataset } from "../datasets/current";
import { OPEN_FILTERS_EVENT } from "../filters/FilterBar";
import { useFilters } from "../filters/useFilters";
import { availableTools, themeName, toolName } from "../toolbox/tools";

type Command = { id: string; group: MessageKey; label: string; hint?: string; icon: LucideIcon; tone?: string; run: () => void };

/** What the palette needs of a registered module (098: every page and tool can be found). */
export type PaletteModule = {
  id: string;
  labelKey: MessageKey;
  icon: LucideIcon;
  path: string;
  scope: string;
  locked?: boolean;
  /** 107: listed only when the open dataset meets these (Vision: an image set). */
  requires?: Requirement[];
  onlyWhenMet?: boolean;
};

type CommandPaletteProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onShortcuts: () => void;
  modules?: PaletteModule[];
};

const MAX_RESULTS = 40;

/** Ctrl+K: go to a page, a variable or a dataset, or run an action, from the keyboard. */
export function CommandPalette({ open, onOpenChange, onShortcuts, modules = [] }: CommandPaletteProps) {
  const navigate = useNavigate();
  const { datasetId, routed } = useCurrentDataset();
  const datasets = useDatasets();
  const format = useDataset(datasetId ?? undefined).data?.format;
  const columns = useColumns(datasetId ?? "", Boolean(datasetId) && open);
  const filters = useFilters(datasetId ?? "");
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);

  const commands = useMemo<Command[]>(() => {
    const go = (to: string, params?: Record<string, string>, search?: Record<string, string>) => () =>
      void navigate({ to, params, ...(search ? { search } : {}) } as never);
    const list: Command[] = [
      { id: "home", group: "palette.pages", label: t("nav.home"), icon: House, run: go("/") },
    ];
    if (datasetId) {
      const pages: [string, MessageKey, LucideIcon][] = [
        ["/d/$datasetId", "nav.overview", LayoutGrid],
        ["/d/$datasetId/columns", "nav.columns", Columns3],
        ["/d/$datasetId/compare", "nav.compare", ScatterChart],
        ["/d/$datasetId/correlations", "nav.correlations", Grid3x3],
        ["/d/$datasetId/regression", "tab.regression", TrendingUp],
        ["/d/$datasetId/data", "nav.data", Rows3],
      ];
      for (const [to, label, icon] of pages) {
        list.push({ id: to, group: "palette.pages", label: t(label), icon, run: go(to, { datasetId }) });
      }
      // 098: the modules of the sidebar, the Toolbox and each of its tools
      const tools = availableTools(modules);
      const covered = new Set(tools.map((tool) => tool.module));
      for (const module of modules) {
        if (module.scope !== "dataset" || covered.has(module.id)) continue;
        if (module.onlyWhenMet && !(module.requires ?? []).every((need) => meets(need, format, []))) continue;
        list.push({
          id: `module:${module.id}`,
          group: "palette.pages",
          label: t(module.labelKey),
          hint: module.locked ? t("lite.locked") : undefined,
          icon: module.icon,
          run: go(module.path, { datasetId }),
        });
      }
      list.push({ id: "toolbox", group: "palette.pages", label: t("nav.toolbox"), icon: Toolbox, run: go("/d/$datasetId/toolbox", { datasetId }) });
      for (const tool of tools) {
        list.push({
          id: `tool:${tool.id}`,
          group: "palette.tools",
          label: t(toolName(tool)),
          hint: tool.locked ? t("lite.locked") : t(themeName(tool.theme)),
          icon: tool.icon,
          run: go(tool.to, { datasetId }, tool.search),
        });
      }
    }
    list.push(
      { id: "help", group: "palette.pages", label: t("nav.help"), icon: BookOpen, run: go("/help") },
      { id: "settings", group: "palette.pages", label: t("nav.settings"), icon: Settings, run: go("/settings") },
    );
    if (datasetId) {
      for (const column of columns.data ?? []) {
        if (column.excluded) continue;
        const meta = TYPE_META[column.type];
        list.push({
          id: `variable:${column.name}`,
          group: "palette.variables",
          label: column.name,
          hint: t(meta.label),
          icon: meta.icon,
          tone: meta.tone,
          run: go("/d/$datasetId/variable/$column", { datasetId, column: column.name }),
        });
      }
      list.push({
        id: "filters",
        group: "palette.actions",
        label: t("palette.openFilters"),
        hint: "F",
        icon: Filter,
        run: () => window.dispatchEvent(new Event(OPEN_FILTERS_EVENT)),
      });
      if (routed && filters.conditions.length) {
        list.push({ id: "clear", group: "palette.actions", label: t("palette.clearFilters"), icon: Eraser, run: filters.clear });
      }
      if (!routed) {
        list.push({ id: "close", group: "palette.actions", label: t("palette.closeDataset"), icon: X, run: closeCurrentDataset });
      }
    }
    for (const dataset of datasets.data ?? []) {
      if (dataset.id === datasetId || dataset.status !== "ready") continue;
      list.push({ id: `dataset:${dataset.id}`, group: "palette.datasets", label: dataset.name, icon: Database, run: go("/d/$datasetId", { datasetId: dataset.id }) });
    }
    const theme = (value: "light" | "dark" | "system") => () => savePreference(themePreference, value);
    list.push(
      { id: "theme:light", group: "palette.actions", label: t("palette.themeLight"), icon: Sun, run: theme("light") },
      { id: "theme:dark", group: "palette.actions", label: t("palette.themeDark"), icon: Moon, run: theme("dark") },
      { id: "theme:system", group: "palette.actions", label: t("palette.themeSystem"), icon: Monitor, run: theme("system") },
      { id: "shortcuts", group: "palette.actions", label: t("palette.shortcuts"), hint: "?", icon: Keyboard, run: onShortcuts },
    );
    return list;
  }, [navigate, datasetId, routed, modules, columns.data, datasets.data, filters, onShortcuts, format]);

  const needle = query.trim().toLowerCase();
  const results = (needle ? commands.filter((c) => c.label.toLowerCase().includes(needle)) : commands).slice(0, MAX_RESULTS);
  const selected = Math.min(active, Math.max(results.length - 1, 0));

  useEffect(() => {
    document.getElementById(`palette-${selected}`)?.scrollIntoView?.({ block: "nearest" });
  }, [selected]);

  const close = (next: boolean) => {
    onOpenChange(next);
    if (!next) {
      setQuery("");
      setActive(0);
    }
  };
  const runAt = (index: number) => {
    const command = results[index];
    if (!command) return;
    close(false);
    command.run();
  };

  return (
    <RadixDialog.Root open={open} onOpenChange={close}>
      <RadixDialog.Portal>
        <RadixDialog.Overlay className="fixed inset-0 z-40 bg-ink-1/30" />
        <RadixDialog.Content
          aria-describedby={undefined}
          className="fixed top-[14vh] left-1/2 z-50 flex max-h-[min(70dvh,560px)] w-[calc(100vw-32px)] max-w-xl -translate-x-1/2 flex-col overflow-hidden rounded-lg border border-line bg-surface-1 shadow-[0_12px_40px_-12px_rgb(0_0_0/0.35)]"
        >
          <RadixDialog.Title className="sr-only">{t("palette.title")}</RadixDialog.Title>
          <div className="flex items-center gap-2 border-b border-line px-3 focus-within:shadow-[inset_0_-2px_0_var(--accent)]">
            <Search size={15} strokeWidth={1.5} className="text-ink-3" aria-hidden />
            <input
              autoFocus
              role="combobox"
              aria-expanded
              aria-controls="palette-results"
              aria-activedescendant={results.length ? `palette-${selected}` : undefined}
              aria-label={t("palette.search")}
              placeholder={t("palette.placeholder")}
              value={query}
              onChange={(event) => {
                setQuery(event.target.value);
                setActive(0);
              }}
              onKeyDown={(event) => {
                if (event.key === "ArrowDown") {
                  event.preventDefault();
                  setActive((selected + 1) % Math.max(results.length, 1));
                } else if (event.key === "ArrowUp") {
                  event.preventDefault();
                  setActive((selected - 1 + results.length) % Math.max(results.length, 1));
                } else if (event.key === "Enter") {
                  event.preventDefault();
                  runAt(selected);
                }
              }}
              className="h-11 flex-1 bg-transparent text-sm text-ink-1 outline-none placeholder:text-ink-3"
            />
            <kbd className="rounded border border-line px-1.5 text-[11px] text-ink-3">Esc</kbd>
          </div>
          <ul id="palette-results" role="listbox" aria-label={t("palette.results")} className="min-h-0 flex-1 overflow-y-auto p-1.5">
            {results.length === 0 && <li className="px-3 py-6 text-center text-ink-3">{t("palette.none")}</li>}
            {results.map((command, index) => {
              const Icon = command.icon;
              const groupStart = index === 0 || results[index - 1]?.group !== command.group;
              return (
                <li key={command.id} role="presentation">
                  {groupStart && <p className="px-2.5 pt-2 pb-1 text-[11px] font-medium tracking-wide text-ink-3 uppercase">{t(command.group)}</p>}
                  <div
                    id={`palette-${index}`}
                    role="option"
                    aria-selected={index === selected}
                    onMouseMove={() => setActive(index)}
                    onClick={() => runAt(index)}
                    className={`flex cursor-default items-center gap-2.5 rounded-md px-2.5 py-1.5 text-[13px] ${index === selected ? "bg-surface-2 text-ink-1" : "text-ink-2"}`}
                  >
                    <Icon size={15} strokeWidth={1.5} className={command.tone ?? "text-ink-3"} aria-hidden />
                    <span className={`min-w-0 flex-1 truncate ${command.group === "palette.variables" ? "font-mono text-xs" : ""}`}>{command.label}</span>
                    {command.hint && <span className="text-xs text-ink-3">{command.hint}</span>}
                  </div>
                </li>
              );
            })}
          </ul>
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </RadixDialog.Root>
  );
}
