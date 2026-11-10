import {
  Calculator,
  FlaskConical,
  Grid3x3,
  Layers,
  LineChart,
  Scale,
  ScanSearch,
  ScatterChart,
  Shapes,
  TrendingUp,
  type LucideIcon,
} from "lucide-react";

import type { AnalysisType } from "../../api/client";
import type { MessageKey } from "../../lib/i18n";

/** 098: the manual tools, grouped by what they do. Each one opens empty (its column pickers
 *  only) from here, and prefilled from the analyses that link to it. */
export type ToolTheme = "time" | "experiments" | "statistics" | "rows";

export type Tool = {
  id: string;
  theme: ToolTheme;
  icon: LucideIcon;
  /** The page it opens, and the URL parameters that pick a tab or a section of that page. */
  to: `/d/$datasetId/${string}`;
  search?: Record<string, string>;
  /** The registered module the page belongs to: its lock in the browser version applies. */
  module?: string;
  /** Column types it needs; greyed out with `requiresKey` otherwise. */
  requires?: AnalysisType[];
  requiresKey?: MessageKey;
};

export const THEMES: ToolTheme[] = ["time", "experiments", "statistics", "rows"];

export const TOOLS: Tool[] = [
  {
    id: "timeseries",
    theme: "time",
    icon: LineChart,
    to: "/d/$datasetId/timeseries",
    module: "timeseries",
    requires: ["datetime"],
    requiresKey: "module.timeseries.needsDate",
  },
  { id: "abtest", theme: "experiments", icon: FlaskConical, to: "/d/$datasetId/experiments", module: "experiments" },
  { id: "plan", theme: "experiments", icon: Calculator, to: "/d/$datasetId/experiments", search: { tab: "plan" }, module: "experiments" },
  { id: "policies", theme: "experiments", icon: Scale, to: "/d/$datasetId/experiments", search: { tab: "policies" }, module: "experiments" },
  { id: "compare", theme: "statistics", icon: ScatterChart, to: "/d/$datasetId/compare" },
  { id: "correlations", theme: "statistics", icon: Grid3x3, to: "/d/$datasetId/correlations" },
  { id: "regression", theme: "statistics", icon: TrendingUp, to: "/d/$datasetId/regression" },
  { id: "structure", theme: "statistics", icon: Layers, to: "/d/$datasetId/correlations" },
  { id: "segments", theme: "rows", icon: Shapes, to: "/d/$datasetId/segments", module: "segments" },
  { id: "unusual", theme: "rows", icon: ScanSearch, to: "/d/$datasetId/segments", search: { section: "unusual" }, module: "segments" },
];

export const toolName = (tool: Tool) => `toolbox.tool.${tool.id}` as MessageKey;
export const toolAnswer = (tool: Tool) => `toolbox.tool.${tool.id}.answers` as MessageKey;
export const toolWhen = (tool: Tool) => `toolbox.tool.${tool.id}.when` as MessageKey;
export const themeName = (theme: ToolTheme) => `toolbox.theme.${theme}` as MessageKey;
export const themeLine = (theme: ToolTheme) => `toolbox.theme.${theme}.line` as MessageKey;

/** The tools of a build: a module that is not registered drops its tools; a module locked in the
 *  browser version locks them (they stay listed, with the lock). */
export function availableTools(modules: { id: string; locked?: boolean }[]): (Tool & { locked: boolean })[] {
  const byId = new Map(modules.map((module) => [module.id, module]));
  return TOOLS.filter((tool) => !tool.module || byId.has(tool.module)).map((tool) => ({
    ...tool,
    locked: Boolean(tool.module && byId.get(tool.module)?.locked),
  }));
}

/** Pages reached through the Toolbox only (no sidebar entry of their own): the Toolbox entry
 *  stays marked while one of them is open. */
const TOOLBOX_ONLY = /^\/d\/[^/]+\/(toolbox|timeseries|experiments)\/?$/;

export function inToolbox(pathname: string): boolean {
  return TOOLBOX_ONLY.test(pathname);
}
