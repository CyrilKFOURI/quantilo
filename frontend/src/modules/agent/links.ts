import type { AgentLink } from "./api";

/** Where an "Open in …" link goes: a route of the app and its parameters (R-AG-T5). */
export type LinkTarget =
  | { to: string; params: Record<string, string>; search?: Record<string, string> }
  /** A file to download (a report): a plain link to the app's own API. */
  | { href: string; download: true };

const ID = /^[\w-]{1,64}$/;

export function linkTarget(datasetId: string, link: AgentLink): LinkTarget | null {
  const p = link.params ?? {};
  switch (link.page) {
    case "overview":
      return { to: "/d/$datasetId", params: { datasetId } };
    case "columns":
      return { to: "/d/$datasetId/columns", params: { datasetId } };
    case "variable":
      return p.column ? { to: "/d/$datasetId/variable/$column", params: { datasetId, column: p.column } } : null;
    case "compare":
      return { to: "/d/$datasetId/compare", params: { datasetId }, search: p };
    case "correlations":
      return { to: "/d/$datasetId/correlations", params: { datasetId } };
    case "regression":
      return {
        to: "/d/$datasetId/regression",
        params: { datasetId },
        search: pick(p, ["target", "columns", "weights", "cluster", "se", "focus"]),
      };
    case "data":
      return { to: "/d/$datasetId/data", params: { datasetId } };
    case "ml":
      return { to: "/d/$datasetId/ml", params: { datasetId }, search: pick(p, ["target", "run", "prediction"]) };
    case "timeseries":
      return { to: "/d/$datasetId/timeseries", params: { datasetId }, search: pick(p, ["time", "value"]) };
    case "kpi":
      return { to: "/d/$datasetId/kpi", params: { datasetId }, search: pick(p, ["kpi"]) };
    case "experiments":
      return { to: "/d/$datasetId/experiments", params: { datasetId }, search: pick(p, ["variant", "metric", "unit", "tab"]) };
    case "segments":
      return { to: "/d/$datasetId/segments", params: { datasetId }, search: pick(p, ["section", "run", "unusual"]) };
    case "synthetic":
      return { to: "/d/$datasetId/synthetic", params: { datasetId }, search: pick(p, ["run"]) };
    case "clean":
      return { to: "/d/$datasetId/clean", params: { datasetId } };
    case "changes":
      return { to: "/d/$datasetId/changes", params: { datasetId }, search: pick(p, ["vs", "column"]) };
    case "recipes":
      return { to: "/recipes", params: {} };
    case "recipe":
      return p.recipe && ID.test(p.recipe)
        ? { to: "/recipes/$recipeId", params: { recipeId: p.recipe }, search: p.run && ID.test(p.run) ? { report: p.run } : {} }
        : null;
    case "dataset":
      return p.dataset && ID.test(p.dataset) ? { to: "/d/$datasetId", params: { datasetId: p.dataset } } : null;
    case "report_file":
      return p.job && ID.test(p.job) ? { href: `/api/v1/reports/jobs/${p.job}/file`, download: true } : null;
    default:
      return null;
  }
}

function pick(params: Record<string, string>, keys: string[]): Record<string, string> {
  return Object.fromEntries(keys.filter((key) => params[key]).map((key) => [key, params[key] as string]));
}

const PAGES: Record<string, string> = {
  "": "Overview",
  columns: "Columns",
  variable: "Variable page",
  compare: "Compare",
  correlations: "Correlations",
  regression: "Regression",
  data: "Data",
  changes: "Changes",
  timeseries: "Time series",
  ml: "Machine learning",
  segments: "Segments",
  kpi: "KPI builder",
  synthetic: "Synthetic data",
  experiments: "Experiments",
  clean: "Clean",
};

/** The page the user is on, in words, so "explain this chart" makes sense to the agent. */
export function describePage(pathname: string, search: Record<string, unknown>): string | null {
  const match = /^\/d\/[^/]+\/?([^/]*)\/?(.*)$/.exec(pathname);
  if (!match) return null;
  const [, section = "", rest = ""] = match;
  const page = PAGES[section] ?? section;
  const details: string[] = [];
  if (section === "variable" && rest) details.push(`column ${decodeURIComponent(rest)}`);
  for (const key of ["x", "y", "z", "color", "target", "columns", "time", "value", "variant", "metric", "tab"]) {
    const value = search[key];
    if (typeof value === "string" && value) details.push(`${key} = ${value}`);
  }
  return details.length ? `${page} (${details.join(", ")})` : page;
}

const APP_PATH = /^\/d\/[\w-]+(\/[\w%.~-]+)*\/?(\?[\w%.~=&-]*)?$/;

/** Model output can be steered by data it read (a column name, a category), so its links may only
 * point inside the app: a same-origin /d/… path, no scheme, no "//", no backslash. Checked where
 * the link is rendered (Markdown's href filter); anything else is shown as plain text. */
export function appHref(href: string): string | null {
  const target = href.trim();
  return APP_PATH.test(target) ? target : null;
}
