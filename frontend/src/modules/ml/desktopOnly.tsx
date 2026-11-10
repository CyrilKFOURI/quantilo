import type { MlCatalogue, MlModel } from "../../api/client";
import { t } from "../../lib/i18n";

/** 102: the models this build offers. In the browser version the heavy ones (XGBoost, LightGBM,
 *  CatBoost) are left to the desktop app: not listed, said once (R-WEB-04). */
export function offered(models: MlModel[]): MlModel[] {
  return models.filter((model) => !model.desktop_only);
}

/** "a, b and c" ("" for none). */
function listing(items: string[]): string {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} and ${items.at(-1)}`;
}

/** The labels of the models left to the desktop app. */
export function desktopOnlyLabels(models: MlModel[]): string[] {
  return models.filter((model) => model.desktop_only).map((model) => model.label);
}

/** One line under a list of models: what is not listed, and where it is. */
export function DesktopOnlyModels({ models, inset = true }: { models: MlModel[]; inset?: boolean }) {
  const labels = desktopOnlyLabels(models);
  if (!labels.length) return null;
  return (
    <p className={inset ? "px-3 text-xs text-ink-3" : "text-xs text-ink-3"}>
      {t("ml.desktopOnly.models", { models: listing(labels) })}
    </p>
  );
}

/** The page's one line in the browser version: what runs here, what is left to the desktop app. */
export function BrowserNote({ catalogue }: { catalogue: MlCatalogue }) {
  const limits = catalogue.browser;
  if (!limits) return null;
  const left = [...desktopOnlyLabels(catalogue.models), ...(limits.tuning ? [] : [t("ml.desktopOnly.tuning")])];
  return (
    <p className="text-xs text-ink-3">
      {t("ml.desktopOnly.page", { rows: limits.max_rows.toLocaleString("en-US") })}
      {left.length > 0 && ` ${t("ml.desktopOnly.left", { items: listing(left) })}`}
    </p>
  );
}
