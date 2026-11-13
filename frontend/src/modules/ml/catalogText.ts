/** The model catalogue in the interface language (M01). The engine sends it in English with an id
 *  for every model and setting; each text is shown from `catalog.json` by that id when the English
 *  text there is the engine's (`engineText`), else as the engine wrote it. */

import type { MlCatalogue, MlModel, MlSetting } from "../../api/client";
import { engineText, hasText, t, variants } from "../../lib/i18n";
// Every key of a kind (badges, reasons…): their texts are matched one by one.
import catalog from "../../lib/i18n/catalog.json";

const slug = (text: string) =>
  text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_|_$/g, "");

/** The keys that may hold each kind of text, the engine's id first. */
export const catalogKeys = {
  modelName: (key: string) => [`ml.model.${key}.name`],
  description: (key: string) => [`ml.model.${key}.description`],
  family: (family: string) => [`ml.family.${slug(family)}`],
  group: (group: string) => [`ml.group.${slug(group)}`],
  settingLabel: (setting: string) => variants(`ml.setting.${setting}.label`),
  option: (setting: string, option: string) => variants(`ml.setting.${setting}.option.${option}`),
  why: (setting: string) => variants(`ml.setting.${setting}.why`),
  badge: () => prefixed("ml.badge."),
  recommended: () => prefixed("ml.recommend."),
  ruleReason: () => prefixed("ml.ruleReason."),
  limit: () => prefixed("ml.limit."),
};

const byPrefix = new Map<string, string[]>();
function prefixed(prefix: string): string[] {
  let found = byPrefix.get(prefix);
  if (!found) {
    found = Object.keys(catalog).filter((key) => key.startsWith(prefix));
    byPrefix.set(prefix, found);
  }
  return found;
}

export const modelName = (key: string, label: string) => engineText(catalogKeys.modelName(key), label);
export const familyName = (family: string) => engineText(catalogKeys.family(family), family);
export const groupName = (group: string) => engineText(catalogKeys.group(group), group);

/** An Auto entry's name: "Random forest · tuned" keeps its suffix in the interface language. */
export function entryName(model: string, label: string): string {
  const suffix = " · tuned";
  if (!label.endsWith(suffix)) return modelName(model, label);
  return hasText("ml.catalog.tuned") ? t("ml.catalog.tuned", { model: modelName(model, label.slice(0, -suffix.length)) }) : label;
}

function localizeSetting(setting: MlSetting): MlSetting {
  return {
    ...setting,
    label: engineText(catalogKeys.settingLabel(setting.key), setting.label),
    why: setting.why && engineText(catalogKeys.why(setting.key), setting.why),
    option_labels: Object.fromEntries(
      setting.options.map((option) => {
        const value = String(option);
        return [value, engineText(catalogKeys.option(setting.key, value), setting.option_labels[value] ?? value)];
      }),
    ),
  };
}

function localizeModel(model: MlModel): MlModel {
  return {
    ...model,
    label: modelName(model.key, model.label),
    family: familyName(model.family),
    description: engineText(catalogKeys.description(model.key), model.description),
    settings: model.settings.map(localizeSetting),
    rules: model.rules.map((rule) => ({ ...rule, reason: engineText(catalogKeys.ruleReason(), rule.reason) })),
    badges: model.badges.map((badge) => engineText(catalogKeys.badge(), badge)),
    unavailable: model.unavailable && engineText(catalogKeys.limit(), model.unavailable),
  };
}

/** The whole catalogue as the page shows it; the setting groups keep their English id (the page
 *  sorts by it) and are named with `groupName` where they are drawn. */
export function localizeCatalogue(catalogue: MlCatalogue): MlCatalogue {
  return {
    ...catalogue,
    families: catalogue.families.map(familyName),
    models: catalogue.models.map(localizeModel),
    recommended: catalogue.recommended.map((r) => ({ ...r, why: engineText(catalogKeys.recommended(), r.why) })),
  };
}
