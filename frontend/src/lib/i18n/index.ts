import en from "./en.json";
// Recipes (030): their texts kept in their own file.
import recipes from "./recipes.json";
// Clean (065): its texts in their own file too.
import clean from "./clean.json";
// Model of a project (088): its texts in their own file too.
import model from "./model.json";
// Pinned rows (099): shared by Data, Unusual rows and Clean.
import pins from "./pins.json";
// Vision (107, 108): its texts in their own file too (the import step keeps vision.import.* in en.json).
import vision from "./vision.json";
// Vision Model and Results tabs (113).
import visionModel from "./vision-model.json";
// Vision Map (109) and Prepare (110), with the generic BeforeAfter: their own file, beside vision.json.
import visionViews from "./visionViews.json";
// Vision explanations (117): "Why?" and the Explain tab.
import visionExplain from "./visionExplain.json";
// Vision Advanced › More (118): fine-tuning, ProtoPNet-lite, concept bottleneck, TCAV, concepts.
import visionMore from "./visionMore.json";
// Vision edits (c3): labels, pins and thresholds; regression, multi-label and the extensions.
import visionEdits from "./visionEdits.json";
// Chart titles, series and hover texts, filter words and the lite engine's steps (127).
import figures from "./figures.json";
// Column roles (127): the engine sends an English label; the page shows it by role id.
import roles from "./roles.json";
// Rows shared with the assistant (026, R-AG-P2): its texts in their own file, French in fr/rows.json.
import rows from "./rows.json";
// The engine's closed lists (M01): model catalogue, KPI suggestion names, kinds of datasets.
import catalog from "./catalog.json";
// AI forecasting models (028 phase B): their names, states, downloads and Settings › Models.
import fcAi from "./fcAi.json";
// Clean's encodings, synonyms and table rules (audit N28, N30, N32), French in fr/clean-misc.json.
import cleanMisc from "./cleanMisc.json";
// Singular forms ("<key>#one", read when {count} is 1), French in fr/plural.json.
import plural from "./plural.json";
// The engine's reasons (M02): their English templates by code, French in fr/reasons.json; see reasons.ts.
import reasons from "./reasons.json";

const all = { ...en, ...recipes, ...clean, ...model, ...pins, ...vision, ...visionViews, ...visionModel, ...visionExplain, ...visionMore, ...visionEdits, ...figures, ...roles, ...rows, ...catalog, ...fcAi, ...cleanMisc, ...plural, ...reasons };

export type MessageKey = keyof typeof all;

const messages: Partial<Record<string, string>> = { ...all };

export type Locale = "en" | "fr";

let current: Locale = "en";

/** The language the texts are shown in (English until `loadLocale` brings another one). */
export function locale(): Locale {
  return current;
}

/** "auto" follows the browser: French when its first language is French, English otherwise. */
export function resolveLocale(choice: "auto" | Locale, languages: readonly string[] = navigator.languages ?? []): Locale {
  if (choice !== "auto") return choice;
  return languages[0]?.toLowerCase().startsWith("fr") ? "fr" : "en";
}

/** French texts (127) live in `fr/*.json`, read only when French is chosen, so the English bundle
 *  stays as it was. A key the French files miss keeps its English text. Call before the first render. */
export async function loadLocale(choice: "auto" | Locale): Promise<Locale> {
  const target = resolveLocale(choice);
  if (target === "fr") {
    const files = import.meta.glob<Record<string, string>>("./fr/*.json", { import: "default" });
    const parts = await Promise.all(Object.values(files).map((load) => load()));
    for (const part of parts) Object.assign(messages, part);
  }
  current = target;
  document.documentElement.lang = target;
  return target;
}

/** Whether a key built at run time has a text (`role.${id}`), so a caller can fall back to its own. */
export function hasText(key: string): boolean {
  return messages[key] !== undefined;
}

/** A column role's name in the interface language; the engine's English label when the id is unknown. */
export function roleName(id: string | null | undefined, label: string): string {
  const key = `role.${id ?? ""}`;
  return id && hasText(key) ? t(key as MessageKey) : label;
}

const englishCatalog: Partial<Record<string, string>> = catalog;
const templates = new Map<string, RegExp>();

/** The `{name}` values that make an English template read `text` ("k = {k}." against "k = 15."). */
export function readTemplate(template: string, text: string): Record<string, string> | null {
  if (!template.includes("{")) return template === text ? {} : null;
  const names = [...template.matchAll(/\{(\w+)\}/g)].map((match) => match[1] ?? "");
  let pattern = templates.get(template);
  if (!pattern) {
    const parts = template.split(/\{\w+\}/).map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    pattern = new RegExp(`^${parts.join("(.+?)")}$`, "s");
    templates.set(template, pattern);
  }
  const match = pattern.exec(text);
  return match ? Object.fromEntries(names.map((name, i) => [name, match[i + 1] ?? ""])) : null;
}

/** The catalogue keys `base`, `base#2`, `base#3`… (several English texts for one id). */
export function variants(base: string): string[] {
  const found: string[] = [];
  for (let i = 1; englishCatalog[i === 1 ? base : `${base}#${i}`] !== undefined; i++) found.push(i === 1 ? base : `${base}#${i}`);
  return found;
}

/** An English text of the engine in the interface language (M01): the first of `keys` whose English
 *  text (catalog.json) is exactly `text`, then the first whose template reads it; null when none
 *  does, so a text the engine changed never shows a stale translation. */
export function findEngineText(keys: readonly string[], text: string): string | null {
  for (const key of keys) if (englishCatalog[key] === text && hasText(key)) return t(key as MessageKey);
  for (const key of keys) {
    const english = englishCatalog[key];
    if (english === undefined || !english.includes("{") || !hasText(key)) continue;
    const vars = readTemplate(english, text);
    if (vars) return t(key as MessageKey, vars);
  }
  return null;
}

/** `findEngineText`, or the engine's English text when no key reads it. */
export function engineText(keys: readonly string[], text: string): string {
  return findEngineText(keys, text) ?? text;
}

/** Translate a key (with `{count}` of 1, its `#one` form when there is one), replacing `{name}` placeholders with `vars.name`. A key built at run time
 *  (`ml.option.${value}`) that has no text shows the key itself instead of breaking the page. */
export function t(key: MessageKey, vars: Record<string, string> = {}): string {
  // A count of exactly 1 reads the key's singular form ("…#one") when it has one.
  const text = (vars.count === "1" ? messages[`${key}#one`] : undefined) ?? messages[key];
  if (text === undefined) {
    if (import.meta.env.DEV) console.warn(`Missing text for "${key}"`);
    return key;
  }
  return text.replace(/\{(\w+)\}/g, (match, name: string) => vars[name] ?? match);
}
