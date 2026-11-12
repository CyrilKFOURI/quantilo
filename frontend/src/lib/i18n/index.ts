import en from "./en.json";
// Recipes (030): their texts kept in their own file.
import recipes from "./recipes.json";
// Clean (065): its texts in their own file too.
import clean from "./clean.json";
// Model of a project (088): its texts in their own file too.
import model from "./model.json";
// Pinned rows (099): shared by Data, Unusual rows and Clean.
import pins from "./pins.json";

const all = { ...en, ...recipes, ...clean, ...model, ...pins };

export type MessageKey = keyof typeof all;

const messages: Partial<Record<string, string>> = all;

/** Translate a key, replacing `{name}` placeholders with `vars.name`. A key built at run time
 *  (`ml.option.${value}`) that has no text shows the key itself instead of breaking the page. */
export function t(key: MessageKey, vars: Record<string, string> = {}): string {
  const text = messages[key];
  if (text === undefined) {
    if (import.meta.env.DEV) console.warn(`Missing text for "${key}"`);
    return key;
  }
  return text.replace(/\{(\w+)\}/g, (match, name: string) => vars[name] ?? match);
}
