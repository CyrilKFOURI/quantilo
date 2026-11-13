// The engine's reasons in the interface language (M02). The engine sends each reason's English
// text with its template code and the vars it cites; the page writes `reason.<code>` with them.
// A reason without a code, a code without a text here, or an English template that no longer
// gives the engine's text (the engine changed): the English text, never a stale translation.
import { hasText, locale, t, type MessageKey } from "./index";
import englishReasons from "./reasons.json";
import englishRoles from "./roles.json";

/** A reason as the API sends it beside its English text. */
export type ReasonDetail = { text: string; code?: string | null; vars?: Record<string, string> | null };

const english: Partial<Record<string, string>> = englishReasons;
const roleKeys = new Map(Object.entries(englishRoles).map(([key, label]) => [label.toLowerCase(), key]));

// The vars that hold the engine's own English words (never a column's name or a value of the data):
// a column role, a word of the engine, a count of rows, words listed with commas, names joined by
// "and" / "or".
const ROLE_VARS = new Set(["role", "label"]);
const WORD_VARS = new Set(["typical", "requested", "coarser", "confidence", "how", "meaning", "words", "language", "what", "label"]);
const ROW_VARS = new Set(["rows", "total"]);
const SPLIT_VARS = new Set(["forms", "shares"]);
const LIST_VARS = new Set(["dates", "states", "columns"]);

export function fill(template: string, vars: Record<string, string>): string {
  return template.replace(/\{(\w+)\}/g, (match, name: string) => vars[name] ?? match);
}

/** The vars of a reason's `index`-th part: a var the part cites with its own value is `name@index`. */
function partVars(vars: Record<string, string>, index: number): Record<string, string> {
  const suffix = `@${index}`;
  const own = { ...vars };
  for (const [name, value] of Object.entries(vars)) if (name.endsWith(suffix)) own[name.slice(0, -suffix.length)] = value;
  return own;
}

/** The English template the engine said a part with (its singular when {count} is 1). */
function englishTemplate(key: string, vars: Record<string, string>): string | undefined {
  return (vars.count === "1" ? english[`${key}#one`] : undefined) ?? english[key];
}

function word(value: string): string | null {
  const key = `reason.word.${value}`;
  return hasText(key) ? t(key as MessageKey) : null;
}

function role(value: string): string | null {
  const key = roleKeys.get(value.toLowerCase()) ?? `role.${value}`;
  return hasText(key) ? t(key as MessageKey).toLocaleLowerCase(locale()) : null;
}

function piece(value: string): string {
  const currency = /^in (.+)$/.exec(value);
  if (currency) return t("reason.in", { currency: currency[1] ?? "" });
  const share = /^(\S+) (\d+%)$/.exec(value); // "French 55%"
  if (share) return `${word(share[1] ?? "") ?? share[1]} ${share[2]}`;
  return word(value) ?? value;
}

/** A value a reason cites, in the interface language when it is one of the engine's English words. */
function translated(name: string, value: string): string {
  if (ROLE_VARS.has(name)) {
    const found = role(value);
    if (found !== null) return found;
  }
  const whole = WORD_VARS.has(name) ? word(value) : null;
  if (whole !== null) return whole;
  const rows = ROW_VARS.has(name) ? /^([\d,]+) rows?$/.exec(value) : null;
  if (rows) return t("reason.unit.rows", { count: rows[1] ?? "" });
  if (SPLIT_VARS.has(name)) return value.split(", ").map(piece).join(", ");
  if (name === "countries") return value.replace(/ and (\d+) more$/, (_, count: string) => ` ${t("reason.and")} ${t("reason.more", { count })}`);
  if (LIST_VARS.has(name)) {
    return value
      .replaceAll(" and ", ` ${t("reason.and")} `)
      .replaceAll(" or ", ` ${t("reason.or")} `)
      .replace(/\((year|month|day|hour|minute)\)/g, (match) => word(match) ?? match);
  }
  return value;
}

/** A coded reason's parts, in English (as the engine filled them) and in the interface language;
 *  null when a part has no template here. */
function written(reason: ReasonDetail): { english: string; shown: string } | null {
  if (!reason.code) return null;
  const vars = reason.vars ?? {};
  let englishText = "";
  let shown = "";
  for (const [index, code] of reason.code.split("+").entries()) {
    const key = `reason.${code}`;
    const own = partVars(vars, index);
    const template = englishTemplate(key, own);
    if (template === undefined) return null;
    englishText += fill(template, own);
    shown += t(key as MessageKey, Object.fromEntries(Object.entries(own).map(([name, value]) => [name, translated(name, value)])));
  }
  return { english: englishText, shown };
}

/** A reason in the interface language; its English text when it cannot be written from its code. */
export function reasonText(reason: ReasonDetail): string {
  if (locale() === "en") return reason.text;
  const found = written(reason);
  return found && found.english === reason.text ? found.shown : reason.text;
}

/** Reasons sent as texts with, from a recent server, their details (same order). */
export function reasonTexts(texts: readonly string[], details?: readonly ReasonDetail[] | null): string[] {
  return texts.map((text, index) => reasonText(details?.[index]?.text === text ? details[index]! : { text }));
}

/** A column's detection reasons in the interface language. */
export function columnReasons(column: { reasons: string[]; reason_details?: ReasonDetail[] | null }): string[] {
  return reasonTexts(column.reasons, column.reason_details);
}

// How the catalogue's note an automatic fix's reason ends with begins (cleaning/catalogue.py).
const NOTE = "A reliable fix:";

type CleanReason = { reason: string; reason_code?: string | null; reason_vars?: Record<string, string> | null };

/** A Clean suggestion's reason in the interface language. `note`: for a fix applied without asking,
 *  the page's own sentence for why it is reliable, which the engine's reason ends with (N04). */
export function suggestionReason(suggestion: CleanReason, note: string | null = null): string {
  const detail = { text: suggestion.reason, code: suggestion.reason_code, vars: suggestion.reason_vars };
  if (note === null || locale() === "en") return reasonText(detail);
  const found = written(detail);
  if (!found) return suggestion.reason;
  // the engine drops the sentence's own "Check your data." and adds the catalogue's note
  const tail = english["reason.CHECK"] ?? "";
  const base = found.english.endsWith(tail) ? found.english.slice(0, -tail.length) : found.english;
  if (!suggestion.reason.startsWith(`${base.trimEnd()} ${NOTE}`)) return suggestion.reason;
  const check = t("reason.CHECK");
  const shown = found.shown.endsWith(check) ? found.shown.slice(0, -check.length) : found.shown;
  return `${shown.trimEnd()} ${note}${check}`;
}
