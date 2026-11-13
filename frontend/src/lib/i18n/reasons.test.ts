import { afterAll, beforeAll, describe, expect, it } from "vitest";

import engine from "../../api/engine-texts.json";
import examples from "../../api/reason-examples.json";
import { loadLocale } from ".";
import fr from "./fr/reasons.json";
import en from "./reasons.json";
import { columnReasons, fill, reasonText, reasonTexts, suggestionReason } from "./reasons";

// M02: every reason template the engine says (dumped by `python -m app.engine_texts_dump`) has its
// English copy, word for word, and its French text with the same placeholders.
const templates: Record<string, { template: string; vars: string[] }> = engine.reasons;
const english: Partial<Record<string, string>> = en;
const french: Partial<Record<string, string>> = fr;
const placeholders = (text: string) => [...new Set([...text.matchAll(/\{(\w+)\}/g)].map((match) => match[1]))].sort();
const CODES = Object.keys(templates);
// French expected texts below are written with plain spaces: the no-break ones before : ; ? !
const nb = (text: string) => text.replace(/ ([:;?!])/g, " $1");

describe("reason templates", () => {
  it.each(CODES)("%s: the English copy is the engine's, the French has the same vars", (code) => {
    const { template, vars } = templates[code]!;
    const key = `reason.${code}`;
    expect(english[key]).toBe(template);
    expect(placeholders(template)).toEqual([...vars].sort());
    const text = french[key];
    expect(text, `${key} has no French text`).toBeDefined();
    expect(placeholders(text ?? "")).toEqual(placeholders(template));
    // a part said after another starts with the same space; French typography: no-break spaces
    expect(text?.startsWith(" "), key).toBe(template.startsWith(" "));
    expect(text, key).not.toMatch(/ [:;?!]|« | »|\bniche\b/i);
    if (template.includes("{count}") && !code.endsWith("#one")) expect(french[`${key}#one`], `${key}#one`).toBeDefined();
  });

  it("has no English text for a code the engine no longer says, and singulars keep their base's vars", () => {
    for (const key of Object.keys(en)) {
      const code = key.slice("reason.".length).replace(/#one$/, "");
      if (/^(word|unit)\.|^(and|or|in|more)$/.test(code)) continue;
      expect(templates[code], key).toBeDefined();
      if (key.endsWith("#one")) {
        expect(placeholders(english[key] ?? "")).toEqual(placeholders(english[`reason.${code}`] ?? ""));
        // the engine's text for a count of 1: its own singular, else the plural template
        expect(english[key]).toBe(templates[`${code}#one`]?.template ?? templates[code]?.template);
      }
    }
    for (const key of Object.keys(fr)) expect(english[key], key).toBeDefined();
  });
});

type Example = (typeof examples)[number];
const detail = (example: Example) => ({ text: example.text, code: example.code, vars: example.vars as Record<string, string> });

describe("reasons shown on dirty_traps and messy_orders", () => {
  it("are their English templates filled with their vars", () => {
    for (const example of examples) {
      const parts = (example.code ?? "").split("+");
      let text = "";
      for (const [index, code] of parts.entries()) {
        const own: Record<string, string> = { ...(example.vars as Record<string, string>) };
        for (const [name, value] of Object.entries(own)) if (name.endsWith(`@${index}`)) own[name.slice(0, -`@${index}`.length)] = value;
        text += fill(english[`reason.${code}${own.count === "1" && english[`reason.${code}#one`] ? "#one" : ""}`] ?? "", own);
      }
      const reason = example.text;
      // an automatic fix's reason ends with the catalogue's note (N04) after its own sentence
      const noted = "measured" in example && example.measured !== null;
      if (noted) expect(reason.startsWith(text.replace(/ Check your data\.$/, "").trimEnd()), reason).toBe(true);
      else expect(text, example.code ?? "").toBe(reason);
    }
  });

  describe("in French", () => {
    beforeAll(async () => {
      await loadLocale("fr");
    });
    afterAll(async () => {
      await loadLocale("en");
    });

    it("are all written in French", () => {
      for (const example of examples) {
        const noted = "measured" in example && example.measured !== null;
        const shown = example.place === "clean" ? suggestionReason({ reason: example.text, reason_code: example.code, reason_vars: example.vars as Record<string, string> }, noted ? "Une correction fiable." : null) : reasonText(detail(example));
        expect(shown, example.text).not.toBe(example.text);
        expect(shown, example.text).not.toMatch(/ [:;?!](\s|$)/);
      }
    });

    it("writes the cited words in French", () => {
      expect(reasonText({ text: "name suggests customer key", code: "TYP.name", vars: { role: "customer key" } })).toBe(nb("le nom indique : clé client"));
      expect(reasonText({ text: "13 distinct values, e.g. a, b", code: "TYP.distinct", vars: { count: "13", examples: "a, b" } })).toBe("13 valeurs distinctes, par ex. a, b");
      const one = reasonText({ text: "1 row without a date is left out.", code: "R-TIME-04.undated", vars: { count: "1" } });
      expect(one).toBe("1 ligne sans date est laissée de côté.");
      const grain = { text: "The dates come about once a month: shown by month instead of by day (R-TIME-02).", code: "R-TIME-02", vars: { typical: "month", requested: "day" } };
      expect(reasonText(grain)).toBe(nb("Les dates arrivent environ une fois par mois : affiché par mois au lieu de par jour (R-TIME-02)."));
    });

    it("writes each part of a reason said in parts, a part's own vars first", () => {
      const text = "R-TIME-04 · 3 rows left out of the time axis: 1 row without a date; 2 rows dated 9999-12-31 (a placeholder meaning no end date, not a real date). Clean can read the placeholders as missing.";
      const vars = { total: "3 rows", rows: "1 row", "rows@3": "2 rows", day: "9999-12-31", meaning: "no end date" };
      const code = "R-TIME-04+R-TIME-04.part_undated+SEP.semicolon+R-TIME-04.part_placeholder+R-TIME-04.end";
      expect(reasonText({ text, code, vars })).toBe(
        nb("R-TIME-04 · 3 lignes hors de l'axe du temps : 1 ligne sans date ; 2 lignes à la date 9999-12-31 (un bouche-trou qui signifie aucune date de fin, pas une vraie date). Nettoyer peut lire les bouche-trous comme manquants."),
      );
    });

    it("keeps the English text when the template no longer gives it, or the code is unknown", () => {
      expect(reasonText({ text: "14 distinct values, e.g. a", code: "TYP.distinct", vars: { count: "13", examples: "a" } })).toBe("14 distinct values, e.g. a");
      expect(reasonText({ text: "something new", code: "TYP.unknown_rule", vars: {} })).toBe("something new");
      expect(reasonText({ text: "an older server", code: null })).toBe("an older server");
      // reasons without details (an older server), or details that are not these reasons
      expect(reasonTexts(["stored as a date"], undefined)).toEqual(["stored as a date"]);
      expect(columnReasons({ reasons: ["stored as a date"], reason_details: [{ text: "other", code: "TYP.stored_date", vars: {} }] })).toEqual(["stored as a date"]);
      expect(columnReasons({ reasons: ["stored as a date"], reason_details: [{ text: "stored as a date", code: "TYP.stored_date", vars: {} }] })).toEqual(["stocké comme une date"]);
    });

    it("ends an automatic fix's reason with the page's note", () => {
      const base = "3 values had extra spaces (at the ends, doubled, tabs or no-break spaces).";
      const reason = `${base} A reliable fix: this rule was right on 99% or more of the cases measured (n = 1,234). Check your data.`;
      const suggestion = { reason, reason_code: "B-01", reason_vars: { count: "3" } };
      expect(suggestionReason(suggestion, "Une correction fiable.")).toBe(
        "3 valeurs avaient des espaces en trop (aux extrémités, doublés, tabulations ou espaces insécables). Une correction fiable. Vérifiez vos données.",
      );
      expect(suggestionReason({ ...suggestion, reason: `${base} Read back from: x.` }, "Une correction fiable.")).toBe(`${base} Read back from: x.`);
    });
  });
});
