import { describe, expect, it } from "vitest";

import engine from "../../api/engine-texts.json";
import { archetypeMessage, kpiName } from "../../modules/kpi/names";
import { catalogKeys, entryName } from "../../modules/ml/catalogText";
import { findEngineText, loadLocale, readTemplate } from ".";
import catalog from "./catalog.json";
import fr from "./fr/catalog.json";

// M01: every text of the engine's closed lists (dumped by `python -m app.engine_texts_dump`) is read
// back exactly by catalog.json in English and has its French text.
type Case = { where: string; keys: string[]; text: string };
const cases: Case[] = [];
const add = (where: string, keys: string[], text: string) => cases.push({ where, keys, text });

const ml = engine.ml;
for (const family of [...ml.families, ...ml.auto.families]) add(`family ${family}`, catalogKeys.family(family), family);
for (const group of ml.groups) add(`group ${group}`, catalogKeys.group(group), group);
for (const text of ml.badges) add(`badge ${text}`, catalogKeys.badge(), text);
for (const text of ml.limits) add(`limit ${text}`, catalogKeys.limit(), text);
for (const text of ml.recommended) add(`recommendation ${text}`, catalogKeys.recommended(), text);
for (const [key, label] of Object.entries(ml.auto.labels)) add(`auto ${key}`, catalogKeys.modelName(key), label);
for (const [key, model] of Object.entries(ml.models)) {
  add(`${key} name`, catalogKeys.modelName(key), model.label);
  add(`${key} description`, catalogKeys.description(key), model.description);
  for (const reason of model.rules) add(`${key} rule`, catalogKeys.ruleReason(), reason);
  for (const [name, setting] of Object.entries(model.settings)) {
    add(`${key}.${name} label`, catalogKeys.settingLabel(name), setting.label);
    for (const [option, label] of Object.entries(setting.options)) add(`${key}.${name} option ${option}`, catalogKeys.option(name, option), label);
    for (const why of setting.whys) add(`${key}.${name} why`, catalogKeys.why(name), why);
  }
}

const kpi = engine.kpi;
const english: Record<string, string> = catalog;
const french: Record<string, string> = fr;
const pattern = (text: string) => text.replace(/\{\w+\}/g, "{}");

describe("engine texts in English", () => {
  it("reads back every text of the model catalogue", () => {
    for (const { where, keys, text } of cases) expect(findEngineText(keys, text), where).toBe(text);
  });

  it("names every kind of dataset and every KPI rule, with the engine's patterns", () => {
    for (const [id, label] of Object.entries(kpi.archetypes)) expect(english[`kpi.archetype.kind.${id}`], id).toBe(label);
    expect(pattern(english["kpi.archetype.looksLike"] ?? "")).toBe(kpi.message);
    for (const [rule, name] of Object.entries(kpi.names)) {
      if (!name) continue; // named from its measure: the agg patterns below
      const text = english[`kpi.suggestName.rule.${rule}`];
      expect(text, rule).toBeDefined();
      if (name !== "{}") expect(pattern(text ?? ""), rule).toBe(name);
    }
    for (const [agg, words] of Object.entries(kpi.aggs)) {
      if (agg !== "share") expect(pattern(english[`kpi.suggestName.agg.${agg}`] ?? ""), agg).toBe(words);
    }
    for (const [agg, word] of Object.entries(kpi.delays)) {
      expect(english[`kpi.suggestName.delay.${agg}`], agg).toBe(`${word} days {column} → {until}`);
    }
  });

  it("reads a template's values and refuses another text", () => {
    expect(readTemplate("√n bounded to 5 to 50: k = {k}.", "√n bounded to 5 to 50: k = 15.")).toEqual({ k: "15" });
    expect(readTemplate("Slow on {rows} rows", "Slow on 200,000 rows")).toEqual({ rows: "200,000" });
    expect(readTemplate("Total {column}", "Average sales")).toBeNull();
  });
});

describe("engine texts in French", () => {
  it("has a French text for every engine text", async () => {
    await loadLocale("fr");
    for (const { where, keys, text } of cases) expect(findEngineText(keys, text), where).not.toBeNull();
    for (const key of Object.keys(english)) expect(french[key], key).toBeDefined();
  });

  it("writes model names, settings and Auto entries in French", async () => {
    await loadLocale("fr");
    expect(findEngineText(catalogKeys.modelName("logistic_regression"), "Logistic regression")).toBe("Régression logistique");
    expect(findEngineText(catalogKeys.option("class_weight", "balanced"), "balanced")).toBe("équilibré");
    expect(findEngineText(catalogKeys.why("n_neighbors"), "√n bounded to 5 to 50: k = 14.")).toBe("√n borné entre 5 et 50 : k = 14.");
    expect(entryName("random_forest", "Random forest · tuned")).toBe("Forêt aléatoire · optimisé");
    // a text the engine changed keeps its English instead of a stale translation
    expect(findEngineText(catalogKeys.modelName("logistic_regression"), "Logistic regression v2")).toBeNull();
  });

  it("writes suggested KPI names and the kind of dataset in French", async () => {
    await loadLocale("fr");
    const sum = { measures: [{ agg: "sum" as const, key: null, until: null }] };
    expect(kpiName({ rule: "R-SUG-02", name: "Total income", definition: sum })).toBe("Total d'income");
    expect(kpiName({ rule: "R-SUG-12", name: "Average basket", definition: sum })).toBe("Panier moyen");
    expect(kpiName({ rule: "R-SUG-15", name: "Income per customer", definition: sum })).toBe("Income par client");
    expect(kpiName({ rule: "R-SUG-02", name: "My own name", definition: sum })).toBe("My own name");
    for (const [id, label] of Object.entries(kpi.archetypes)) {
      const message = kpi.message.replace("{}", label).replace("{}", "5");
      expect(archetypeMessage({ id, label, message }), id).toMatch(/^On dirait des données .* · 5 KPI$/);
    }
  });
});
