import { describe, expect, it } from "vitest";

import { resolveLocale, t, type MessageKey } from ".";
import en from "./en.json";

describe("t", () => {
  it("fills placeholders", () => {
    expect(t("ml.compare.selected", { count: "3" })).toBe("3 models");
  });

  it("shows a missing key instead of breaking the page", () => {
    expect(t("ml.option.unknown_value" as MessageKey)).toBe("ml.option.unknown_value");
  });
});

describe("resolveLocale", () => {
  it("follows the browser on auto and keeps an explicit choice", () => {
    expect(resolveLocale("auto", ["fr-FR", "en"])).toBe("fr");
    expect(resolveLocale("auto", ["en-GB", "fr"])).toBe("en");
    expect(resolveLocale("auto", [])).toBe("en");
    expect(resolveLocale("en", ["fr-FR"])).toBe("en");
  });
});

// Every French text (127) must answer an English key and keep its {placeholders}.
const english: Record<string, string> = {
  ...en,
  ...Object.assign({}, ...Object.values(import.meta.glob<Record<string, string>>("./*.json", { eager: true, import: "default" }))),
};
const french = import.meta.glob<Record<string, string>>("./fr/*.json", { eager: true, import: "default" });
const placeholders = (text: string) => [...text.matchAll(/\{(\w+)\}/g)].map((match) => match[1]).sort();

describe("French texts", () => {
  for (const [file, texts] of Object.entries(french)) {
    it(`${file} matches the English keys and placeholders`, () => {
      for (const [key, text] of Object.entries(texts)) {
        expect(english[key], `${file}: ${key} has no English text`).toBeDefined();
        expect(placeholders(text), `${file}: ${key}`).toEqual(placeholders(english[key] ?? ""));
        expect(text, `${file}: ${key}`).not.toMatch(/\bniche\b/i);
      }
    });
  }
});

describe("singular", () => {
  it("reads the #one form for a count of 1 only", () => {
    expect(t("clean.preview.rows", { count: "1" })).toBe("1 row removed");
    expect(t("clean.preview.rows", { count: "2" })).toBe("2 rows removed");
  });

  it("keeps the placeholders of the base key in every plural.json entry", async () => {
    const singulars = (await import("./plural.json")).default as Record<string, string>;
    const frenchSingulars = (await import("./fr/plural.json")).default as Record<string, string>;
    expect(Object.keys(frenchSingulars).sort()).toEqual(Object.keys(singulars).sort());
    for (const [key, text] of Object.entries(singulars)) {
      expect(key.endsWith("#one"), key).toBe(true);
      const base = english[key.slice(0, -"#one".length)];
      expect(base, `${key} has no base key`).toBeDefined();
      expect(placeholders(text), key).toEqual(placeholders(base ?? ""));
    }
  });
});
