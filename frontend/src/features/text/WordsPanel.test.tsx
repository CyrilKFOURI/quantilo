import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { DistinctiveWords } from "../../api/client";
import { WordsPanel } from "./WordsPanel";

const words: DistinctiveWords = {
  column: "review",
  by: "rating",
  texts: 1200,
  sampled: false,
  groups_left: 0,
  sentence: "“refund” marks rating = 1: 40% of its texts use it, 2% of the others.",
  rules: ["R-WRD-01: words of 2 letters or more"],
  groups: [
    {
      group: "1",
      texts: 300,
      note: null,
      words: [{ word: "refund", score: 12.5, share: 0.4, elsewhere: 0.02, texts: 120 }],
    },
    { group: "5", texts: 15, note: "15 texts: fewer than 20", words: [] },
  ],
};

describe("WordsPanel", () => {
  it("shows the conclusion, a card per group and why a group has no words", () => {
    render(<WordsPanel words={words} />);
    expect(screen.getByText(words.sentence)).toBeTruthy();
    expect(screen.getByText("refund")).toBeTruthy();
    expect(screen.getByText("40%")).toBeTruthy();
    expect(screen.getByText("15 texts: fewer than 20")).toBeTruthy();
  });
});
