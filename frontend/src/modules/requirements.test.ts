import { describe, expect, it } from "vitest";

import { meets } from "./types";

describe("page requirements (106)", () => {
  it("reads 'images' from the dataset's format and column types from the columns", () => {
    const columns = [
      { type: "datetime" as const, excluded: false },
      { type: "text" as const, excluded: true },
    ];
    expect(meets("images", "images", [])).toBe(true);
    expect(meets("images", "csv", columns)).toBe(false);
    expect(meets("datetime", "csv", columns)).toBe(true);
    expect(meets("text", "csv", columns)).toBe(false);
  });
});
