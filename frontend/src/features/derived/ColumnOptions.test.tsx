import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { Column } from "../../api/client";
import { ColumnOptions } from "./ColumnOptions";

const column = (name: string, derivedFrom?: string): Column =>
  ({
    name,
    position: 0,
    type: derivedFrom ? "numeric_discrete" : "text",
    parser: derivedFrom
      ? { kind: "derived", derived: { source: derivedFrom, part: "length", option: null, topics: [] } }
      : { kind: "text" },
    reasons: derivedFrom ? [`derived from '${derivedFrom}': number of characters (R-DER-01)`] : [],
  }) as unknown as Column;

describe("ColumnOptions", () => {
  it("groups each source's derived columns under it, with their rule as a tooltip (R-DER-08)", () => {
    const columns = [column("review"), column("review › Length", "review"), column("review › Words", "review"), column("rating")];
    const { container } = render(
      <select>
        <ColumnOptions columns={columns} unavailable={(c) => (c.name === "rating" ? "not a text" : null)} />
      </select>,
    );
    const group = container.querySelector("optgroup");
    expect(group?.getAttribute("label")).toBe("Derived from review");
    expect([...(group?.querySelectorAll("option") ?? [])].map((o) => o.value)).toEqual(["review › Length", "review › Words"]);
    expect(group?.querySelector("option")?.getAttribute("title")).toContain("(R-DER-01). Check that your data was read correctly.");
    const rating = container.querySelector<HTMLOptionElement>('option[value="rating"]');
    expect(rating?.disabled).toBe(true);
    expect(rating?.textContent).toBe("rating (not a text)");
    expect(container.querySelectorAll("select > option")).toHaveLength(2);
  });
});
