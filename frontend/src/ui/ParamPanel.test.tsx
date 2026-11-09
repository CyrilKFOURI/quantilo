import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ParamPanel } from "./ParamPanel";

describe("ParamPanel (041)", () => {
  it("sums the settings up in one line and opens them on demand", () => {
    render(
      <ParamPanel label="Series settings" summary={["date", "units", null, false, "Week"]}>
        <label>
          Time <select aria-label="Time" />
        </label>
      </ParamPanel>,
    );

    const toggle = screen.getByRole("button", { name: /date · units · Week/ });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByLabelText("Time")).not.toBeInTheDocument();

    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByLabelText("Time")).toBeInTheDocument();
  });

  it("starts open when asked (nothing chosen yet)", () => {
    render(
      <ParamPanel label="Variables" summary={["Choose X and Y"]} defaultOpen>
        <input aria-label="X" />
      </ParamPanel>,
    );
    expect(screen.getByLabelText("X")).toBeInTheDocument();
  });
});
