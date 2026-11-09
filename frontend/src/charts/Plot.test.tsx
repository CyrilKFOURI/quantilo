import { render, screen } from "@testing-library/react";
import type { Data } from "plotly.js-dist-min";
import { afterEach, describe, expect, it, vi } from "vitest";

import { isThreeD, Plot } from "./Plot";

vi.mock("plotly.js-dist-min", () => {
  const plotly = { react: vi.fn(() => Promise.resolve()), purge: vi.fn() };
  return { default: plotly, ...plotly };
});

function touchScreen(touch: boolean) {
  vi.stubGlobal("matchMedia", (query: string) => ({
    matches: touch && query === "(pointer: coarse)",
    addEventListener: () => {},
    removeEventListener: () => {},
  }));
}

afterEach(() => vi.unstubAllGlobals());

const figure = () => ({ data: [{ type: "bar", x: [1], y: [2] } as Data] });

describe("Charts on a touch screen (044)", () => {
  it("knows a 3D chart, which keeps turning under the finger", () => {
    expect(isThreeD([{ type: "scatter3d" } as Data])).toBe(true);
    expect(isThreeD([{ type: "bar" } as Data, { type: "scatter" } as Data])).toBe(false);
  });

  it("offers full screen on a touch screen, not with a mouse", () => {
    touchScreen(true);
    const { unmount } = render(<Plot figure={figure} height={200} label="Sales" />);
    expect(screen.getByRole("button", { name: "Open full screen to zoom" })).toBeInTheDocument();
    unmount();

    touchScreen(false);
    render(<Plot figure={figure} height={200} label="Sales" />);
    expect(screen.queryByRole("button", { name: "Open full screen to zoom" })).not.toBeInTheDocument();
  });

  it("adds no button where the card already opens full screen", () => {
    touchScreen(true);
    render(<Plot figure={figure} height={200} label="Sales" zoom="none" />);
    expect(screen.queryByRole("button", { name: "Open full screen to zoom" })).not.toBeInTheDocument();
  });
});
