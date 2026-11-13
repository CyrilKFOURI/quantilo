import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { BeforeAfter, clampView, differencePixels, placement } from "./BeforeAfter";
import { parseColor, rampTable, sequentialStops } from "./colorRamp";

const A = "data:image/png;base64,AAAA";
const B = "data:image/png;base64,BBBB";
const C = "data:image/png;base64,CCCC";

const two = [
  { src: A, label: "Before", alt: "cat before", caption: <span>224 × 224 px</span> },
  { src: B, label: "After", alt: "cat after" },
] as const;

describe("BeforeAfter (110)", () => {
  it("shows the two images side by side with an arrow and the captions", () => {
    render(<BeforeAfter label="cat: before and after" panels={two} />);
    const group = screen.getByRole("group", { name: "cat: before and after" });
    expect(within(group).getByAltText("cat before")).toBeInTheDocument();
    expect(within(group).getByAltText("cat after")).toBeInTheDocument();
    expect(within(group).getByText("224 × 224 px")).toBeInTheDocument();
    expect(within(group).getByRole("radio", { name: "Side by side" })).toHaveAttribute("aria-checked", "true");
  });

  it("switches modes with the keys: slider moves with the arrows, toggle with Space", () => {
    render(<BeforeAfter label="cmp" panels={two} />);
    const group = screen.getByRole("group", { name: "cmp" });
    fireEvent.keyDown(group, { key: "2" });
    const handle = screen.getByRole("slider", { name: "Reveal" });
    expect(handle).toHaveAttribute("aria-valuenow", "50");
    fireEvent.keyDown(group, { key: "ArrowRight" });
    fireEvent.keyDown(group, { key: "ArrowRight", shiftKey: true });
    expect(handle).toHaveAttribute("aria-valuenow", "62");
    fireEvent.keyDown(group, { key: "3" });
    expect(screen.getByText("Showing: After")).toBeInTheDocument();
    fireEvent.keyDown(group, { key: " " });
    expect(screen.getByText("Showing: Before")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show After (Space)" }));
    expect(screen.getByText("Showing: After")).toBeInTheDocument();
  });

  it("refuses the difference when the step changes the geometry, and says why", () => {
    render(<BeforeAfter label="cmp" panels={two} differenceBlocked="Changes the geometry." />);
    const group = screen.getByRole("group", { name: "cmp" });
    fireEvent.keyDown(group, { key: "4" });
    expect(screen.getByRole("radio", { name: "Side by side" })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByText("Changes the geometry.")).toBeInTheDocument();
  });

  it("lays out three panels (input, reconstruction, error) without a computed difference", () => {
    render(
      <BeforeAfter
        label="reconstruction"
        header={<p>Score 0.42</p>}
        panels={[
          { src: A, label: "Input", alt: "input" },
          { src: B, label: "Reconstruction", alt: "reconstruction" },
          { src: C, label: "Error", alt: "error" },
        ]}
      />,
    );
    for (const alt of ["input", "reconstruction", "error"]) expect(screen.getByAltText(alt)).toBeInTheDocument();
    expect(screen.getByText("Score 0.42")).toBeInTheDocument();
    expect(screen.queryByRole("radio", { name: "Difference" })).not.toBeInTheDocument();
  });

  it("can be driven from outside without its own switch", () => {
    const { rerender } = render(<BeforeAfter label="cmp" panels={two} mode="slider" showModes={false} />);
    expect(screen.queryByRole("radiogroup")).not.toBeInTheDocument();
    expect(screen.getByRole("slider")).toBeInTheDocument();
    rerender(<BeforeAfter label="cmp" panels={two} mode="side" showModes={false} />);
    expect(screen.queryByRole("slider")).not.toBeInTheDocument();
  });
});

describe("BeforeAfter geometry and difference", () => {
  it("places an image to fit, then zooms around the centre", () => {
    expect(placement(400, 200, 100, 100, { zoom: 1, cx: 0.5, cy: 0.5 })).toEqual({ scale: 2, x: 100, y: 0 });
    const zoomed = placement(400, 200, 100, 100, { zoom: 2, cx: 0.5, cy: 0.5 });
    expect(zoomed.scale).toBe(4);
    expect(zoomed.x).toBe(0);
  });

  it("keeps the view on the image", () => {
    expect(clampView({ zoom: 1, cx: 0.9, cy: 0.1 }, 200, 200, 100, 100)).toEqual({ zoom: 1, cx: 0.5, cy: 0.5 });
    const far = clampView({ zoom: 4, cx: 1, cy: 0 }, 200, 200, 100, 100);
    expect(far.cx).toBeCloseTo(0.875);
    expect(far.cy).toBeCloseTo(0.125);
    expect(clampView({ zoom: 99, cx: 0.5, cy: 0.5 }, 200, 200, 100, 100).zoom).toBe(16);
  });

  it("colours |after - before| scaled to its 99th percentile", () => {
    const before = new Uint8ClampedArray([0, 0, 0, 255, 10, 10, 10, 255]);
    const after = new Uint8ClampedArray([0, 0, 0, 255, 10, 60, 10, 255]);
    const ramp = rampTable([[0, 0, 0], [255, 255, 255]]);
    const found = differencePixels(before, after, ramp);
    expect(found.top).toBe(50);
    expect(found.mean).toBe(25);
    expect([...found.pixels.slice(0, 4)]).toEqual([0, 0, 0, 255]);
    expect([...found.pixels.slice(4, 8)]).toEqual([255, 255, 255, 255]);
  });

  it("reads theme colours and builds a ramp from the surface to the ink", () => {
    expect(parseColor("#2a78d6")).toEqual([42, 120, 214]);
    expect(parseColor("#fff")).toEqual([255, 255, 255]);
    expect(parseColor("rgb(1, 2, 3)")).toEqual([1, 2, 3]);
    const stops = sequentialStops({ surface: "#ffffff", palette: ["#0000ff"], ink1: "#000000" });
    const table = rampTable(stops);
    expect([...table.slice(0, 3)]).toEqual([255, 255, 255]);
    expect([...table.slice(255 * 3)]).toEqual([0, 0, 0]);
  });
});
