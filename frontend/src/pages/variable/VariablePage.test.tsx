import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { forwardRef } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import { categoricalProfile, fit, numericProfile, outliers, textProfile } from "../../test/variable";

// jsdom can't draw Plotly: render a labelled placeholder instead.
vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot({ label }, ref) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

const columns = [
  column({ name: "price", position: 0 }),
  column({ name: "city", position: 1, type: "categorical_nominal", detected_type: "categorical_nominal" }),
  column({ name: "hidden", position: 2, excluded: true }),
  column({ name: "survey_answer", position: 3, type: "text", detected_type: "text", role: "free_text" }),
  column({ name: "item", position: 4, type: "text", detected_type: "text", role: "label" }),
];

const histogram = {
  kind: "histogram",
  view: "histogram",
  views: ["histogram", "box"],
  rows: 1200,
  missing: 10,
  distinct: 900,
  bars: [],
  others: null,
  ordered: false,
  bins: [{ start: 0, end: 10, count: 5 }],
  stats: null,
  box: null,
  step: null,
  steps: [],
  examples: [],
  duplicates: 0,
  message: null,
};

function setup(profile: unknown = numericProfile()) {
  return mockApi({
    "GET /health": health,
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": columns,
    "GET /column-roles": roles,
    "GET /datasets/d1/variable": profile,
    "GET /datasets/d1/variable/fit": fit,
    "GET /datasets/d1/variable/outliers": outliers,
    "GET /datasets/d1/columns/chart": {
      ...histogram,
      ...(profile === categoricalProfile
        ? { kind: "bars", view: "bars", views: ["bars"], bins: [], bars: [{ label: "Paris", count: 90 }] }
        : {}),
    },
  });
}

beforeEach(() => localStorage.clear());
afterEach(() => vi.unstubAllGlobals());

describe("Variable page", () => {
  it("shows a numeric variable's statistics with explanations", async () => {
    setup();
    renderAt("/d/d1/variable/price");

    expect(await screen.findByRole("heading", { name: "price" })).toBeInTheDocument();
    expect(await screen.findByText("Excess kurtosis")).toBeInTheDocument();
    expect(screen.getByText("Not normal")).toBeInTheDocument();
    expect(screen.getByText("1 of 4")).toBeInTheDocument(); // excluded columns are skipped
    expect(screen.getAllByRole("button", { name: "Skewness" }).length).toBeGreaterThan(0); // info tips
    expect(await screen.findByRole("img", { name: "Distribution of price" })).toBeInTheDocument();
  });

  it("explains normality tests and re-judges them when α changes", async () => {
    setup();
    renderAt("/d/d1/variable/price");
    fireEvent.click(await screen.findByRole("tab", { name: "Normality" }));

    expect(screen.getByText("5 of 5 tests reject normality at α = 0.05.")).toBeInTheDocument();
    expect(screen.getByText("≤ 0.01")).toBeInTheDocument(); // Anderson's table floor

    fireEvent.click(screen.getByRole("radio", { name: "0.01" }));

    expect(screen.getByText("3 of 5 tests reject normality at α = 0.01.")).toBeInTheDocument();
    expect(localStorage.getItem("de.variable.alpha")).toBe('"0.01"');
  });

  it("ranks fitted laws and previews transformations", async () => {
    setup();
    renderAt("/d/d1/variable/price");
    fireEvent.click(await screen.findByRole("tab", { name: "Distribution" }));

    expect(await screen.findByText("Log-normal describes this variable best (lowest AIC).")).toBeInTheDocument();
    expect(screen.getByText("A log transformation would bring it close to normal.")).toBeInTheDocument();
    const table = screen.getByRole("table");
    expect(within(table).getAllByRole("row")).toHaveLength(3);
    expect(screen.getByRole("heading", { name: "Log" })).toBeInTheDocument();
  });

  it("switches outlier rules", async () => {
    const calls = setup();
    renderAt("/d/d1/variable/price");
    fireEvent.click(await screen.findByRole("tab", { name: "Outliers" }));

    expect(await screen.findByRole("cell", { name: "3,348" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: "Robust z" }));

    await waitFor(() =>
      expect(calls.some((call) => call.key === "GET /datasets/d1/variable/outliers")).toBe(true),
    );
    expect(screen.getByText(/Iglewicz & Hoaglin/)).toBeInTheDocument();
  });

  it("filters the dataset on the outliers of a rule", async () => {
    setup();
    const router = renderAt("/d/d1/variable/price");
    fireEvent.click(await screen.findByRole("tab", { name: "Outliers" }));

    fireEvent.click(await screen.findByRole("button", { name: "Show only these rows" }));

    await waitFor(() =>
      expect(router.state.location.search).toEqual({
        filters: {
          dataset: "d1",
          conditions: [{ column: "price", op: "between", low: -114, high: 317, negate: true }],
        },
      }),
    );
  });

  it("profiles a category and warns about imbalance", async () => {
    setup(categoricalProfile);
    renderAt("/d/d1/variable/city");

    expect(await screen.findByText(/"Paris" alone makes up/)).toBeInTheDocument();
    expect(screen.getByText("0.52 bits")).toBeInTheDocument();
    expect(screen.getByText(/not equally frequent \(p < 0.001\)/)).toBeInTheDocument();
    expect(await screen.findByRole("cell", { name: "Paris" })).toBeInTheDocument();
  });

  it("profiles a text: language, expressions, examples and repeated texts (050)", async () => {
    setup(textProfile);
    renderAt("/d/d1/variable/survey_answer");

    expect(await screen.findByText("French")).toBeInTheDocument();
    expect(screen.getByText("90% of texts")).toBeInTheDocument();
    expect(screen.getByText("“livraison” comes up most often: 95 times.")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("radio", { name: "2 words" }));
    expect(screen.getByText("“très rapide” comes up most often: 92 times.")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: /Examples/ }));
    expect(screen.getByText("Delivery was fine selon moi.")).toBeInTheDocument();
    expect(screen.getByText(/Texts found more than once: 1, on 0.6% of rows/)).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "La qualité est parfait et je recommande." })).toBeInTheDocument();
  });

  it("names a short text after its role (050)", async () => {
    setup(textProfile);
    renderAt("/d/d1/variable/item");

    expect(await screen.findByRole("button", { name: "Change type" })).toHaveTextContent("Short text");
  });

  it("links to the previous and next variables", async () => {
    setup();
    renderAt("/d/d1/variable/price");

    const next = await screen.findByRole("link", { name: /Next variable: city/ });
    expect(next).toHaveAttribute("href", "/d/d1/variable/city");
    expect(screen.getByRole("button", { name: "Previous variable" })).toBeDisabled();
  });

  it("says when the column doesn't exist", async () => {
    setup();
    renderAt("/d/d1/variable/nope");

    expect(await screen.findByText("This column doesn't exist in the dataset.")).toBeInTheDocument();
  });
});
