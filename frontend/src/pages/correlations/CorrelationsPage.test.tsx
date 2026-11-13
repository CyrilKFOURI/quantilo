import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { forwardRef } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot({ label }, ref) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

const columns = [
  column({ name: "units", position: 0 }),
  column({ name: "revenue", position: 1 }),
  column({ name: "store", position: 2, type: "categorical_nominal", detected_type: "categorical_nominal" }),
];

const correlations = {
  matrix: {
    method: "pearson",
    columns: [
      { name: "units", type: "numeric_continuous", role: "num" },
      { name: "revenue", type: "numeric_continuous", role: "num" },
      { name: "store", type: "categorical_nominal", role: "cat" },
    ],
    values: [
      [1, 0.997, 0.7],
      [0.997, 1, 0.69],
      [0.7, 0.69, 1],
    ],
    metrics: [
      [null, "pearson", "eta"],
      ["pearson", null, "eta"],
      ["eta", "eta", null],
    ],
    n: [
      [100, 100, 100],
      [100, 100, 100],
      [100, 100, 100],
    ],
    clustered: [2, 0, 1],
    rows: 100,
    total: 100,
    sampled: false,
    excluded: [{ name: "date", reason: "Dates: compare them with a numeric or category on the Compare page." }],
    truncated: 0,
    pairs: [
      { a: "units", b: "revenue", metric: "pearson", value: 0.997, p_value: 0.0001, n: 100, q_value: 0.0002, evidence: "strong" },
      { a: "units", b: "store", metric: "eta", value: 0.7, p_value: 0.0001, n: 100, q_value: 0.0002, evidence: "strong" },
      { a: "revenue", b: "store", metric: "eta", value: 0.3, p_value: 0.04, n: 100, q_value: 0.06, evidence: "weak" },
    ],
    multiple: {
      method: "holm",
      label: "Holm",
      tests: 3,
      significant: 3,
      kept: 2,
      alpha: 0.05,
      auto: true,
      reason: "3 tests: Holm (automatic below 10 tests).",
      sentence: "2 of the 3 significant links stay significant after correcting for 3 tests (Holm).",
    },
  },
  insights: [
    {
      id: "redundant:units|revenue",
      kind: "redundant",
      severity: "warning",
      text: "units and revenue are almost identical (r = 0.997): probably redundant, keep one of them in models.",
      columns: ["units", "revenue"],
      link: { page: "compare", x: "units", y: "revenue" },
      score: 4.9,
      evidence: "strong",
    },
    {
      id: "group_difference:revenue|store",
      kind: "group_difference",
      severity: "info",
      text: "revenue seems to differ across the groups of store, but this could be chance (η = 0.30, adj. p = 0.200).",
      columns: ["revenue", "store"],
      link: { page: "compare", x: "store", y: "revenue" },
      score: 1.3,
      evidence: "weak",
    },
    {
      id: "skewed:revenue",
      kind: "skewed",
      severity: "info",
      text: "revenue is strongly skewed to the right.",
      columns: ["revenue"],
      link: { page: "variable", x: "revenue", y: null },
      score: 0.9,
    },
  ],
};

function setup() {
  return mockApi({
    "GET /health": health,
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": columns,
    "GET /column-roles": roles,
    "GET /datasets/d1/correlations": correlations,
  });
}

beforeEach(() => localStorage.clear());
afterEach(() => vi.unstubAllGlobals());

describe("Correlations page", () => {
  it("says how many links survive the correction and marks those that no longer hold", async () => {
    const calls = setup();
    renderAt("/d/d1/correlations");

    expect(await screen.findByText(/2 of the 3 significant links stay significant/)).toBeInTheDocument();
    expect(screen.getAllByText("no longer holds")).toHaveLength(1);
    fireEvent.change(screen.getByRole("combobox", { name: /Correction/ }), { target: { value: "bonferroni" } });
    await waitFor(() => {
      const last = calls.filter((c) => c.key === "GET /datasets/d1/correlations").at(-1);
      expect(last?.params.get("correction")).toBe("bonferroni");
    });
  });

  it("shows insights with links to their proof", async () => {
    setup();
    renderAt("/d/d1/correlations");

    const insight = await screen.findByText(/almost identical/);
    const card = insight.closest("li") as HTMLElement;
    expect(within(card).getByRole("link", { name: "View" })).toHaveAttribute(
      "href",
      "/d/d1/compare?x=units&y=revenue",
    );
    const skewed = (screen.getByText(/strongly skewed/).closest("li") as HTMLElement);
    expect(within(skewed).getByRole("link", { name: "View" })).toHaveAttribute("href", "/d/d1/variable/revenue");
  });

  it("groups insights by theme and colours variable names", async () => {
    setup();
    renderAt("/d/d1/correlations");

    expect(await screen.findByRole("heading", { name: "Redundant variables" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Distributions" })).toBeInTheDocument();
    const names = screen.getAllByText("revenue", { selector: "span" });
    expect(names.some((name) => name.className.includes("text-type-numeric"))).toBe(true);
  });

  it("draws the matrix, ranks pairs and lists excluded columns", async () => {
    setup();
    renderAt("/d/d1/correlations");

    expect(await screen.findByRole("img", { name: "Association matrix" })).toBeInTheDocument();
    const table = screen.getByRole("table");
    expect(within(table).getAllByRole("row")).toHaveLength(4);
    expect(within(within(table).getAllByRole("row")[3] as HTMLElement).getByText("Weak")).toBeInTheDocument();
    expect(within(table).getByText("0.997")).toBeInTheDocument();
    expect(screen.getByText("Columns left out (1)")).toBeInTheDocument();
  });

  it("shows links by level of evidence and remembers the choice", async () => {
    setup();
    renderAt("/d/d1/correlations");

    await screen.findByText(/almost identical/);
    expect(screen.queryByText(/could be chance/)).not.toBeInTheDocument();
    expect(screen.getByText(/1 weaker link hidden/)).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "+ Moderate · 1" })).toHaveAttribute("aria-checked", "true");

    fireEvent.click(screen.getByRole("radio", { name: "+ Weak · 2" }));
    const weak = (await screen.findByText(/could be chance/)).closest("li") as HTMLElement;
    expect(within(weak).getByText("Weak")).toBeInTheDocument();
    expect(screen.getByText(/strongly skewed/)).toBeInTheDocument(); // distributions are never filtered
    expect(localStorage.getItem("de.insights.evidence")).toBe('"weak"');

    fireEvent.click(screen.getByRole("radio", { name: "Strong · 1" }));
    expect(screen.queryByText(/could be chance/)).not.toBeInTheDocument();
  });

  it("focuses on a variable and switches method", async () => {
    const calls = setup();
    renderAt("/d/d1/correlations");
    await screen.findByRole("img", { name: "Association matrix" });

    fireEvent.click(screen.getByRole("button", { name: "store" }));
    const ranked = screen.getByRole("list", { name: "Variables most related to store" });
    expect(within(ranked).getAllByRole("listitem")[0]).toHaveTextContent("units");
    expect(within(ranked).getByRole("link", { name: "Compare store with units" })).toHaveAttribute(
      "href",
      "/d/d1/compare?x=store&y=units",
    );

    fireEvent.click(screen.getByRole("radio", { name: "Spearman" }));
    await waitFor(() =>
      expect(calls.some((c) => c.key === "GET /datasets/d1/correlations" && c.params.get("method") === "spearman")).toBe(
        true,
      ),
    );
  });
});

const structure = {
  columns: ["units", "revenue", "cost", "weight", "volume"],
  left_out: [{ name: "flat", reason: "Constant: it carries no variance." }],
  rows: 100,
  total: 100,
  sampled: false,
  dropped: 0,
  filled: 0,
  components: [
    { index: 1, eigenvalue: 2.6, explained: 0.52, cumulative: 0.52, threshold: 1.3, retained: true, label: "units, revenue, cost" },
    { index: 2, eigenvalue: 1.4, explained: 0.28, cumulative: 0.8, threshold: 1.15, retained: true, label: "weight, volume" },
    { index: 3, eigenvalue: 0.6, explained: 0.12, cumulative: 0.92, threshold: 1.0, retained: false, label: "cost" },
  ],
  retained: 2,
  kaiser: 2,
  loadings: [
    [0.9, 0.1],
    [0.88, 0.05],
    [0.8, -0.1],
    [0.1, 0.9],
    [0.05, 0.85],
  ],
  shown: 2,
  biplot: { x: [0.1, -0.2], y: [0.3, 0.1], points: 2, of: 100, arrows: [{ column: "units", x: 0.9, y: 0.1 }] },
  conclusion: "2 components explain 80% of the variance of 5 numeric columns: PC1 = units, revenue, cost; PC2 = weight, volume.",
  permutations: 100,
  parallel_rows: 100,
  notes: [],
  reason: null,
};

const factors = {
  columns: structure.columns,
  left_out: [],
  rows: 100,
  sampled: false,
  k: 2,
  rotation: "varimax",
  loadings: structure.loadings,
  communalities: [0.82, 0.78, 0.65, 0.82, 0.73],
  uniqueness: [0.18, 0.22, 0.35, 0.18, 0.27],
  variance: [0.45, 0.31],
  labels: ["units, revenue, cost", "weight, volume"],
  kmo: 0.71,
  kmo_label: "middling",
  msa: [0.75, 0.72, 0.68, 0.45, 0.7],
  bartlett_chi2: 312.4,
  bartlett_df: 10,
  bartlett_p: 0.00001,
  conclusion: "KMO = 0.71 (middling): the columns share enough to be factored.",
  notes: [],
  reason: null,
};

function openFold(text: string) {
  const details = screen.getByText(text).closest("details") as HTMLDetailsElement;
  details.open = true;
  fireEvent(details, new Event("toggle"));
}

describe("Correlations page: Structure (078)", () => {
  const proposed = { proposed: true, numeric: 5, strong: 4, reason: "5 numeric columns, 4 pairs with |r| ≥ 0.5: they may measure fewer underlying dimensions." };

  function withStructure(rule = proposed) {
    return mockApi({
      "GET /health": health,
      "GET /datasets/d1": dataset(),
      "GET /datasets/d1/columns": columns,
      "GET /column-roles": roles,
      "GET /datasets/d1/correlations": { ...correlations, structure: rule },
      "GET /datasets/d1/correlations/structure": structure,
      "GET /datasets/d1/correlations/factors": factors,
    });
  }

  it("is proposed with its reason, folded, and computed only once opened", async () => {
    const calls = withStructure();
    renderAt("/d/d1/correlations");

    expect(await screen.findByText(/they may measure fewer underlying dimensions/)).toBeInTheDocument();
    expect(calls.some((c) => c.key === "GET /datasets/d1/correlations/structure")).toBe(false);

    openFold("Structure");
    expect(await screen.findByText(/2 components explain 80% of the variance/)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Scree plot" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Loadings" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Biplot" })).toBeInTheDocument();
    expect(screen.getByText(/keep 80% of the variance/)).toBeInTheDocument();
    expect(screen.getByText("flat")).toBeInTheDocument();
    expect(calls.some((c) => c.key === "GET /datasets/d1/correlations/factors")).toBe(false);
  });

  it("computes the factor analysis only once More is opened", async () => {
    const calls = withStructure();
    renderAt("/d/d1/correlations");
    await screen.findByText(/they may measure fewer underlying dimensions/);
    openFold("Structure");
    await screen.findByText(/2 components explain/);

    openFold("More: factor analysis, KMO, Bartlett");
    expect(await screen.findByText(/KMO = 0.71 \(middling\)/)).toBeInTheDocument();
    expect(calls.some((c) => c.key === "GET /datasets/d1/correlations/factors")).toBe(true);
    const table = screen.getByRole("table", { name: "Loadings after rotation" });
    expect(within(table).getAllByRole("row")).toHaveLength(7); // header, 5 columns, share of variance
    expect(within(table).getByText("45%")).toBeInTheDocument();
    expect(screen.getByText("312")).toBeInTheDocument();
  });

  it("is not shown when the rule does not propose it", async () => {
    withStructure({ ...proposed, proposed: false, reason: "Needs at least 3 pairs with |r| ≥ 0.5 (1 here)." });
    renderAt("/d/d1/correlations");
    await screen.findByRole("img", { name: "Association matrix" });
    expect(screen.queryByText("Structure")).not.toBeInTheDocument();
  });
});
