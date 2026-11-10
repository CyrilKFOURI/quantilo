import { fireEvent, screen, within } from "@testing-library/react";
import { forwardRef } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { readTokens } from "../../charts/theme";
import { column, roles } from "../../test/columns";
import fixtures from "../../test/experiments.json";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import { linkTarget } from "../agent/links";
import { modules } from "../registry";
import type { BanditReplay, BanditSimulation, ExperimentMetricResult, PolicyQini, SampleSize } from "./api";
import { banditRegretFigure, banditSharesFigure, banditSimulationFigure, intervalsFigure, policyValuesFigure, qiniFigure, sampleSizeFigure } from "./figures";
import { validateExperimentsSearch } from "./search";

vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot({ label }, ref) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

/* Responses written by the real engine on samples/ab_test.csv (src/test/experiments.json). */
const { candidates, loose, byUser, srm, plan, simulation, policy, policyQini, decisions, noDecisions, evaluation, learnt, estimated } = fixtures;

function setup(
  analyze: (body: { unit?: string | null; expected?: unknown }) => unknown = (body) => (body.unit ? byUser : loose),
  logged: unknown = noDecisions,
  found: unknown = candidates,
) {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset()],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": [column({ name: "revenue" })],
    "GET /column-roles": roles,
    "GET /modules/experiments/datasets/d1/candidates": found,
    "POST /modules/experiments/datasets/d1/analyze": analyze,
    "GET /modules/experiments/datasets/d1/baseline": { column: "converted", kind: "rate", value: 0.1, sd: 0.3, rows: 4023 },
    "POST /modules/experiments/sample-size": plan,
    "POST /modules/experiments/bandit-simulation": simulation,
    "POST /modules/experiments/datasets/d1/policy": (body: { qini?: boolean }) => (body.qini ? policyQini : policy),
    "GET /modules/experiments/datasets/d1/decisions": logged,
    "POST /modules/experiments/datasets/d1/evaluate-policy": (body: { propensity?: string | null; learn?: boolean }) =>
      body.learn ? learnt : body.propensity ? evaluation : estimated,
    "GET /modules/ml/datasets/d1/runs": [],
  });
}

/** 098: the tab opens empty (pickers and the rule's suggestion); the tests take the suggestion. */
async function openSuggested() {
  renderAt("/d/d1/experiments");
  fireEvent.click(await screen.findByRole("button", { name: "Use it" }));
}

afterEach(() => vi.unstubAllGlobals());

describe("Experiments page", () => {
  it("opens empty from the Toolbox: the pickers and the suggestion with its reason, nothing analysed (098)", async () => {
    const ruled = {
      ...candidates,
      variants: candidates.variants.map((v) => ({ ...v, planned: v.column === "variant", why: v.column === "variant" ? "Its name says a variant (variant)." : "" })),
    };
    const calls = setup(undefined, noDecisions, ruled);
    const router = renderAt("/d/d1/experiments");

    expect(await screen.findByRole("heading", { name: "Describe the test" })).toBeInTheDocument();
    expect(screen.getByText(/Suggested: variant against converted, one user_id at a time/)).toBeInTheDocument();
    expect(screen.getByText("R-EXP-01 · Its name says a variant (variant).")).toBeInTheDocument();
    expect(calls.some((c) => c.key === "POST /modules/experiments/datasets/d1/analyze")).toBe(false);

    fireEvent.change(screen.getByRole("combobox", { name: "Variant" }), { target: { value: "variant" } });
    expect(calls.some((c) => c.key === "POST /modules/experiments/datasets/d1/analyze")).toBe(false);
    fireEvent.change(screen.getByRole("combobox", { name: "Metric" }), { target: { value: "converted" } });

    expect(await screen.findByRole("combobox", { name: "Unit" })).toBeInTheDocument();
    expect(router.state.location.search).toMatchObject({ design: { variant: "variant", metrics: [{ column: "converted" }] } });
    const sent = calls.find((c) => c.key === "POST /modules/experiments/datasets/d1/analyze");
    expect(sent?.body).toMatchObject({ variant: "variant", metrics: [{ column: "converted" }] });
  });

  it("opens on the suggested experiment: by user, with its sentence, chart and Holm line", async () => {
    const calls = setup();
    await openSuggested();

    expect(await screen.findByRole("heading", { name: "Experiments" })).toBeInTheDocument();
    expect(await screen.findByText(/new_checkout reaches 25.0 % vs 19.8 % for control/)).toBeInTheDocument();
    expect(screen.getAllByText("Significant · higher").length).toBeGreaterThan(0);
    expect(screen.getByText("p adjusted for 2 comparisons (Holm)")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Difference of converted to the control, with its interval" })).toBeInTheDocument();
    const table = screen.getByRole("table", { name: "converted by variant" });
    expect(within(table).getByText("+5.2 pts")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Unit" })).toHaveValue("user_id");
    expect(screen.getByRole("heading", { name: /converted over time, cumulative by day/ })).toBeInTheDocument();
    expect(screen.getByText(/Subgroups are hypotheses to confirm/)).toBeInTheDocument();
    const sent = calls.find((c) => c.key === "POST /modules/experiments/datasets/d1/analyze");
    expect(sent?.body).toMatchObject({ variant: "variant", unit: "user_id", metrics: [{ column: "converted" }] });
  });

  it("opens nothing on observed categories, says why and lets one compare anyway (R-EXP-01)", async () => {
    const note = "No column looks like an experiment's variant (R-EXP-01): variant is an observed category.";
    const observed = {
      ...candidates,
      variants: candidates.variants.map((v) => ({ ...v, planned: false, why: "Not named like a variant and its levels are uneven." })),
      suggested: null,
      note,
    };
    const calls = setup(undefined, noDecisions, observed);
    renderAt("/d/d1/experiments");

    expect(await screen.findByText(note)).toBeInTheDocument();
    expect(calls.some((c) => c.key === "POST /modules/experiments/datasets/d1/analyze")).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: `Compare ${candidates.variants[0]?.column ?? ""} anyway` }));
    expect(await screen.findByRole("combobox", { name: "Unit" })).toBeInTheDocument();
  });

  it("adds the bandit view: P(best) and expected loss per variant, the guardrail, the replay on demand", async () => {
    setup();
    await openSuggested();

    const section = (await screen.findByRole("heading", { name: "Bandit view" })).closest("section") as HTMLElement;
    expect(within(section).getByText(byUser.bandit.sentence)).toBeInTheDocument();
    const list = within(section).getByRole("list", { name: "P(best) and expected loss of each variant" });
    expect(within(list).getByText("P(best) 99.7 % · expected loss < 0.01 pt")).toBeInTheDocument();
    expect(within(section).getByText(/complements the A\/B test above, it does not replace it/)).toBeInTheDocument();
    expect(within(section).queryByText(/reinforcement/i)).toBeNull();
    expect(within(section).queryByRole("img")).toBeNull();

    const details = within(section).getByText("What a bandit would have done").closest("details") as HTMLDetailsElement;
    details.open = true;
    fireEvent(details, new Event("toggle"));
    expect(await within(section).findByRole("img", { name: "Cumulative regret of the fixed split and of the bandit over time" })).toBeInTheDocument();
    expect(within(section).getByText(byUser.bandit.replay.sentence)).toBeInTheDocument();
  });

  it("says why there is no replay without a date", async () => {
    setup(() => loose);
    renderAt("/d/d1/experiments?variant=variant&metric=converted");

    const section = (await screen.findByRole("heading", { name: "Bandit view" })).closest("section") as HTMLElement;
    expect(within(section).getByText(/Choose a date under More/)).toBeInTheDocument();
  });

  it("warns when an identifier repeats and switches to it as the unit", async () => {
    const calls = setup();
    renderAt("/d/d1/experiments?variant=variant&metric=converted");

    expect(await screen.findByText(/user_id repeats \(2.0 rows per value\)/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Use user_id as the unit" }));

    expect(await screen.findByText(/new_checkout reaches 25.0 %/)).toBeInTheDocument();
    const last = calls.filter((c) => c.key === "POST /modules/experiments/datasets/d1/analyze").at(-1);
    expect(last?.body).toMatchObject({ unit: "user_id" });
  });

  it("puts the sample ratio alarm before any result", async () => {
    setup(() => srm);
    await openSuggested();

    const alarm = await screen.findByRole("alert");
    expect(alarm).toHaveTextContent("The groups are not the size you planned: the results may be biased.");
  });

  it("learns who should get which variant under Advanced, and the Qini curve under More, each when opened", async () => {
    const calls = setup();
    await openSuggested();

    const section = (await screen.findByRole("heading", { name: "Who should get which variant" })).closest("section") as HTMLElement;
    const policyCalls = () => calls.filter((c) => c.key === "POST /modules/experiments/datasets/d1/policy");
    expect(policyCalls()).toHaveLength(0);
    const advanced = within(section).getByText("Advanced · Learn a rule").closest("details") as HTMLDetailsElement;
    advanced.open = true;
    fireEvent(advanced, new Event("toggle"));

    expect(await within(section).findByText(policy.rule)).toBeInTheDocument();
    expect(within(section).getByText(/Hypothesis to confirm with a new test/)).toBeInTheDocument();
    expect(within(section).getByText(policy.sentence)).toBeInTheDocument();
    expect(within(section).getByText("Worth a new test")).toBeInTheDocument();
    expect(within(section).getByRole("img", { name: "Value of the rule and of each variant given to everyone, with their intervals" })).toBeInTheDocument();
    expect(within(section).queryByRole("img", { name: /Gain of giving new_checkout first/ })).toBeNull();
    expect(policyCalls().every((c) => !(c.body as { qini?: boolean }).qini)).toBe(true);
    expect(within(section).queryByText(/reinforcement/i)).toBeNull();

    const more = within(section).getByText("More · Qini curve: who gains most").closest("details") as HTMLDetailsElement;
    more.open = true;
    fireEvent(more, new Event("toggle"));
    expect(await within(section).findByRole("img", { name: /Gain of giving new_checkout first/ })).toBeInTheDocument();
    expect(policyCalls().at(-1)?.body).toMatchObject({ qini: true });

    fireEvent.click(within(section).getByRole("button", { name: "spend_before" }));
    await within(section).findByText(policy.rule);
    const last = policyCalls().filter((c) => !(c.body as { qini?: boolean }).qini).at(-1);
    expect(last?.body).toMatchObject({ design: { variant: "variant" }, covariates: ["device", "orders", "revenue"] });
  });

  it("shows the Policies tab only for logged decisions: diagnostics first, estimators under Advanced, a rule under More", async () => {
    const calls = setup(undefined, decisions);
    renderAt("/d/d1/experiments?tab=policies");

    expect(await screen.findByRole("tab", { name: "Policies" })).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "Diagnostics" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Decisions / Policy" })).toBeInTheDocument();
    for (const p of evaluation.policies) expect(screen.getByText(p.sentence)).toBeInTheDocument();
    expect(screen.getByText(evaluation.diagnostics.sentences[0] as string)).toBeInTheDocument();
    expect(screen.queryByText(/reinforcement/i)).toBeNull();
    const evaluations = () => calls.filter((c) => c.key === "POST /modules/experiments/datasets/d1/evaluate-policy");
    expect(evaluations().every((c) => !(c.body as { learn?: boolean }).learn)).toBe(true);

    const advanced = screen.getByText("Advanced · Every estimator and saved models").closest("details") as HTMLDetailsElement;
    advanced.open = true;
    const table = screen.getByRole("table", { name: "Policies by estimator" });
    expect(within(table).getByText("Always sms")).toBeInTheDocument();
    const more = screen.getByText("More · Learn a rule from the log").closest("details") as HTMLDetailsElement;
    more.open = true;
    fireEvent(more, new Event("toggle"));
    expect(await screen.findByText(learnt.rule as string)).toBeInTheDocument();
    expect(evaluations().at(-1)?.body).toMatchObject({ learn: true });

    fireEvent.change(screen.getByRole("combobox", { name: "Propensity" }), { target: { value: "" } });
    expect(await screen.findByText(/Observational: valid only if no hidden factor drove the decisions/)).toBeInTheDocument();
  });

  it("hides the Policies tab when the data is not a log of decisions", async () => {
    setup();
    renderAt("/d/d1/experiments?tab=policies");

    expect(await screen.findByRole("heading", { name: "Describe the test" })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Policies" })).toBeNull();
  });

  it("plans the sample size: per group, total and the curve", async () => {
    const calls = setup();
    renderAt("/d/d1/experiments?tab=plan");

    expect(await screen.findByText("57,756")).toBeInTheDocument();
    expect(screen.getByText(plan.sentence)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Units per group against the smallest detectable lift" })).toBeInTheDocument();
    const sent = calls.find((c) => c.key === "POST /modules/experiments/sample-size");
    expect(sent?.body).toMatchObject({ kind: "rate", baseline: 0.1, effect: 0.05, relative: true, alpha: 0.05, power: 0.8 });

    fireEvent.change(screen.getByRole("combobox", { name: "Read from" }), { target: { value: "converted" } });
    expect(await screen.findByDisplayValue("10")).toBeInTheDocument();
  });

  it("simulates fixed split vs bandit only once its section is opened", async () => {
    const calls = setup();
    renderAt("/d/d1/experiments?tab=plan");

    const summary = await screen.findByText("Fixed split vs bandit");
    expect(calls.some((c) => c.key === "POST /modules/experiments/bandit-simulation")).toBe(false);
    const details = summary.closest("details") as HTMLDetailsElement;
    details.open = true;
    fireEvent(details, new Event("toggle"));

    expect(await screen.findByText(simulation.sentence)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Cumulative regret of the fixed split and of a bandit over the simulated test" })).toBeInTheDocument();
    expect(screen.getByText(simulation.guardrail)).toBeInTheDocument();
    const sent = calls.find((c) => c.key === "POST /modules/experiments/bandit-simulation");
    expect(sent?.body).toMatchObject({ kind: "rate", baseline: 0.1, effect: 0.05 });
  });
});

describe("Experiments module", () => {
  it("is available in the browser-only build too (not locked)", () => {
    const found = modules.find((m) => m.id === "experiments");
    expect(found?.path).toBe("/d/$datasetId/experiments");
    expect(found?.locked).toBeFalsy();
  });

  it("keeps a well-formed experiment from the URL and drops the rest", () => {
    expect(validateExperimentsSearch({ tab: "plan", design: { variant: "g", metrics: [{ column: "y" }] } })).toEqual({
      tab: "plan",
      design: { variant: "g", metrics: [{ column: "y" }] },
    });
    expect(validateExperimentsSearch({ tab: "x", design: { variant: "g", metrics: [] }, metric: 3 })).toEqual({});
  });

  it("opens from an assistant link", () => {
    expect(linkTarget("d1", { page: "experiments", params: { variant: "variant", metric: "converted", unit: "user_id" }, label: "" })).toEqual({
      to: "/d/$datasetId/experiments",
      params: { datasetId: "d1" },
      search: { variant: "variant", metric: "converted", unit: "user_id" },
    });
  });

  it("draws each variant in its palette colour with the lift written, and the plan on the curve", () => {
    const tokens = { ...readTokens(), palette: ["#111111", "#222222"], surface: "#ffffff", ink2: "#555555", ink3: "#777777", ink1: "#000000" };
    const metric = byUser.metrics[0] as unknown as ExperimentMetricResult;
    const intervals = intervalsFigure(metric)(tokens);
    const [line, point] = intervals.data as { x: number[]; line?: { color: string }; text?: string[] }[];
    expect(line?.line?.color).toBe("#111111");
    expect(line?.x[0]).toBeCloseTo((metric.comparisons[0]?.diff_low ?? 0) * 100);
    expect(point?.text?.[0]).toBe("+5.22 pts (+26.4 %)");

    const replay = byUser.bandit.replay as unknown as BanditReplay;
    const shares = banditSharesFigure(replay)(tokens).data as { name: string; line: { color: string } }[];
    expect(shares.map((d) => d.name)).toEqual(["control", "new_checkout"]);
    expect(shares[0]?.line.color).toBe("#777777");
    expect(shares[1]?.line.color).toBe("#111111");
    const regret = banditRegretFigure(replay, "rate")(tokens);
    expect((regret.data as { name: string }[]).map((d) => d.name)).toEqual(["Fixed split", "Bandit"]);
    expect(regret.layout?.yaxis?.title).toMatchObject({ text: "Cumulative regret (conversions)" });
    const simulated = banditSimulationFigure(simulation as unknown as BanditSimulation)(tokens);
    expect((simulated.data[1] as { fill: string }).fill).toBe("tonexty");

    const curve = sampleSizeFigure(plan as unknown as SampleSize)(tokens);
    const marker = curve.data[1] as { x: number[]; y: number[] };
    expect(marker.x[0]).toBe(57_756);
    expect(marker.y[0]).toBeCloseTo(5);
  });

  it("draws the policies against their reference and the Qini curve against random targeting", () => {
    const tokens = { ...readTokens(), palette: ["#111111", "#222222"], surface: "#ffffff", ink3: "#777777" };
    const points = policy.values.map((v) => ({ label: v.label, value: v.value, low: v.low, high: v.high }));
    const values = policyValuesFigure(points, "rate", 0.2)(tokens);
    const segment = values.data[0] as { x: number[]; line: { color: string } };
    expect(segment.x[0]).toBeCloseTo((policy.values[0]?.low ?? 0) * 100);
    expect(segment.line.color).toBe("#111111");
    expect(values.layout?.shapes?.[0]).toMatchObject({ x0: 20, line: { dash: "dot" } });
    const qini = qiniFigure(policyQini.qini[0] as unknown as PolicyQini, "rate")(tokens);
    const [random, ranked] = qini.data as { name: string; x: number[] }[];
    expect(random?.name).toBe("Random targeting");
    expect(ranked?.x.at(-1)).toBe(100);
  });
});
