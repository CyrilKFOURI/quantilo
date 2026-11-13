import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { mockApi } from "../../test/mockApi";
import type { ImageRow } from "./model";
import { anomalyTask } from "./modelTestData";
import { ConceptsBlock, ExplainMore, LooksLike, PatchCrop } from "./MoreExplain";
import { MoreOptions } from "./MoreOptions";
import { conceptProbabilities, type MoreKey } from "./moreApi";

afterEach(() => vi.unstubAllGlobals());

/** jsdom does not toggle <details> on a click: open it as the browser would. */
function open(title: string) {
  const details = screen.getByText(title).closest("details") as HTMLDetailsElement;
  details.open = true;
  fireEvent(details, new Event("toggle"));
}

function wrap(node: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{node}</QueryClientProvider>);
}

const RUN = "abcdef012345";
const BASE = `/modules/vision/datasets/d1/runs/${RUN}`;
const row = { id: "eeeeeeeeeeeeeeee", key: "test/a.jpg", name: "a.jpg" } as unknown as ImageRow;

const more = [
  { key: "finetune" as const, title: "Fine-tuned DINOv2 ViT-S/14 (top blocks)", available: true, reason: null, estimate: "≈ 1 min on NVIDIA GeForce RTX 4050 Laptop GPU", seconds: 64, note: "Fine-tunes the last 2 blocks." },
  { key: "protopnet" as const, title: "ProtoPNet-lite on DINOv2 ViT-S/14 (this looks like that)", available: false, reason: "Download the weights first: DINOv2 ViT-S/14, 88 MB.", estimate: null, seconds: null, note: "Prototypes." },
  { key: "concept_bottleneck" as const, title: "Concept bottleneck (your concepts, then the class)", available: true, reason: null, estimate: "≈ 11 s on the CPU (numpy)", seconds: 11, note: "Concepts, then the class." },
];
const concept = (key: string, name: string, images: number) => ({ key, name, kind: "binary" as const, source: "group" as const, images, note: "", bottleneck: true, tcav: images >= 20, id: null, created_at: null });
const task = {
  ...anomalyTask,
  choice: { ...anomalyTask.choice, task: "classification" as const },
  more,
  concepts: [concept("group:aaaaaaaaaaaa", "spots", 40), concept("group:bbbbbbbbbbbb", "yellow edges", 25)],
};

describe("Model › Advanced › More (118)", () => {
  it("greys an option with its reason, shows estimates, and asks nothing of the server", async () => {
    const calls = mockApi({});
    let picked: MoreKey[] = [];
    let chosen: string[] | null = null;
    const view = wrap(<MoreOptions out={task} more={[]} onMore={(next) => (picked = next)} concepts={null} onConcepts={(next) => (chosen = next)} />);
    open("More");
    expect(screen.getByText(/Download the weights first/)).toBeInTheDocument();
    expect(screen.getByText(/Estimated ≈ 1 min on NVIDIA/)).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /ProtoPNet-lite/ })).toBeDisabled(); // no weights
    fireEvent.click(screen.getByRole("checkbox", { name: /Fine-tuned/ }));
    expect(picked).toEqual(["finetune"]);
    view.rerender(
      <QueryClientProvider client={new QueryClient()}>
        <MoreOptions out={task} more={["concept_bottleneck"]} onMore={(next) => (picked = next)} concepts={null} onConcepts={(next) => (chosen = next)} />
      </QueryClientProvider>,
    );
    expect(await screen.findByText("Concepts it reads (2 chosen)")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: /yellow edges/ }));
    expect(chosen).toEqual(["group:aaaaaaaaaaaa"]);
    expect(calls).toHaveLength(0);
  });
});

describe("the concept bottleneck's head (R-VCB-05)", () => {
  it("is a sigmoid with one row of weights and a softmax otherwise", () => {
    const binary = { classes: ["a", "b"], mean: [0.5], scale: [0.5], weight: [[2]], bias: [0], setting: "" };
    const [pa, pb] = conceptProbabilities(binary, [1]);
    expect(pb).toBeCloseTo(1 / (1 + Math.exp(-2)));
    expect((pa ?? 0) + (pb ?? 0)).toBeCloseTo(1);
    const three = { classes: ["a", "b", "c"], mean: [0, 0], scale: [1, 1], weight: [[1, 0], [0, 1], [0, 0]], bias: [0, 0, 0], setting: "" };
    const p = conceptProbabilities(three, [2, 0]);
    expect(p.reduce((s, v) => s + v, 0)).toBeCloseTo(1);
    expect(p[0]).toBeGreaterThan(p[1] ?? 1);
  });
});

describe("Why? of the More models", () => {
  it("shows this looks like that with the numbers of the score", async () => {
    mockApi({
      [`GET ${BASE}/explain/images/${row.id}/looks`]: {
        label: "bean_rust",
        probability: 0.9,
        looks: [
          { prototype: 3, label: "bean_rust", similarity: 3.15, weight: 2.02, points: 6.37, box: [0.4, 0.1, 0.46, 0.16], source: { row: 7, image_id: "ffffffffffffffff", label: "bean_rust", similarity: null, distance: null, share: null, witness: null }, source_box: [0.2, 0.2, 0.26, 0.26] },
        ],
        total: 6.37,
        sentence: "This patch looks like that prototype of bean_rust (from a training image): similarity 3.15 × weight 2.02 = 6.37 points towards bean_rust.",
        tested: true,
        notes: [],
      },
    });
    wrap(<LooksLike datasetId="d1" runId={RUN} row={row} nameOf={(ref) => ref.image_id.slice(0, 4)} openRef={() => undefined} />);
    expect(await screen.findByText(/This patch looks like that prototype of bean_rust/)).toBeInTheDocument();
    expect(screen.getByText("similarity 3.15 × weight 2.02 = 6.37")).toBeInTheDocument();
    expect(screen.getByText(/not a post-hoc map/)).toBeInTheDocument();
  });

  it("recomputes the class when a concept is corrected", async () => {
    mockApi({
      [`GET ${BASE}/explain/images/${row.id}/concepts`]: {
        space: "the frozen DINOv2 ViT-S/14 vectors",
        concepts: [{ key: "group:aaaaaaaaaaaa", name: "spots", kind: "binary", predicted: 0.1, truth: 1, positive: "in the group", note: "" }],
        head: { classes: ["healthy", "rust"], mean: [0.5], scale: [0.5], weight: [[3]], bias: [0], setting: "L1 logistic, C = 1" },
        tested: true,
        sentence: "The bottleneck read spots 0.10 in this image.",
        notes: [],
      },
    });
    wrap(<ConceptsBlock datasetId="d1" runId={RUN} row={row} />);
    expect(await screen.findByText("The bottleneck read spots 0.10 in this image.")).toBeInTheDocument();
    expect(screen.getByText("The class, from the concepts as read:")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: "Has it" }));
    expect(screen.getByText("The class, from the concepts as you set them:")).toBeInTheDocument();
    // spots = 1 -> z = 1 -> p(rust) = sigmoid(3) = 0.95
    expect(screen.getByText("0.95")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Back to the concepts as read/ }));
    expect(screen.getByText("The class, from the concepts as read:")).toBeInTheDocument();
  });
});

describe("Explain › More (118)", () => {
  it("asks nothing before it is opened, then runs TCAV only on click", async () => {
    const calls = mockApi({
      [`GET ${BASE}/explain/prototypes`]: null,
      [`GET ${BASE}/explain/tcav/options`]: {
        available: true,
        reason: null,
        concepts: [concept("group:aaaaaaaaaaaa", "spots", 40)],
        layers: ["block 1", "block 9"],
        layer: "block 9",
        cost: "about 20 s on a GPU",
      },
      [`GET ${BASE}/explain/tcav`]: {
        concept: "spots",
        concept_key: "group:aaaaaaaaaaaa",
        layer: "block 9",
        layers: ["block 1", "block 9"],
        images: 40,
        random_images: 40,
        cav_accuracy: 0.92,
        random_accuracy: 0.51,
        classes: [
          { label: "rust", images: 40, score: 0.9, spread: 0.05, random: 0.5, random_spread: 0.2, p_value: 0.0001, significant: true, sentence: "Concept found." },
          { label: "healthy", images: 40, score: 0.45, spread: 0.1, random: 0.5, random_spread: 0.2, p_value: 0.4, significant: false, sentence: "Not significant." },
        ],
        sentence: "Spots matters for rust (p < 0.05) at block 9.",
        seconds: 18,
        cached: false,
        notes: [],
      },
    });
    wrap(<ExplainMore datasetId="d1" runId={RUN} byId={new Map()} onWhy={() => undefined} />);
    expect(calls).toHaveLength(0);
    open("More");
    expect(await screen.findByText(/This run has no ProtoPNet-lite/)).toBeInTheDocument();
    await screen.findByRole("button", { name: /Test the concept/ });
    expect(calls.some((c) => c.key.endsWith("/explain/tcav"))).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: /Test the concept/ }));
    expect(await screen.findByText("Spots matters for rust (p < 0.05) at block 9.")).toBeInTheDocument();
    expect(screen.getByText("Concept found")).toBeInTheDocument();
    expect(screen.getByText("Not significant")).toBeInTheDocument();
    const asked = calls.find((c) => c.key.endsWith("/explain/tcav"));
    expect(asked?.params.get("concept")).toBe("group:aaaaaaaaaaaa");
    await waitFor(() => expect(asked?.params.get("layer")).toBeNull()); // the default layer
  });

  it("crops a patch with its context and outlines it", () => {
    const { container } = render(<PatchCrop src="x.png" box={[0.5, 0.5, 0.5625, 0.5625]} alt="patch" />);
    const image = container.querySelector("img") as HTMLImageElement;
    expect(image.style.width).toBe(`${100 / (0.0625 * 3)}%`);
    const outline = container.querySelector("span.ring-accent") as HTMLElement;
    expect(Number.parseFloat(outline.style.left)).toBeCloseTo(100 / 3);
  });
});
