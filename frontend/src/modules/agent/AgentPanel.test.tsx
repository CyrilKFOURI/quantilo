import { fireEvent, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

const providers = [
  { id: "anthropic", label: "Anthropic (Claude)", configured: false, console: "console.anthropic.com", needs_key: true },
  { id: "gemini", label: "Google Gemini", configured: false, console: "aistudio.google.com", needs_key: true },
];

const settings = (configured: boolean, vendor: "anthropic" | "gemini" = "anthropic", sharing = "never") => ({
  provider: "auto",
  active: vendor,
  active_label: vendor === "anthropic" ? "Anthropic (Claude)" : "Google Gemini",
  providers: providers.map((p) => ({ ...p, configured: configured && p.id === vendor })),
  model: "",
  known_models: vendor === "anthropic" ? ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"] : [],
  base_url: null,
  effort: "high",
  effort_supported: vendor === "anthropic",
  budget_usd: 1,
  key: { configured, source: configured ? "file" : "none", hint: configured ? "…abcd" : null },
  sdk_installed: true,
  ready: configured,
  token_limit: 2_000_000,
  rows_sharing: sharing,
});

const conversation = {
  id: "c1",
  dataset_id: "d1",
  title: "New conversation",
  model: "claude-opus-5-5",
  cost_usd: 0,
  tokens: 0,
  created_at: "2026-10-21T10:00:00Z",
  updated_at: "2026-10-21T10:00:00Z",
  running: false,
};

const turn = [
  { type: "spent", turn_usd: 0.01, conversation_usd: 0.01, tokens: 900, conversation_tokens: 900, priced: true },
  { type: "step_started", id: "s1", tool: "rank_associations_with", label: "Ranking what is linked to Survived", input: {} },
  {
    type: "step_done",
    id: "s1",
    ok: true,
    summary: "Strongest link with Survived: Sex (Cramér's V 0.54)",
    link: { page: "correlations", params: {}, label: "Open the correlations" },
    ms: 420,
  },
  { type: "text", text: "**Sex** is the strongest driver." },
  { type: "done", end: "answered", message: null, turn_usd: 0.02, conversation_usd: 0.02, conversation_tokens: 1800, priced: true },
];

const saved = [
  { role: "user", text: "What is most linked to Survived, and how?", thinking: "", steps: [] },
  {
    role: "assistant",
    text: "**Sex** is the strongest driver.",
    thinking: "",
    unverified: ["46.8"],
    steps: [
      {
        id: "s1",
        tool: "rank_associations_with",
        label: "Ranking what is linked to Survived",
        input: {},
        ok: true,
        summary: "Strongest link with Survived: Sex (Cramér's V 0.54)",
        link: { page: "correlations", params: {}, label: "Open the correlations" },
      },
    ],
  },
];

let notes: { id: string; text: string; source: string; created_at: string }[] = [];

function setup(configured = true, vendor: "anthropic" | "gemini" = "anthropic", events = turn, sharing = "never") {
  let answered = false;
  const calls = mockApi({
    "GET /health": health,
    "GET /datasets/d1": dataset({ types_reviewed_at: "2026-10-13T10:00:00Z" }),
    "POST /datasets/d1/open": dataset(),
    "GET /datasets/d1/columns": [
      column({ name: "Survived", type: "boolean", detected_type: "boolean" }),
      column({ name: "Age", position: 1 }),
    ],
    "GET /column-roles": roles,
    "GET /modules/agent/settings": settings(configured, vendor, sharing),
    "PUT /modules/agent/settings": (body: unknown) => ({ ...settings(configured, vendor, sharing), ...(body as object) }),
    "GET /modules/agent/datasets/d1/personal-columns": [{ name: "email", mask: "email", unmasked: false }],
    "PUT /modules/agent/datasets/d1/personal-columns": (body: unknown) => [
      { name: "email", mask: "email", unmasked: (body as { unmasked: string[] }).unmasked.includes("email") },
    ],
    "GET /modules/agent/settings/models": { models: ["gemini-2.5-flash", "gemini-2.5-pro"], default: "gemini-2.5-pro" },
    "GET /modules/agent/datasets/d1/conversations": [],
    "GET /modules/agent/datasets/d1/notes": () => notes,
    "POST /modules/agent/datasets/d1/notes": (body: unknown) => {
      const note = { id: `n${notes.length + 1}`, text: (body as { text: string }).text, source: "user", created_at: "" };
      notes.push(note);
      return note;
    },
    "POST /modules/agent/datasets/d1/conversations": conversation,
    "GET /modules/agent/conversations/c1": () => ({ conversation, messages: answered ? saved : [] }),
  });
  const json = globalThis.fetch;
  vi.stubGlobal("fetch", async (input: RequestInfo | URL, init?: RequestInit) => {
    if (String(input).endsWith("/conversations/c1/messages")) {
      calls.push({ key: "POST /modules/agent/conversations/c1/messages", body: JSON.parse(String(init?.body)), params: new URLSearchParams() });
      answered = true;
      const body = events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("");
      return new Response(body, { headers: { "Content-Type": "text/event-stream" } });
    }
    return json(input, init);
  });
  return calls;
}

beforeEach(() => {
  localStorage.clear();
  notes = [{ id: "n0", text: "Survived is the target", source: "agent", created_at: "" }];
});
afterEach(() => vi.unstubAllGlobals());

describe("Assistant panel", () => {
  it("asks a suggested question, shows the analysis it ran and the answer", async () => {
    const calls = setup();
    renderAt("/d/d1/columns");

    fireEvent.click(await screen.findByRole("button", { name: "Ask the assistant" }));
    const panel = await screen.findByRole("complementary", { name: "Assistant" });
    fireEvent.click(await within(panel).findByRole("button", { name: "What is most linked to Survived, and how?" }));

    expect(await within(panel).findByText("Sex")).toBeInTheDocument();
    expect(within(panel).getByText("Ranking what is linked to Survived")).toBeInTheDocument();
    expect(within(panel).getByRole("link", { name: /Open the correlations/ })).toHaveAttribute("href", "/d/d1/correlations");
    const sent = calls.find((c) => c.key === "POST /modules/agent/conversations/c1/messages");
    expect(sent?.body).toMatchObject({ text: "What is most linked to Survived, and how?", filters: [], page: "Columns" });
    expect(localStorage.getItem("de.agent.open")).toBe("true");
    expect(await within(panel).findByText("1 figure not found in the analyses: 46.8")).toBeInTheDocument();
  });

  it("explains how to add a key when none is set", async () => {
    setup(false);
    renderAt("/d/d1/columns");
    await screen.findByRole("button", { name: "Ask the assistant" });

    fireEvent.keyDown(window, { key: "j", ctrlKey: true });
    const panel = await screen.findByRole("complementary", { name: "Assistant" });

    expect(await within(panel).findByText("Set up the assistant")).toBeInTheDocument();
    expect(within(panel).getByRole("textbox", { name: "Message to the assistant" })).toBeDisabled();
    fireEvent.click(within(panel).getByRole("button", { name: "Add an API key" }));
    expect(await within(panel).findByLabelText("Anthropic (Claude) key")).toBeInTheDocument();
  });

  it("works with a Gemini key: models listed from the API, tokens instead of dollars", async () => {
    const tokensTurn = [
      { type: "text", text: "Done." },
      { type: "done", end: "answered", message: null, turn_usd: 0, conversation_usd: 0, conversation_tokens: 12_400, priced: false },
    ];
    setup(true, "gemini", tokensTurn);
    renderAt("/d/d1/columns");
    fireEvent.click(await screen.findByRole("button", { name: "Ask the assistant" }));
    const panel = await screen.findByRole("complementary", { name: "Assistant" });

    fireEvent.change(await within(panel).findByRole("textbox", { name: "Message to the assistant" }), {
      target: { value: "Hi" },
    });
    fireEvent.click(within(panel).getByRole("button", { name: "Send" }));
    expect(await within(panel).findByText("12k tokens / 2M")).toBeInTheDocument();

    fireEvent.click(within(panel).getByRole("button", { name: "Assistant settings" }));
    expect(await within(panel).findByRole("option", { name: "Default · gemini-2.5-pro" })).toBeInTheDocument();
    expect(within(panel).getByRole("option", { name: "gemini-2.5-flash" })).toBeInTheDocument();
    expect(within(panel).queryByLabelText("Effort")).not.toBeInTheDocument();
    expect(within(panel).getByText(/each conversation stops at 2M tokens/)).toBeInTheDocument();
  });

  it("lists, adds and attributes the dataset notes", async () => {
    const calls = setup();
    renderAt("/d/d1/columns");
    fireEvent.click(await screen.findByRole("button", { name: "Ask the assistant" }));
    const panel = await screen.findByRole("complementary", { name: "Assistant" });

    fireEvent.click(await within(panel).findByRole("button", { name: "Dataset notes" }));
    const list = await within(panel).findByRole("list", { name: "Dataset notes" });
    expect(within(list).getByText("Survived is the target")).toBeInTheDocument();
    expect(within(list).getByLabelText("Saved by the assistant")).toBeInTheDocument();

    fireEvent.change(within(panel).getByRole("textbox", { name: "New note" }), { target: { value: "Fare is in pounds" } });
    fireEvent.click(within(panel).getByRole("button", { name: "Add" }));

    expect(await within(panel).findByText("Fare is in pounds")).toBeInTheDocument();
    expect(calls.find((c) => c.key === "POST /modules/agent/datasets/d1/notes")?.body).toEqual({ text: "Fare is in pounds" });
  });

  it("stays out of the way outside a dataset", async () => {
    mockApi({ "GET /health": health, "GET /datasets": [], "GET /samples": [] });
    renderAt("/");

    await screen.findAllByRole("heading");
    expect(screen.queryByRole("button", { name: "Ask the assistant" })).not.toBeInTheDocument();
  });
});

describe("Assistant confirmations", () => {
  it("asks before a change and sends the user's answer", async () => {
    const calls = setup();
    let push: ((event: object) => void) | null = null;
    let close: () => void = () => {};
    const json = globalThis.fetch;
    vi.stubGlobal("fetch", async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/conversations/c1/messages")) {
        const encoder = new TextEncoder();
        const body = new ReadableStream({
          start(controller) {
            push = (event) => controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
            close = () => controller.close();
          },
        });
        return new Response(body, { headers: { "Content-Type": "text/event-stream" } });
      }
      if (url.endsWith("/conversations/c1/confirm/s1")) {
        calls.push({ key: "POST confirm", body: JSON.parse(String(init?.body)), params: new URLSearchParams() });
        return new Response(null, { status: 204 });
      }
      return json(input, init);
    });
    renderAt("/d/d1/columns");
    fireEvent.click(await screen.findByRole("button", { name: "Ask the assistant" }));
    const panel = await screen.findByRole("complementary", { name: "Assistant" });
    fireEvent.change(await within(panel).findByRole("textbox", { name: "Message to the assistant" }), {
      target: { value: "Save a revenue KPI" },
    });
    fireEvent.click(within(panel).getByRole("button", { name: "Send" }));

    await vi.waitFor(() => expect(push).not.toBeNull());
    const send = (event: object) => push?.(event);
    send({ type: "step_started", id: "s1", tool: "save_kpi", label: "Saving the KPI", input: {} });
    send({ type: "confirm", id: "s1", tool: "save_kpi", text: "Save the KPI “Revenue” in the KPI builder." });

    const card = await within(panel).findByRole("group", { name: "The assistant asks before changing something" });
    expect(within(card).getByText("Save the KPI “Revenue” in the KPI builder.")).toBeInTheDocument();
    fireEvent.click(within(card).getByRole("button", { name: "Confirm" }));

    await vi.waitFor(() => expect(calls.find((c) => c.key === "POST confirm")?.body).toEqual({ approved: true }));
    send({ type: "done", end: "answered", message: null, turn_usd: 0.01, conversation_usd: 0.01, conversation_tokens: 900, priced: true });
    close();
  });
});

describe("Assistant and rows (R-AG-P2)", () => {
  it("asks before sharing rows and sends the columns the user lets through unmasked", async () => {
    const calls = setup(true, "anthropic", turn, "ask");
    let push: ((event: object) => void) | null = null;
    let close: () => void = () => {};
    const json = globalThis.fetch;
    vi.stubGlobal("fetch", async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/conversations/c1/messages")) {
        const encoder = new TextEncoder();
        const body = new ReadableStream({
          start(controller) {
            push = (event) => controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
            close = () => controller.close();
          },
        });
        return new Response(body, { headers: { "Content-Type": "text/event-stream" } });
      }
      if (url.endsWith("/conversations/c1/confirm/s1")) {
        calls.push({ key: "POST confirm", body: JSON.parse(String(init?.body)), params: new URLSearchParams() });
        return new Response(null, { status: 204 });
      }
      return json(input, init);
    });
    renderAt("/d/d1/columns");
    fireEvent.click(await screen.findByRole("button", { name: "Ask the assistant" }));
    const panel = await screen.findByRole("complementary", { name: "Assistant" });
    // the empty conversation says rows can leave, on the user's card
    expect(await within(panel).findByText(/also up to 20 rows at a time/)).toBeInTheDocument();
    fireEvent.change(within(panel).getByRole("textbox", { name: "Message to the assistant" }), {
      target: { value: "Show me some e-mails" },
    });
    fireEvent.click(within(panel).getByRole("button", { name: "Send" }));

    await vi.waitFor(() => expect(push).not.toBeNull());
    const send = (event: object) => push?.(event);
    send({ type: "step_started", id: "s1", tool: "sample_rows", label: "Reading 5 rows", input: {} });
    send({
      type: "confirm",
      id: "s1",
      tool: "sample_rows",
      text: "Share up to 5 rows of email, city with the assistant.",
      rows: { count: 5, columns: ["email", "city"], masked: [{ name: "email", mask: "email" }], filtered: true },
    });

    const card = await within(panel).findByRole("group", { name: "The assistant asks to see rows" });
    expect(within(card).getByText(/Up to 5 rows with these columns: email, city\. From the filtered rows only\./)).toBeInTheDocument();
    fireEvent.click(within(card).getByRole("checkbox", { name: "email (e-mail)" }));
    fireEvent.click(within(card).getByRole("button", { name: "Allow once" }));

    await vi.waitFor(() =>
      expect(calls.find((c) => c.key === "POST confirm")?.body).toEqual({ approved: true, unmask: ["email"] }),
    );
    expect(within(panel).queryByRole("group", { name: "The assistant asks to see rows" })).not.toBeInTheDocument();
    send({ type: "done", end: "answered", message: null, turn_usd: 0.01, conversation_usd: 0.01, conversation_tokens: 900, priced: true });
    close();
  });

  it("sets the sharing level and the personal columns shared unmasked in the settings", async () => {
    const calls = setup(true, "anthropic", turn, "ask");
    renderAt("/d/d1/columns");
    fireEvent.click(await screen.findByRole("button", { name: "Ask the assistant" }));
    const panel = await screen.findByRole("complementary", { name: "Assistant" });
    fireEvent.click(await within(panel).findByRole("button", { name: "Assistant settings" }));

    expect(await within(panel).findByText(/With row sharing on/)).toBeInTheDocument();
    expect(within(panel).getByText(/the assistant asks first/)).toBeInTheDocument();
    fireEvent.click(await within(panel).findByRole("checkbox", { name: "email (e-mail)" }));
    await vi.waitFor(() =>
      expect(calls.find((c) => c.key === "PUT /modules/agent/datasets/d1/personal-columns")?.body).toEqual({ unmasked: ["email"] }),
    );
    fireEvent.change(within(panel).getByRole("combobox", { name: "Sharing rows" }), { target: { value: "never" } });
    await vi.waitFor(() =>
      expect(calls.find((c) => c.key === "PUT /modules/agent/settings")?.body).toEqual({ rows_sharing: "never" }),
    );
    expect(await within(panel).findByText(/never sees a row of your data/)).toBeInTheDocument();
    expect(within(panel).getByText(/Never whole rows/)).toBeInTheDocument();
  });
});
