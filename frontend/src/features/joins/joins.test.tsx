import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { MlPreparation } from "../../api/client";
import { TextRepresentation } from "../../modules/ml/TextRepresentation";
import { keyLabel } from "./JoinDialog";

describe("089 join keys", () => {
  it("names the columns, the cardinality and the share found", () => {
    const text = keyLabel({
      left_columns: ["customer_id"],
      right_columns: ["customer_id"],
      mode: "lookup",
      cardinality: "n:1",
      confidence: "high",
      matched_share: 0.983,
      reasons: [],
    });
    expect(text).toBe("customer_id = customer_id · n:1 · 98 % found");
  });
});

describe("091 text representation", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("greys sentence embeddings with the reason when the extra is missing", async () => {
    const reason = "Needs the optional extra `embed`.";
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(
          JSON.stringify({ available: false, installed: false, downloaded: false, reason, model: "m", size: "0.5 GB", license: "MIT", install: "", download: "idle", error: null }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      ),
    );
    const preparation = {
      decisions: [{ column: "review", type: "text", use: true, encoding: "words", options: { encoding: [{ value: "words", reason: null }, { value: "embeddings", reason }] } }],
    } as unknown as MlPreparation;
    render(
      <QueryClientProvider client={new QueryClient()}>
        <TextRepresentation preparation={preparation} onColumn={() => {}} />
      </QueryClientProvider>,
    );
    const option = screen.getByRole("option", { name: /Sentence embeddings/ }) as HTMLOptionElement;
    expect(option.disabled).toBe(true);
    expect(option.textContent).toContain(reason);
    expect(await screen.findByText(reason)).toBeTruthy();
  });
});
