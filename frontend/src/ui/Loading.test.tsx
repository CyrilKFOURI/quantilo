import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ActivityBar } from "./ActivityBar";
import { Loading } from "./Loading";

describe("Loading (042)", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("says what is computed, then how long it has taken, then that it is still working", () => {
    render(<Loading label="Computing the trend…" />);
    const status = screen.getByRole("status");
    expect(status).toHaveAttribute("aria-busy", "true");
    expect(status).toHaveTextContent("Computing the trend…");
    expect(status).not.toHaveTextContent(/\d s/);

    act(() => vi.advanceTimersByTime(3_000));
    expect(status).toHaveTextContent("3 s");
    expect(status).not.toHaveTextContent(/Still working/);

    act(() => vi.advanceTimersByTime(5_000));
    expect(status).toHaveTextContent(/Still working/);
  });

  it("keeps a small spot to the label and the time", () => {
    render(<Loading label="Drawing the chart…" compact />);
    act(() => vi.advanceTimersByTime(10_000));
    expect(screen.getByRole("status")).not.toHaveTextContent(/Still working/);
  });
});

describe("ActivityBar (042)", () => {
  it("shows after a short wait for data the page does not have yet, and goes when it arrives", async () => {
    const client = new QueryClient();
    let answer: (value: number) => void = () => {};
    render(
      <QueryClientProvider client={client}>
        <ActivityBar />
      </QueryClientProvider>,
    );
    act(() => {
      void client.fetchQuery({ queryKey: ["slow"], queryFn: () => new Promise<number>((resolve) => (answer = resolve)) });
    });
    await act(() => new Promise((resolve) => setTimeout(resolve, 100)));
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
    expect(await screen.findByRole("progressbar")).toBeInTheDocument();

    act(() => answer(1));
    await waitFor(() => expect(screen.queryByRole("progressbar")).not.toBeInTheDocument());
  });

  it("stays off while data the page already has is checked again", async () => {
    const client = new QueryClient();
    client.setQueryData(["health"], { ok: true });
    render(
      <QueryClientProvider client={client}>
        <ActivityBar />
      </QueryClientProvider>,
    );
    act(() => {
      void client.fetchQuery({ queryKey: ["health"], queryFn: () => new Promise(() => {}), staleTime: 0 });
    });
    await act(() => new Promise((resolve) => setTimeout(resolve, 500)));
    expect(client.isFetching()).toBe(1);
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
  });
});
