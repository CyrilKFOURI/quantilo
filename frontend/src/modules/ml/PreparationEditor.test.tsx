import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { MlPreparation } from "../../api/client";
import { AdvancedEncodings } from "./PreparationEditor";

const temp = {
  column: "temp",
  type: "numeric_continuous",
  use: true,
  transform: null,
  options: {
    transform: [
      { value: "none", reason: null },
      { value: "cycle", reason: "Only for a column that goes round." },
      { value: "past_mean", reason: null },
      { value: "past_ewm", reason: null },
    ],
  },
};

const preparation = (reason: string | null = null) =>
  ({
    decisions: [
      {
        ...temp,
        options: { transform: temp.options.transform.map((o) => (o.value.startsWith("past") ? { ...o, reason } : o)) },
      },
    ],
    folds: { kind: "periods", column: "era", why: "Target encoding cross-fitted on 5 blocks of consecutive whole periods of era." },
  }) as unknown as MlPreparation;

describe("Advanced › More: causal lags (063)", () => {
  it("keeps the lags folded twice, then sets one for a column", () => {
    const onColumn = vi.fn();
    render(<AdvancedEncodings preparation={preparation()} onColumn={onColumn} />);

    expect(screen.queryByLabelText("Past values — temp")).toBeNull(); // not rendered while folded
    fireEvent.click(screen.getByText("Advanced"));
    expect(screen.getByText(/blocks of consecutive whole periods of era/)).toBeInTheDocument();
    const more = screen.getByText("More");
    (more.parentElement as HTMLDetailsElement).open = true;
    fireEvent(more.parentElement as HTMLDetailsElement, new Event("toggle"));

    fireEvent.change(screen.getByLabelText("Past values — temp"), { target: { value: "past_mean" } });
    expect(onColumn).toHaveBeenCalledWith("temp", "transform", "past_mean");
  });

  it("says why lags are impossible without ordered rows", () => {
    render(<AdvancedEncodings preparation={preparation("Needs ordered rows: choose panel walk-forward.")} onColumn={vi.fn()} />);
    const more = screen.getByText("More");
    (more.parentElement as HTMLDetailsElement).open = true;
    fireEvent(more.parentElement as HTMLDetailsElement, new Event("toggle"));

    expect(screen.getByText(/Needs ordered rows/)).toBeInTheDocument();
    expect(screen.getByLabelText("Past values — temp")).toBeDisabled();
  });
});
